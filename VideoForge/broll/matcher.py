import logging
import re
import string
from typing import Dict, List

import numpy as np

from VideoForge.adapters.embedding_adapter import encode_text_clip
from VideoForge.broll.db import LibraryDB
from VideoForge.broll.keyword_tools import translate_keywords
from VideoForge.config.config_manager import Config


DEFAULT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
}


class QueryGenerator:
    def __init__(self, stopwords: set[str] | None = None, top_n: int = 6) -> None:
        self.stopwords = stopwords or DEFAULT_STOPWORDS
        self.top_n = top_n

    def generate(self, sentence_text: str) -> Dict[str, str]:
        fallback = sentence_text.strip()
        tokens = [t for t in sentence_text.split() if t.strip()]
        cleaned = []
        for token in tokens:
            token = token.strip(string.punctuation)
            token = self._collapse_repeats(token)
            token = token.lower()
            if not token:
                continue
            if token in self.stopwords:
                continue
            cleaned.append(token)

        scored = [(self._score_token(tok), tok) for tok in cleaned]
        scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        primary_tokens = [tok for _score, tok in scored[: self.top_n]]
        translated_tokens = translate_keywords(primary_tokens)
        if translated_tokens:
            seen = set()
            merged: List[str] = []
            for tok in translated_tokens + primary_tokens:
                if tok in seen:
                    continue
                seen.add(tok)
                merged.append(tok)
            primary_tokens = merged
        primary = " ".join(primary_tokens).strip()

        boost_tokens = [tok for tok in cleaned if self._is_boost(tok)]
        boost = " ".join(sorted(set(boost_tokens)))

        if not primary:
            primary = fallback
        return {"primary": primary, "fallback": fallback, "boost": boost}

    @staticmethod
    def _collapse_repeats(token: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1\1", token)

    @staticmethod
    def _score_token(token: str) -> int:
        score = 0
        if len(token) >= 2:
            score += 1
        if any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token):
            score += 1
        if QueryGenerator._has_camelcase(token):
            score += 1
        if re.search(r"[a-zA-Z]", token):
            score += 1
        return score

    @staticmethod
    def _has_camelcase(token: str) -> bool:
        return bool(re.search(r"[a-z][A-Z]", token) or re.search(r"^[A-Z][a-z]", token))

    @staticmethod
    def _is_boost(token: str) -> bool:
        return bool(
            re.search(r"[a-zA-Z]", token)
            or re.search(r"\d", token)
            or QueryGenerator._has_camelcase(token)
        )


class CooldownManager:
    def __init__(self, db: LibraryDB, cooldown_sec: float, penalty: float) -> None:
        self.db = db
        self.cooldown_sec = cooldown_sec
        self.penalty = penalty

    def get_recent_ids(self, sentence_t0: float, recent: List[Dict[str, float]]) -> List[str]:
        return [
            r["clip_id"]
            for r in recent
            if sentence_t0 - float(r["t0"]) <= self.cooldown_sec
        ]

    def apply_penalty(self, recent_ids: List[str]) -> None:
        if recent_ids:
            self.db.set_cooldown(recent_ids, penalty=self.penalty)


class VisualSimilarityFilter:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.recent_embeddings: List[np.ndarray] = []

    def is_too_similar(self, candidate_emb: np.ndarray | None) -> bool:
        if candidate_emb is None:
            return False
        return _is_visually_similar(candidate_emb, self.recent_embeddings, threshold=self.threshold)

    def remember(self, candidate_emb: np.ndarray | None) -> None:
        if candidate_emb is not None:
            self.recent_embeddings.append(candidate_emb)


class ScriptIndexSelector:
    def __init__(self, db: LibraryDB, visual_filter: VisualSimilarityFilter, threshold: float) -> None:
        self.db = db
        self.visual_filter = visual_filter
        self.threshold = threshold

    def select(
        self,
        results: List[Dict],
        recent_ids: List[str],
        script_index: str,
        clip_map: Dict[str, Dict],
        embedding_map: Dict[str, np.ndarray | None],
    ) -> Dict | None:
        candidates = []
        for result in results:
            if result["clip_id"] in recent_ids or result["score"] < self.threshold:
                continue
            clip = clip_map.get(result["clip_id"]) or {}
            if not BrollMatcher._clip_has_script_index(clip, script_index):
                continue
            candidate_emb = embedding_map.get(result["clip_id"])
            if self.visual_filter.is_too_similar(candidate_emb):
                continue
            self.visual_filter.remember(candidate_emb)
            candidates.append((result, clip))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0]["score"], reverse=True)
        return candidates[0][0]


class BrollMatcher:
    def __init__(self, db: LibraryDB | None, settings: Dict[str, Dict[str, float]]) -> None:
        if db is None:
            global_db_path = Config.get("global_library_db_path", "")
            local_db_path = Config.get("local_library_db_path", "")
            db = LibraryDB(
                global_db_path=global_db_path or None,
                local_db_path=local_db_path or None,
            )
        self.db = db
        self.settings = settings
        self.query_generator = QueryGenerator(top_n=settings["query"]["top_n_tokens"])
        self.logger = logging.getLogger(__name__)

    def match(self, sentences: List[Dict]) -> List[Dict]:
        matches: List[Dict] = []
        cooldown = float(self.settings["broll"]["cooldown_sec"])
        threshold = float(self.settings["broll"]["matching_threshold"])
        visual_threshold = float(self.settings["broll"].get("visual_similarity_threshold", 0.85))
        cooldown_penalty = float(self.settings["broll"].get("cooldown_penalty", 0.4))
        recent: List[Dict[str, float]] = []
        cooldown_manager = CooldownManager(self.db, cooldown, cooldown_penalty)
        visual_filter = VisualSimilarityFilter(visual_threshold)
        script_selector = ScriptIndexSelector(self.db, visual_filter, threshold)
        embedding_cache: Dict[str, np.ndarray] = {}

        for idx, sentence in enumerate(sorted(sentences, key=lambda s: s["t0"]), start=1):
            script_index = self._extract_script_index(sentence, idx)
            query = self.query_generator.generate(sentence["text"])
            query_text = query["primary"]
            embedding = embedding_cache.get(query_text)
            if embedding is None:
                embedding = encode_text_clip(query_text).astype(np.float32)
                embedding_cache[query_text] = embedding

            recent_ids = cooldown_manager.get_recent_ids(float(sentence["t0"]), recent)
            cooldown_manager.apply_penalty(recent_ids)

            weights = self.settings["broll"].get("hybrid_weights")
            results = self.db.hybrid_search(
                query["primary"], embedding, limit=5, weights=weights
            )
            clip_ids = [r["clip_id"] for r in results if r.get("clip_id")]
            clip_map = self.db.get_clips_by_ids(clip_ids)
            embedding_map = self.db.get_embeddings_by_ids(clip_ids)
            selected = None
            if script_index:
                selected = script_selector.select(
                    results, recent_ids, script_index, clip_map, embedding_map
                )
            for result in results:
                if result["clip_id"] not in recent_ids and result["score"] >= threshold:
                    candidate_emb = embedding_map.get(result["clip_id"])
                    if visual_filter.is_too_similar(candidate_emb):
                        continue
                    selected = result
                    visual_filter.remember(candidate_emb)
                    break

            if not selected:
                continue

            clip = clip_map.get(selected["clip_id"]) or self.db.get_clip(selected["clip_id"]) or {}
            match = {
                "sentence_id": sentence["id"],
                "clip_id": selected["clip_id"],
                "score": selected["score"],
                "metadata": {
                    "sentence": sentence["text"],
                    "query": query,
                    "clip": clip,
                    "script_index": script_index,
                    "script_match": bool(script_index and self._clip_has_script_index(clip, script_index)),
                    "explain": selected.get("explain", {}),
                },
            }
            matches.append(match)
            recent.append({"clip_id": selected["clip_id"], "t0": float(sentence["t0"])})

        return matches

    @staticmethod
    def _extract_script_index(sentence: Dict, fallback_index: int) -> str | None:
        if "script_index" in sentence and sentence["script_index"]:
            return str(sentence["script_index"])
        metadata = sentence.get("metadata") or {}
        if metadata.get("script_index"):
            return str(metadata["script_index"])
        return str(fallback_index) if fallback_index > 0 else None

    @staticmethod
    def _clip_has_script_index(clip: Dict, script_index: str) -> bool:
        folder_tags = str(clip.get("folder_tags") or "")
        return f"script {script_index}".lower() in folder_tags.lower()

def _is_visually_similar(
    new_embedding: np.ndarray, recent_embeddings: List[np.ndarray], threshold: float = 0.85
) -> bool:
    if not recent_embeddings:
        return False
    new_embedding = new_embedding.astype(np.float32)
    for emb in recent_embeddings:
        score = float(np.dot(new_embedding, emb))
        if score >= threshold:
            return True
    return False

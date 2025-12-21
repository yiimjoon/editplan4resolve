import re
import string
from typing import Dict, List

import numpy as np

from VideoForge.adapters.embedding_adapter import encode_text_clip
from VideoForge.broll.db import LibraryDB


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


class BrollMatcher:
    def __init__(self, db: LibraryDB, settings: Dict[str, Dict[str, float]]) -> None:
        self.db = db
        self.settings = settings
        self.query_generator = QueryGenerator(top_n=settings["query"]["top_n_tokens"])

    def match(self, sentences: List[Dict]) -> List[Dict]:
        matches: List[Dict] = []
        cooldown = float(self.settings["broll"]["cooldown_sec"])
        threshold = float(self.settings["broll"]["matching_threshold"])
        recent: List[Dict[str, float]] = []

        for sentence in sorted(sentences, key=lambda s: s["t0"]):
            query = self.query_generator.generate(sentence["text"])
            embedding = encode_text_clip(query["primary"]).astype(np.float32)

            recent_ids = [r["clip_id"] for r in recent if sentence["t0"] - r["t0"] <= cooldown]
            self.db.set_cooldown(recent_ids)

            results = self.db.hybrid_search(query["primary"], embedding, limit=5)
            selected = None
            for result in results:
                if result["clip_id"] not in recent_ids and result["score"] >= threshold:
                    selected = result
                    break

            if not selected:
                continue

            clip = self.db.get_clip(selected["clip_id"]) or {}
            match = {
                "sentence_id": sentence["id"],
                "clip_id": selected["clip_id"],
                "score": selected["score"],
                "metadata": {
                    "sentence": sentence["text"],
                    "query": query,
                    "clip": clip,
                    "explain": selected.get("explain", {}),
                },
            }
            matches.append(match)
            recent.append({"clip_id": selected["clip_id"], "t0": float(sentence["t0"])})

        return matches

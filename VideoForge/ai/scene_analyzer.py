"""LLM-based scene analysis for B-roll generation."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from VideoForge.ai.llm_hooks import (
    _call_gemini,
    _extract_json,
    get_llm_provider,
    is_llm_enabled,
)
from VideoForge.broll.keyword_tools import translate_keywords

logger = logging.getLogger("VideoForge.ai.scene")

KOREAN_PARTICLES = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "에서",
    "의",
    "와",
    "과",
    "도",
    "만",
    "으로",
    "로",
    "까지",
    "부터",
    "보다",
    "께서",
    "에게",
    "에게서",
    "조차",
    "마저",
    "밖에",
    "라도",
    "야",
    "아",
    "여",
    "하고",
    "랑",
    "이랑",
}

KOREAN_STOPWORDS = {
    "그래서",
    "하지만",
    "그리고",
    "그러나",
    "그런데",
    "근데",
    "그러니까",
    "그렇죠",
    "그렇습니다",
    "이제",
    "정말",
    "사실",
    "그냥",
    "진짜",
    "조금",
    "좀",
    "아직",
    "초보",
    "죄송",
    "죄송합니다",
    "자랑",
    "아니지만",
    "하필이면",
    "수입",
    "수입도",
    "없어요",
    "없습니다",
    "없다",
    "있어요",
    "있습니다",
    "있다",
    "합니다",
    "했습니다",
    "했어요",
    "했죠",
    "나니",
    "지금",
    "매일",
    "분의",
    "또",
    "또는",
    "아니지만",
    "하필이면",
    "제가",
    "저는",
    "나는",
    "우리는",
    "당신은",
    "여러분",
    "우리",
    "저희",
    "그거",
    "이거",
    "저거",
    "그것",
    "이것",
    "저것",
    "그런",
    "이런",
    "저런",
    "이렇게",
    "저렇게",
}

KOREAN_SUFFIXES = (
    "입니다",
    "이다",
    "이라",
    "인데",
    "지만",
    "합니다",
    "했습니다",
    "했어요",
    "했죠",
    "했는데",
    "했지만",
    "습니다",
    "었어요",
    "아요",
    "어요",
    "했다",
    "했다",
    "한다",
    "하는",
    "하며",
    "해서",
    "하고",
    "하면",
)
EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "didn",
    "do",
    "does",
    "doesn",
    "doing",
    "don",
    "down",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "im",
    "in",
    "into",
    "is",
    "isn",
    "it",
    "its",
    "just",
    "may",
    "me",
    "mean",
    "might",
    "mightn",
    "more",
    "most",
    "my",
    "mine",
    "must",
    "mustn",
    "no",
    "not",
    "of",
    "on",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "out",
    "over",
    "really",
    "s",
    "same",
    "she",
    "should",
    "shouldn",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "under",
    "up",
    "very",
    "was",
    "wasn",
    "we",
    "were",
    "weren",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "without",
    "would",
    "wouldn",
    "you",
    "your",
    "yours",
}


class SceneAnalyzer:
    """Analyze sentences and generate visual scene descriptions for B-roll."""

    def __init__(self) -> None:
        self.provider = get_llm_provider()

    def analyze_sentences(
        self,
        sentences: List[Dict[str, Any]],
        max_scenes: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentences and generate B-roll scene descriptions.

        Args:
            sentences: List of sentence dicts with "id", "text", "uid"

        Returns:
            List of scene descriptions:
            [
                {
                    "sentence_id": 1,
                    "uid": "abc123",
                    "scene_description": "coffee beans close-up, brown tones, vintage",
                    "visual_tags": ["coffee", "beans", "brown", "vintage"],
                    "camera": "close-up",
                    "mood": "nostalgic",
                    "subject": "coffee beans",
                    "environment": "studio",
                    "lighting": "soft",
                }
            ]
        """
        if max_scenes is None or max_scenes <= 0:
            from VideoForge.config.config_manager import Config

            try:
                max_scenes = int(Config.get("broll_max_scenes") or 15)
            except (TypeError, ValueError):
                max_scenes = 15

        if is_llm_enabled():
            try:
                scenes = self._analyze_with_llm(sentences, max_scenes)
                if scenes:
                    logger.info("LLM selected %d scenes", len(scenes))
                    return scenes
                logger.warning("LLM returned 0 scenes, falling back to heuristic")
            except Exception as exc:
                logger.error("LLM analysis failed: %s", exc)
        else:
            logger.info("LLM disabled, using heuristic scene selection")

        return self._heuristic_select_scenes(sentences, max_scenes)

    def select_scenes(
        self,
        sentences: List[Dict[str, Any]],
        max_scenes: int,
    ) -> List[Dict[str, Any]]:
        """
        Select only the most visually relevant sentences for B-roll.

        Args:
            sentences: Full transcript sentences
            max_scenes: Maximum number of scenes to return
        """
        if not sentences or max_scenes <= 0:
            return []

        if not is_llm_enabled():
            logger.warning("LLM not enabled. Selecting scenes via heuristic fallback.")
            return self._heuristic_select_scenes(sentences, max_scenes)

        try:
            scenes = self._select_with_llm(sentences, max_scenes)
            if scenes:
                return scenes
            logger.warning("LLM returned 0 scenes, falling back to heuristic.")
            return self._heuristic_select_scenes(sentences, max_scenes)
        except Exception as exc:
            logger.error("LLM scene selection failed: %s", exc)
            return self._heuristic_select_scenes(sentences, max_scenes)

    def _analyze_with_llm(
        self, sentences: List[Dict[str, Any]], max_scenes: int
    ) -> List[Dict[str, Any]]:
        """Use Gemini to analyze scenes."""
        prompt = self._build_prompt(sentences, max_scenes)
        response = _call_gemini(prompt)
        scenes = self._parse_response(
            response, sentences, merge_with_sentences=False, prompt_name="analysis"
        )
        if not scenes:
            retry_prompt = self._build_selection_prompt(sentences, max_scenes)
            retry_response = _call_gemini(retry_prompt)
            scenes = self._parse_response(
                retry_response,
                sentences,
                merge_with_sentences=False,
                prompt_name="analysis_retry",
            )
        if len(scenes) > max_scenes:
            scenes = scenes[:max_scenes]
        return scenes

    def _build_prompt(self, sentences: List[Dict[str, Any]], max_scenes: int) -> str:
        """Build Gemini prompt for scene analysis."""
        sentences_text = "\n".join(
            [f"{i+1}. {s.get('text', '')}" for i, s in enumerate(sentences)]
        )

        prompt = f"""You are a professional cinematographer selecting B-roll moments.
Pick up to {max_scenes} sentences that benefit most from visual footage and provide scene descriptions.

For each sentence, provide:
- scene_description: Detailed visual description (1-2 sentences, cinematic style)
- visual_tags: 3-5 short keywords for stock/AI search (nouns only, 1-2 words each)
- camera: Shot type (wide, medium, close-up, extreme close-up, etc.)
- mood: Emotional tone (professional, nostalgic, energetic, calm, etc.)
- subject: Main visual subject
- environment: Setting/location type
- lighting: Lighting style (natural, studio, soft, dramatic, etc.)

Keyword rules:
- Prefer concrete objects, actions, or places (visual nouns).
- No filler words or particles.
- If the transcript is Korean, translate keywords to English for stock search.
- Return at least 1 scene if any sentence has visualizable content.

Transcript (sentence_id. text):
{sentences_text}

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "scenes": [
    {{
      "sentence_id": 1,
      "scene_description": "Coffee beans scattered on rustic wooden table, shallow depth of field, warm brown tones, vintage aesthetic",
      "visual_tags": ["coffee", "beans", "rustic", "wooden", "vintage"],
      "camera": "close-up",
      "mood": "nostalgic",
      "subject": "coffee beans",
      "environment": "studio",
      "lighting": "soft natural"
    }}
  ]
}}"""
        return prompt

    def _build_selection_prompt(
        self, sentences: List[Dict[str, Any]], max_scenes: int
    ) -> str:
        """Build Gemini prompt for selecting the most visual sentences."""
        sentences_text = "\n".join(
            [f"{s.get('id')}. {s.get('text', '')}" for s in sentences]
        )

        prompt = f"""You are a professional cinematographer and keyword extractor.
Pick up to {max_scenes} sentences that benefit most from visual footage, then provide stock-search keywords.

Rules:
- Prefer concrete objects, actions, places, or key concepts.
- Skip greetings, fillers, and transitions.
- Return fewer than {max_scenes} only if the transcript has almost no visual content.
- visual_tags must be short nouns (1-2 words each), no particles or full sentences.
- If transcript is Korean, translate visual_tags to English for stock search.

Transcript (sentence_id. text):
{sentences_text}

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "scenes": [
    {{
      "sentence_id": 3,
      "scene_description": "Close-up of coffee beans on rustic table",
      "visual_tags": ["coffee", "beans", "rustic"],
      "camera": "close-up",
      "mood": "warm",
      "subject": "coffee beans",
      "environment": "studio",
      "lighting": "soft"
    }}
  ]
}}"""
        return prompt

    def _select_with_llm(
        self, sentences: List[Dict[str, Any]], max_scenes: int
    ) -> List[Dict[str, Any]]:
        """Use Gemini to select a subset of sentences for B-roll."""
        prompt = self._build_selection_prompt(sentences, max_scenes)
        response = _call_gemini(prompt)
        scenes = self._parse_response(
            response, sentences, merge_with_sentences=False, prompt_name="selection"
        )
        if scenes:
            return scenes
        logger.warning("Selection prompt returned 0 scenes, retrying analysis prompt.")
        return self._analyze_with_llm(sentences, max_scenes)

    def _parse_response(
        self,
        response: str,
        sentences: List[Dict[str, Any]],
        merge_with_sentences: bool = True,
        prompt_name: str = "analysis",
    ) -> List[Dict[str, Any]]:
        """Parse LLM JSON response."""
        data: Any = None
        try:
            data = _extract_json(response)
        except Exception as exc:
            data = self._extract_json_list(response)
            if data is None:
                logger.warning("Failed to extract JSON from LLM response: %s", exc)
                self._dump_llm_response(response, f"{prompt_name}_json_error")
                return self._fallback_analysis(sentences) if merge_with_sentences else []

        if isinstance(data, list):
            scenes = data
        elif isinstance(data, dict):
            scenes = data.get("scenes") or data.get("data") or []
        else:
            scenes = []
        if not scenes:
            logger.warning("LLM returned no scenes")
            self._dump_llm_response(response, f"{prompt_name}_empty")
            return self._fallback_analysis(sentences) if merge_with_sentences else []

        if not merge_with_sentences:
            return self._sanitize_scene_list(scenes, sentences)

        scene_map = {s.get("sentence_id"): s for s in scenes if s.get("sentence_id") is not None}
        results = []
        for sentence in sentences:
            scene = scene_map.get(sentence.get("id"), {})
            results.append(self._build_scene_entry(sentence, scene))
        return results

    @staticmethod
    def _extract_json_list(text: str) -> List[Dict[str, Any]] | None:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None

    @staticmethod
    def _dump_llm_response(response: str, reason: str) -> None:
        appdata = os.environ.get("APPDATA")
        if appdata:
            log_dir = (
                Path(appdata)
                / "Blackmagic Design"
                / "DaVinci Resolve"
                / "Support"
                / "VideoForge"
                / "logs"
            )
        else:
            log_dir = Path(__file__).resolve().parents[1] / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "scene_llm_debug.log"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snippet = response or ""
        if len(snippet) > 4000:
            snippet = snippet[:4000] + "\n... (truncated) ..."
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n--- {stamp} reason={reason} ---\n")
            handle.write(snippet)
            handle.write("\n")

    def _fallback_analysis(
        self, sentences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback: basic analysis without LLM."""
        results = []
        for sentence in sentences:
            text = sentence.get("text", "")
            words = self._extract_keywords(text, limit=5)
            results.append(self._build_scene_entry(sentence, {"visual_tags": words}))
        return results

    def _heuristic_select_scenes(
        self, sentences: List[Dict[str, Any]], max_scenes: int
    ) -> List[Dict[str, Any]]:
        """Heuristic selection when LLM is disabled or fails."""
        scored: List[Dict[str, Any]] = []
        total_sentences = len(sentences)
        bucket_size = max(1, (total_sentences + max_scenes - 1) // max_scenes)
        for position, sentence in enumerate(sentences):
            text = str(sentence.get("text", "")).strip()
            if not text:
                continue

            words = [w.strip() for w in text.split() if w.strip()]
            word_count = len(words)
            if word_count < 5:
                continue

            nouns = []
            for word in words:
                token = self._normalize_token(word)
                if not token:
                    continue
                if not self._is_candidate_noun(token):
                    continue
                nouns.append(token)

            noun_count = len(nouns)
            score = noun_count * 2.0 + word_count * 0.5
            keywords = self._extract_keywords(text, limit=5)
            if not keywords:
                fallback_tokens = []
                for word in words:
                    token = self._normalize_token(word)
                    if not token:
                        continue
                    if token in KOREAN_PARTICLES:
                        continue
                    if token in KOREAN_STOPWORDS or token in EN_STOPWORDS:
                        continue
                    fallback_tokens.append(token)
                fallback_tokens = self._dedupe_preserve_order(fallback_tokens)
                keywords = fallback_tokens[:3]
            keywords = self._dedupe_preserve_order(keywords)
            translated = translate_keywords(keywords)
            if translated:
                keywords = self._dedupe_preserve_order(translated + keywords)
            bucket = min(position // bucket_size, max_scenes - 1)
            scored.append(
                {
                    "sentence": sentence,
                    "score": score,
                    "keywords": keywords,
                    "noun_count": noun_count,
                    "word_count": word_count,
                    "position": position,
                    "bucket": bucket,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        bucketed: Dict[int, List[Dict[str, Any]]] = {}
        for item in scored:
            bucketed.setdefault(int(item["bucket"]), []).append(item)

        selected: List[Dict[str, Any]] = []
        selected_keys = set()
        for bucket in range(max_scenes):
            items = bucketed.get(bucket, [])
            if not items:
                continue
            best = max(items, key=lambda item: item["score"])
            key = best["sentence"].get("id") or best["position"]
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(best)

        if len(selected) < max_scenes:
            for item in scored:
                key = item["sentence"].get("id") or item["position"]
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                selected.append(item)
                if len(selected) >= max_scenes:
                    break

        selected.sort(key=lambda item: item["position"])
        results = []
        for item in selected:
            sentence = item["sentence"]
            keywords = item["keywords"]
            results.append(
                {
                    "sentence_id": sentence.get("id"),
                    "uid": sentence.get("metadata", {}).get("uid") or sentence.get("uid", ""),
                    "keywords": keywords,
                    "reason": f"heuristic (nouns={item['noun_count']}, words={item['word_count']})",
                    "scene_description": sentence.get("text", ""),
                    "visual_tags": keywords,
                    "camera": "medium shot",
                    "mood": "neutral",
                    "subject": keywords[0] if keywords else "",
                    "environment": "general",
                    "lighting": "natural",
                }
            )

        logger.info(
            "Heuristic selected %d scenes from %d candidates", len(results), len(scored)
        )
        return results

    def _sanitize_scene_list(
        self, scenes: Iterable[Dict[str, Any]], sentences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        sentence_map = {s.get("id"): s for s in sentences if s.get("id") is not None}
        results = []
        for scene in scenes:
            sentence_id = scene.get("sentence_id")
            if sentence_id is None:
                continue
            sentence = sentence_map.get(sentence_id)
            if not sentence:
                continue
            results.append(self._build_scene_entry(sentence, scene))
        return results

    def _build_scene_entry(self, sentence: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
        text = sentence.get("text", "")
        visual_tags = scene.get("visual_tags") or self._extract_keywords(text, limit=5)
        return {
            "sentence_id": sentence.get("id"),
            "uid": sentence.get("metadata", {}).get("uid") or "",
            "scene_description": scene.get("scene_description", text),
            "visual_tags": visual_tags,
            "camera": scene.get("camera", "medium"),
            "mood": scene.get("mood", "neutral"),
            "subject": scene.get("subject", visual_tags[0] if visual_tags else ""),
            "environment": scene.get("environment", ""),
            "lighting": scene.get("lighting", "natural"),
        }

    def _extract_keywords(self, text: str, limit: int = 5) -> List[str]:
        tokens = self._tokenize(text)
        keywords = []
        for token in tokens:
            normalized = self._normalize_token(token)
            if not normalized:
                continue
            if self._is_candidate_noun(normalized):
                keywords.append(normalized)
        keywords = self._dedupe_preserve_order(keywords)
        return keywords[:limit]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[0-9A-Za-z가-힣]+", text)
        return [t.lower() for t in tokens if t]

    @staticmethod
    def _normalize_token(token: str) -> str:
        cleaned = re.sub(r"[^A-Za-z가-힣]", "", str(token))
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        for suffix in KOREAN_SUFFIXES:
            if lowered.endswith(suffix) and len(lowered) > len(suffix) + 1:
                lowered = lowered[: -len(suffix)]
                break
        for suffix in sorted(KOREAN_PARTICLES, key=len, reverse=True):
            if lowered.endswith(suffix) and len(lowered) > len(suffix) + 1:
                lowered = lowered[: -len(suffix)]
                break
        return lowered

    @staticmethod
    def _is_candidate_noun(token: str) -> bool:
        if token in KOREAN_PARTICLES or token in KOREAN_STOPWORDS or token in EN_STOPWORDS:
            return False
        if token.isdigit():
            return False
        if re.fullmatch(r"[가-힣]+", token):
            return len(token) >= 2
        return len(token) >= 3

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        output: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
        return output
        return True

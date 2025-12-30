"""Optional LLM tagging for multicam segments."""

from __future__ import annotations

import threading
from typing import Dict, List

from VideoForge.ai.llm_hooks import call_llm, get_llm_provider
from VideoForge.config.config_manager import Config


class LLMTagger:
    """Generate context tags for segments via the configured LLM provider."""

    def __init__(self) -> None:
        provider = str(Config.get("llm_provider", "") or "").strip().lower()
        if not provider or provider == "disabled":
            provider = get_llm_provider() or ""
        self.enabled = bool(provider)
        self.batch_size = 10
        self.timeout_sec = 5.0

    def tag_segments(self, segments: List[Dict]) -> List[str]:
        if not self.enabled:
            return ["neutral"] * len(segments)

        all_tags: List[str] = []
        for idx in range(0, len(segments), self.batch_size):
            batch = segments[idx : idx + self.batch_size]
            tags = self._tag_batch(batch)
            all_tags.extend(tags)

        while len(all_tags) < len(segments):
            all_tags.append("neutral")

        return all_tags[: len(segments)]

    def _tag_batch(self, segments: List[Dict]) -> List[str]:
        prompt = self._build_prompt(segments)
        result: Dict[str, str | None] = {"text": None}

        def _run() -> None:
            try:
                result["text"] = call_llm(prompt)
            except Exception:
                result["text"] = None

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_sec)

        if thread.is_alive() or result["text"] is None:
            return ["neutral"] * len(segments)

        tags = self._parse_response(result["text"], len(segments))
        if len(tags) < len(segments):
            tags.extend(["neutral"] * (len(segments) - len(tags)))
        return tags[: len(segments)]

    def _build_prompt(self, segments: List[Dict]) -> str:
        lines = []
        for idx, seg in enumerate(segments, start=1):
            start_sec = float(seg.get("start_sec", 0.0))
            end_sec = float(seg.get("end_sec", 0.0))
            text = str(seg.get("text", "") or "").strip().replace("\n", " ")
            text = text[:80]
            lines.append(f"{idx}. [{start_sec:.1f}-{end_sec:.1f}s] \"{text}\"")

        segment_list = "\n".join(lines)
        return (
            "Classify each video segment by its context:\n"
            "- speaking: Normal dialogue or narration\n"
            "- action: Physical movement or demonstration\n"
            "- emphasis: Important points or emotional moments\n"
            "- neutral: Unclear or mixed context\n\n"
            "Segments:\n"
            f"{segment_list}\n\n"
            "Reply with just the numbers and tags:\n"
            "1. speaking\n2. action\n"
        )

    @staticmethod
    def _parse_response(response: str, expected_count: int) -> List[str]:
        valid_tags = {"speaking", "action", "emphasis", "neutral"}
        tags: List[str] = []
        for line in str(response).strip().splitlines():
            parts = line.split(".", 1)
            if len(parts) != 2:
                continue
            tag = parts[1].strip().lower()
            if tag in valid_tags:
                tags.append(tag)
        while len(tags) < expected_count:
            tags.append("neutral")
        return tags[:expected_count]

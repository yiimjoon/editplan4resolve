"""LLM-based prompt generator for ComfyUI."""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from VideoForge.ai.llm_hooks import call_llm, is_llm_enabled

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generate cinematographic prompts for AI image generation."""

    def generate_prompts(self, scene_analyses: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """
        Generate positive/negative prompts for ComfyUI.

        Args:
            scene_analyses: SceneAnalyzer output.

        Returns:
            [{"sentence_id": 1, "positive": "...", "negative": "...", "seed": None}, ...]
        """
        if not scene_analyses:
            return []

        if not is_llm_enabled():
            logger.warning("LLM disabled, using fallback prompts")
            return self._fallback_prompts(scene_analyses)

        try:
            return self._generate_with_llm(scene_analyses)
        except Exception as exc:
            logger.error("LLM prompt generation failed: %s", exc)
            return self._fallback_prompts(scene_analyses)

    def _generate_with_llm(self, scene_analyses: List[Dict[str, object]]) -> List[Dict[str, object]]:
        prompt = self._build_prompt(scene_analyses)
        response = call_llm(prompt)
        return self._parse_response(response, scene_analyses)

    def _build_prompt(self, scene_analyses: List[Dict[str, object]]) -> str:
        scenes_lines: List[str] = []
        for idx, scene in enumerate(scene_analyses, 1):
            tags = scene.get("visual_tags") or []
            tags_text = ", ".join([str(t) for t in tags if t])
            scenes_lines.append(
                "\n".join(
                    [
                        f"Scene {idx}:",
                        f"- sentence_id: {scene.get('sentence_id')}",
                        f"- Description: {scene.get('scene_description', '')}",
                        f"- Visual tags: {tags_text}",
                        f"- Camera: {scene.get('camera', 'medium shot')}",
                        f"- Mood: {scene.get('mood', 'neutral')}",
                        f"- Subject: {scene.get('subject', '')}",
                        f"- Environment: {scene.get('environment', '')}",
                        f"- Lighting: {scene.get('lighting', 'natural')}",
                    ]
                )
            )

        scenes_text = "\n\n".join(scenes_lines)
        return f"""You are a professional AI image prompt engineer for ComfyUI using z_image_turbo.

Generate one **positive** and one **negative** prompt for each scene.

Requirements:
- Output must be ONLY valid JSON array (no markdown, no explanation).
- Keep prompts concise but specific and cinematic.
- Positive must be a concrete visual description that illustrates the sentence (subject, action, setting, time, light, and camera framing).
- Avoid empty quality buzzwords (no 4k/8k/high detail lists).
- Negative must include: "cartoon, illustration, anime, text, watermark, signature".
- If visual tags are Korean, translate them to English in the prompt.

Scenes:
{scenes_text}

Return JSON:
[
  {{
    "sentence_id": 1,
    "positive": "close-up shot of roasted coffee beans spilling from a burlap sack on a wooden table, warm morning light through a window, soft shadows, shallow depth of field, calm mood",
    "negative": "cartoon, illustration, anime, text, watermark, signature"
  }}
]
"""

    def _parse_response(
        self, response: str, scene_analyses: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        data = self._extract_json_array(response)
        if not isinstance(data, list):
            logger.warning("Prompt JSON not a list, using fallback")
            return self._fallback_prompts(scene_analyses)

        results: List[Dict[str, object]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            positive = item.get("positive")
            negative = item.get("negative")
            if not positive or not negative:
                continue
            results.append(
                {
                    "sentence_id": item.get("sentence_id"),
                    "positive": str(positive),
                    "negative": str(negative),
                    "seed": item.get("seed", None),
                }
            )

        if not results:
            return self._fallback_prompts(scene_analyses)

        logger.info("LLM generated %d prompts", len(results))
        return results

    @staticmethod
    def _extract_json_array(text: str) -> object:
        candidate = (text or "").strip()
        if not candidate:
            return None

        fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", candidate, re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1).strip()

        try:
            return json.loads(candidate)
        except Exception:
            pass

        start = candidate.find("[")
        end = candidate.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except Exception:
                return None
        return None

    def _fallback_prompts(self, scene_analyses: List[Dict[str, object]]) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for scene in scene_analyses:
            desc = str(scene.get("scene_description", "") or "").strip()
            camera = str(scene.get("camera", "medium shot") or "medium shot")
            mood = str(scene.get("mood", "neutral") or "neutral")
            lighting = str(scene.get("lighting", "natural") or "natural")
            tags = scene.get("visual_tags") or []
            tags_text = ", ".join([str(t) for t in tags if t])

            positive_parts = [camera, "photograph"]
            if desc:
                positive_parts.append(desc)
            if tags_text:
                positive_parts.append(tags_text)
            positive_parts.append(f"{mood} mood")
            positive_parts.append(f"{lighting} lighting")
            positive_parts.append(
                "professional photography, 8k resolution, highly detailed, photorealistic"
            )
            positive = ", ".join(positive_parts)

            negative = (
                "cartoon, illustration, anime, painting, low quality, blurry, artificial, "
                "lowres, jpeg artifacts, text, watermark, signature"
            )

            results.append(
                {
                    "sentence_id": scene.get("sentence_id"),
                    "positive": positive,
                    "negative": negative,
                    "seed": None,
                }
            )

        logger.info("Fallback prompts: %d", len(results))
        return results

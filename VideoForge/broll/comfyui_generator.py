"""ComfyUI image generation integration."""
from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, Optional

import requests

from VideoForge.broll.metadata import BrollMetadata, now_iso

logger = logging.getLogger(__name__)


class ComfyUIGenerator:
    """Generate AI images using ComfyUI with z_image_turbo workflow."""

    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        workflow_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize ComfyUI generator.

        Args:
            comfyui_url: ComfyUI server URL
            workflow_path: Path to workflow JSON template
        """
        self.comfyui_url = comfyui_url.rstrip("/")
        self.prompt_url = f"{self.comfyui_url}/prompt"
        self.workflow_path = workflow_path
        self.workflow_template: Optional[Dict] = None

        if workflow_path and workflow_path.exists():
            self._load_workflow()

    def _load_workflow(self) -> None:
        """Load workflow template from JSON file."""
        try:
            with open(self.workflow_path, "r", encoding="utf-8") as f:
                self.workflow_template = json.load(f)
            logger.info("Loaded ComfyUI workflow: %s", self.workflow_path.name)
        except Exception as exc:
            logger.error("Failed to load workflow: %s", exc)
            self.workflow_template = None

    def is_available(self) -> bool:
        """Check if ComfyUI server is running."""
        try:
            response = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def generate_image(
        self,
        positive_prompt: str,
        negative_prompt: str = "cartoon, illustration, anime, low quality, blurry",
        filename_prefix: str = "ComfyUI",
        seed: Optional[int] = None,
        width: int = 1920,
        height: int = 1024,
    ) -> Optional[str]:
        """
        Generate an image with ComfyUI.

        Args:
            positive_prompt: Positive prompt text
            negative_prompt: Negative prompt text
            filename_prefix: Output filename prefix
            seed: Random seed (None = random)
            width: Image width
            height: Image height

        Returns:
            Prompt ID if successful, None if failed
        """
        if not self.workflow_template:
            logger.error("Workflow template not loaded")
            return None

        # Make a copy of the workflow template
        workflow = json.loads(json.dumps(self.workflow_template))

        # Modify workflow nodes
        try:
            # Node 6: Positive prompt (CLIPTextEncode)
            workflow["6"]["inputs"]["text"] = positive_prompt

            # Node 7: Negative prompt (CLIPTextEncode)
            workflow["7"]["inputs"]["text"] = negative_prompt

            # Node 3: Seed (KSampler)
            workflow["3"]["inputs"]["seed"] = seed or random.randint(1, 999999999999)

            # Node 9: Filename (SaveImage)
            workflow["9"]["inputs"]["filename_prefix"] = filename_prefix

            # Node 13: Image size (EmptySD3LatentImage)
            workflow["13"]["inputs"]["width"] = width
            workflow["13"]["inputs"]["height"] = height

        except KeyError as exc:
            logger.error("Workflow node modification failed: %s", exc)
            return None

        # Send to ComfyUI
        payload = {"prompt": workflow}

        try:
            response = requests.post(self.prompt_url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id", "unknown")
                logger.info("ComfyUI prompt queued: %s", prompt_id)
                return prompt_id

            logger.error("ComfyUI error: HTTP %d %s", response.status_code, response.text)
            return None

        except requests.exceptions.ConnectionError:
            logger.error("ComfyUI not running at %s", self.comfyui_url)
            return None
        except Exception as exc:
            logger.error("ComfyUI request failed: %s", exc)
            return None

    def wait_for_completion(
        self, prompt_id: str, timeout_sec: int = 300, poll_interval: float = 2.0
    ) -> bool:
        """
        Wait for ComfyUI to finish generating an image.

        Args:
            prompt_id: Prompt ID from generate_image()
            timeout_sec: Max wait time
            poll_interval: Polling interval in seconds

        Returns:
            True if completed successfully
        """
        elapsed = 0.0

        while elapsed < timeout_sec:
            try:
                # Check queue status
                response = requests.get(f"{self.comfyui_url}/queue", timeout=10)
                if response.status_code == 200:
                    queue_data = response.json()
                    queue_running = queue_data.get("queue_running", [])
                    queue_pending = queue_data.get("queue_pending", [])

                    # Check if our prompt is still in queue
                    in_queue = any(
                        item[1] == prompt_id
                        for item in queue_running + queue_pending
                        if len(item) > 1
                    )

                    if not in_queue:
                        logger.info("ComfyUI prompt %s completed", prompt_id)
                        return True

                time.sleep(poll_interval)
                elapsed += poll_interval

            except Exception as exc:
                logger.warning("Queue check failed: %s", exc)
                time.sleep(poll_interval)
                elapsed += poll_interval

        logger.warning("ComfyUI timeout after %ds", timeout_sec)
        return False

    def generate_and_wait(
        self,
        positive_prompt: str,
        negative_prompt: str = "cartoon, illustration, anime, low quality, blurry",
        filename_prefix: str = "ComfyUI",
        seed: Optional[int] = None,
        timeout_sec: int = 300,
    ) -> bool:
        """
        Generate image and wait for completion.

        Args:
            positive_prompt: Positive prompt
            negative_prompt: Negative prompt
            filename_prefix: Output filename prefix
            seed: Random seed
            timeout_sec: Max wait time

        Returns:
            True if successful
        """
        prompt_id = self.generate_image(
            positive_prompt, negative_prompt, filename_prefix, seed
        )

        if not prompt_id:
            return False

        return self.wait_for_completion(prompt_id, timeout_sec)

    def build_metadata(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: Optional[int],
        width: int,
        height: int,
    ) -> BrollMetadata:
        keywords = self._keywords_from_prompt(positive_prompt)
        raw_metadata = {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "workflow": str(self.workflow_path) if self.workflow_path else "",
        }
        return BrollMetadata(
            source="comfyui",
            keywords=keywords,
            description=positive_prompt,
            visual_elements=list(keywords),
            created_at=now_iso(),
            raw_metadata=raw_metadata,
        )

    @staticmethod
    def _keywords_from_prompt(prompt: str, limit: int = 12) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9]+", prompt.lower())
        seen = set()
        keywords = []
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= limit:
                break
        return keywords

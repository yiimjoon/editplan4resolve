"""ComfyUI image generation integration."""
from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

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
        self._server_process: Optional[subprocess.Popen] = None
        self._server_started_by_us = False

        if workflow_path and workflow_path.exists():
            self._load_workflow()

    def _autostart_enabled(self) -> bool:
        from VideoForge.config.config_manager import Config

        value = Config.get("comfyui_autostart")
        if value is None:
            return True
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def _get_run_script_path(self) -> Optional[Path]:
        from VideoForge.config.config_manager import Config

        custom = Config.get("comfyui_run_script")
        if custom:
            candidate = Path(str(custom))
            if candidate.exists():
                return candidate
        default_path = Path("C:/ComfyUI/ComfyUI/run_comfyui.bat")
        if default_path.exists():
            return default_path
        return None

    def ensure_running(self, timeout_sec: float = 60.0) -> bool:
        """Start ComfyUI if needed and wait until available."""
        if self.is_available():
            return True
        if not self._autostart_enabled():
            logger.warning("ComfyUI not running and autostart disabled")
            return False

        run_script = self._get_run_script_path()
        if not run_script:
            logger.error("ComfyUI run script not found; set comfyui_run_script")
            return False

        try:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            self._server_process = subprocess.Popen(
                str(run_script),
                shell=True,
                creationflags=creationflags,
            )
            self._server_started_by_us = True
        except Exception as exc:
            logger.error("Failed to start ComfyUI: %s", exc)
            return False

        from VideoForge.config.config_manager import Config

        cfg_timeout = Config.get("comfyui_start_timeout")
        if cfg_timeout:
            try:
                timeout_sec = float(cfg_timeout)
            except (TypeError, ValueError):
                pass

        start_time = time.time()
        while (time.time() - start_time) < float(timeout_sec):
            if self.is_available():
                logger.info("ComfyUI server started")
                return True
            time.sleep(1.0)

        logger.error("ComfyUI did not become available within %.1fs", float(timeout_sec))
        return False

    def stop_server(self) -> None:
        """Stop ComfyUI server if it was started by this process."""
        if not self._server_started_by_us or not self._server_process:
            return
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(self._server_process.pid)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info("Stopped ComfyUI server (pid=%s)", self._server_process.pid)
        except Exception as exc:
            logger.warning("Failed to stop ComfyUI server: %s", exc)
        finally:
            self._server_process = None
            self._server_started_by_us = False

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
        if not self.workflow_template and self.workflow_path and self.workflow_path.exists():
            self._load_workflow()

        if not self.workflow_template:
            logger.error("Workflow template not loaded")
            return None

        # Make a copy of the workflow template
        workflow = json.loads(json.dumps(self.workflow_template))

        # Modify workflow nodes
        try:
            seed_value = seed if seed is not None else self.random_seed()
            # Node 6: Positive prompt (CLIPTextEncode)
            workflow["6"]["inputs"]["text"] = positive_prompt

            # Node 7: Negative prompt (CLIPTextEncode)
            workflow["7"]["inputs"]["text"] = negative_prompt

            # Node 3: Seed (KSampler)
            workflow["3"]["inputs"]["seed"] = seed_value

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
        self,
        prompt_id: str,
        timeout_sec: int = 300,
        poll_interval: float = 2.0,
        filename_prefix: Optional[str] = None,
        output_dir: Optional[Path] = None,
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

            if filename_prefix and self._output_exists(
                filename_prefix, output_dir=output_dir
            ):
                logger.info(
                    "Detected ComfyUI output for %s; assuming completion",
                    filename_prefix,
                )
                return True

        logger.warning("ComfyUI timeout after %ds", timeout_sec)
        return False

    def generate_and_wait(
        self,
        positive_prompt: str,
        negative_prompt: str = "cartoon, illustration, anime, low quality, blurry",
        filename_prefix: str = "ComfyUI",
        seed: Optional[int] = None,
        width: int = 1920,
        height: int = 1024,
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
            positive_prompt,
            negative_prompt,
            filename_prefix,
            seed,
            width=width,
            height=height,
        )

        if not prompt_id:
            return False

        return self.wait_for_completion(
            prompt_id,
            timeout_sec,
            filename_prefix=filename_prefix,
        )

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
    def random_seed() -> int:
        """Return a 12-digit random seed."""
        return random.randint(10**11, 10**12 - 1)

    def find_output_images(
        self,
        filename_prefix: str,
        output_dir: Optional[Path] = None,
        timeout_sec: float = 60.0,
    ) -> List[Path]:
        """
        Find ComfyUI output images under output_dir by filename prefix.

        Args:
            filename_prefix: SaveImage node filename_prefix (e.g. "scene_001")
            output_dir: ComfyUI output directory (None uses Config override or default path)
            timeout_sec: Poll timeout in seconds

        Returns:
            List of output paths (latest 1 item), or [] if none found.
        """
        filename_prefix = str(filename_prefix or "").strip()
        if not filename_prefix:
            return []

        resolved_dir = self._resolve_output_dir(output_dir)
        if not resolved_dir:
            logger.error("ComfyUI output directory not found: %s", output_dir)
            return []

        output_dir = resolved_dir
        start_time = time.time()
        found_files: List[Path] = []

        while (time.time() - start_time) < float(timeout_sec):
            pattern = f"{filename_prefix}_*.png"
            matches = list(output_dir.glob(pattern))
            if matches:
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                found_files = [matches[0]]
                logger.info("Found ComfyUI output: %s", found_files[0].name)
                break
            time.sleep(1.0)

        if not found_files:
            logger.warning(
                "ComfyUI output not found: %s (timeout: %.1fs)",
                filename_prefix,
                float(timeout_sec),
            )

        return found_files

    def _resolve_output_dir(self, output_dir: Optional[Path]) -> Optional[Path]:
        if output_dir is not None:
            candidate = Path(output_dir)
            return candidate if candidate.exists() else None
        from VideoForge.config.config_manager import Config

        custom_path = Config.get("comfyui_output_dir")
        if custom_path:
            candidate = Path(str(custom_path))
            if candidate.exists():
                return candidate
        default_path = Path("C:/ComfyUI/ComfyUI/output")
        return default_path if default_path.exists() else None

    def _output_exists(
        self, filename_prefix: str, output_dir: Optional[Path] = None
    ) -> bool:
        filename_prefix = str(filename_prefix or "").strip()
        if not filename_prefix:
            return False
        resolved_dir = self._resolve_output_dir(output_dir)
        if not resolved_dir:
            return False
        pattern = f"{filename_prefix}_*.png"
        return any(resolved_dir.glob(pattern))

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

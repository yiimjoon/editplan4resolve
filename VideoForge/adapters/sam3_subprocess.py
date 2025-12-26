from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class SAM3Subprocess:
    def __init__(
        self,
        wsl_distro: Optional[str] = None,
        wsl_python: Optional[str] = None,
        wsl_venv: Optional[str] = None,
    ) -> None:
        try:
            from VideoForge.config.config_manager import Config

            self.wsl_distro = wsl_distro or str(Config.get("sam3_wsl_distro") or "Ubuntu")
            self.wsl_python = wsl_python or str(Config.get("sam3_wsl_python") or "python3")
            self.wsl_venv = wsl_venv or str(Config.get("sam3_wsl_venv") or "")
        except Exception:
            self.wsl_distro = wsl_distro or "Ubuntu"
            self.wsl_python = wsl_python or "python3"
            self.wsl_venv = wsl_venv or ""

        self.worker_script = Path(__file__).resolve().parent / "sam3_worker.py"

    def segment_video(
        self,
        video_path: Path,
        text_prompt: str,
        model_size: str = "large",
        max_frames: int = 300,
        timeout_sec: float = 300.0,
    ) -> Optional[Dict[str, Any]]:
        worker_path = self._to_wsl_path(self.worker_script)
        wsl_video = self._to_wsl_path(video_path)
        if not worker_path or not wsl_video:
            logger.error("Failed to convert paths for SAM3 worker")
            return None

        args = [
            "segment_video",
            "--video",
            wsl_video,
            "--prompt",
            text_prompt,
            "--model-size",
            model_size,
            "--max-frames",
            str(int(max_frames)),
        ]
        try:
            result = self._run_worker(args, timeout_sec)
        except Exception as exc:
            logger.warning("SAM3 worker failed: %s", exc)
            return None
        return result

    def _run_worker(self, args: List[str], timeout_sec: float) -> Dict[str, Any]:
        cmd = self._build_wsl_command(args)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "sam3 worker failed")
        payload = proc.stdout.strip()
        if not payload:
            return {}
        return json.loads(payload)

    def _build_wsl_command(self, args: List[str]) -> List[str]:
        worker = self._to_wsl_path(self.worker_script)
        if not worker:
            raise RuntimeError("sam3 worker path unavailable")
        python_cmd = shlex.quote(self.wsl_python)
        arg_str = " ".join(shlex.quote(arg) for arg in [worker, *args])
        if self.wsl_venv:
            venv_path = shlex.quote(self.wsl_venv)
            cmd_str = f"source {venv_path}/bin/activate && {python_cmd} {arg_str}"
        else:
            cmd_str = f"{python_cmd} {arg_str}"
        return ["wsl", "-d", self.wsl_distro, "bash", "-lc", cmd_str]

    @staticmethod
    def _to_wsl_path(path: Path) -> Optional[str]:
        path_str = str(path)
        if path_str.startswith("/mnt/"):
            return path_str
        if path_str.startswith("/"):
            return path_str
        if ":" not in path_str:
            return None
        drive, rest = path_str.split(":", 1)
        drive = drive.strip().lower()
        rest = rest.replace("\\", "/")
        return f"/mnt/{drive}{rest}"

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class SAMAudioSubprocess:
    def __init__(
        self,
        wsl_distro: Optional[str] = None,
        wsl_python: Optional[str] = None,
        wsl_venv: Optional[str] = None,
    ) -> None:
        try:
            from VideoForge.config.config_manager import Config

            self.wsl_distro = wsl_distro or str(
                Config.get("sam_audio_wsl_distro")
                or Config.get("sam3_wsl_distro")
                or "Ubuntu"
            )
            self.wsl_python = wsl_python or str(
                Config.get("sam_audio_wsl_python")
                or Config.get("sam3_wsl_python")
                or "python3"
            )
            self.wsl_venv = wsl_venv or str(
                Config.get("sam_audio_wsl_venv")
                or Config.get("sam3_wsl_venv")
                or ""
            )
        except Exception:
            self.wsl_distro = wsl_distro or "Ubuntu"
            self.wsl_python = wsl_python or "python3"
            self.wsl_venv = wsl_venv or ""

        self.worker_script = Path(__file__).resolve().parent / "sam_audio_worker.py"

    def separate_speech(
        self,
        audio_path: Path,
        model_size: str = "large",
        output_path: Optional[Path] = None,
        timeout_sec: float = 300.0,
    ) -> Optional[Dict[str, Any]]:
        worker_path = self._to_wsl_path(self.worker_script)
        wsl_audio = self._to_wsl_path(audio_path)
        if not worker_path or not wsl_audio:
            logger.error("Failed to convert paths for SAM Audio worker")
            return None

        args = [
            "separate_speech",
            "--audio",
            wsl_audio,
            "--model-size",
            model_size,
        ]
        if output_path:
            wsl_output = self._to_wsl_path(output_path)
            if wsl_output:
                args += ["--output", wsl_output]
        try:
            result = self._run_worker(args, timeout_sec)
        except Exception as exc:
            logger.warning("SAM Audio worker failed: %s", exc)
            return None
        return result

    def _run_worker(self, args: list[str], timeout_sec: float) -> Dict[str, Any]:
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
            raise RuntimeError(proc.stderr.strip() or "sam audio worker failed")
        payload = proc.stdout.strip()
        if not payload:
            return {}
        return json.loads(payload)

    def _build_wsl_command(self, args: list[str]) -> list[str]:
        worker = self._to_wsl_path(self.worker_script)
        if not worker:
            raise RuntimeError("sam audio worker path unavailable")
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

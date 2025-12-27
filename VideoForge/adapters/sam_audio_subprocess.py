from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
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
            self.use_server = bool(Config.get("sam_audio_use_server", True))
            self.server_port = int(Config.get("sam_audio_server_port") or 8789)
            self.server_start_timeout = float(
                Config.get("sam_audio_server_start_timeout_sec") or 600.0
            )
        except Exception:
            self.wsl_distro = wsl_distro or "Ubuntu"
            self.wsl_python = wsl_python or "python3"
            self.wsl_venv = wsl_venv or ""
            self.use_server = True
            self.server_port = 8789
            self.server_start_timeout = 600.0

        self.worker_script = Path(__file__).resolve().parent / "sam_audio_worker.py"

    def separate_speech(
        self,
        audio_path: Path,
        prompt: str,
        model_size: str = "large",
        output_path: Optional[Path] = None,
        timeout_sec: float = 1000.0,
    ) -> Optional[Dict[str, Any]]:
        prompt_text = prompt.strip()
        if not prompt_text:
            logger.warning("SAM Audio prompt is required; skipping.")
            return None
        worker_path = self._to_wsl_path(self.worker_script)
        wsl_audio = self._to_wsl_path(audio_path)
        if not worker_path or not wsl_audio:
            logger.error("Failed to convert paths for SAM Audio worker")
            return None

        wsl_output = None
        if output_path:
            wsl_output = self._to_wsl_path(output_path)

        if self.use_server and self._run_server_if_needed(model_size=model_size):
            result = self._request_server(
                wsl_audio,
                wsl_output,
                prompt_text,
                model_size,
                timeout_sec,
            )
            if result:
                return result
            logger.warning("SAM Audio server did not return a result; falling back to worker.")

        args = [
            "separate_speech",
            "--audio",
            wsl_audio,
            "--prompt",
            prompt_text,
            "--model-size",
            model_size,
        ]
        if wsl_output:
            args += ["--output", wsl_output]
        try:
            result = self._run_worker(args, timeout_sec)
        except Exception as exc:
            logger.warning("SAM Audio worker failed: %s", exc)
            return None
        return result

    def _run_server_if_needed(self, model_size: str) -> bool:
        if self._check_server():
            return True
        try:
            logger.info(
                "Starting SAM Audio server on port %d (model_size=%s)",
                self.server_port,
                model_size,
            )
            self._start_server(model_size)
        except Exception as exc:
            logger.warning("SAM Audio server start failed: %s", exc)
            return False
        ready = self._wait_for_server(timeout_sec=self.server_start_timeout)
        if ready:
            logger.info("SAM Audio server ready on port %d", self.server_port)
        else:
            logger.warning(
                "SAM Audio server did not become ready after %.1fs",
                self.server_start_timeout,
            )
        return ready

    def _check_server(self) -> bool:
        try:
            import requests

            response = requests.get(
                f"http://127.0.0.1:{self.server_port}/health",
                timeout=1.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _wait_for_server(self, timeout_sec: float = 20.0) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._check_server():
                return True
            time.sleep(0.5)
        return False

    def _start_server(self, model_size: str) -> None:
        worker = self._to_wsl_path(self.worker_script)
        if not worker:
            raise RuntimeError("sam audio worker path unavailable")
        python_cmd = shlex.quote(self.wsl_python)
        arg_str = " ".join(
            shlex.quote(arg)
            for arg in [worker, "serve", "--host", "127.0.0.1", "--port", str(self.server_port), "--model-size", model_size]
        )
        if self.wsl_venv:
            venv_path = shlex.quote(self.wsl_venv)
            cmd_str = f"cd ~ && source {venv_path}/bin/activate && nohup {python_cmd} {arg_str} > /tmp/vf_sam_audio_server.log 2>&1 &"
        else:
            cmd_str = f"cd ~ && nohup {python_cmd} {arg_str} > /tmp/vf_sam_audio_server.log 2>&1 &"
        subprocess.run(
            ["wsl", "-d", self.wsl_distro, "bash", "-lc", cmd_str],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="ignore",
        )

    def _request_server(
        self,
        audio_path: str,
        output_path: Optional[str],
        prompt: str,
        model_size: str,
        timeout_sec: float,
    ) -> Optional[Dict[str, Any]]:
        try:
            import requests

            payload: Dict[str, Any] = {
                "audio_path": audio_path,
                "prompt": prompt,
                "model_size": model_size,
            }
            if output_path:
                payload["output_path"] = output_path
            response = requests.post(
                f"http://127.0.0.1:{self.server_port}/separate",
                json=payload,
                timeout=timeout_sec,
            )
            if response.status_code != 200:
                logger.warning("SAM Audio server error: %s", response.text)
                return None
            data = response.json()
            if data.get("status") != "ok":
                logger.warning("SAM Audio server failure: %s", data.get("message"))
                return None
            return data
        except Exception as exc:
            logger.warning("SAM Audio server request failed: %s", exc)
            return None

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
            cmd_str = f"cd ~ && source {venv_path}/bin/activate && {python_cmd} {arg_str}"
        else:
            cmd_str = f"cd ~ && {python_cmd} {arg_str}"
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

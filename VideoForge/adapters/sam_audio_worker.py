from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional


def _get_pipeline(model_size: str) -> "SAMAudioPipeline":
    return SAMAudioPipeline(model_size)


class SAMAudioPipeline:
    def __init__(self, model_size: str) -> None:
        from sam_audio import SAMAudio, SAMAudioProcessor

        self.model_size = model_size
        self.model_id = f"facebook/sam-audio-{model_size}"
        self.model = SAMAudio.from_pretrained(self.model_id)
        self.processor = SAMAudioProcessor.from_pretrained(self.model_id)

    def separate(
        self,
        audio_path: str,
        output_path: Optional[str],
        prompt: str,
    ) -> Dict[str, Any]:
        prompt_text = prompt.strip()
        if not prompt_text:
            raise RuntimeError("SAM Audio prompt is required")
        batch = self.processor(
            audios=[audio_path],
            descriptions=[prompt_text],
        )
        result = self.model.separate(batch, predict_spans=False)
        out_path = output_path or audio_path.replace(".wav", "_speech.wav")

        target_audio = None
        if hasattr(result, "target"):
            target_audio = result.target
        elif isinstance(result, dict):
            target_audio = result.get("target")

        if isinstance(target_audio, list):
            target_audio = target_audio[0] if target_audio else None

        if target_audio is None:
            raise RuntimeError("SAM Audio output does not contain target audio")

        if hasattr(target_audio, "detach"):
            target_audio = target_audio.detach().cpu()
        if hasattr(target_audio, "numpy"):
            target_audio = target_audio.numpy()

        self._save_audio(out_path, target_audio)
        return {"output_path": out_path, "model_id": self.model_id}

    def _save_audio(self, out_path: str, target_audio: Any) -> None:
        try:
            import soundfile as sf

            sf.write(out_path, target_audio, int(getattr(self.model, "sample_rate", 16000)))
            return
        except Exception:
            from scipy.io import wavfile
            import numpy as np

            sample_rate = int(getattr(self.model, "sample_rate", 16000))
            audio_array = np.asarray(target_audio)
            if audio_array.dtype != np.int16:
                audio_array = np.clip(audio_array, -1.0, 1.0)
                audio_array = (audio_array * 32767).astype(np.int16)
            wavfile.write(out_path, sample_rate, audio_array)


def _separate_speech(
    audio_path: str,
    model_size: str,
    output_path: str | None,
    prompt: str,
) -> Dict[str, Any]:
    pipeline = _get_pipeline(model_size)
    return pipeline.separate(audio_path, output_path, prompt)


class _SAMAudioRequestHandler(BaseHTTPRequestHandler):
    pipeline: SAMAudioPipeline | None = None

    def log_message(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            model_size = self.pipeline.model_size if self.pipeline else None
            self._send_json({"status": "ok", "model_size": model_size})
        else:
            self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/separate":
            self._send_json({"error": "not_found"}, status=404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            self._send_json({"error": "invalid_json"}, status=400)
            return

        audio_path = str(data.get("audio_path") or "")
        output_path = data.get("output_path")
        prompt = str(data.get("prompt") or "")
        if not audio_path:
            self._send_json({"error": "audio_path_required"}, status=400)
            return
        if not prompt.strip():
            self._send_json({"error": "prompt_required"}, status=400)
            return
        try:
            result = self.pipeline.separate(audio_path, output_path, prompt)
            self._send_json({"status": "ok", **result})
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, status=500)


def _serve(
    host: str,
    port: int,
    model_size: str,
) -> None:
    pipeline = _get_pipeline(model_size)
    handler = _SAMAudioRequestHandler
    handler.pipeline = pipeline
    server = HTTPServer((host, port), handler)
    server.serve_forever()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM Audio worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sep = subparsers.add_parser("separate_speech", help="Separate speech audio")
    sep.add_argument("--audio", required=True)
    sep.add_argument("--model-size", default="large")
    sep.add_argument("--output")
    sep.add_argument("--prompt", required=True)

    serve = subparsers.add_parser("serve", help="Serve SAM Audio requests")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8789)
    serve.add_argument("--model-size", default="large")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "separate_speech":
        payload = _separate_speech(args.audio, args.model_size, args.output, args.prompt)
        print(json.dumps(payload))
        return 0
    if args.command == "serve":
        _serve(args.host, int(args.port), args.model_size)
        return 0
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from VideoForge.adapters.beat_detector import BeatDetector


def _detect_beats(args: argparse.Namespace) -> Dict[str, Any]:
    detector = BeatDetector(
        mode=str(args.mode or "onset").strip().lower(),
        hop_length=int(args.hop_length),
        onset_threshold=float(args.onset_threshold),
        min_bpm=float(args.min_bpm),
        max_bpm=float(args.max_bpm),
    )
    result = detector.detect(Path(args.audio), mode=str(args.mode or "onset"))
    debug = detector.last_debug or {}
    error = detector.last_error
    debug.update(
        {
            "worker_path": str(Path(__file__).resolve()),
            "worker_cwd": os.getcwd(),
            "worker_exe": sys.executable,
            "sys_path0": sys.path[0] if sys.path else None,
            "audio_path": str(args.audio),
            "mode": str(args.mode),
        }
    )
    payload = {"result": result, "debug": debug}
    if error:
        payload["error"] = error
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    detect = sub.add_parser("detect_beats")
    detect.add_argument("--audio", required=True)
    detect.add_argument("--mode", default="onset")
    detect.add_argument("--hop-length", type=int, default=512)
    detect.add_argument("--onset-threshold", type=float, default=0.5)
    detect.add_argument("--min-bpm", type=float, default=60.0)
    detect.add_argument("--max-bpm", type=float, default=180.0)

    args = parser.parse_args()

    if args.cmd == "detect_beats":
        payload = _detect_beats(args)
        print(json.dumps(payload))


if __name__ == "__main__":
    main()

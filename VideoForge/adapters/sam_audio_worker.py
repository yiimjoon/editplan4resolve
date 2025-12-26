from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict


def _separate_speech(
    audio_path: str,
    model_size: str,
    output_path: str | None,
) -> Dict[str, Any]:
    from sam_audio import SAMAudio, SAMAudioProcessor

    model_id = f"facebook/sam-audio-{model_size}"
    model = SAMAudio.from_pretrained(model_id)
    processor = SAMAudioProcessor.from_pretrained(model_id)

    batch = processor(
        audios=[audio_path],
        descriptions=["human speech, voice, talking"],
    )
    result = model.separate(batch, predict_spans=False)
    out_path = output_path or audio_path.replace(".wav", "_speech.wav")

    if hasattr(result, "save"):
        result.save(out_path)
    elif isinstance(result, dict) and "save" in result:
        result["save"](out_path)
    else:
        raise RuntimeError("SAM Audio output does not support saving")

    return {"output_path": out_path, "model_id": model_id}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM Audio worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sep = subparsers.add_parser("separate_speech", help="Separate speech audio")
    sep.add_argument("--audio", required=True)
    sep.add_argument("--model-size", default="large")
    sep.add_argument("--output")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "separate_speech":
        payload = _separate_speech(args.audio, args.model_size, args.output)
        print(json.dumps(payload))
        return 0
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

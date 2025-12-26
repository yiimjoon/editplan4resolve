from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List


def _segment_video(
    video_path: str,
    text_prompt: str,
    model_size: str,
    max_frames: int,
) -> Dict[str, Any]:
    from sam3 import Sam3Processor, build_sam3_video_predictor

    model_id = f"facebook/sam3-{model_size}"
    predictor = build_sam3_video_predictor(model_id)
    _ = Sam3Processor.from_pretrained(model_id)

    predictor.start_session(video_path)
    predictor.set_text_prompt(text_prompt)

    fps = float(getattr(predictor, "fps", 0.0) or 0.0)
    total_frames = int(getattr(predictor, "num_frames", 0) or 0)
    limit = min(max_frames, total_frames) if total_frames else max_frames

    masks: List[Dict[str, Any]] = []
    for frame_idx in range(limit):
        mask = predictor.propagate(frame_idx)
        try:
            total = float(mask.numel()) if hasattr(mask, "numel") else float(mask.size)
            summed = mask.sum()
            summed = float(summed.item() if hasattr(summed, "item") else summed)
        except Exception:
            continue
        if total <= 0.0:
            continue
        coverage = max(0.0, min(1.0, summed / total))
        if coverage <= 0.0:
            continue
        timestamp = frame_idx / fps if fps else 0.0
        masks.append(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "coverage": coverage,
            }
        )

    return {
        "video_path": video_path,
        "text_prompt": text_prompt,
        "model_id": model_id,
        "fps": fps,
        "detected_frames": len(masks),
        "masks": masks[:50],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment = subparsers.add_parser("segment_video", help="Segment video via SAM3")
    segment.add_argument("--video", required=True)
    segment.add_argument("--prompt", required=True)
    segment.add_argument("--model-size", default="large")
    segment.add_argument("--max-frames", type=int, default=300)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "segment_video":
        payload = _segment_video(
            args.video,
            args.prompt,
            args.model_size,
            int(args.max_frames),
        )
        print(json.dumps(payload))
        return 0
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

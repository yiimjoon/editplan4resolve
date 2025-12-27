from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable as IterableType
from typing import Any, Dict, Iterable, List, Optional


def _segment_video(
    video_path: str,
    text_prompt: str,
    model_size: str,
    max_frames: int,
) -> Dict[str, Any]:
    from sam3.model_builder import build_sam3_video_predictor

    model_id = f"sam3-{model_size}"
    try:
        predictor = build_sam3_video_predictor(model_size)
    except Exception:
        predictor = build_sam3_video_predictor()

    start_payload = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": video_path,
        }
    )
    session_id = start_payload.get("session_id") or start_payload.get("session", {}).get("session_id")
    if not session_id:
        raise RuntimeError("SAM3 start_session did not return session_id")

    fps = float(start_payload.get("fps") or getattr(predictor, "fps", 0.0) or 0.0)
    total_frames = int(start_payload.get("num_frames") or getattr(predictor, "num_frames", 0) or 0)
    limit = min(max_frames, total_frames) if total_frames else max_frames

    add_payload = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt,
        }
    )

    masks: List[Dict[str, Any]] = []
    _collect_masks_from_outputs(
        add_payload.get("outputs"),
        frame_idx=add_payload.get("frame_index", 0),
        fps=fps,
        masks=masks,
    )

    stream_iter = predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
            "start_frame_index": 0,
            "max_frame_num_to_track": limit,
        }
    )
    for payload in _ensure_iterable(stream_iter):
        _collect_masks_from_outputs(
            payload.get("outputs"),
            frame_idx=payload.get("frame_index", 0),
            fps=fps,
            masks=masks,
        )

    try:
        predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
            }
        )
    except Exception:
        pass

    return {
        "video_path": video_path,
        "text_prompt": text_prompt,
        "model_id": model_id,
        "fps": fps,
        "detected_frames": len(masks),
        "masks": masks[:50],
    }


def _ensure_iterable(value: Any) -> Iterable[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, IterableType):
        return value
    return [value]


def _collect_masks_from_outputs(
    outputs: Any,
    frame_idx: int,
    fps: float,
    masks: List[Dict[str, Any]],
) -> None:
    best_coverage = _coverage_from_outputs(outputs)
    if best_coverage <= 0.0:
        return
    timestamp = frame_idx / fps if fps else 0.0
    masks.append(
        {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "coverage": best_coverage,
        }
    )


def _coverage_from_outputs(outputs: Any) -> float:
    if outputs is None:
        return 0.0
    if isinstance(outputs, dict):
        if "masks" in outputs:
            candidates = outputs["masks"]
        else:
            candidates = list(outputs.values())
    elif isinstance(outputs, (list, tuple)):
        candidates = list(outputs)
    else:
        candidates = [outputs]

    best = 0.0
    for candidate in candidates:
        if isinstance(candidate, dict) and "mask" in candidate:
            mask = candidate["mask"]
        else:
            mask = candidate
        coverage = _mask_coverage(mask)
        if coverage > best:
            best = coverage
    return best


def _mask_coverage(mask: Any) -> float:
    try:
        if hasattr(mask, "numel"):
            total = float(mask.numel())
        elif hasattr(mask, "size"):
            size = mask.size
            if callable(size):
                size = size()
            if isinstance(size, int):
                total = float(size)
            else:
                total = 1.0
                for dim in size:
                    total *= float(dim)
        elif hasattr(mask, "shape"):
            total = 1.0
            for dim in mask.shape:
                total *= float(dim)
        else:
            return 0.0

        summed = mask.sum()
        summed = float(summed.item() if hasattr(summed, "item") else summed)
    except Exception:
        return 0.0

    if total <= 0.0:
        return 0.0
    return max(0.0, min(1.0, summed / total))


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

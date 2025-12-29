from __future__ import annotations

import argparse
import json
from pathlib import Path

from VideoForge.core.sync_matcher import SyncMatcher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoForge sync worker")
    parser.add_argument("--reference", required=True, help="Reference video path")
    parser.add_argument("--target", required=True, help="Target video path")
    parser.add_argument("--mode", required=True, choices=["same", "inverse", "voice", "content"])
    parser.add_argument("--max-offset", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    matcher = SyncMatcher()
    result = matcher.find_sync_offset(
        reference_video=Path(args.reference),
        target_video=Path(args.target),
        mode=args.mode,
        max_offset=args.max_offset,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from pathlib import Path

from VideoForge.broll.sidecar_tools import generate_scene_sidecars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_sidecars",
        description="Generate sidecar JSON metadata files for Scene_* assets based on matching Scene_* video metadata.",
    )
    parser.add_argument("broll_dir", type=Path, help="Folder containing Scene_* assets (images/videos).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecar .json files.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written.")
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Also generate sidecars for the video files themselves (videos already embed metadata).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = generate_scene_sidecars(
        broll_dir=args.broll_dir,
        overwrite=args.overwrite,
        include_video=args.include_video,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"Wrote {written} sidecar files.")


if __name__ == "__main__":
    main()

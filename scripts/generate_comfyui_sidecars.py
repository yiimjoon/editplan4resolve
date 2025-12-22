import argparse
from pathlib import Path

from VideoForge.broll.sidecar_tools import generate_comfyui_sidecars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_comfyui_sidecars",
        description="Generate sidecar JSON files for ComfyUI_*.png using embedded prompt metadata.",
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Assign scene numbers sequentially when not present in metadata.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = generate_comfyui_sidecars(
        directory=args.directory,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        sequence=args.sequence,
    )
    if not args.dry_run:
        print(f"Wrote {written} sidecar files.")


if __name__ == "__main__":
    main()

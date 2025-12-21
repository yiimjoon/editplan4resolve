import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict

import yaml

from VideoForge.broll.db import LibraryDB
from VideoForge.broll.indexer import scan_library
from VideoForge.broll.matcher import BrollMatcher
from VideoForge.core.segment_store import SegmentStore
from VideoForge.core.silence_processor import process_silence
from VideoForge.core.timeline_builder import TimelineBuilder
from VideoForge.core.transcriber import transcribe_segments
from VideoForge.integrations.edl_generator import generate_edl
from VideoForge.integrations.resolve_python import export_timeline_file


def load_settings() -> Dict:
    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    data.setdefault("settings", {})
    return data


def project_id_for_file(path: str) -> str:
    stat = os.stat(path)
    payload = f"{path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def project_db_path(project_id: str) -> str:
    return str(Path(__file__).resolve().parents[1] / "data" / "projects" / f"{project_id}.db")


def cmd_library_scan(args: argparse.Namespace) -> None:
    settings = load_settings()
    library_db_path = args.library or str(Path(__file__).resolve().parents[1] / "data" / "cache" / "library.db")
    db = LibraryDB(library_db_path, schema_path=str(Path(__file__).resolve().parents[1] / "config" / "schema.sql"))
    count = scan_library(args.directory, db)
    print(f"Indexed {count} clips into {library_db_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    settings = load_settings()
    project_id = project_id_for_file(args.main)
    db_path = project_db_path(project_id)
    store = SegmentStore(db_path)
    store.save_artifact("project_id", project_id)
    store.save_artifact("whisper_model", args.whisper_model)

    if args.stage in ("segments", "all"):
        if not args.force and store.get_segments():
            print("Segments already exist. Use --force to recompute.")
        else:
            segments = process_silence(
                video_path=args.main,
                threshold_db=settings["silence"]["threshold_db"],
                min_keep=settings["silence"]["min_keep_duration"],
                buffer_before=settings["silence"]["buffer_before"],
                buffer_after=settings["silence"]["buffer_after"],
            )
            store.save_segments(segments)
            print(f"Saved {len(segments)} segments")

    if args.stage in ("speech", "all"):
        if not args.force and store.get_sentences():
            print("Sentences already exist. Use --force to recompute.")
        else:
            segments = store.get_segments()
            sentences = transcribe_segments(args.main, segments, args.whisper_model)
            saved = store.save_sentences(sentences)
            print(f"Saved {len(saved)} sentences")

    print(f"Project DB: {db_path}")


def cmd_match(args: argparse.Namespace) -> None:
    settings = load_settings()
    if args.threshold is not None:
        settings["broll"]["matching_threshold"] = float(args.threshold)
    store = SegmentStore(args.project_db)
    sentences = store.get_sentences()
    if not sentences:
        print("No sentences found. Run analyze --stage speech first.")
        return

    db = LibraryDB(args.library, schema_path=str(Path(__file__).resolve().parents[1] / "config" / "schema.sql"))
    matcher = BrollMatcher(db, settings)
    matches = matcher.match(sentences)
    store.save_matches(matches)
    print(f"Saved {len(matches)} matches")


def cmd_build(args: argparse.Namespace) -> None:
    settings = load_settings()
    store = SegmentStore(args.project_db)
    builder = TimelineBuilder(store, settings)
    timeline = builder.build(args.main)
    Path(args.output).write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


def cmd_export(args: argparse.Namespace) -> None:
    if args.to == "resolve":
        settings = load_settings()
        export_timeline_file(args.timeline, settings["resolve"]["timeline_name_prefix"])
        print("Exported to DaVinci Resolve")
    elif args.format == "edl":
        data = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
        edl_text = generate_edl(data)
        out_path = args.output or "output.edl"
        Path(out_path).write_text(edl_text, encoding="utf-8")
        print(f"Wrote {out_path}")


def cmd_debug(args: argparse.Namespace) -> None:
    store = SegmentStore(args.project_db)
    if args.show_segments:
        print(json.dumps(store.get_segments(), indent=2))
    if args.show_sentences:
        print(json.dumps(store.get_sentences(), indent=2))
    if args.show_matches:
        print(json.dumps(store.get_matches(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videoforge")
    sub = parser.add_subparsers(dest="command", required=True)

    lib = sub.add_parser("library")
    lib_sub = lib.add_subparsers(dest="action", required=True)
    scan = lib_sub.add_parser("scan")
    scan.add_argument("directory")
    scan.add_argument("--library", default=None)
    scan.set_defaults(func=cmd_library_scan)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("main")
    analyze.add_argument("--stage", choices=["segments", "speech", "all"], default="all")
    analyze.add_argument("--whisper-model", default="large-v3")
    analyze.add_argument("--force", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

    match = sub.add_parser("match")
    match.add_argument("project_db")
    match.add_argument("--library", required=True)
    match.add_argument("--threshold", type=float, default=None)
    match.set_defaults(func=cmd_match)

    build = sub.add_parser("build")
    build.add_argument("project_db")
    build.add_argument("--main", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(func=cmd_build)

    export = sub.add_parser("export")
    export.add_argument("timeline")
    export.add_argument("--to", default=None, choices=["resolve"])
    export.add_argument("--format", default=None, choices=["edl"])
    export.add_argument("--output", default=None)
    export.set_defaults(func=cmd_export)

    debug = sub.add_parser("debug")
    debug.add_argument("project_db")
    debug.add_argument("--show-segments", action="store_true")
    debug.add_argument("--show-sentences", action="store_true")
    debug.add_argument("--show-matches", action="store_true")
    debug.set_defaults(func=cmd_debug)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

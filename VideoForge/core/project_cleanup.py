import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    removed: int
    kept: int
    skipped_recent: int
    errors: int


def cleanup_project_files(
    projects_dir: Path,
    current_db: str | None,
    keep_latest: int = 5,
    min_age_days: int = 14,
) -> CleanupResult:
    """Delete stale project DB/SRT files while preserving recent and active ones."""
    projects_dir = Path(projects_dir)
    if not projects_dir.exists():
        return CleanupResult(removed=0, kept=0, skipped_recent=0, errors=0)

    current_path = Path(current_db).resolve() if current_db else None
    db_files = sorted(
        projects_dir.glob("*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    keep_set = set(db_files[: max(0, keep_latest)])
    cutoff = datetime.now() - timedelta(days=max(0, min_age_days))

    removed = kept = skipped_recent = errors = 0

    for db_file in db_files:
        if current_path and db_file.resolve() == current_path:
            kept += 1
            continue
        if db_file in keep_set:
            kept += 1
            continue
        mtime = datetime.fromtimestamp(db_file.stat().st_mtime)
        if mtime > cutoff:
            skipped_recent += 1
            continue
        try:
            db_file.unlink(missing_ok=True)
            srt_file = db_file.with_suffix(".srt")
            srt_file.unlink(missing_ok=True)
            removed += 1
        except Exception as exc:
            errors += 1
            logger.warning("Failed to remove %s: %s", db_file, exc)

    return CleanupResult(
        removed=removed,
        kept=kept,
        skipped_recent=skipped_recent,
        errors=errors,
    )

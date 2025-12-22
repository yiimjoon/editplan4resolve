from pathlib import Path

from VideoForge.broll.db import LibraryDB


def test_fts_search_handles_apostrophes(tmp_path: Path):
    db_path = tmp_path / "library.db"
    schema_path = Path("VideoForge/config/schema.sql")
    db = LibraryDB(str(db_path), schema_path=str(schema_path))
    clip_id = db.upsert_clip(
        {
            "path": str(tmp_path / "a.png"),
            "folder_tags": "Scene 1 Script 1",
            "vision_tags": "dont stop",
            "description": "test",
        },
        clip_embedding=__import__("numpy").ones(512, dtype="float32"),
    )
    results = db.fts_search("don't stop", limit=5)
    assert any(r["clip_id"] == clip_id for r in results)

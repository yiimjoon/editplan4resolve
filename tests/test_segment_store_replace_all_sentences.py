from __future__ import annotations

from pathlib import Path

from VideoForge.core.segment_store import SegmentStore


def test_replace_all_sentences_replaces_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "project.db"
    store = SegmentStore(str(db_path))

    store.save_sentences(
        [
            {"t0": 0.0, "t1": 1.0, "text": "old-1", "confidence": 0.1, "metadata": {}},
            {"t0": 1.0, "t1": 2.0, "text": "old-2", "confidence": 0.2, "metadata": {}},
        ]
    )
    assert len(store.get_sentences()) == 2

    store.replace_all_sentences(
        [
            {
                "id": 10,
                "segment_id": 1,
                "t0": 0.0,
                "t1": 1.5,
                "text": "new-1",
                "confidence": None,
                "metadata": {"source": "test"},
            },
            {
                "id": 11,
                "segment_id": 1,
                "t0": 1.5,
                "t1": 2.0,
                "text": "new-2",
                "confidence": 0.9,
                "metadata": {},
            },
        ]
    )

    sentences = store.get_sentences()
    assert [s["id"] for s in sentences] == [10, 11]
    assert [s["text"] for s in sentences] == ["new-1", "new-2"]


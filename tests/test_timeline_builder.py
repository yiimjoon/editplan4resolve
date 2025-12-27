import json
import tempfile

from VideoForge.core.segment_store import SegmentStore
from VideoForge.core.timeline_builder import TimelineBuilder


def test_timeline_builder_required_fields():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/project.db"
        store = SegmentStore(db_path)
        store.save_artifact("project_id", "test_project")
        segments = store.save_segments([{"t0": 0.0, "t1": 2.0, "type": "speech"}])
        sentences = store.save_sentences([
            {"t0": 0.2, "t1": 1.2, "text": "hello world", "confidence": 0.9, "segment_id": segments[0]["id"]}
        ])
        store.save_matches([
            {"sentence_id": sentences[0]["id"], "clip_id": "clip1", "score": 0.8, "metadata": {"clip": {"path": "broll.mp4"}}}
        ])
        settings = {
            "silence": {"threshold_db": -34},
            "broll": {"matching_threshold": 0.65, "max_overlay_duration": 3.0, "fit_mode": "center", "crossfade": 0.3, "cooldown_sec": 30},
            "query": {"top_n_tokens": 6},
            "settings": {},
        }
        builder = TimelineBuilder(store, settings)
        timeline = builder.build("main.mp4")

        # Updated to match SegmentStore.SCHEMA_VERSION.
        assert timeline["schema_version"] == "1.1"
        assert "metadata" in timeline
        assert "tracks" in timeline
        assert timeline["tracks"][0]["id"] == "V1"

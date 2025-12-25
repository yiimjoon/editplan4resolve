"""Library-first orchestration for B-roll generation."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List

import numpy as np

from VideoForge.adapters.embedding_adapter import encode_text_clip
from VideoForge.broll.db import LibraryDB
from VideoForge.broll.matcher import QueryGenerator

logger = logging.getLogger(__name__)


@dataclass
class LibraryMatch:
    sentence_id: int
    clip_id: str
    path: str
    score: float
    query: str


class BrollOrchestrator:
    """Resolve library matches before generation for library_* modes."""

    def __init__(self, library_db: LibraryDB, settings: Dict[str, Dict[str, float]]) -> None:
        self.db = library_db
        self.settings = settings
        top_n = int(settings.get("query", {}).get("top_n_tokens", 6))
        self.query_generator = QueryGenerator(top_n=top_n)

    def find_library_matches(self, sentences: Iterable[Dict]) -> Dict[int, LibraryMatch]:
        threshold = float(self.settings["broll"]["matching_threshold"])
        weights = self.settings["broll"].get("hybrid_weights")
        used_ids = set()
        matches: Dict[int, LibraryMatch] = {}

        for sentence in sentences:
            text = str(sentence.get("text", "")).strip()
            if not text:
                continue
            sentence_id = sentence.get("id")
            if sentence_id is None or sentence_id in matches:
                continue

            query = self.query_generator.generate(text)["primary"]
            try:
                embedding = encode_text_clip(query).astype(np.float32)
            except Exception as exc:
                logger.warning("Library search embedding failed: %s", exc)
                continue

            results = self.db.hybrid_search(query, embedding, limit=5, weights=weights)
            selected = None
            for result in results:
                if result["score"] < threshold:
                    continue
                clip_id = result["clip_id"]
                if clip_id in used_ids:
                    continue
                clip = self.db.get_clip(clip_id) or {}
                path = clip.get("path")
                if not path:
                    continue
                selected = LibraryMatch(
                    sentence_id=int(sentence_id),
                    clip_id=str(clip_id),
                    path=str(path),
                    score=float(result["score"]),
                    query=str(query),
                )
                used_ids.add(clip_id)
                break

            if selected:
                matches[int(sentence_id)] = selected

        return matches

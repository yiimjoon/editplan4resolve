import hashlib
import json
import logging
import sqlite3
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from VideoForge.config.config_manager import Config

logger = logging.getLogger(__name__)


def _sha1_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


def _clean_db_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass
class VectorResult:
    clip_id: str
    score: float


class VectorBackend:
    def add(self, _clip_id: str, _embedding: np.ndarray) -> None:
        return None

    def search(self, _query_embedding: np.ndarray, _limit: int) -> List[VectorResult]:
        return []


class NumpyVectorBackend(VectorBackend):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def search(self, query_embedding: np.ndarray, limit: int) -> List[VectorResult]:
        query_embedding = _normalize(query_embedding.astype(np.float32))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, embedding FROM clips WHERE embedding IS NOT NULL").fetchall()
        results: List[VectorResult] = []
        for clip_id, blob in rows:
            if blob is None:
                continue
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.size == 0:
                continue
            emb = _normalize(emb)
            score = float(np.dot(query_embedding, emb))
            results.append(VectorResult(clip_id=clip_id, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]


class LibraryDB:
    def __init__(
        self,
        global_db_path: Optional[str] = None,
        local_db_path: Optional[str] = None,
        schema_path: Optional[str] = None,
    ) -> None:
        if global_db_path is None and local_db_path is None:
            global_db_path = _clean_db_path(Config.get("global_library_db_path", ""))
            local_db_path = _clean_db_path(Config.get("local_library_db_path", ""))
        self.global_db_path = _clean_db_path(global_db_path)
        self.local_db_path = _clean_db_path(local_db_path)
        self.schema_path = schema_path
        self._recent_clip_ids: set[str] = set()
        self._cooldown_penalty = 0.4
        self.search_scope = str(Config.get("library_search_scope", "both") or "both").strip().lower()
        if self.search_scope not in {"both", "global", "local"}:
            self.search_scope = "both"
        self._vector_backend_global = (
            NumpyVectorBackend(self.global_db_path) if self.global_db_path else None
        )
        self._vector_backend_local = (
            NumpyVectorBackend(self.local_db_path) if self.local_db_path else None
        )
        self._ensure_schema(self.global_db_path)
        self._ensure_schema(self.local_db_path)

    @staticmethod
    def _connect_db(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self, db_path: Optional[str]) -> None:
        if not db_path or not self.schema_path:
            return
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect_db(db_path) as conn:
            schema = Path(self.schema_path).read_text(encoding="utf-8")
            conn.executescript(schema)

    def _iter_db_handles(self) -> List[Tuple[str, str, VectorBackend]]:
        handles: List[Tuple[str, str, VectorBackend]] = []
        if self.global_db_path and self._vector_backend_global:
            handles.append(("global", self.global_db_path, self._vector_backend_global))
        if self.local_db_path and self._vector_backend_local:
            handles.append(("local", self.local_db_path, self._vector_backend_local))
        return handles

    def _iter_search_handles(self) -> List[Tuple[str, str, VectorBackend]]:
        scope = self.search_scope or "both"
        handles = []
        for name, path, backend in self._iter_db_handles():
            if scope in ("both", name):
                handles.append((name, path, backend))
        return handles

    def _resolve_write_target(self, target_db: str) -> Tuple[str, VectorBackend, str]:
        choice = str(target_db or "auto").strip().lower()
        if choice == "global":
            if self.global_db_path and self._vector_backend_global:
                return self.global_db_path, self._vector_backend_global, "global"
            raise ValueError("Global DB not configured")
        if choice == "local":
            if self.local_db_path and self._vector_backend_local:
                return self.local_db_path, self._vector_backend_local, "local"
            raise ValueError("Local DB not configured")
        if self.local_db_path and self._vector_backend_local:
            return self.local_db_path, self._vector_backend_local, "local"
        if self.global_db_path and self._vector_backend_global:
            return self.global_db_path, self._vector_backend_global, "global"
        raise ValueError("No library DB configured")

    def set_cooldown(self, recent_clip_ids: Iterable[str], penalty: float = 0.4) -> None:
        self._recent_clip_ids = set(recent_clip_ids)
        self._cooldown_penalty = penalty

    def upsert_clip(
        self,
        metadata: Dict,
        clip_embedding: np.ndarray,
        target_db: str = "auto",
    ) -> str:
        db_path, _backend, _label = self._resolve_write_target(target_db)
        clip_id = metadata.get("id") or _sha1_id(metadata["path"])
        meta = {
            "id": clip_id,
            "path": metadata["path"],
            "duration": metadata.get("duration"),
            "fps": metadata.get("fps"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "folder_tags": metadata.get("folder_tags", ""),
            "vision_tags": metadata.get("vision_tags", ""),
            "description": metadata.get("description", ""),
            "created_at": metadata.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }
        emb_blob = clip_embedding.astype(np.float32).tobytes()
        with self._connect_db(db_path) as conn:
            conn.execute(
                """
                INSERT INTO clips (id, path, duration, fps, width, height, folder_tags, vision_tags, description, embedding, created_at)
                VALUES (:id, :path, :duration, :fps, :width, :height, :folder_tags, :vision_tags, :description, :embedding, :created_at)
                ON CONFLICT(id) DO UPDATE SET
                  path=excluded.path,
                  duration=excluded.duration,
                  fps=excluded.fps,
                  width=excluded.width,
                  height=excluded.height,
                  folder_tags=excluded.folder_tags,
                  vision_tags=excluded.vision_tags,
                  description=excluded.description,
                  embedding=excluded.embedding,
                  created_at=excluded.created_at
                """,
                {**meta, "embedding": emb_blob},
            )
        return clip_id

    def _fts_search_db(self, db_path: str, query: str, limit: int, library_type: str) -> List[Dict]:
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []
        with self._connect_db(db_path) as conn:
            rows = conn.execute(
                """
                SELECT clips.id, clips.path, clips.folder_tags, clips.vision_tags, clips.description,
                       bm25(clips_fts) as rank
                FROM clips_fts
                JOIN clips ON clips_fts.rowid = clips.rowid
                WHERE clips_fts MATCH ?
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        results = []
        for row in rows:
            rank = float(row["rank"]) if row["rank"] is not None else 0.0
            text_score = 1.0 / (1.0 + max(rank, 0.0))
            results.append(
                {
                    "clip_id": row["id"],
                    "score": text_score,
                    "row": dict(row),
                    "library": library_type,
                }
            )
        return results

    def fts_search(self, query: str, limit: int = 10) -> List[Dict]:
        results: List[Dict] = []
        for library_type, db_path, _backend in self._iter_search_handles():
            results.extend(self._fts_search_db(db_path, query, limit, library_type))
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def _vector_search_db(
        backend: VectorBackend, query_embedding: np.ndarray, limit: int, library_type: str
    ) -> List[Dict]:
        results = backend.search(query_embedding, limit)
        return [
            {"clip_id": r.clip_id, "score": r.score, "library": library_type} for r in results
        ]

    def vector_search(self, query_embedding: np.ndarray, limit: int = 10) -> List[Dict]:
        results: List[Dict] = []
        for library_type, _db_path, backend in self._iter_search_handles():
            results.extend(self._vector_search_db(backend, query_embedding, limit, library_type))
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def _search_single_db(
        self,
        library_type: str,
        db_path: str,
        backend: VectorBackend,
        text_query: str,
        query_embedding: np.ndarray,
        limit: int,
        weights: Dict[str, float] | None = None,
    ) -> List[Dict]:
        text_results = {
            r["clip_id"]: r
            for r in self._fts_search_db(db_path, text_query, limit=limit * 2, library_type=library_type)
        }
        vector_results = {
            r["clip_id"]: r
            for r in self._vector_search_db(backend, query_embedding, limit=limit * 2, library_type=library_type)
        }

        weights = weights or {}
        vector_weight = float(weights.get("vector", 0.5))
        text_weight = float(weights.get("text", 0.5))
        prior_weight = float(weights.get("prior", 0.1))

        clip_ids = set(text_results.keys()) | set(vector_results.keys())
        combined: List[Dict] = []
        for clip_id in clip_ids:
            text_score = text_results.get(clip_id, {}).get("score", 0.0)
            vector_score = vector_results.get(clip_id, {}).get("score", 0.0)
            row = text_results.get(clip_id, {}).get("row", {})
            folder_tags = row.get("folder_tags", "")
            folder_hit = any(token for token in text_query.split() if token and token in folder_tags)
            prior = (1.0 if folder_hit else 0.0) - (
                self._cooldown_penalty if clip_id in self._recent_clip_ids else 0.0
            )
            final = vector_weight * vector_score + text_weight * text_score + prior_weight * prior
            combined.append(
                {
                    "clip_id": clip_id,
                    "score": final,
                    "library": library_type,
                    "explain": {
                        "vector_score": vector_score,
                        "text_score": text_score,
                        "folder_hit": bool(folder_hit),
                        "prior": prior,
                        "final": final,
                        "weights": {
                            "vector": vector_weight,
                            "text": text_weight,
                            "prior": prior_weight,
                        },
                    },
                }
            )
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:limit]

    def hybrid_search(
        self,
        text_query: str,
        query_embedding: np.ndarray,
        limit: int = 10,
        weights: Dict[str, float] | None = None,
    ) -> List[Dict]:
        if not getattr(self, "_logged_scope", False):
            logger.info("Library search scope: %s", self.search_scope)
            self._logged_scope = True
        combined: Dict[str, Dict] = {}
        for library_type, db_path, backend in self._iter_search_handles():
            for result in self._search_single_db(
                library_type,
                db_path,
                backend,
                text_query,
                query_embedding,
                limit=limit,
                weights=weights,
            ):
                clip_id = result["clip_id"]
                if clip_id not in combined or result["score"] > combined[clip_id]["score"]:
                    combined[clip_id] = result
        merged = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
        return merged[:limit]

    def get_clip(self, clip_id: str) -> Optional[Dict]:
        for _library_type, db_path, _backend in self._iter_search_handles():
            with self._connect_db(db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM clips WHERE id = ?",
                    (clip_id,),
                ).fetchone()
                if not row:
                    continue
                data = dict(row)
                if "embedding" in data:
                    data.pop("embedding")
                return data
        return None

    def get_embedding(self, clip_id: str) -> Optional[np.ndarray]:
        for _library_type, db_path, _backend in self._iter_search_handles():
            with self._connect_db(db_path) as conn:
                row = conn.execute(
                    "SELECT embedding FROM clips WHERE id = ?",
                    (clip_id,),
                ).fetchone()
            if not row or row["embedding"] is None:
                continue
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if emb.size == 0:
                return None
            return _normalize(emb)
        return None

    def get_clips_by_ids(self, clip_ids: Iterable[str]) -> Dict[str, Dict]:
        ids = [str(cid) for cid in clip_ids if cid]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT * FROM clips WHERE id IN ({placeholders})"
        results: Dict[str, Dict] = {}
        for _library_type, db_path, _backend in self._iter_search_handles():
            with self._connect_db(db_path) as conn:
                rows = conn.execute(query, ids).fetchall()
            for row in rows:
                data = dict(row)
                if "embedding" in data:
                    data.pop("embedding")
                results[str(data.get("id"))] = data
        return results

    def get_embeddings_by_ids(self, clip_ids: Iterable[str]) -> Dict[str, Optional[np.ndarray]]:
        ids = [str(cid) for cid in clip_ids if cid]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT id, embedding FROM clips WHERE id IN ({placeholders})"
        results: Dict[str, Optional[np.ndarray]] = {}
        for _library_type, db_path, _backend in self._iter_search_handles():
            with self._connect_db(db_path) as conn:
                rows = conn.execute(query, ids).fetchall()
            for row in rows:
                clip_id = str(row["id"])
                blob = row["embedding"]
                if blob is None:
                    results[clip_id] = None
                    continue
                emb = np.frombuffer(blob, dtype=np.float32)
                if emb.size == 0:
                    results[clip_id] = None
                    continue
                results[clip_id] = _normalize(emb)
        return results

    def delete_clips_by_path_prefix(self, prefix: str) -> int:
        prefix = str(prefix or "").strip()
        if not prefix:
            return 0
        like_pattern = prefix.rstrip("\\/") + "%"
        removed = 0
        for _library_type, db_path, _backend in self._iter_db_handles():
            with self._connect_db(db_path) as conn:
                rows = conn.execute(
                    "SELECT id FROM clips WHERE path LIKE ?",
                    (like_pattern,),
                ).fetchall()
                conn.execute("DELETE FROM clips WHERE path LIKE ?", (like_pattern,))
                removed += len(rows)
        return removed


def _to_fts_query(raw: str) -> str:
    """
    Convert arbitrary user text into a safe FTS5 MATCH query.

    FTS5 treats many punctuation characters (quotes/apostrophes/operators) as syntax.
    We extract word-like tokens and OR them together.
    """
    tokens = re.findall(r"[0-9A-Za-z_]+", raw or "")
    tokens = [t for t in tokens if t.strip()]
    if not tokens:
        return ""
    # OR provides recall; scoring still comes from bm25()
    return " OR ".join(tokens)

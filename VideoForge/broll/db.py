import hashlib
import json
import sqlite3
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _sha1_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


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
    def __init__(self, db_path: str, schema_path: Optional[str] = None) -> None:
        self.db_path = db_path
        self.schema_path = schema_path
        self._recent_clip_ids: set[str] = set()
        self._cooldown_penalty = 0.4
        self._ensure_schema()
        self._vector_backend: VectorBackend = NumpyVectorBackend(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            if self.schema_path:
                schema = Path(self.schema_path).read_text(encoding="utf-8")
                conn.executescript(schema)

    def set_cooldown(self, recent_clip_ids: Iterable[str], penalty: float = 0.4) -> None:
        self._recent_clip_ids = set(recent_clip_ids)
        self._cooldown_penalty = penalty

    def upsert_clip(self, metadata: Dict, clip_embedding: np.ndarray) -> str:
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
        with self._connect() as conn:
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

    def fts_search(self, query: str, limit: int = 10) -> List[Dict]:
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []
        with self._connect() as conn:
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
            results.append({"clip_id": row["id"], "score": text_score, "row": dict(row)})
        return results

    def vector_search(self, query_embedding: np.ndarray, limit: int = 10) -> List[Dict]:
        results = self._vector_backend.search(query_embedding, limit)
        return [{"clip_id": r.clip_id, "score": r.score} for r in results]

    def hybrid_search(
        self,
        text_query: str,
        query_embedding: np.ndarray,
        limit: int = 10,
        weights: Dict[str, float] | None = None,
    ) -> List[Dict]:
        text_results = {r["clip_id"]: r for r in self.fts_search(text_query, limit=limit * 2)}
        vector_results = {r["clip_id"]: r for r in self.vector_search(query_embedding, limit=limit * 2)}

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
            prior = (1.0 if folder_hit else 0.0) - (self._cooldown_penalty if clip_id in self._recent_clip_ids else 0.0)
            final = vector_weight * vector_score + text_weight * text_score + prior_weight * prior
            combined.append(
                {
                    "clip_id": clip_id,
                    "score": final,
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

    def get_clip(self, clip_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            if "embedding" in data:
                data.pop("embedding")
            return data

    def get_embedding(self, clip_id: str) -> Optional[np.ndarray]:
        with self._connect() as conn:
            row = conn.execute("SELECT embedding FROM clips WHERE id = ?", (clip_id,)).fetchone()
        if not row or row["embedding"] is None:
            return None
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        if emb.size == 0:
            return None
        return _normalize(emb)

    def get_clips_by_ids(self, clip_ids: Iterable[str]) -> Dict[str, Dict]:
        ids = [str(cid) for cid in clip_ids if cid]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT * FROM clips WHERE id IN ({placeholders})"
        with self._connect() as conn:
            rows = conn.execute(query, ids).fetchall()
        results: Dict[str, Dict] = {}
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
        with self._connect() as conn:
            rows = conn.execute(query, ids).fetchall()
        results: Dict[str, Optional[np.ndarray]] = {}
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
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM clips WHERE path LIKE ?",
                (like_pattern,),
            ).fetchall()
            conn.execute("DELETE FROM clips WHERE path LIKE ?", (like_pattern,))
        return len(rows)


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

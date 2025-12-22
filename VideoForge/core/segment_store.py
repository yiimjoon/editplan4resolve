import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List


class SegmentStore:
    SCHEMA_VERSION = "1.0"
    SCHEMA_VERSION_INT = 1
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connect_ctx(self) -> sqlite3.Connection:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect_ctx() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = DELETE;
                CREATE TABLE IF NOT EXISTS schema_version (
                  version INTEGER PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS segments (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  t0 REAL NOT NULL,
                  t1 REAL NOT NULL,
                  type TEXT NOT NULL,
                  metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS sentences (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  segment_id INTEGER,
                  t0 REAL NOT NULL,
                  t1 REAL NOT NULL,
                  text TEXT NOT NULL,
                  confidence REAL,
                  metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS matches (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  sentence_id INTEGER NOT NULL,
                  clip_id TEXT NOT NULL,
                  score REAL,
                  metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  key TEXT UNIQUE NOT NULL,
                  value TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(t0, t1);
                CREATE INDEX IF NOT EXISTS idx_sentences_time ON sentences(t0, t1);
                CREATE INDEX IF NOT EXISTS idx_matches_sentence ON matches(sentence_id);
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION_INT,),
            )

    def save_segments(self, segments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self._connect_ctx() as conn:
            return self._insert_segments(conn, segments)

    def save_sentences(self, sentences: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self._connect_ctx() as conn:
            return self._insert_sentences(conn, sentences)

    def save_matches(self, matches: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self._connect_ctx() as conn:
            return self._insert_matches(conn, matches)

    def save_project_data(
        self,
        segments: Iterable[Dict[str, Any]] | None = None,
        sentences: Iterable[Dict[str, Any]] | None = None,
        matches: Iterable[Dict[str, Any]] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Save segments, sentences, and matches in a single transaction."""
        with self._connect_ctx() as conn:
            saved_segments = self._insert_segments(conn, segments or [])
            saved_sentences = self._insert_sentences(conn, sentences or [])
            saved_matches = self._insert_matches(conn, matches or [])
        return {
            "segments": saved_segments,
            "sentences": saved_sentences,
            "matches": saved_matches,
        }

    def get_segments(self) -> List[Dict[str, Any]]:
        with self._connect_ctx() as conn:
            rows = conn.execute("SELECT * FROM segments ORDER BY t0").fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_sentences(self) -> List[Dict[str, Any]]:
        with self._connect_ctx() as conn:
            rows = conn.execute("SELECT * FROM sentences ORDER BY t0").fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_matches(self) -> List[Dict[str, Any]]:
        with self._connect_ctx() as conn:
            rows = conn.execute("SELECT * FROM matches ORDER BY id").fetchall()
            return [self._row_to_dict(row) for row in rows]

    def save_artifact(self, key: str, value: Dict[str, Any]) -> None:
        with self._connect_ctx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    def get_artifact(self, key: str) -> Dict[str, Any] | None:
        with self._connect_ctx() as conn:
            row = conn.execute("SELECT value FROM artifacts WHERE key = ?", (key,)).fetchone()
            if not row:
                return None
            return json.loads(row["value"])

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        if "metadata" in data and data["metadata"]:
            try:
                data["metadata"] = json.loads(data["metadata"])
            except json.JSONDecodeError:
                data["metadata"] = {}
        return data

    @staticmethod
    def _insert_segments(
        conn: sqlite3.Connection, segments: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        saved = []
        cur = conn.cursor()
        for seg in segments:
            cur.execute(
                "INSERT INTO segments (t0, t1, type, metadata) VALUES (?, ?, ?, ?)",
                (
                    float(seg["t0"]),
                    float(seg["t1"]),
                    seg.get("type", "speech"),
                    json.dumps(seg.get("metadata", {})),
                ),
            )
            seg_id = cur.lastrowid
            saved.append({**seg, "id": seg_id})
        return saved

    @staticmethod
    def _insert_sentences(
        conn: sqlite3.Connection, sentences: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        saved = []
        cur = conn.cursor()
        for sent in sentences:
            cur.execute(
                """
                INSERT INTO sentences (segment_id, t0, t1, text, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sent.get("segment_id"),
                    float(sent["t0"]),
                    float(sent["t1"]),
                    sent["text"],
                    float(sent.get("confidence", 0.0)),
                    json.dumps(sent.get("metadata", {})),
                ),
            )
            sent_id = cur.lastrowid
            saved.append({**sent, "id": sent_id})
        return saved

    @staticmethod
    def _insert_matches(
        conn: sqlite3.Connection, matches: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        saved = []
        cur = conn.cursor()
        for match in matches:
            cur.execute(
                """
                INSERT INTO matches (sentence_id, clip_id, score, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    match["sentence_id"],
                    match["clip_id"],
                    float(match.get("score", 0.0)),
                    json.dumps(match.get("metadata", {})),
                ),
            )
            match_id = cur.lastrowid
            saved.append({**match, "id": match_id})
        return saved

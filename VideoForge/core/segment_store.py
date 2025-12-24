import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List


class SegmentStore:
    SCHEMA_VERSION = "1.1"
    SCHEMA_VERSION_INT = 2
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
            conn.commit()
        except Exception:
            conn.rollback()
            raise
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
                CREATE TABLE IF NOT EXISTS draft_order (
                  uid TEXT PRIMARY KEY,
                  position INTEGER NOT NULL
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
                CREATE INDEX IF NOT EXISTS idx_draft_order_position ON draft_order(position);
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

    def update_sentences(self, sentences: Iterable[Dict[str, Any]]) -> None:
        """Update existing sentences by id."""
        with self._connect_ctx() as conn:
            for sentence in sentences:
                sentence_id = sentence.get("id")
                if sentence_id is None:
                    continue
                self._ensure_sentence_uid(sentence)
                conn.execute(
                    """
                    UPDATE sentences
                    SET t0 = ?, t1 = ?, text = ?, confidence = ?, segment_id = ?, metadata = ?
                    WHERE id = ?
                    """,
                    (
                        float(sentence.get("t0", 0.0)),
                        float(sentence.get("t1", 0.0)),
                        sentence.get("text", ""),
                        sentence.get("confidence"),
                        sentence.get("segment_id"),
                        json.dumps(sentence.get("metadata", {})),
                        int(sentence_id),
                    ),
                )

    def replace_sentence(self, sentence_id: int, new_sentences: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Replace a single sentence row with one or more new sentence rows."""
        with self._connect_ctx() as conn:
            conn.execute("DELETE FROM sentences WHERE id = ?", (int(sentence_id),))
            return self._insert_sentences(conn, new_sentences)

    def replace_all_sentences(self, sentences: Iterable[Dict[str, Any]]) -> None:
        """Replace all sentences while preserving ids when provided."""
        with self._connect_ctx() as conn:
            conn.execute("DELETE FROM sentences")
            cur = conn.cursor()
            for sentence in sentences:
                self._ensure_sentence_uid(sentence)
                cur.execute(
                    """
                    INSERT INTO sentences (id, segment_id, t0, t1, text, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sentence.get("id"),
                        sentence.get("segment_id"),
                        float(sentence.get("t0", 0.0)),
                        float(sentence.get("t1", 0.0)),
                        sentence.get("text", ""),
                        sentence.get("confidence"),
                        json.dumps(sentence.get("metadata", {})),
                    ),
                )

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

    def clear_analysis_data(self) -> None:
        """Clear segments, sentences, and matches before a new analysis run."""
        with self._connect_ctx() as conn:
            conn.execute("DELETE FROM matches")
            conn.execute("DELETE FROM sentences")
            conn.execute("DELETE FROM segments")
            conn.execute("DELETE FROM draft_order")

    def clear_matches(self) -> None:
        """Clear existing matches before re-matching."""
        with self._connect_ctx() as conn:
            conn.execute("DELETE FROM matches")

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

    def save_artifact(self, key: str, value: Any) -> None:
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

    def save_draft_order(self, uids: List[str]) -> None:
        with self._connect_ctx() as conn:
            conn.execute("DELETE FROM draft_order")
            cur = conn.cursor()
            for idx, uid in enumerate(uids):
                cur.execute(
                    "INSERT INTO draft_order (uid, position) VALUES (?, ?)",
                    (uid, int(idx)),
                )

    def get_draft_order(self) -> List[str]:
        with self._connect_ctx() as conn:
            rows = conn.execute(
                "SELECT uid FROM draft_order ORDER BY position"
            ).fetchall()
            return [row["uid"] for row in rows]

    def clear_draft_order(self) -> None:
        with self._connect_ctx() as conn:
            conn.execute("DELETE FROM draft_order")

    def has_draft_order(self) -> bool:
        with self._connect_ctx() as conn:
            row = conn.execute("SELECT 1 FROM draft_order LIMIT 1").fetchone()
            return bool(row)

    def ensure_sentence_uids(self) -> int:
        updated = 0
        with self._connect_ctx() as conn:
            rows = conn.execute("SELECT id, metadata FROM sentences").fetchall()
            for row in rows:
                metadata = {}
                if row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata = {}
                if metadata.get("uid"):
                    continue
                metadata["uid"] = str(uuid.uuid4())
                conn.execute(
                    "UPDATE sentences SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata), int(row["id"])),
                )
                updated += 1
        return updated

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
            SegmentStore._ensure_sentence_uid(sent)
            confidence = sent.get("confidence", 0.0)
            if confidence is None:
                confidence = 0.0
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
                    float(confidence),
                    json.dumps(sent.get("metadata", {})),
                ),
            )
            sent_id = cur.lastrowid
            saved.append({**sent, "id": sent_id})
        return saved

    @staticmethod
    def _ensure_sentence_uid(sentence: Dict[str, Any]) -> None:
        metadata = sentence.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if not metadata.get("uid"):
            metadata["uid"] = str(uuid.uuid4())
        sentence["metadata"] = metadata

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

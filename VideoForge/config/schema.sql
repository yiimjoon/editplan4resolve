-- Library database schema for VideoForge

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS clips (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  duration REAL,
  fps REAL,
  width INTEGER,
  height INTEGER,
  library_folder TEXT,
  folder_tags TEXT,
  vision_tags TEXT,
  description TEXT,
  embedding BLOB,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS library_folders (
  name TEXT PRIMARY KEY
);

CREATE VIRTUAL TABLE IF NOT EXISTS clips_fts USING fts5(
  folder_tags,
  vision_tags,
  description,
  content='clips',
  content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS clips_ai AFTER INSERT ON clips BEGIN
  INSERT INTO clips_fts(rowid, folder_tags, vision_tags, description)
  VALUES (new.rowid, new.folder_tags, new.vision_tags, new.description);
END;

CREATE TRIGGER IF NOT EXISTS clips_ad AFTER DELETE ON clips BEGIN
  INSERT INTO clips_fts(clips_fts, rowid, folder_tags, vision_tags, description)
  VALUES('delete', old.rowid, old.folder_tags, old.vision_tags, old.description);
END;

CREATE TRIGGER IF NOT EXISTS clips_au AFTER UPDATE ON clips BEGIN
  INSERT INTO clips_fts(clips_fts, rowid, folder_tags, vision_tags, description)
  VALUES('delete', old.rowid, old.folder_tags, old.vision_tags, old.description);
  INSERT INTO clips_fts(rowid, folder_tags, vision_tags, description)
  VALUES (new.rowid, new.folder_tags, new.vision_tags, new.description);
END;

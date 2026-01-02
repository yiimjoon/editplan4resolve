from __future__ import annotations

import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from VideoForge.config.config_manager import Config

try:
    from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QUrl
    from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMenuBar,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTableView,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    try:
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        from PySide6.QtMultimediaWidgets import QVideoWidget
    except Exception:
        QAudioOutput = None
        QMediaPlayer = None
        QVideoWidget = None
    PYSIDE6 = True
except Exception:
    from PySide2.QtCore import QAbstractTableModel, QModelIndex, Qt, QUrl
    from PySide2.QtGui import QBrush, QColor, QImage, QPixmap
    from PySide2.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMenuBar,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTableView,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    try:
        from PySide2.QtMultimedia import QMediaContent, QMediaPlayer
        from PySide2.QtMultimediaWidgets import QVideoWidget
    except Exception:
        QMediaContent = None
        QMediaPlayer = None
        QVideoWidget = None
    PYSIDE6 = False


PAGE_SIZE = 1000


def _is_video(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _is_image(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _build_fts_query(text: str) -> str:
    tokens = re.findall(r"[0-9A-Za-z_]+", text or "")
    tokens = [t for t in tokens if t.strip()]
    if not tokens:
        return ""
    return " OR ".join(tokens)


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    return f"{num_bytes / (1024 ** 3):.2f} GB"


class ClipsTableModel(QAbstractTableModel):
    def __init__(self, columns: Sequence[Tuple[str, str]]) -> None:
        super().__init__()
        self._columns = list(columns)
        self._rows: List[Dict[str, object]] = []

    def set_rows(self, rows: List[Dict[str, object]]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, _parent: QModelIndex = QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, _parent: QModelIndex = QModelIndex()) -> int:
        return len(self._columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < len(self._columns):
                return self._columns[section][1]
            return None
        return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        key = self._columns[index.column()][0]
        if role == Qt.DisplayRole:
            value = row.get(key)
            if value is None:
                return ""
            if key == "duration":
                try:
                    return f"{float(value):.2f}"
                except Exception:
                    return str(value)
            return str(value)
        if role == Qt.ToolTipRole and key == "path":
            return str(row.get(key) or "")
        if role == Qt.ForegroundRole and row.get("_missing"):
            return QBrush(QColor(200, 30, 30))
        return None

    def row_at(self, row: int) -> Optional[Dict[str, object]]:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None


class LibraryManager(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("editρlan Library Manager")
        self.resize(1280, 760)

        self._db_path: Optional[Path] = None
        self._columns: List[Tuple[str, str]] = []
        self._page = 0
        self._total_rows = 0
        self._sort_column = "created_at"
        self._sort_order = "DESC"
        self._search_text = ""
        self._folder_filter = "all"
        self._folder_syncing = False
        self._folder_tree_items: Dict[str, QTreeWidgetItem] = {}
        self._stats_dirty = True
        self._library_type = "Custom"
        self._db_columns: set[str] = set()

        self._player: Optional[QMediaPlayer] = None
        self._audio_output: Optional[QAudioOutput] = None
        self._video_widget: Optional[QVideoWidget] = None
        self._video_playing = False

        self._init_ui()

    def _init_ui(self) -> None:
        menu_bar = QMenuBar(self)
        file_menu = QMenu("File", self)
        open_action = file_menu.addAction("Open Database")
        open_action.triggered.connect(self._open_database)
        switch_menu = QMenu("Switch Library", self)
        switch_global = switch_menu.addAction("Global Library")
        switch_global.triggered.connect(lambda: self._switch_library("global"))
        switch_local = switch_menu.addAction("Local Library")
        switch_local.triggered.connect(lambda: self._switch_library("local"))
        file_menu.addMenu(switch_menu)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        menu_bar.addMenu(file_menu)
        self.setMenuBar(menu_bar)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.db_label = QLabel("Database: (not loaded)")
        layout.addWidget(self.db_label)
        self.library_type_label = QLabel("Library Type: -")
        layout.addWidget(self.library_type_label)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search (FTS5): folder_tags, vision_tags, description")
        search_button = QPushButton("Search")
        search_button.clicked.connect(self._run_search)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_search)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_database)
        folder_label = QLabel("Folder")
        self.folder_filter_combo = QComboBox()
        self.folder_filter_combo.addItems(["All", "Unassigned"])
        self.folder_filter_combo.currentTextChanged.connect(self._on_folder_filter_changed)
        search_row.addWidget(self.search_input)
        search_row.addWidget(folder_label)
        search_row.addWidget(self.folder_filter_combo)
        search_row.addWidget(search_button)
        search_row.addWidget(clear_button)
        search_row.addWidget(refresh_button)
        layout.addLayout(search_row)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)

        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_header = QHBoxLayout()
        tree_label = QLabel("Folders")
        self.new_folder_button = QPushButton("New")
        self.new_folder_button.clicked.connect(self._create_folder)
        self.delete_folder_button = QPushButton("Delete")
        self.delete_folder_button.clicked.connect(self._delete_folder)
        tree_header.addWidget(tree_label)
        tree_header.addStretch()
        tree_header.addWidget(self.new_folder_button)
        tree_header.addWidget(self.delete_folder_button)
        tree_layout.addLayout(tree_header)

        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderHidden(True)
        self.folder_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.folder_tree.itemSelectionChanged.connect(self._on_folder_tree_selection_changed)
        tree_layout.addWidget(self.folder_tree)
        tree_panel.setMinimumWidth(200)
        splitter.addWidget(tree_panel)

        self.table = QTableView()
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        splitter.addWidget(self.table)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        preview_label = QLabel("Preview")
        right_layout.addWidget(preview_label)

        self.thumbnail_label = QLabel("No selection")
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setMinimumHeight(240)
        self.thumbnail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.preview_container = QWidget()
        preview_layout = QVBoxLayout(self.preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.thumbnail_label)

        self._init_video_player(preview_layout)
        right_layout.addWidget(self.preview_container, stretch=1)

        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_play)
        right_layout.addWidget(self.play_button)

        self.add_to_timeline_button = QPushButton("Add to Timeline")
        self.add_to_timeline_button.setEnabled(False)
        self.add_to_timeline_button.clicked.connect(self._on_add_to_timeline)
        right_layout.addWidget(self.add_to_timeline_button)

        edit_label = QLabel("Metadata")
        right_layout.addWidget(edit_label)

        self.folder_tags_edit = QLineEdit()
        self.folder_tags_edit.setPlaceholderText("Folder tags (comma-separated)")
        right_layout.addWidget(self.folder_tags_edit)

        self.vision_tags_edit = QLineEdit()
        self.vision_tags_edit.setPlaceholderText("Vision tags (comma-separated)")
        right_layout.addWidget(self.vision_tags_edit)

        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Description")
        right_layout.addWidget(self.description_edit)

        actions_row = QHBoxLayout()
        self.save_button = QPushButton("Save Metadata")
        self.save_button.clicked.connect(self._save_metadata)
        self.save_button.setEnabled(False)
        self.remove_missing_button = QPushButton("Remove Missing Files")
        self.remove_missing_button.clicked.connect(self._remove_missing_files)
        actions_row.addWidget(self.save_button)
        actions_row.addWidget(self.remove_missing_button)
        right_layout.addLayout(actions_row)

        bulk_row = QHBoxLayout()
        self.add_folder_tags_button = QPushButton("Add Folder Tags")
        self.add_folder_tags_button.clicked.connect(lambda: self._add_tags_to_selected("folder_tags"))
        self.add_vision_tags_button = QPushButton("Add Vision Tags")
        self.add_vision_tags_button.clicked.connect(lambda: self._add_tags_to_selected("vision_tags"))
        bulk_row.addWidget(self.add_folder_tags_button)
        bulk_row.addWidget(self.add_vision_tags_button)
        right_layout.addLayout(bulk_row)

        reindex_row = QHBoxLayout()
        self.reindex_button = QPushButton("Re-index Selected")
        self.reindex_button.clicked.connect(self._reindex_selected)
        reindex_row.addWidget(self.reindex_button)
        right_layout.addLayout(reindex_row)

        delete_row = QHBoxLayout()
        self.delete_db_button = QPushButton("Delete Selected (DB Only)")
        self.delete_db_button.clicked.connect(lambda: self._delete_selected(False))
        self.delete_files_button = QPushButton("Delete Selected (Delete Files)")
        self.delete_files_button.clicked.connect(lambda: self._delete_selected(True))
        delete_row.addWidget(self.delete_db_button)
        delete_row.addWidget(self.delete_files_button)
        right_layout.addLayout(delete_row)

        organize_label = QLabel("Organize")
        right_layout.addWidget(organize_label)

        organize_row = QHBoxLayout()
        self.move_folder_button = QPushButton("Set Folder")
        self.move_folder_button.clicked.connect(self._move_selected_to_folder)
        organize_row.addWidget(self.move_folder_button)
        right_layout.addLayout(organize_row)

        auto_row = QHBoxLayout()
        self.group_date_button = QPushButton("Group by Date")
        self.group_date_button.clicked.connect(self._group_selected_by_date)
        self.group_tag_button = QPushButton("Group by Tag")
        self.group_tag_button.clicked.connect(self._group_selected_by_tag)
        auto_row.addWidget(self.group_date_button)
        auto_row.addWidget(self.group_tag_button)
        right_layout.addLayout(auto_row)

        stats_label = QLabel("Stats")
        right_layout.addWidget(stats_label)
        self.stats_library_label = QLabel("Library Type: -")
        self.stats_count_label = QLabel("Total Clips: -")
        self.stats_size_label = QLabel("Total Size: -")
        self.stats_duration_label = QLabel("Avg Duration: -")
        self.stats_resolution_label = QLabel("Resolutions: -")
        self.stats_format_label = QLabel("Formats: -")
        right_layout.addWidget(self.stats_library_label)
        right_layout.addWidget(self.stats_count_label)
        right_layout.addWidget(self.stats_size_label)
        right_layout.addWidget(self.stats_duration_label)
        right_layout.addWidget(self.stats_resolution_label)
        right_layout.addWidget(self.stats_format_label)

        nav_row = QHBoxLayout()
        self.prev_button = QPushButton("Prev")
        self.prev_button.clicked.connect(self._prev_page)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._next_page)
        self.page_label = QLabel("Page 0/0")
        self.status_label = QLabel("")
        nav_row.addWidget(self.prev_button)
        nav_row.addWidget(self.next_button)
        nav_row.addWidget(self.page_label)
        nav_row.addStretch(1)
        nav_row.addWidget(self.status_label)
        layout.addLayout(nav_row)

        self._refresh_buttons()
        self._update_selection_buttons()
        self._update_folder_buttons()

    def _init_video_player(self, preview_layout: QVBoxLayout) -> None:
        if QMediaPlayer is None:
            return
        try:
            self._video_widget = QVideoWidget()
            self._video_widget.setMinimumHeight(240)
            preview_layout.addWidget(self._video_widget)
            self._video_widget.hide()

            self._player = QMediaPlayer(self)
            if PYSIDE6:
                self._audio_output = QAudioOutput(self)
                self._player.setAudioOutput(self._audio_output)
            self._player.setVideoOutput(self._video_widget)
        except Exception:
            self._video_widget = None
            self._player = None
            self._audio_output = None

    def _open_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Database", "", "SQLite DB (*.db)")
        if not path:
            return
        self.load_database(path)

    def _switch_library(self, library_type: str) -> None:
        key = "global_library_db_path" if library_type == "global" else "local_library_db_path"
        path = str(Config.get(key, "") or "").strip()
        if not path:
            QMessageBox.information(
                self,
                "Library Manager",
                f"Set {library_type} library DB path in Settings first.",
            )
            return
        self.load_database(path)

    def _ensure_schema(self, path: Path) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "config" / "schema.sql"
        if not schema_path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            schema = schema_path.read_text(encoding="utf-8")
            conn.executescript(schema)

    @staticmethod
    def _normalize_path(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def _resolve_library_type(self, path: Path) -> str:
        global_path = str(Config.get("global_library_db_path", "") or "").strip()
        local_path = str(Config.get("local_library_db_path", "") or "").strip()
        target = self._normalize_path(str(path))
        if global_path and self._normalize_path(global_path) == target:
            return "Global"
        if local_path and self._normalize_path(local_path) == target:
            return "Local"
        return "Custom"

    def load_database(self, path: str) -> None:
        self._db_path = Path(path)
        self._ensure_schema(self._db_path)
        self._ensure_library_folder_column()
        self._ensure_library_folder_table()
        self._library_type = self._resolve_library_type(self._db_path)
        self.db_label.setText(f"Database: {path}")
        self.library_type_label.setText(f"Library Type: {self._library_type}")
        self.stats_library_label.setText(f"Library Type: {self._library_type}")
        self.setWindowTitle(f"editρlan Library Manager ({self._library_type})")
        self._load_columns()
        self._page = 0
        self._search_text = ""
        self._stats_dirty = True
        self.search_input.setText("")
        self._refresh_folder_filter()
        self._load_page()
        self._update_stats()

    def _connect(self) -> sqlite3.Connection:
        if not self._db_path:
            raise RuntimeError("Database not loaded")
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_library_folder_column(self) -> None:
        if not self._db_path:
            return
        with self._connect() as conn:
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(clips)").fetchall()}
            if "library_folder" not in cols:
                conn.execute("ALTER TABLE clips ADD COLUMN library_folder TEXT")

    def _ensure_library_folder_table(self) -> None:
        if not self._db_path:
            return
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS library_folders (name TEXT PRIMARY KEY)"
            )

    def _load_columns(self) -> None:
        if not self._db_path:
            return
        with self._connect() as conn:
            cols = [row["name"] for row in conn.execute("PRAGMA table_info(clips)").fetchall()]
        self._db_columns = set(cols)
        preferred = [
            ("id", "ID"),
            ("path", "Path"),
            ("format", "Format"),
            ("library_folder", "Folder"),
            ("duration", "Duration"),
            ("width", "Width"),
            ("height", "Height"),
            ("folder_tags", "Folder Tags"),
            ("vision_tags", "Vision Tags"),
            ("description", "Description"),
            ("created_at", "Created"),
        ]
        self._columns = []
        for key, label in preferred:
            if key == "format":
                self._columns.append((key, label))
            elif key in cols:
                self._columns.append((key, label))
        if self._sort_column not in self._db_columns and self._db_columns:
            for key, _label in self._columns:
                if key in self._db_columns:
                    self._sort_column = key
                    break
        model = ClipsTableModel(self._columns)
        self.table.setModel(model)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

    def _load_page(self) -> None:
        if not self._db_path:
            return
        offset = self._page * PAGE_SIZE
        order_clause = self._safe_order_clause()
        fts_query = _build_fts_query(self._search_text)
        if fts_query:
            filter_clause, filter_params = self._folder_filter_clause("AND")
            rows_query = (
                "SELECT clips.* FROM clips_fts "
                "JOIN clips ON clips_fts.rowid = clips.rowid "
                f"WHERE clips_fts MATCH ? {filter_clause} "
                f"{order_clause} LIMIT ? OFFSET ?"
            )
            count_query = (
                "SELECT COUNT(*) FROM clips_fts "
                "JOIN clips ON clips_fts.rowid = clips.rowid "
                f"WHERE clips_fts MATCH ? {filter_clause}"
            )
            params = (fts_query, *filter_params, PAGE_SIZE, offset)
            count_params = (fts_query, *filter_params)
        else:
            filter_clause, filter_params = self._folder_filter_clause("WHERE")
            rows_query = f"SELECT * FROM clips {filter_clause} {order_clause} LIMIT ? OFFSET ?"
            count_query = f"SELECT COUNT(*) FROM clips {filter_clause}"
            params = (*filter_params, PAGE_SIZE, offset)
            count_params = tuple(filter_params)
        with self._connect() as conn:
            rows = conn.execute(rows_query, params).fetchall()
            total = conn.execute(count_query, count_params).fetchone()[0]
        self._total_rows = int(total or 0)
        data = []
        for row in rows:
            item = dict(row)
            path = str(item.get("path") or "")
            item["_missing"] = not path or not Path(path).exists()
            item["format"] = Path(path).suffix.lower().lstrip(".")
            data.append(item)
        if self._sort_column not in self._db_columns:
            data.sort(
                key=lambda r: str(r.get(self._sort_column) or "").lower(),
                reverse=self._sort_order == "DESC",
            )
        model = self.table.model()
        if isinstance(model, ClipsTableModel):
            model.set_rows(data)
        self._update_status()
        self._update_stats()
        self._refresh_buttons()
        self._update_selection_buttons()

    def _safe_order_clause(self) -> str:
        if self._sort_column not in self._db_columns:
            return ""
        order = "DESC" if self._sort_order == "DESC" else "ASC"
        return f"ORDER BY {self._sort_column} {order}"

    def _run_search(self) -> None:
        self._search_text = self.search_input.text().strip()
        self._page = 0
        self._load_page()

    def _clear_search(self) -> None:
        self.search_input.setText("")
        self._search_text = ""
        self._page = 0
        self._load_page()

    def _refresh_database(self) -> None:
        if not self._db_path:
            return
        self._load_columns()
        self._stats_dirty = True
        self._page = 0
        self._refresh_folder_filter()
        self._load_page()
        self._update_stats()

    def _folder_filter_label(self) -> str:
        if self._folder_filter == "__unassigned__":
            return "Unassigned"
        if self._folder_filter in ("", "all"):
            return "All"
        return self._folder_filter

    def _set_folder_filter_from_label(self, label: str) -> None:
        value = str(label or "").strip()
        if value.lower() == "all":
            self._folder_filter = "all"
        elif value.lower() == "unassigned":
            self._folder_filter = "__unassigned__"
        else:
            self._folder_filter = value

    def _fetch_folder_names(self) -> List[str]:
        if not self._db_path:
            return []
        names: set[str] = set()
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM library_folders ORDER BY name").fetchall()
            names.update(str(row[0]) for row in rows if row and row[0])
            rows = conn.execute(
                "SELECT DISTINCT library_folder FROM clips WHERE library_folder IS NOT NULL AND library_folder != ''"
            ).fetchall()
            names.update(str(row[0]) for row in rows if row and row[0])
        return sorted(names, key=lambda name: name.lower())

    def _refresh_folder_filter(self) -> None:
        if not self._db_path:
            return
        folders = self._fetch_folder_names()
        current_label = self._folder_filter_label()
        if current_label not in {"All", "Unassigned"} and current_label not in folders:
            current_label = "All"
            self._set_folder_filter_from_label(current_label)
        self._folder_syncing = True
        self.folder_filter_combo.blockSignals(True)
        self.folder_filter_combo.clear()
        self.folder_filter_combo.addItems(["All", "Unassigned"] + folders)
        self.folder_filter_combo.setCurrentText(current_label)
        self.folder_filter_combo.blockSignals(False)
        self._refresh_folder_tree(folders, current_label)
        self._folder_syncing = False
        self._update_folder_buttons()

    def _on_folder_filter_changed(self, value: str) -> None:
        label = str(value or "").strip()
        self._set_folder_filter_from_label(label)
        if not self._folder_syncing:
            self._folder_syncing = True
            self._select_folder_tree_label(label)
            self._folder_syncing = False
        self._page = 0
        self._load_page()
        self._update_folder_buttons()

    def _refresh_folder_tree(self, folders: List[str], selected: str) -> None:
        if not hasattr(self, "folder_tree"):
            return
        self.folder_tree.blockSignals(True)
        self.folder_tree.clear()
        self._folder_tree_items.clear()
        for label in ["All", "Unassigned"] + folders:
            item = QTreeWidgetItem([label])
            self.folder_tree.addTopLevelItem(item)
            self._folder_tree_items[label] = item
        item = self._folder_tree_items.get(selected) or self._folder_tree_items.get("All")
        if item:
            self.folder_tree.setCurrentItem(item)
        self.folder_tree.blockSignals(False)

    def _select_folder_tree_label(self, label: str) -> None:
        if not hasattr(self, "folder_tree"):
            return
        item = self._folder_tree_items.get(label) or self._folder_tree_items.get("All")
        if item:
            self.folder_tree.setCurrentItem(item)

    def _select_folder_combo_label(self, label: str) -> None:
        if not hasattr(self, "folder_filter_combo"):
            return
        self.folder_filter_combo.blockSignals(True)
        self.folder_filter_combo.setCurrentText(label)
        self.folder_filter_combo.blockSignals(False)

    def _on_folder_tree_selection_changed(self) -> None:
        if self._folder_syncing or not hasattr(self, "folder_tree"):
            return
        items = self.folder_tree.selectedItems()
        if not items:
            return
        label = items[0].text(0)
        self._folder_syncing = True
        self._set_folder_filter_from_label(label)
        self._select_folder_combo_label(label)
        self._folder_syncing = False
        self._page = 0
        self._load_page()
        self._update_folder_buttons()

    def _update_folder_buttons(self) -> None:
        if not hasattr(self, "new_folder_button") or not hasattr(self, "delete_folder_button"):
            return
        has_db = bool(self._db_path)
        self.new_folder_button.setEnabled(has_db)
        deletable = has_db and self._folder_filter not in ("", "all", "__unassigned__")
        self.delete_folder_button.setEnabled(deletable)

    def _create_folder(self) -> None:
        if not self._db_path:
            QMessageBox.warning(self, "New Folder", "Database not loaded.")
            return
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if not ok:
            return
        name = self._sanitize_folder_name(name)
        if not name:
            QMessageBox.information(self, "New Folder", "Folder name is required.")
            return
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO library_folders (name) VALUES (?)", (name,))
        self._set_folder_filter_from_label(name)
        self._refresh_folder_filter()
        self._page = 0
        self._load_page()

    def _delete_folder(self) -> None:
        if not self._db_path:
            QMessageBox.warning(self, "Delete Folder", "Database not loaded.")
            return
        label = self._folder_filter_label()
        if label in {"All", "Unassigned"}:
            QMessageBox.information(self, "Delete Folder", "Select a folder first.")
            return
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM clips WHERE library_folder = ?",
                (label,),
            ).fetchone()[0]
        if count:
            reply = QMessageBox.question(
                self,
                "Delete Folder",
                f"Remove folder '{label}' and clear it from {count} clips?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        else:
            reply = QMessageBox.question(
                self,
                "Delete Folder",
                f"Remove empty folder '{label}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        with self._connect() as conn:
            conn.execute("DELETE FROM library_folders WHERE name = ?", (label,))
            conn.execute("UPDATE clips SET library_folder = NULL WHERE library_folder = ?", (label,))
        self._set_folder_filter_from_label("All")
        self._refresh_folder_filter()
        self._page = 0
        self._load_page()

    def _folder_filter_clause(self, prefix: str) -> tuple[str, List[str]]:
        if self._folder_filter in ("", "all"):
            return "", []
        if self._folder_filter == "__unassigned__":
            clause = f"{prefix} (library_folder IS NULL OR library_folder = '')"
            return clause, []
        clause = f"{prefix} library_folder = ?"
        return clause, [self._folder_filter]

    def _on_header_clicked(self, index: int) -> None:
        if not self._columns:
            return
        column = self._columns[index][0]
        if column == self._sort_column:
            self._sort_order = "ASC" if self._sort_order == "DESC" else "DESC"
        else:
            self._sort_column = column
            self._sort_order = "DESC"
        self.table.horizontalHeader().setSortIndicator(index, Qt.DescendingOrder if self._sort_order == "DESC" else Qt.AscendingOrder)
        self._load_page()

    def _on_selection_changed(self) -> None:
        self._stop_player()
        selected = self._current_row()
        if not selected:
            self._set_metadata_fields(None)
            self._show_thumbnail(None)
            self.play_button.setEnabled(False)
            self._update_selection_buttons()
            return
        self._set_metadata_fields(selected)
        self._show_thumbnail(selected)
        self.play_button.setEnabled(_is_video(str(selected.get("path") or "")) and self._player is not None)
        self.save_button.setEnabled(True)
        self._update_selection_buttons()

    def _current_row(self) -> Optional[Dict[str, object]]:
        model = self.table.model()
        if not isinstance(model, ClipsTableModel):
            return None
        index = self.table.currentIndex()
        if not index.isValid():
            return None
        return model.row_at(index.row())

    def _selected_rows(self) -> List[Dict[str, object]]:
        model = self.table.model()
        if not isinstance(model, ClipsTableModel):
            return []
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return []
        rows: List[Dict[str, object]] = []
        for index in selection_model.selectedRows():
            row = model.row_at(index.row())
            if row:
                rows.append(row)
        return rows

    def _set_metadata_fields(self, row: Optional[Dict[str, object]]) -> None:
        if not row:
            self.folder_tags_edit.setText("")
            self.vision_tags_edit.setText("")
            self.description_edit.setPlainText("")
            self.save_button.setEnabled(False)
            return
        self.folder_tags_edit.setText(str(row.get("folder_tags") or ""))
        self.vision_tags_edit.setText(str(row.get("vision_tags") or ""))
        self.description_edit.setPlainText(str(row.get("description") or ""))

    def _show_thumbnail(self, row: Optional[Dict[str, object]]) -> None:
        self.thumbnail_label.show()
        if self._video_widget:
            self._video_widget.hide()
        if not row:
            self.thumbnail_label.setText("No selection")
            self.thumbnail_label.setPixmap(QPixmap())
            return
        path = str(row.get("path") or "")
        if not path or not Path(path).exists():
            self.thumbnail_label.setText("Missing file")
            self.thumbnail_label.setPixmap(QPixmap())
            return
        if _is_image(path):
            pixmap = QPixmap(path)
            if pixmap.isNull():
                self.thumbnail_label.setText("Preview unavailable")
                return
            self._set_thumbnail_pixmap(pixmap)
            return
        if _is_video(path):
            frame = self._extract_video_frame(path)
            if frame is None:
                self.thumbnail_label.setText("Preview unavailable")
                return
            pixmap = self._pixmap_from_frame(frame)
            self._set_thumbnail_pixmap(pixmap)
            return
        self.thumbnail_label.setText("Preview unavailable")
        self.thumbnail_label.setPixmap(QPixmap())

    def _extract_video_frame(self, path: str) -> Optional[object]:
        try:
            from VideoForge.adapters.video_analyzer import VideoAnalyzer

            return VideoAnalyzer.extract_frame_at_time(Path(path), 0.0)
        except Exception:
            return None

    def _pixmap_from_frame(self, frame: object) -> QPixmap:
        try:
            import numpy as np

            if not isinstance(frame, np.ndarray):
                return QPixmap()
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, channels = rgb.shape
            bytes_per_line = channels * w
            image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(image)
        except Exception:
            return QPixmap()

    def _set_thumbnail_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            self.thumbnail_label.setText("Preview unavailable")
            return
        scaled = pixmap.scaled(self.thumbnail_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.thumbnail_label.setPixmap(scaled)

    def _toggle_play(self) -> None:
        row = self._current_row()
        if not row or not self._player:
            return
        path = str(row.get("path") or "")
        if not path or not Path(path).exists():
            return
        if not PYSIDE6 and "QMediaContent" in globals() and QMediaContent is None:
            return
        if self._video_playing:
            self._player.pause()
            self._video_playing = False
            self.play_button.setText("Play")
            return
        if PYSIDE6:
            self._player.setSource(QUrl.fromLocalFile(path))
        else:
            self._player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        if self._video_widget:
            self.thumbnail_label.hide()
            self._video_widget.show()
        self._player.play()
        self._video_playing = True
        self.play_button.setText("Pause")

    def _on_add_to_timeline(self) -> None:
        row = self._current_row()
        if not row:
            QMessageBox.information(self, "Add to Timeline", "No clip selected.")
            return
        path = str(row.get("path") or "")
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Add to Timeline", "File path not found.")
            return
        try:
            from VideoForge.integrations.resolve_api import ResolveAPI

            resolve_api = ResolveAPI()
        except Exception as exc:
            QMessageBox.information(self, "Add to Timeline", f"Resolve API unavailable: {exc}")
            return
        try:
            timeline = resolve_api.get_current_timeline()
        except Exception as exc:
            QMessageBox.warning(self, "Add to Timeline", f"Timeline unavailable: {exc}")
            return

        start_frame = None
        base_track = 0
        primary = resolve_api.get_primary_clip()
        if primary:
            start_frame, _ = resolve_api._get_item_range(primary)
            base_track = int(resolve_api._safe_call(primary, "GetTrackIndex") or 1)
        if start_frame is None:
            start_frame = resolve_api._safe_call(timeline, "GetCurrentFrame")
        if start_frame is None:
            start_frame = 0

        media_item = resolve_api.import_media_if_needed(path)
        if media_item is None:
            QMessageBox.warning(self, "Add to Timeline", "Failed to import media to Resolve.")
            return

        fps = resolve_api.get_timeline_fps()
        duration = resolve_api._get_media_item_duration_frames(media_item, fps)
        if duration <= 0:
            duration = 1
        end_frame = int(start_frame) + int(duration) - 1
        target_track = self._resolve_insert_track_index(
            resolve_api, int(start_frame), int(end_frame), base_track
        )
        pre_items = timeline.GetItemListInTrack("video", int(target_track)) or []
        inserted = resolve_api.insert_clip_at_position_with_range(
            "video",
            int(target_track),
            media_item,
            int(start_frame),
            0,
            max(0, int(duration) - 1),
        )
        if inserted is None:
            inserted = resolve_api._find_latest_item_by_media_path(
                "video", int(target_track), path
            )
        if inserted is None:
            inserted = resolve_api._find_item_near_frame(
                "video", int(target_track), int(start_frame)
            )
        post_items = timeline.GetItemListInTrack("video", int(target_track)) or []
        if inserted is None and len(post_items) <= len(pre_items):
            QMessageBox.warning(self, "Add to Timeline", "Failed to insert clip into timeline.")
            return
        QMessageBox.information(
            self,
            "Add to Timeline",
            f"Inserted clip on V{target_track} at frame {start_frame}.",
        )

    @staticmethod
    def _resolve_insert_track_index(
        resolve_api, start_frame: int, end_frame: int, base_track: int
    ) -> int:
        timeline = resolve_api.get_current_timeline()
        track_count = int(timeline.GetTrackCount("video") or 0)
        top_index = 0
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("video", track_index) or []
            for item in items:
                s, e = resolve_api._get_item_range(item)
                if s is None or e is None:
                    continue
                if s <= end_frame and e >= start_frame:
                    top_index = max(top_index, track_index)
                    break
        anchor = max(top_index, int(base_track) if base_track else 0)
        if anchor <= 0:
            return 1
        return anchor + 1

    def _stop_player(self) -> None:
        if not self._player:
            return
        self._player.stop()
        self._video_playing = False
        self.play_button.setText("Play")

    def _save_metadata(self) -> None:
        row = self._current_row()
        if not row or not self._db_path:
            return
        clip_id = row.get("id")
        if not clip_id:
            return
        folder_tags = self.folder_tags_edit.text().strip()
        vision_tags = self.vision_tags_edit.text().strip()
        description = self.description_edit.toPlainText().strip()
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE clips SET folder_tags = ?, vision_tags = ?, description = ? WHERE id = ?",
                    (folder_tags, vision_tags, description, str(clip_id)),
                )
            self._load_page()
            QMessageBox.information(self, "Saved", "Metadata saved.")
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to save metadata: {exc}")

    def _merge_tags(self, existing: str, additions: List[str]) -> str:
        current = [t.strip() for t in (existing or "").split(",") if t.strip()]
        combined = current[:]
        for tag in additions:
            if tag and tag not in combined:
                combined.append(tag)
        return ", ".join(combined)

    def _add_tags_to_selected(self, field: str) -> None:
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "Add Tags", "No clips selected.")
            return
        title = "Add Folder Tags" if field == "folder_tags" else "Add Vision Tags"
        text, ok = QInputDialog.getText(self, title, "Tags (comma-separated):")
        if not ok:
            return
        tags = [t.strip() for t in (text or "").split(",") if t.strip()]
        if not tags:
            return
        try:
            with self._connect() as conn:
                for row in rows:
                    clip_id = row.get("id")
                    if not clip_id:
                        continue
                    existing = str(row.get(field) or "")
                    merged = self._merge_tags(existing, tags)
                    conn.execute(
                        f"UPDATE clips SET {field} = ? WHERE id = ?",
                        (merged, str(clip_id)),
                    )
            self._load_page()
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to add tags: {exc}")

    def _delete_selected(self, delete_files: bool) -> None:
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "Delete", "No clips selected.")
            return
        message = "Delete selected clips from database?"
        if delete_files:
            message = "Delete selected clips and remove files from disk?"
        confirm = QMessageBox.question(
            self,
            "Confirm Delete",
            message,
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        ids_to_delete: List[str] = []
        failed_files: List[str] = []
        for row in rows:
            clip_id = row.get("id")
            if not clip_id:
                continue
            path = str(row.get("path") or "")
            if delete_files and path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    failed_files.append(path)
                    continue
            ids_to_delete.append(str(clip_id))
        if not ids_to_delete:
            QMessageBox.warning(self, "Delete", "No clips were deleted.")
            return
        try:
            placeholders = ",".join("?" for _ in ids_to_delete)
            with self._connect() as conn:
                conn.execute(f"DELETE FROM clips WHERE id IN ({placeholders})", ids_to_delete)
            self._stats_dirty = True
            self._page = 0
            self._load_page()
            if failed_files:
                QMessageBox.warning(
                    self,
                    "Delete",
                    f"Deleted {len(ids_to_delete)} clips. Failed to remove {len(failed_files)} files.",
                )
            else:
                QMessageBox.information(self, "Delete", f"Deleted {len(ids_to_delete)} clips.")
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to delete clips: {exc}")

    def _remove_missing_files(self) -> None:
        if not self._db_path:
            return
        with self._connect() as conn:
            rows = conn.execute("SELECT id, path FROM clips").fetchall()
        missing_ids = [row["id"] for row in rows if not row["path"] or not Path(row["path"]).exists()]
        if not missing_ids:
            QMessageBox.information(self, "Remove Missing Files", "No missing files found.")
            return
        confirm = QMessageBox.question(
            self,
            "Remove Missing Files",
            f"Delete {len(missing_ids)} missing clips from the database?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            placeholders = ",".join("?" for _ in missing_ids)
            with self._connect() as conn:
                conn.execute(f"DELETE FROM clips WHERE id IN ({placeholders})", [str(cid) for cid in missing_ids])
            self._stats_dirty = True
            self._page = 0
            self._load_page()
            QMessageBox.information(self, "Remove Missing Files", f"Removed {len(missing_ids)} clips.")
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to remove missing files: {exc}")

    def _reindex_selected(self) -> None:
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "Re-index", "No clips selected.")
            return
        progress = QProgressDialog("Re-indexing clips...", "Cancel", 0, len(rows), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        try:
            import numpy as np
            from VideoForge.adapters.embedding_adapter import encode_image_clip, encode_video_clip

            completed = 0
            with self._connect() as conn:
                for row in rows:
                    if progress.wasCanceled():
                        break
                    clip_id = row.get("id")
                    path = str(row.get("path") or "")
                    if not clip_id or not path or not Path(path).exists():
                        completed += 1
                        progress.setValue(completed)
                        continue
                    try:
                        if _is_video(path):
                            embedding = encode_video_clip(path)
                        elif _is_image(path):
                            embedding = encode_image_clip(path)
                        else:
                            completed += 1
                            progress.setValue(completed)
                            continue
                        emb_blob = embedding.astype(np.float32).tobytes()
                        conn.execute("UPDATE clips SET embedding = ? WHERE id = ?", (emb_blob, str(clip_id)))
                    except Exception:
                        pass
                    completed += 1
                    progress.setValue(completed)
        finally:
            progress.close()
        QMessageBox.information(self, "Re-index", "Re-indexing complete.")

    def _move_selected_to_folder(self) -> None:
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "Set Folder", "No clips selected.")
            return
        folder_name, ok = QInputDialog.getText(
            self, "Set Folder", "Folder name (leave empty to clear):"
        )
        if not ok:
            return
        folder_name = self._sanitize_folder_name(folder_name)
        self._assign_rows_to_folders(
            rows, lambda _row: folder_name, "Set Folder"
        )

    def _group_selected_by_date(self) -> None:
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "Group by Date", "No clips selected.")
            return
        confirm = QMessageBox.question(
            self,
            "Group by Date",
            f"Assign date folders to {len(rows)} clips?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        def folder_for_row(row):
            date_key = self._extract_date_key(row)
            return date_key

        self._assign_rows_to_folders(rows, folder_for_row, "Group by Date")

    def _group_selected_by_tag(self) -> None:
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "Group by Tag", "No clips selected.")
            return
        tag_source, ok = QInputDialog.getItem(
            self,
            "Group by Tag",
            "Tag source:",
            ["Folder Tags", "Vision Tags"],
            0,
            False,
        )
        if not ok:
            return
        field = "folder_tags" if tag_source.startswith("Folder") else "vision_tags"
        confirm = QMessageBox.question(
            self,
            "Group by Tag",
            f"Assign tag folders to {len(rows)} clips?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        def folder_for_row(row):
            tag = self._extract_tag_key(row, field)
            return tag

        self._assign_rows_to_folders(rows, folder_for_row, "Group by Tag")

    @staticmethod
    def _sanitize_folder_name(name: str) -> str:
        cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", str(name or "").strip())
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" _-")

    def _extract_date_key(self, row: Dict[str, object]) -> str:
        created_at = str(row.get("created_at") or "").strip()
        if created_at:
            text = created_at.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(text)
                return dt.date().isoformat()
            except Exception:
                pass
        path = str(row.get("path") or "")
        if path:
            try:
                ts = Path(path).stat().st_mtime
                return datetime.fromtimestamp(ts).date().isoformat()
            except Exception:
                pass
        return "unknown_date"

    def _extract_tag_key(self, row: Dict[str, object], field: str) -> str:
        raw = str(row.get(field) or "")
        tokens = [t.strip() for t in re.split(r"[,\s;|]+", raw) if t.strip()]
        tag = tokens[0] if tokens else "unclassified"
        sanitized = self._sanitize_folder_name(tag)
        return sanitized or "unclassified"

    def _assign_rows_to_folders(
        self,
        rows: List[Dict[str, object]],
        folder_resolver,
        title: str,
    ) -> None:
        if not self._db_path:
            QMessageBox.warning(self, title, "Database not loaded.")
            return
        progress = QProgressDialog(
            "Updating folders...", "Cancel", 0, len(rows), self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        updated = 0
        failed = 0
        processed = 0
        try:
            with self._connect() as conn:
                for row in rows:
                    if progress.wasCanceled():
                        break
                    processed += 1
                    clip_id = row.get("id")
                    if not clip_id:
                        failed += 1
                        progress.setValue(processed)
                        continue
                    folder_name = folder_resolver(row)
                    folder_name = self._sanitize_folder_name(folder_name)
                    try:
                        if folder_name:
                            conn.execute(
                                "INSERT OR IGNORE INTO library_folders (name) VALUES (?)",
                                (folder_name,),
                            )
                        conn.execute(
                            "UPDATE clips SET library_folder = ? WHERE id = ?",
                            (folder_name or None, str(clip_id)),
                        )
                        updated += 1
                    except Exception:
                        failed += 1
                    progress.setValue(processed)
        finally:
            progress.close()
        self._stats_dirty = True
        self._page = 0
        self._load_page()
        self._refresh_folder_filter()
        summary = f"Updated {updated} clips."
        if failed:
            summary += f" Failed: {failed}."
        QMessageBox.information(self, title, summary)

    def _prev_page(self) -> None:
        if self._page <= 0:
            return
        self._page -= 1
        self._load_page()

    def _next_page(self) -> None:
        if (self._page + 1) * PAGE_SIZE >= self._total_rows:
            return
        self._page += 1
        self._load_page()

    def _update_status(self) -> None:
        if self._total_rows == 0:
            self.page_label.setText("Page 0/0")
            self.status_label.setText("0 rows")
            return
        total_pages = max(1, (self._total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
        self.page_label.setText(f"Page {self._page + 1}/{total_pages}")
        start = self._page * PAGE_SIZE + 1
        end = min((self._page + 1) * PAGE_SIZE, self._total_rows)
        self.status_label.setText(f"Rows {start}-{end} of {self._total_rows}")

    def _update_stats(self) -> None:
        if not self._db_path or not self._stats_dirty:
            return
        try:
            with self._connect() as conn:
                total = int(conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] or 0)
                avg_duration = conn.execute("SELECT AVG(duration) FROM clips").fetchone()[0]
                res_rows = conn.execute(
                    "SELECT width, height, COUNT(*) as cnt FROM clips GROUP BY width, height ORDER BY cnt DESC LIMIT 5"
                ).fetchall()
                paths = conn.execute("SELECT path FROM clips").fetchall()
            total_bytes = 0
            ext_counts: Dict[str, int] = {}
            for row in paths:
                path = str(row["path"] or "")
                if not path:
                    continue
                ext = Path(path).suffix.lower()
                if ext:
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                try:
                    total_bytes += Path(path).stat().st_size
                except Exception:
                    continue
            res_parts = []
            for row in res_rows:
                width = row["width"] or 0
                height = row["height"] or 0
                cnt = row["cnt"] or 0
                if width and height:
                    res_parts.append(f"{width}x{height}: {cnt}")
            res_text = ", ".join(res_parts) if res_parts else "-"
            self.stats_count_label.setText(f"Total Clips: {total}")
            self.stats_size_label.setText(f"Total Size: {_format_bytes(total_bytes)}")
            if avg_duration is None:
                self.stats_duration_label.setText("Avg Duration: -")
            else:
                self.stats_duration_label.setText(f"Avg Duration: {float(avg_duration):.2f}s")
            self.stats_resolution_label.setText(f"Resolutions: {res_text}")
            format_parts = []
            for ext, cnt in sorted(ext_counts.items(), key=lambda item: (-item[1], item[0])):
                label = ext.lstrip(".") or "unknown"
                format_parts.append(f"{label}: {cnt}")
                if len(format_parts) >= 6:
                    break
            format_text = ", ".join(format_parts) if format_parts else "-"
            self.stats_format_label.setText(f"Formats: {format_text}")
            self._stats_dirty = False
        except Exception:
            self.stats_count_label.setText("Total Clips: -")
            self.stats_size_label.setText("Total Size: -")
            self.stats_duration_label.setText("Avg Duration: -")
            self.stats_resolution_label.setText("Resolutions: -")
            self.stats_format_label.setText("Formats: -")

    def _refresh_buttons(self) -> None:
        self.prev_button.setEnabled(self._page > 0)
        self.next_button.setEnabled((self._page + 1) * PAGE_SIZE < self._total_rows)

    def _update_selection_buttons(self) -> None:
        has_selection = bool(self._selected_rows())
        self.add_folder_tags_button.setEnabled(has_selection)
        self.add_vision_tags_button.setEnabled(has_selection)
        self.reindex_button.setEnabled(has_selection)
        self.delete_db_button.setEnabled(has_selection)
        self.delete_files_button.setEnabled(has_selection)
        self.add_to_timeline_button.setEnabled(has_selection)
        self.move_folder_button.setEnabled(has_selection)
        self.group_date_button.setEnabled(has_selection)
        self.group_tag_button.setEnabled(has_selection)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Delete and self.table.hasFocus():
            self._prompt_delete_selected()
            return
        super().keyPressEvent(event)

    def _prompt_delete_selected(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Delete Selected")
        dialog.setText("Delete selected clips?")
        db_only = dialog.addButton("DB Only", QMessageBox.AcceptRole)
        delete_files = dialog.addButton("Delete Files", QMessageBox.DestructiveRole)
        cancel = dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(cancel)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == db_only:
            self._delete_selected(False)
        elif clicked == delete_files:
            self._delete_selected(True)

    @classmethod
    def show_as_dialog(cls, parent: Optional[QWidget] = None, db_path: Optional[str] = None) -> "LibraryManager":
        window = cls()
        if parent is not None:
            window.setParent(parent, Qt.Window)
        if db_path:
            window.load_database(db_path)
        window.show()
        return window


def main() -> int:
    app = QApplication(sys.argv)
    window = LibraryManager()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

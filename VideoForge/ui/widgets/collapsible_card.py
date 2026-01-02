"""Collapsible settings card widget."""

from __future__ import annotations

from VideoForge.config.config_manager import Config
from VideoForge.ui.qt_compat import QFrame, QPushButton, QVBoxLayout, QWidget, Qt


class CollapsibleCard(QFrame):
    """Card with a clickable header that persists its collapsed state."""

    def __init__(
        self,
        title: str,
        section_id: str,
        parent: QWidget | None = None,
        collapsed: bool | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SectionCard")
        self._title = title
        self.section_id = section_id

        default_state = True if collapsed is None else bool(collapsed)
        stored = Config.get(f"ui_collapse_{section_id}")
        self.collapsed = self._coerce_bool(stored, default_state)

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.toggle_btn = QPushButton(self._header_text())
        self.toggle_btn.setObjectName("CollapsibleHeader")
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.setFlat(True)
        self.toggle_btn.clicked.connect(self._toggle)
        self.layout.addWidget(self.toggle_btn)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self.layout.addWidget(self.content)

        self.content.setVisible(not self.collapsed)

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)

    def add_layout(self, layout: QVBoxLayout) -> None:
        self.content_layout.addLayout(layout)

    def _toggle(self) -> None:
        self.collapsed = not self.collapsed
        self.content.setVisible(not self.collapsed)
        self.toggle_btn.setText(self._header_text())
        Config.set(f"ui_collapse_{self.section_id}", self.collapsed)

    def _header_text(self) -> str:
        indicator = ">" if self.collapsed else "v"
        return f"{indicator} {self._title}"

    @staticmethod
    def _coerce_bool(value, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

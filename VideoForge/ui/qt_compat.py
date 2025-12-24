"""Qt compatibility layer for PySide6/PySide2."""

from __future__ import annotations

try:
    from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QPushButton,
        QProgressBar,
        QSlider,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    from PySide2.QtCore import QObject, QThread, Signal, Qt, QTimer
    from PySide2.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QPushButton,
        QProgressBar,
        QSlider,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

__all__ = [
    "QApplication",
    "QCheckBox",
    "QComboBox",
    "QFileDialog",
    "QFrame",
    "QHBoxLayout",
    "QHeaderView",
    "QInputDialog",
    "QLabel",
    "QLineEdit",
    "QObject",
    "QProgressBar",
    "QPushButton",
    "QSlider",
    "QTabWidget",
    "QTableWidget",
    "QTableWidgetItem",
    "QThread",
    "Qt",
    "QTimer",
    "QVBoxLayout",
    "QWidget",
    "Signal",
]

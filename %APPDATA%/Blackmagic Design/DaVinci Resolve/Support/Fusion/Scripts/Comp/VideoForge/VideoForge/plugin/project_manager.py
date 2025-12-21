from pathlib import Path
from typing import Any, Optional


class ProjectManager:
    """Resolve project metadata manager for VideoForge."""

    def __init__(self, resolve_api: Any) -> None:
        self.resolve_api = resolve_api

    def get_project_db_path(self) -> str:
        """Return project-scoped DB path."""
        project = self.resolve_api.get_current_project()
        name = project.GetName() if project else "default"
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        base = Path(__file__).resolve().parents[1] / "data" / "projects"
        base.mkdir(parents=True, exist_ok=True)
        return str(base / f"{safe}.db")

    def save_library_path(self, library_path: str) -> None:
        """Save library path in a sidecar file for the current project."""
        project = self.resolve_api.get_current_project()
        name = project.GetName() if project else "default"
        base = Path(__file__).resolve().parents[1] / "data" / "projects"
        base.mkdir(parents=True, exist_ok=True)
        sidecar = base / f"{name}.library.txt"
        sidecar.write_text(library_path, encoding="utf-8")

    def load_library_path(self) -> Optional[str]:
        """Load the saved library path if present."""
        project = self.resolve_api.get_current_project()
        name = project.GetName() if project else "default"
        sidecar = Path(__file__).resolve().parents[1] / "data" / "projects" / f"{name}.library.txt"
        if not sidecar.exists():
            return None
        return sidecar.read_text(encoding="utf-8").strip()

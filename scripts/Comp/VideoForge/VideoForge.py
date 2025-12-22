import logging
import os
import sys
from pathlib import Path

"""
VideoForge - DaVinci Resolve Auto B-roll Plugin
Place this folder in the Resolve Fusion Scripts/Comp directory.
"""


def main() -> None:
    log_dir = Path(os.environ.get("APPDATA", ".")) / "Blackmagic Design" / "DaVinci Resolve" / "Support"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "VideoForge.log"
    os.environ.setdefault("VIDEOFORGE_LOG_PATH", str(log_path))
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    try:
        script_path = Path(__file__).resolve()
    except NameError:
        script_path = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd()
    script_dir = script_path.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    package_base = os.environ.get("VIDEOFORGE_ROOT")
    candidates = [Path(package_base)] if package_base else []
    candidates.append(script_dir)
    for base_dir in candidates:
        package_dir = base_dir / "VideoForge"
        if package_dir.is_dir():
            if str(base_dir) not in sys.path:
                sys.path.insert(0, str(base_dir))
            if __name__ == "VideoForge":
                module = sys.modules.get(__name__)
                if module is not None and not hasattr(module, "__path__"):
                    module.__path__ = [str(package_dir)]
            break

    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        try:
            from PySide2.QtWidgets import QApplication
        except Exception as exc:
            print("PySide6/PySide2 is not available in the Resolve Python environment.")
            print(exc)
            return

    from ui.main_panel import VideoForgePanel

    app = None
    if not QApplication.instance():
        app = QApplication(sys.argv)

    panel = VideoForgePanel()
    panel.show()

    if app:
        sys.exit(app.exec())


if __name__ == "__main__":
    main()

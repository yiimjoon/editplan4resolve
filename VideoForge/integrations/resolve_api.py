import subprocess
from pathlib import Path
from typing import Any, List, Optional

from VideoForge.integrations.resolve_python import _get_resolve


class ResolveAPI:
    """Convenience wrapper for DaVinciResolveScript API."""

    def __init__(self) -> None:
        """Initialize Resolve scripting instance."""
        try:
            self._resolve = _get_resolve()
        except Exception as exc:
            raise RuntimeError(
                "DaVinci Resolve scripting API is not available. Start Resolve and enable scripting."
            ) from exc

    def get_current_project(self) -> Any:
        """Return the current Resolve project."""
        manager = self._resolve.GetProjectManager()
        project = manager.GetCurrentProject()
        if project is None:
            project = manager.CreateProject("VideoForge")
        return project

    def get_current_timeline(self) -> Any:
        """Return the active timeline."""
        project = self.get_current_project()
        timeline = project.GetCurrentTimeline()
        if timeline is None:
            raise RuntimeError("No active timeline. Create or open a timeline first.")
        return timeline

    def get_selected_clips(self) -> List[Any]:
        """Return the currently selected timeline clips."""
        timeline = self.get_current_timeline()
        selected: List[Any] = []
        track_count = int(timeline.GetTrackCount("video") or 0)
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("video", track_index) or []
            for item in items:
                try:
                    if item.GetSelected():
                        selected.append(item)
                except Exception:
                    continue
        return selected

    def get_media_pool(self) -> Any:
        """Return the project media pool."""
        project = self.get_current_project()
        return project.GetMediaPool()

    def export_clip_audio(self, clip: Any, output_path: str) -> str:
        """Export clip audio to a WAV file using ffmpeg."""
        source = clip.GetClipProperty("File Path")
        if not source:
            raise RuntimeError("Selected clip has no file path.")
        output = str(Path(output_path))
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            output,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8",
                errors="ignore",
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
        return output

    def insert_clip_at_position(
        self, track_type: str, track_index: int, clip: Any, timeline_position: int
    ) -> None:
        """Insert a media pool clip at a timeline position (frames)."""
        media_pool = self.get_media_pool()
        media_pool.InsertClips([clip], track_type, track_index, timeline_position)

    def get_timeline_fps(self) -> float:
        """Return timeline FPS or 24 if missing."""
        project = self.get_current_project()
        try:
            fps = float(project.GetSetting("timelineFrameRate"))
        except Exception:
            fps = 24.0
        return fps

    def ensure_track(self, track_type: str, track_index: int) -> None:
        """Ensure a timeline track index exists."""
        timeline = self.get_current_timeline()
        track_count = int(timeline.GetTrackCount(track_type) or 0)
        while track_count < track_index:
            timeline.AddTrack(track_type)
            track_count += 1

    def import_media_if_needed(self, path: str) -> Optional[Any]:
        """Import media if missing and return the media pool clip."""
        media_pool = self.get_media_pool()
        root = media_pool.GetRootFolder()
        for clip in root.GetClipList() or []:
            if clip.GetClipProperty("File Path") == path:
                return clip
        imported = media_pool.ImportMedia([path])
        return imported[0] if imported else None

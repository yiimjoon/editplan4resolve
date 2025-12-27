import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from VideoForge.adapters.embedding_adapter import encode_image_clip, encode_video_clip
from VideoForge.broll.db import LibraryDB
from VideoForge.broll.metadata import coerce_metadata_dict
from VideoForge.config.config_manager import Config

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class LibraryIndexer:
    def __init__(self, clip_model=None) -> None:
        self.clip_model = clip_model
        global_db_path = str(Config.get("global_library_db_path", "") or "").strip()
        local_db_path = str(Config.get("local_library_db_path", "") or "").strip()
        self.db = LibraryDB(
            global_db_path=global_db_path or None,
            local_db_path=local_db_path or None,
        )

    def index_library(
        self,
        library_path: Path,
        library_type: str = "global",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        if library_type not in ("global", "local"):
            raise ValueError(f"Invalid library_type: {library_type}")
        target_db_path = (
            Config.get("global_library_db_path")
            if library_type == "global"
            else Config.get("local_library_db_path")
        )
        target_db_path = str(target_db_path or "").strip()
        if not target_db_path:
            raise ValueError(f"{library_type} library DB path not configured")
        logger.info(
            "Indexing %s library: %s -> %s",
            library_type,
            library_path,
            target_db_path,
        )
        return scan_library(
            str(library_path),
            self.db,
            target_db=library_type,
            progress_callback=progress_callback,
        )


def smart_frame_sampling(
    duration: float,
    max_frames: int = 12,
    base_frames: int = 6,
    video_path: str | None = None,
) -> List[float]:
    if duration <= 0:
        return [0.0]

    if video_path:
        try:
            from VideoForge.adapters.scene_detector import SceneDetector
            from VideoForge.config.config_manager import Config

            use_scene_detection = Config.get("use_scene_detection")
            if use_scene_detection is None:
                use_scene_detection = True

            if use_scene_detection:
                raw_threshold = Config.get("scene_detection_threshold")
                try:
                    threshold = float(raw_threshold) if raw_threshold is not None else 30.0
                except (TypeError, ValueError):
                    threshold = 30.0
                detector = SceneDetector(threshold=threshold)
                scene_changes = detector.detect_scene_changes(
                    Path(video_path),
                    sample_rate=5,
                )

                if scene_changes and len(scene_changes) > 1:
                    if len(scene_changes) <= max_frames:
                        return scene_changes
                    step = len(scene_changes) / max_frames
                    return [scene_changes[int(i * step)] for i in range(max_frames)]
        except Exception as exc:
            logger.warning("Scene detection failed, using time-based: %s", exc)

    if duration < 3.0:
        return _unique_sorted([i * duration / 2 for i in range(3)], duration)

    excl = min(2.0, duration * 0.15)

    if 3.0 <= duration <= 10.0:
        start_anchor = min(excl, duration * 0.1)
        end_anchor = max(duration - excl, duration * 0.9)
        anchors = [
            start_anchor,
            duration * 0.3,
            duration * 0.5,
            duration * 0.7,
            end_anchor,
        ]
        return _unique_sorted(anchors, duration)

    target_frames = min(max_frames, max(base_frames, int(duration / 3)))
    inner_count = max(0, target_frames - 2)
    inner_start = excl
    inner_end = max(excl, duration - excl)

    inner_points: List[float] = []
    if inner_count > 0:
        step = (inner_end - inner_start) / (inner_count + 1)
        for i in range(1, inner_count + 1):
            inner_points.append(inner_start + i * step)

    anchors = [min(excl, duration * 0.1), max(duration - excl, duration * 0.9)]
    return _unique_sorted(anchors + inner_points, duration)


def _unique_sorted(points: Iterable[float], duration: float) -> List[float]:
    clamped = [min(max(0.0, p), duration) for p in points]
    unique = sorted(set(round(p, 3) for p in clamped))
    return unique


def _ffprobe_basic(path: str) -> Dict[str, float | int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
            encoding="utf-8",
            errors="ignore",
        )
        data = json.loads(proc.stdout)
        stream = (data.get("streams") or [{}])[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        fps_raw = stream.get("avg_frame_rate", "0/1")
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
        duration = float(data.get("format", {}).get("duration", 0.0))
        return {"width": width, "height": height, "fps": fps, "duration": duration}
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
    except Exception:
        return {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0}


def _ffprobe_tags(path: str) -> Dict[str, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format_tags=comment,title,description",
        "-of",
        "json",
        path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            encoding="utf-8",
            errors="ignore",
        )
        data = json.loads(proc.stdout)
        tags = data.get("format", {}).get("tags", {}) or {}
        return {k: str(v) for k, v in tags.items()}
    except Exception:
        return {}


def _parse_metadata_json(raw: str | bytes | None) -> Dict:
    if not raw:
        return {}
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return json.loads(raw)
    except Exception:
        return {}


def _normalize_prompt_text(text: str) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:500] if len(cleaned) > 500 else cleaned


def _extract_prompt_from_comfyui_graph(graph: Dict) -> str:
    texts: List[str] = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "CLIPTextEncode":
            continue
        title = str((node.get("_meta") or {}).get("title", ""))
        if "Negative" in title:
            continue
        text = (node.get("inputs") or {}).get("text")
        if text:
            texts.append(_normalize_prompt_text(str(text)))
    return " ".join(texts).strip()


def _extract_prompt_from_comfyui_workflow(workflow: Dict) -> str:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return ""
    texts: List[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("type") != "CLIPTextEncode":
            continue
        title = str(node.get("title") or "")
        if "Negative" in title:
            continue
        widgets = node.get("widgets_values") or []
        if widgets:
            texts.append(_normalize_prompt_text(str(widgets[0])))
    return " ".join(texts).strip()


def _coerce_comfyui_metadata(meta: Dict) -> Dict:
    prompt = ""
    if any(isinstance(v, dict) and "class_type" in v for v in meta.values()):
        prompt = _extract_prompt_from_comfyui_graph(meta)
    if not prompt and "nodes" in meta:
        prompt = _extract_prompt_from_comfyui_workflow(meta)
    if not prompt:
        return {}
    return {"prompt": prompt, "source": "comfyui"}


def _read_image_metadata(path: Path) -> Dict:
    if not PIL_AVAILABLE:
        return {}
    sidecar = _read_sidecar_metadata(path)
    if sidecar:
        return sidecar
    try:
        with Image.open(path) as img:
            info = img.info or {}
            for key in ("metadata", "prompt", "parameters", "comment"):
                raw = info.get(key)
                if not raw:
                    continue
                parsed = _parse_metadata_json(raw)
                if parsed:
                    comfy = _coerce_comfyui_metadata(parsed)
                    return coerce_metadata_dict(comfy or parsed)
                if key in ("prompt", "parameters") and isinstance(raw, str):
                    return coerce_metadata_dict({"prompt": _normalize_prompt_text(raw)})
            return {}
    except Exception:
        return {}


def _read_video_metadata(path: Path) -> Dict:
    sidecar = _read_sidecar_metadata(path)
    if sidecar:
        return sidecar
    tags = _ffprobe_tags(str(path))
    return coerce_metadata_dict(_parse_metadata_json(tags.get("comment")))


def _read_sidecar_metadata(path: Path) -> Dict:
    candidates = [
        path.with_suffix(path.suffix + ".json"),
        path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            raw = _parse_metadata_json(candidate.read_text(encoding="utf-8", errors="ignore"))
            return coerce_metadata_dict(raw)
    return {}


def _convert_maker_metadata(meta: Dict) -> Dict[str, str]:
    if not meta:
        return {}
    raw = meta.get("raw_metadata") if isinstance(meta.get("raw_metadata"), dict) else {}
    tags = []
    source = meta.get("source") or raw.get("source")
    model = raw.get("model") or meta.get("model")
    scene_num = raw.get("scene_num") or meta.get("scene_num")
    script_index = raw.get("script_index") or meta.get("script_index")
    if source:
        tags.append(str(source))
    if model:
        tags.append(str(model))
    if scene_num:
        tags.append(f"Scene {scene_num}")
    if script_index:
        tags.append(f"Script {script_index}")
    folder_tags = ", ".join(tags)
    keywords = meta.get("keywords") or []
    visual_elements = meta.get("visual_elements") or []
    vision_tokens = []
    if isinstance(keywords, list):
        vision_tokens.extend([str(k) for k in keywords])
    elif isinstance(keywords, str):
        vision_tokens.append(keywords)
    if isinstance(visual_elements, list):
        vision_tokens.extend([str(v) for v in visual_elements])
    elif isinstance(visual_elements, str):
        vision_tokens.append(visual_elements)
    if not vision_tokens:
        vision_tokens.append(str(meta.get("prompt") or meta.get("query") or ""))
    vision_tags = ", ".join([t for t in vision_tokens if t])
    description = meta.get("description") or meta.get("generated_from_text") or ""
    created_at = meta.get("created_at") or ""
    return {
        "folder_tags": folder_tags,
        "vision_tags": vision_tags,
        "description": description,
        "created_at": created_at,
    }


def scan_library(
    library_dir: str,
    db: LibraryDB,
    target_db: str = "auto",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    library_path = Path(library_dir)
    if not library_path.exists():
        raise FileNotFoundError(library_dir)

    total_videos = 0
    total_images = 0
    for path in library_path.rglob("*"):
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTS:
            total_videos += 1
        elif suffix in IMAGE_EXTS:
            total_images += 1

    try:
        use_scene_detection = Config.get("use_scene_detection")
        if use_scene_detection is None:
            use_scene_detection = True
        logger.info(
            "Library scan: %d videos, %d images. Scene detection=%s",
            total_videos,
            total_images,
            "on" if use_scene_detection else "off",
        )
    except Exception:
        logger.info(
            "Library scan: %d videos, %d images.",
            total_videos,
            total_images,
        )

    count = 0
    total_items = total_videos + total_images
    for path in library_path.rglob("*"):
        suffix = path.suffix.lower()
        if suffix not in VIDEO_EXTS and suffix not in IMAGE_EXTS:
            continue

        folder_tags = " ".join(part for part in path.parent.parts if part)
        metadata = {
            "path": str(path),
            "duration": 0.0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "folder_tags": folder_tags,
            "vision_tags": "",
            "description": "",
        }

        if suffix in VIDEO_EXTS:
            info = _ffprobe_basic(str(path))
            metadata.update(
                {
                    "duration": info["duration"],
                    "fps": info["fps"],
                    "width": info["width"],
                    "height": info["height"],
                }
            )
            embedded = _read_video_metadata(path)
            sample_points = smart_frame_sampling(
                info["duration"],
                max_frames=12,
                base_frames=6,
                video_path=str(path),
            )
            embedding = encode_video_clip(str(path), sample_points=sample_points)
        else:
            embedded = _read_image_metadata(path)
            if PIL_AVAILABLE:
                try:
                    with Image.open(path) as img:
                        metadata.update({"width": img.width, "height": img.height})
                except Exception:
                    pass
            embedding = encode_image_clip(str(path))

        merged = _convert_maker_metadata(embedded)
        for key, value in merged.items():
            if value:
                metadata[key] = value

        if suffix in VIDEO_EXTS:
            try:
                use_sam3 = Config.get("use_sam3_tagging", False)
            except Exception:
                use_sam3 = False

            if use_sam3:
                logger.info(
                    "SAM3 auto-tagging is disabled in the indexing pipeline. Use the SAM3 tool in Misc tab."
                )

        db.upsert_clip(metadata, embedding, target_db=target_db)
        count += 1
        if progress_callback:
            progress_callback(count, total_items)
    return count

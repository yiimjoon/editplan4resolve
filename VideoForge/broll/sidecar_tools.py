import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCENE_RE = re.compile(r"(?i)scene[_\-\s]*(\d{1,4})")
STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
}


def _parse_metadata_json(raw: str | bytes | None) -> Dict[str, Any]:
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


def _extract_prompt(meta: Dict) -> str:
    prompt = ""
    if any(isinstance(v, dict) and "class_type" in v for v in meta.values()):
        prompt = _extract_prompt_from_comfyui_graph(meta)
    if not prompt and "nodes" in meta:
        prompt = _extract_prompt_from_comfyui_workflow(meta)
    if not prompt:
        prompt = str(meta.get("prompt") or meta.get("parameters") or "").strip()
    return _normalize_prompt_text(prompt) if prompt else ""


def _keywords_from_prompt(prompt: str, limit: int = 12) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", prompt.lower())
    seen = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token in seen:
            continue
        seen.append(token)
        if len(seen) >= limit:
            break
    return ", ".join(seen)


def _scene_number_from_name(path: Path) -> str | None:
    match = SCENE_RE.search(path.stem)
    if not match:
        return None
    return str(int(match.group(1)))


def _sidecar_path(asset_path: Path) -> Path:
    return asset_path.with_suffix(asset_path.suffix + ".json")


def _write_sidecar(sidecar_path: Path, meta: Dict[str, Any], overwrite: bool) -> bool:
    if sidecar_path.exists() and not overwrite:
        return False
    sidecar_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _ffprobe_comment_json(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format_tags=comment",
        "-of",
        "json",
        str(path),
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
        data = json.loads(proc.stdout or "{}")
        raw = ((data.get("format") or {}).get("tags") or {}).get("comment")
        if not raw:
            return {}
        if isinstance(raw, str):
            raw = raw.strip()
        return json.loads(raw)
    except Exception:
        return {}


def generate_scene_sidecars(
    broll_dir: Path,
    overwrite: bool = False,
    include_video: bool = False,
    dry_run: bool = False,
) -> int:
    videos = sorted([p for p in broll_dir.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv"}])
    meta_by_scene: dict[str, Dict[str, Any]] = {}
    for video in videos:
        scene = _scene_number_from_name(video)
        if not scene:
            continue
        meta = _ffprobe_comment_json(video)
        if not meta:
            continue
        meta_by_scene.setdefault(scene, meta)

    assets = sorted(
        [
            p
            for p in broll_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".mkv"}
        ]
    )
    written = 0
    for asset in assets:
        if asset.suffix.lower() in {".mp4", ".mov", ".mkv"} and not include_video:
            continue
        scene = _scene_number_from_name(asset)
        if not scene:
            continue
        meta = meta_by_scene.get(scene)
        if not meta:
            continue
        sidecar = _sidecar_path(asset)
        if dry_run:
            continue
        if _write_sidecar(sidecar, meta, overwrite=overwrite):
            written += 1
    return written


def generate_comfyui_sidecars(
    directory: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    sequence: bool = False,
) -> int:
    files = sorted(directory.rglob("ComfyUI_*.png"))
    if not files:
        return 0

    seq_map: Dict[Path, str] = {}
    if sequence:
        for idx, path in enumerate(files, start=1):
            seq_map[path] = str(idx)

    written = 0
    for file in files:
        from PIL import Image

        with Image.open(file) as img:
            info = img.info or {}
        raw = None
        for key in ("metadata", "prompt", "parameters", "comment"):
            raw = info.get(key)
            if raw:
                break
        meta = _parse_metadata_json(raw)
        prompt = _extract_prompt(meta)
        if not prompt:
            continue
        scene_num = (
            str(meta.get("scene_num") or "").strip()
            or _scene_number_from_name(file)
            or seq_map.get(file)
        )
        if not scene_num:
            continue
        script_index = str(meta.get("script_index") or scene_num)
        sidecar = {
            "scene_num": str(scene_num),
            "script_index": script_index,
            "description": prompt,
            "keywords": _keywords_from_prompt(prompt),
            "prompt": prompt,
            "source": "comfyui",
            "workflow": "ComfyUI",
        }
        sidecar_path = _sidecar_path(file)
        if dry_run:
            continue
        if _write_sidecar(sidecar_path, sidecar, overwrite=overwrite):
            written += 1
    return written

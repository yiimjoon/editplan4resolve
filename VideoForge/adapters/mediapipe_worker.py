from __future__ import annotations

import argparse
import bz2
import ctypes
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

_DEFAULT_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_DEFAULT_DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"


def _to_short_path(path: Path) -> Optional[Path]:
    if os.name != "nt":
        return None
    try:
        buffer_len = 512
        buf = ctypes.create_unicode_buffer(buffer_len)
        result = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, buffer_len)
        if result == 0:
            return None
        return Path(buf.value)
    except Exception:
        return None


def _ensure_face_landmarker_model(debug: Dict[str, Any]) -> Tuple[Optional[Path], Optional[str]]:
    env_path = os.environ.get("VIDEOFORGE_MEDIAPIPE_MODEL", "").strip()
    if env_path:
        path = Path(env_path)
        debug["model_path"] = str(path)
        debug["model_source"] = "env"
        if path.exists():
            return path, None
        return None, f"model_path_not_found: {path}"

    cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache" / "mediapipe"
    model_path = cache_dir / "face_landmarker.task"
    debug["model_path"] = str(model_path)
    if model_path.exists():
        debug["model_source"] = "cache"
        return model_path, None

    url = os.environ.get("VIDEOFORGE_MEDIAPIPE_MODEL_URL", _DEFAULT_FACE_LANDMARKER_URL)
    debug["model_url"] = url
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if not data:
            return None, "model_download_failed: empty download"
        model_path.write_bytes(data)
        debug["model_downloaded"] = True
        debug["model_source"] = "download"
        return model_path, None
    except Exception as exc:
        return None, f"model_download_failed: {exc}"


def _solve_yaw_from_landmarks(
    landmarks,
    frame_shape: Tuple[int, int, int],
    debug: Dict[str, Any],
    cv2,
    np,
) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    h, w = frame_shape[:2]
    debug["frame_shape"] = [int(h), int(w), int(frame_shape[2]) if len(frame_shape) > 2 else 1]

    def _pt(idx: int) -> Tuple[float, float]:
        lm = landmarks[idx]
        return float(lm.x * w), float(lm.y * h)

    image_points = np.array(
        [
            _pt(1),    # nose tip
            _pt(152),  # chin
            _pt(33),   # left eye
            _pt(263),  # right eye
            _pt(61),   # mouth left
            _pt(291),  # mouth right
        ],
        dtype="double",
    )

    xs = [float(lm.x) for lm in landmarks]
    ys = [float(lm.y) for lm in landmarks]
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        face_area_ratio = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
    else:
        face_area_ratio = 0.0
    return _solve_yaw_from_image_points(
        image_points=image_points,
        frame_shape=frame_shape,
        debug=debug,
        cv2=cv2,
        np=np,
        face_area_ratio=face_area_ratio,
    )


def _solve_yaw_from_image_points(
    image_points,
    frame_shape: Tuple[int, int, int],
    debug: Dict[str, Any],
    cv2,
    np,
    face_area_ratio: Optional[float] = None,
) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    h, w = frame_shape[:2]
    debug["frame_shape"] = [int(h), int(w), int(frame_shape[2]) if len(frame_shape) > 2 else 1]

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype="double",
    )

    focal_length = float(w)
    center = (float(w) / 2.0, float(h) / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        debug["solvepnp_failed"] = True
        return 0.0, 0.0, debug, "solvepnp_failed"

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = np.hstack((rotation_mat, translation_vec))
    _proj = cv2.decomposeProjectionMatrix(pose_mat)
    euler_angles = _proj[6]
    pitch = float(euler_angles[0])
    yaw = float(euler_angles[1])
    roll = float(euler_angles[2])
    debug["pose_deg"] = {"pitch": pitch, "yaw": yaw, "roll": roll}

    area = float(face_area_ratio or 0.0)
    confidence_divisor = float(os.environ.get("VIDEOFORGE_MEDIAPIPE_FACE_AREA", "0.15") or 0.15)
    if confidence_divisor <= 0.0:
        confidence_divisor = 0.15
    confidence = min(1.0, max(0.0, area / confidence_divisor))
    debug["face_area"] = area
    debug["confidence"] = confidence
    debug["confidence_divisor"] = confidence_divisor

    yaw = max(-90.0, min(90.0, yaw))
    debug["yaw_deg"] = float(yaw)
    return float(yaw), float(confidence), debug, None


def _ensure_dlib_model(debug: Dict[str, Any]) -> Tuple[Optional[Path], Optional[str]]:
    env_path = os.environ.get("VIDEOFORGE_DLIB_MODEL", "").strip()
    if env_path:
        path = Path(env_path)
        debug["dlib_model_path"] = str(path)
        debug["dlib_model_source"] = "env"
        if path.exists():
            return path, None
        return None, f"dlib_model_not_found: {path}"

    cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache" / "dlib"
    model_path = cache_dir / "shape_predictor_68_face_landmarks.dat"
    debug["dlib_model_path"] = str(model_path)
    if model_path.exists():
        debug["dlib_model_source"] = "cache"
        return model_path, None

    auto_download = os.environ.get("VIDEOFORGE_DLIB_AUTO_DOWNLOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not auto_download:
        return None, "dlib_model_missing"

    url = os.environ.get("VIDEOFORGE_DLIB_MODEL_URL", _DEFAULT_DLIB_MODEL_URL)
    debug["dlib_model_url"] = url
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        if not payload:
            return None, "dlib_model_download_failed: empty download"
        if url.endswith(".bz2"):
            payload = bz2.decompress(payload)
        model_path.write_bytes(payload)
        debug["dlib_model_downloaded"] = True
        debug["dlib_model_source"] = "download"
        return model_path, None
    except Exception as exc:
        return None, f"dlib_model_download_failed: {exc}"


def _estimate_yaw_from_frame_dlib(
    frame,
    debug: Dict[str, Any],
    cv2,
    np,
) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    try:
        import dlib  # type: ignore
    except Exception as exc:
        debug["dlib_error"] = f"dlib_import_failed: {exc}"
        return 0.0, 0.0, debug, f"dlib_import_failed: {exc}"

    model_path, error = _ensure_dlib_model(debug)
    if error or not model_path:
        debug["dlib_error"] = error or "dlib_model_missing"
        return 0.0, 0.0, debug, debug["dlib_error"]

    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(model_path))
    except Exception as exc:
        debug["dlib_error"] = f"dlib_init_failed: {exc}"
        return 0.0, 0.0, debug, debug["dlib_error"]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces: List[Any] = []
    opencv_boxes = debug.get("opencv_face_boxes")
    if isinstance(opencv_boxes, list) and opencv_boxes:
        try:
            box = max(opencv_boxes, key=lambda b: int(b.get("w", 0)) * int(b.get("h", 0)))
            x = int(box.get("x", 0))
            y = int(box.get("y", 0))
            w = int(box.get("w", 0))
            h = int(box.get("h", 0))
            if w > 0 and h > 0:
                pad_ratio = float(os.environ.get("VIDEOFORGE_DLIB_FACE_PAD", "0.2") or 0.2)
                pad = int(max(w, h) * pad_ratio)
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(frame.shape[1], x + w + pad)
                y1 = min(frame.shape[0], y + h + pad)
                rect = dlib.rectangle(int(x0), int(y0), int(x1), int(y1))
                debug["dlib_face_box"] = {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)}
                faces = [rect]
                debug["detect_variant"] = "dlib_opencv_box"
        except Exception as exc:
            debug["dlib_opencv_box_error"] = str(exc)

    if not faces:
        upsample = int(os.environ.get("VIDEOFORGE_DLIB_UPSAMPLE", "1") or 1)
        if upsample < 0:
            upsample = 0
        debug["dlib_upsample"] = upsample
        faces = detector(gray, upsample)
        debug["dlib_face_count"] = int(len(faces))
        if not faces:
            return 0.0, 0.0, debug, "no_face_dlib"
        rect = max(faces, key=lambda r: r.width() * r.height())
        x = int(rect.left())
        y = int(rect.top())
        w = int(rect.width())
        h = int(rect.height())
        debug["dlib_face_box"] = {"x": x, "y": y, "w": w, "h": h}
    else:
        rect = faces[0]
        x = int(rect.left())
        y = int(rect.top())
        w = int(rect.right() - rect.left())
        h = int(rect.bottom() - rect.top())

    shape = predictor(gray, rect)
    debug["dlib_landmark_count"] = int(shape.num_parts)
    indices = {
        "nose": 30,
        "chin": 8,
        "left_eye": 36,
        "right_eye": 45,
        "mouth_left": 48,
        "mouth_right": 54,
    }
    image_points = np.array(
        [
            (float(shape.part(indices["nose"]).x), float(shape.part(indices["nose"]).y)),
            (float(shape.part(indices["chin"]).x), float(shape.part(indices["chin"]).y)),
            (float(shape.part(indices["left_eye"]).x), float(shape.part(indices["left_eye"]).y)),
            (float(shape.part(indices["right_eye"]).x), float(shape.part(indices["right_eye"]).y)),
            (float(shape.part(indices["mouth_left"]).x), float(shape.part(indices["mouth_left"]).y)),
            (float(shape.part(indices["mouth_right"]).x), float(shape.part(indices["mouth_right"]).y)),
        ],
        dtype="double",
    )
    frame_h, frame_w = frame.shape[:2]
    face_area_ratio = float((w * h) / float(frame_w * frame_h)) if frame_w and frame_h else 0.0
    debug["dlib_face_area"] = face_area_ratio
    if "detect_variant" not in debug:
        debug["detect_variant"] = "dlib"
    debug["no_face"] = False
    return _solve_yaw_from_image_points(
        image_points=image_points,
        frame_shape=frame.shape,
        debug=debug,
        cv2=cv2,
        np=np,
        face_area_ratio=face_area_ratio,
    )


def _detect_faces_with_opencv(frame, debug: Dict[str, Any], cv2) -> List[Tuple[int, int, int, int]]:
    faces: List[Tuple[int, int, int, int]] = []
    try:
        cascade_root = getattr(cv2.data, "haarcascades", "")
        cascades = [
            ("frontal_default", "haarcascade_frontalface_default.xml"),
            ("frontal_alt2", "haarcascade_frontalface_alt2.xml"),
            ("profile", "haarcascade_profileface.xml"),
        ]
        scale = float(os.environ.get("VIDEOFORGE_OPENCV_FACE_SCALE", "1.05") or 1.05)
        min_neighbors = int(os.environ.get("VIDEOFORGE_OPENCV_FACE_NEIGHBORS", "3") or 3)
        min_size = int(os.environ.get("VIDEOFORGE_OPENCV_FACE_MINSIZE", "24") or 24)
        if scale < 1.01:
            scale = 1.01
        if min_neighbors < 1:
            min_neighbors = 1
        if min_size < 10:
            min_size = 10
        debug["opencv_face_params"] = {
            "scaleFactor": scale,
            "minNeighbors": min_neighbors,
            "minSize": min_size,
        }
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sources: List[Dict[str, Any]] = []
        for name, filename in cascades:
            cascade_path = os.path.join(cascade_root, filename)
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                continue
            detected = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale,
                minNeighbors=min_neighbors,
                minSize=(min_size, min_size),
            )
            boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detected]
            if name == "profile":
                flipped = cv2.flip(gray, 1)
                detected_flip = face_cascade.detectMultiScale(
                    flipped,
                    scaleFactor=scale,
                    minNeighbors=min_neighbors,
                    minSize=(min_size, min_size),
                )
                w_img = frame.shape[1]
                for (x, y, w, h) in detected_flip:
                    boxes.append((int(w_img - x - w), int(y), int(w), int(h)))
            if boxes:
                faces.extend(boxes)
            sources.append({"cascade": name, "count": int(len(boxes))})
        if sources:
            debug["opencv_face_sources"] = sources
        if not faces:
            debug["opencv_face_detector"] = "no_faces"
            return faces

        debug["opencv_face_count"] = int(len(faces))
        if len(faces) > 0:
            areas: List[int] = []
            boxes: List[Dict[str, int]] = []
            for (x, y, w, h) in faces[:3]:
                areas.append(int(w * h))
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
            debug["opencv_face_max_area"] = int(max(areas))
            debug["opencv_face_boxes"] = boxes
    except Exception as exc:
        debug["opencv_face_detector_error"] = str(exc)
    return faces


def _detect_landmarks_tasks(
    landmarker,
    frame,
    cv2,
    mp,
) -> Optional[Any]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_image)
    if results and results.face_landmarks:
        return results.face_landmarks[0]
    return None


def _estimate_yaw_from_frame_tasks(
    frame,
    debug: Dict[str, Any],
    cv2,
    np,
    mp,
) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    try:
        from mediapipe.tasks.python import vision  # type: ignore
        from mediapipe.tasks.python.core import base_options  # type: ignore
    except Exception as exc:
        debug["error"] = f"mediapipe_tasks_import_failed: {exc}"
        return 0.0, 0.0, debug, f"mediapipe_tasks_import_failed: {exc}"

    model_path, error = _ensure_face_landmarker_model(debug)
    if error or not model_path:
        debug["error"] = error or "model_missing"
        return 0.0, 0.0, debug, error or "model_missing"

    min_detect = float(os.environ.get("VIDEOFORGE_MEDIAPIPE_MIN_DETECT", "0.3") or 0.3)
    min_presence = float(os.environ.get("VIDEOFORGE_MEDIAPIPE_MIN_PRESENCE", "0.3") or 0.3)
    min_track = float(os.environ.get("VIDEOFORGE_MEDIAPIPE_MIN_TRACK", "0.3") or 0.3)
    debug["mediapipe_thresholds"] = {
        "min_detect": min_detect,
        "min_presence": min_presence,
        "min_track": min_track,
    }

    options = vision.FaceLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=min_detect,
        min_face_presence_confidence=min_presence,
        min_tracking_confidence=min_track,
    )

    landmarker = vision.FaceLandmarker.create_from_options(options)
    try:
        landmarks = _detect_landmarks_tasks(landmarker, frame, cv2, mp)
        if landmarks:
            debug["detect_variant"] = "full"
            debug["landmark_count"] = len(landmarks)
            return _solve_yaw_from_landmarks(landmarks, frame.shape, debug, cv2, np)

        max_side = int(os.environ.get("VIDEOFORGE_MEDIAPIPE_MAX_SIDE", "1280") or 1280)
        h, w = frame.shape[:2]
        if max_side > 0 and max(h, w) > max_side:
            scale = float(max_side) / float(max(h, w))
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            landmarks = _detect_landmarks_tasks(landmarker, resized, cv2, mp)
            if landmarks:
                debug["detect_variant"] = "downscale"
                debug["landmark_count"] = len(landmarks)
                return _solve_yaw_from_landmarks(landmarks, resized.shape, debug, cv2, np)

        faces = _detect_faces_with_opencv(frame, debug, cv2)
        if faces:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            pad_ratio = float(os.environ.get("VIDEOFORGE_MEDIAPIPE_FACE_PAD", "0.25") or 0.25)
            pad = int(max(w, h) * pad_ratio)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad)
            y1 = min(frame.shape[0], y + h + pad)
            crop = frame[y0:y1, x0:x1]
            debug["opencv_face_crop"] = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
            landmarks = _detect_landmarks_tasks(landmarker, crop, cv2, mp)
            if landmarks:
                debug["detect_variant"] = "opencv_crop"
                debug["landmark_count"] = len(landmarks)
                return _solve_yaw_from_landmarks(landmarks, crop.shape, debug, cv2, np)
    finally:
        try:
            landmarker.close()
        except Exception:
            pass

    debug["no_face"] = True
    try:
        debug["image_mean"] = float(np.mean(frame))
        debug["image_std"] = float(np.std(frame))
    except Exception:
        pass
    if "opencv_face_count" not in debug:
        _detect_faces_with_opencv(frame, debug, cv2)
    yaw, confidence, debug, error = _estimate_yaw_from_frame_dlib(frame, debug, cv2, np)
    if not error:
        return yaw, confidence, debug, None
    return 0.0, 0.0, debug, error or "no_face"


def _estimate_yaw_from_frame(frame) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    debug: Dict[str, Any] = {}
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        debug["error"] = f"mediapipe_import_failed: {exc}"
        return 0.0, 0.0, debug, f"mediapipe_import_failed: {exc}"

    solutions = getattr(mp, "solutions", None)
    face_mesh = getattr(solutions, "face_mesh", None) if solutions else None
    if not face_mesh:
        debug["mediapipe_mode"] = "tasks"
        return _estimate_yaw_from_frame_tasks(frame, debug, cv2, np, mp)

    debug["mediapipe_mode"] = "solutions"
    mp_face_mesh = face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    if not results or not results.multi_face_landmarks:
        debug["no_face"] = True
        try:
            import numpy as np  # type: ignore

            debug["image_mean"] = float(np.mean(frame))
            debug["image_std"] = float(np.std(frame))
        except Exception:
            pass
        _detect_faces_with_opencv(frame, debug, cv2)
        yaw, confidence, debug, error = _estimate_yaw_from_frame_dlib(frame, debug, cv2, np)
        if not error:
            return yaw, confidence, debug, None
        return 0.0, 0.0, debug, error or "no_face"

    landmarks = results.multi_face_landmarks[0].landmark
    debug["landmark_count"] = len(landmarks)
    return _solve_yaw_from_landmarks(landmarks, frame.shape, debug, cv2, np)


def estimate_yaw_from_frame(frame) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    return _estimate_yaw_from_frame(frame)


def estimate_yaw_from_video(
    video_path: Path, timestamp_sec: float
) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    debug: Dict[str, Any] = {
        "worker_path": str(Path(__file__).resolve()),
        "worker_cwd": os.getcwd(),
        "worker_exe": sys.executable,
        "sys_path0": sys.path[0] if sys.path else None,
    }
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return 0.0, 0.0, debug, f"opencv_import_failed: {exc}"

    used_path = video_path
    used_short_path = False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        short_path = _to_short_path(video_path)
        if short_path and short_path != video_path:
            used_path = short_path
            used_short_path = True
            cap = cv2.VideoCapture(str(short_path))
    if not cap.isOpened():
        debug["path_used"] = str(used_path)
        debug["used_short_path"] = used_short_path
        return 0.0, 0.0, debug, f"cap_open_failed: {video_path}"

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    debug["path_used"] = str(used_path)
    debug["used_short_path"] = used_short_path
    debug["fps"] = fps
    if fps <= 0.0:
        cap.release()
        return 0.0, 0.0, debug, f"fps_missing: {video_path}"

    frame_idx = max(0, int(round(float(timestamp_sec) * fps)))
    debug["frame_idx"] = frame_idx

    radius = int(os.environ.get("VIDEOFORGE_MEDIAPIPE_FRAME_RADIUS", "2") or 2)
    if radius < 0:
        radius = 0
    candidates = [frame_idx]
    for offset in range(1, radius + 1):
        candidates.extend([frame_idx - offset, frame_idx + offset])
    candidates = [idx for idx in candidates if idx >= 0]
    debug["frame_idx_candidates"] = candidates

    best = None
    best_conf = -1.0
    last_error = None
    last_debug: Optional[Dict[str, Any]] = None

    for idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            last_error = f"frame_read_failed: {video_path} @ {idx}"
            continue
        yaw, confidence, frame_debug, error = estimate_yaw_from_frame(frame)
        frame_debug = dict(frame_debug or {})
        frame_debug["frame_idx"] = idx
        last_debug = frame_debug
        if error:
            if error != "no_face":
                last_error = error
            continue
        if confidence > best_conf:
            best = (yaw, confidence, frame_debug)
            best_conf = confidence
            if confidence >= 0.9:
                break

    cap.release()
    if best:
        yaw, confidence, frame_debug = best
        debug.update(frame_debug)
        debug["frame_idx_used"] = frame_debug.get("frame_idx")
        return float(yaw), float(confidence), debug, None

    if last_debug:
        debug.update(last_debug)
        debug["frame_idx_used"] = last_debug.get("frame_idx")
    return 0.0, 0.0, debug, last_error or "no_face"


def _handle_gaze_score(video_path: Path, timestamp_sec: float) -> Dict[str, Any]:
    yaw, confidence, debug, error = estimate_yaw_from_video(video_path, timestamp_sec)
    payload = {
        "yaw_deg": float(yaw),
        "confidence": float(confidence),
        "debug": debug,
    }
    if error:
        payload["error"] = error
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    gaze = sub.add_parser("gaze_score")
    gaze.add_argument("--video", required=True)
    gaze.add_argument("--timestamp", type=float, default=0.0)

    args = parser.parse_args()

    if args.cmd == "gaze_score":
        payload = _handle_gaze_score(Path(args.video), float(args.timestamp))
        print(json.dumps(payload))


if __name__ == "__main__":
    main()

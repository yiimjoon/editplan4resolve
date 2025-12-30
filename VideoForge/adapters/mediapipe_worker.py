from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


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


def _estimate_yaw_from_frame(frame) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    debug: Dict[str, Any] = {}
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        return 0.0, 0.0, debug, f"mediapipe_import_failed: {exc}"

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    if not results or not results.multi_face_landmarks:
        return 0.0, 0.0, debug, "no_face"

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]
    debug["frame_shape"] = [int(h), int(w), int(frame.shape[2]) if frame.ndim > 2 else 1]

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
        return 0.0, 0.0, debug, "solvepnp_failed"

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = np.hstack((rotation_mat, translation_vec))
    _proj = cv2.decomposeProjectionMatrix(pose_mat)
    euler_angles = _proj[6]
    pitch = float(euler_angles[0])
    yaw = float(euler_angles[1])
    roll = float(euler_angles[2])
    debug["pose_deg"] = {"pitch": pitch, "yaw": yaw, "roll": roll}

    xs = [float(lm.x) for lm in landmarks]
    ys = [float(lm.y) for lm in landmarks]
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
    else:
        area = 0.0
    confidence = min(1.0, max(0.0, area / 0.15))
    debug["face_area"] = area
    debug["confidence"] = confidence

    yaw = max(-90.0, min(90.0, yaw))
    return float(yaw), float(confidence), debug, None


def estimate_yaw_from_frame(frame) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    return _estimate_yaw_from_frame(frame)


def estimate_yaw_from_video(
    video_path: Path, timestamp_sec: float
) -> Tuple[float, float, Dict[str, Any], Optional[str]]:
    debug: Dict[str, Any] = {}
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return 0.0, 0.0, debug, f"frame_read_failed: {video_path} @ {timestamp_sec:.2f}s"

    yaw, confidence, frame_debug, error = estimate_yaw_from_frame(frame)
    debug.update(frame_debug)
    return yaw, confidence, debug, error


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

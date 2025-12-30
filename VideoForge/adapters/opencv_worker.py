from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from VideoForge.config.config_manager import Config


def _scene_detect(video_path: Path, threshold: float, sample_rate: int) -> List[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [0.0]

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        return [0.0]

    prev_hist = None
    scene_timestamps: List[float] = [0.0]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if sample_rate > 1 and frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) * 100.0
            if diff > threshold:
                scene_timestamps.append(frame_idx / fps)

        prev_hist = hist
        frame_idx += 1

    duration = frame_idx / fps if fps > 0 else 0.0
    cap.release()

    if not scene_timestamps or scene_timestamps[-1] < max(0.0, duration - 1.0):
        scene_timestamps.append(duration)

    return scene_timestamps


def _quality_check(video_path: Path, sample_frames: int) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"blur_score": 0.0, "brightness": 0.0, "noise_level": 0.0, "quality_score": 0.0}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return {"blur_score": 0.0, "brightness": 0.0, "noise_level": 0.0, "quality_score": 0.0}

    sample_frames = max(1, sample_frames)
    start_idx = int(frame_count * 0.1)
    end_idx = max(start_idx + 1, int(frame_count * 0.9))
    frame_indices = np.linspace(start_idx, end_idx, sample_frames, dtype=int)

    blur_scores = []
    brightness_scores = []
    noise_scores = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_scores.append(float(laplacian.var()))
        brightness_scores.append(float(gray.mean()))

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(sobelx**2 + sobely**2)
        noise_scores.append(float(edge_strength.std()))

    cap.release()

    if not blur_scores:
        return {"blur_score": 0.0, "brightness": 0.0, "noise_level": 0.0, "quality_score": 0.0}

    avg_blur = float(np.mean(blur_scores))
    avg_brightness = float(np.mean(brightness_scores))
    avg_noise = float(np.mean(noise_scores))

    blur_quality = min(avg_blur / 150.0, 1.0)
    brightness_quality = 1.0 - min(abs(avg_brightness - 128.0) / 128.0, 1.0)
    noise_quality = max(1.0 - (avg_noise / 30.0), 0.0)
    quality_score = blur_quality * 0.5 + brightness_quality * 0.3 + noise_quality * 0.2

    return {
        "blur_score": avg_blur,
        "brightness": avg_brightness,
        "noise_level": avg_noise,
        "quality_score": float(quality_score),
    }


def _edge_hist_variance(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
    return float(np.var(hist))


def _calc_motion(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> float:
    if prev_gray is None:
        return 0.0
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=120, qualityLevel=0.01, minDistance=7)
    if prev_pts is None:
        return 0.0
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    if next_pts is None or status is None:
        return 0.0
    valid = status.flatten() == 1
    if not valid.any():
        return 0.0
    displacements = next_pts[valid] - prev_pts[valid]
    magnitudes = np.linalg.norm(displacements, axis=2)
    return float(np.mean(magnitudes))


def _face_box_score(x: float, y: float, w: float, h: float) -> float:
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
    center_score = max(0.0, 1.0 - min(distance / 0.5, 1.0))
    size_score = min((w * h) / 0.2, 1.0)
    return center_score * size_score


def _detect_face_dnn(frame: np.ndarray) -> Optional[float]:
    model_dir = str(Config.get("multicam_face_model_dir") or "").strip()
    if not model_dir:
        model_dir = str(Path(__file__).resolve().parents[1] / "models" / "face_detection")
    model_dir_path = Path(model_dir)
    prototxt = model_dir_path / "deploy.prototxt"
    caffemodel = model_dir_path / "res10_300x300_ssd_iter_140000.caffemodel"
    if not (prototxt.exists() and caffemodel.exists()):
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    except Exception:
        return None
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()
    if detections is None or detections.shape[2] == 0:
        return None
    best = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue
        box = detections[0, 0, i, 3:7]
        x0, y0, x1, y1 = box.tolist()
        score = _face_box_score(x0, y0, x1 - x0, y1 - y0)
        if best is None or score > best:
            best = score
    return best


def _detect_face_haar(frame: np.ndarray) -> float:
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
    except Exception:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if faces is None or len(faces) == 0:
        return 0.0
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    h_frame, w_frame = gray.shape[:2]
    return _face_box_score(
        float(x) / float(w_frame),
        float(y) / float(h_frame),
        float(w) / float(w_frame),
        float(h) / float(h_frame),
    )


def _face_score(frame: np.ndarray, face_detector: str) -> float:
    detector = str(face_detector or "opencv_dnn").strip().lower()
    if detector == "mediapipe":
        try:
            import mediapipe as mp  # type: ignore

            mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb)
            if not results or not results.detections:
                return 0.0
            detection = results.detections[0]
            box = detection.location_data.relative_bounding_box
            return _face_box_score(
                float(box.xmin),
                float(box.ymin),
                float(box.width),
                float(box.height),
            )
        except Exception:
            detector = "opencv_dnn"

    if detector == "opencv_dnn":
        score = _detect_face_dnn(frame)
        if score is not None:
            return score

    return _detect_face_haar(frame)


def _angle_score(video_path: Path, timestamp: float, face_detector: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"sharpness": 0.0, "motion": 0.0, "stability": 0.0, "face_score": 0.0}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        return {"sharpness": 0.0, "motion": 0.0, "stability": 0.0, "face_score": 0.0}

    frame_idx = max(0, int(round(float(timestamp) * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return {"sharpness": 0.0, "motion": 0.0, "stability": 0.0, "face_score": 0.0}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = None
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
        prev_ret, prev_frame = cap.read()
        if prev_ret and prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cap.release()

    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0, 1.0)
    motion = min(_calc_motion(prev_gray, gray) / 10.0, 1.0)
    stability = 1.0 - min(_edge_hist_variance(gray) / 10000.0, 1.0)
    face_score = min(max(_face_score(frame, face_detector), 0.0), 1.0)

    return {
        "sharpness": float(max(sharpness, 0.0)),
        "motion": float(max(motion, 0.0)),
        "stability": float(max(stability, 0.0)),
        "face_score": float(face_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    scene = sub.add_parser("scene_detect")
    scene.add_argument("--video", required=True)
    scene.add_argument("--threshold", type=float, default=30.0)
    scene.add_argument("--sample-rate", type=int, default=5)

    quality = sub.add_parser("quality_check")
    quality.add_argument("--video", required=True)
    quality.add_argument("--sample-frames", type=int, default=5)

    angle = sub.add_parser("angle_score")
    angle.add_argument("--video", required=True)
    angle.add_argument("--timestamp", type=float, default=0.0)
    angle.add_argument("--face-detector", default="opencv_dnn")

    args = parser.parse_args()

    if args.cmd == "scene_detect":
        scenes = _scene_detect(Path(args.video), float(args.threshold), int(args.sample_rate))
        print(json.dumps({"scenes": scenes}))
        return

    if args.cmd == "quality_check":
        metrics = _quality_check(Path(args.video), int(args.sample_frames))
        print(json.dumps({"metrics": metrics}))
        return

    metrics = _angle_score(Path(args.video), float(args.timestamp), str(args.face_detector))
    print(json.dumps({"metrics": metrics}))


if __name__ == "__main__":
    main()


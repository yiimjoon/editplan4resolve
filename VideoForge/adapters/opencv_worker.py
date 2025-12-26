from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


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

    args = parser.parse_args()

    if args.cmd == "scene_detect":
        scenes = _scene_detect(Path(args.video), float(args.threshold), int(args.sample_rate))
        print(json.dumps({"scenes": scenes}))
        return

    metrics = _quality_check(Path(args.video), int(args.sample_frames))
    print(json.dumps({"metrics": metrics}))


if __name__ == "__main__":
    main()


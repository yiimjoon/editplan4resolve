"""OpenCV-based angle scoring utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from VideoForge.config.config_manager import Config

logger = logging.getLogger(__name__)


class AngleScorer:
    """Compute quality scores for an angle using OpenCV metrics."""

    def __init__(self, face_detector: Optional[str] = None) -> None:
        self.face_detector = (
            str(face_detector or Config.get("multicam_face_detector", "opencv_dnn"))
            .strip()
            .lower()
        )
        self._prev_gray = None
        self._face_net = None
        self._face_proto: Optional[Path] = None
        self._face_model: Optional[Path] = None
        self._face_detector_type = self.face_detector
        self._init_face_detector()

    def _init_face_detector(self) -> None:
        if self.face_detector != "opencv_dnn":
            return
        try:
            from VideoForge.adapters.opencv_subprocess import should_use_subprocess

            if should_use_subprocess():
                return
        except Exception:
            pass

        model_dir = self._resolve_face_model_dir()
        if not model_dir:
            self._face_detector_type = "haar"
            return
        self._face_proto = model_dir / "deploy.prototxt"
        self._face_model = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        if not (self._face_proto.exists() and self._face_model.exists()):
            self._face_detector_type = "haar"
            return

        try:
            import cv2  # type: ignore

            self._face_net = cv2.dnn.readNetFromCaffe(str(self._face_proto), str(self._face_model))
        except Exception:
            self._face_detector_type = "haar"

    @staticmethod
    def _resolve_face_model_dir() -> Optional[Path]:
        default_dir = Path(__file__).resolve().parent.parent / "models" / "face_detection"
        if default_dir.exists():
            return default_dir
        configured = str(Config.get("multicam_face_model_dir", "") or "").strip()
        if configured:
            return Path(configured).expanduser()
        return None

    def score_frame(self, frame) -> Dict[str, float]:
        """Score a single frame. Requires OpenCV in-process."""
        try:
            from VideoForge.adapters.opencv_subprocess import should_use_subprocess

            if should_use_subprocess():
                raise RuntimeError("OpenCV subprocess mode enabled; use score_video_frame().")
        except Exception:
            pass

        import cv2  # type: ignore
        import numpy as np  # type: ignore

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = self._normalize_sharpness(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        stability = self._normalize_stability(self._edge_hist_variance(gray, cv2))
        motion = self._normalize_motion(self._calc_motion(self._prev_gray, gray, cv2, np))
        face_score = self._normalize_face_score(self._detect_face_score(frame, cv2))
        self._prev_gray = gray
        return {
            "sharpness": sharpness,
            "motion": motion,
            "stability": stability,
            "face_score": face_score,
        }

    def score_video_frame(self, video_path: Path, timestamp_sec: float) -> Dict[str, float]:
        """Score a frame from a video path at the given timestamp."""
        try:
            from VideoForge.adapters.opencv_subprocess import run_angle_score, should_use_subprocess

            if should_use_subprocess():
                return run_angle_score(
                    video_path=Path(video_path),
                    timestamp_sec=float(timestamp_sec),
                    face_detector=self.face_detector,
                )
        except Exception as exc:
            logger.debug("OpenCV subprocess unavailable: %s", exc)

        import cv2  # type: ignore
        import numpy as np  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            cap.release()
            raise RuntimeError("Invalid FPS for video scoring.")
        frame_idx = max(0, int(round(float(timestamp_sec) * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError("Failed to read target frame.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        prev_gray = None
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
            prev_ret, prev_frame = cap.read()
            if prev_ret and prev_frame is not None:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        cap.release()

        sharpness = self._normalize_sharpness(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        stability = self._normalize_stability(self._edge_hist_variance(gray, cv2))
        motion = self._normalize_motion(self._calc_motion(prev_gray, gray, cv2, np))
        face_score = self._normalize_face_score(self._detect_face_score(frame, cv2))
        return {
            "sharpness": sharpness,
            "motion": motion,
            "stability": stability,
            "face_score": face_score,
        }

    @staticmethod
    def _edge_hist_variance(gray, cv2) -> float:
        edges = cv2.Canny(gray, 100, 200)
        hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        try:
            import numpy as np  # type: ignore

            return float(np.var(hist))
        except Exception:
            return float(hist.var())

    @staticmethod
    def _calc_motion(prev_gray, gray, cv2, np) -> float:
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

    def _detect_face_score(self, frame, cv2) -> float:
        detector = self._face_detector_type
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
                return self._face_box_score(
                    float(box.xmin),
                    float(box.ymin),
                    float(box.width),
                    float(box.height),
                )
            except Exception:
                detector = "opencv_dnn"

        if detector == "opencv_dnn":
            dnn_score = self._detect_face_dnn(frame, cv2)
            if dnn_score is not None:
                return dnn_score

        return self._detect_face_haar(frame, cv2)

    def _detect_face_dnn(self, frame, cv2) -> Optional[float]:
        if self._face_detector_type != "opencv_dnn":
            return None
        if self._face_net is None:
            if not self._load_face_net(cv2):
                return None
        net = self._face_net
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
            score = AngleScorer._face_box_score(x0, y0, x1 - x0, y1 - y0)
            if best is None or score > best:
                best = score
        return best

    def _load_face_net(self, cv2) -> bool:
        if self._face_net is not None:
            return True
        model_dir = self._resolve_face_model_dir()
        if not model_dir:
            self._face_detector_type = "haar"
            return False
        self._face_proto = model_dir / "deploy.prototxt"
        self._face_model = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        if not (self._face_proto.exists() and self._face_model.exists()):
            self._face_detector_type = "haar"
            return False
        try:
            self._face_net = cv2.dnn.readNetFromCaffe(str(self._face_proto), str(self._face_model))
            return True
        except Exception:
            self._face_detector_type = "haar"
            return False

    @staticmethod
    def _detect_face_haar(frame, cv2) -> float:
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
        return AngleScorer._face_box_score(
            float(x) / float(w_frame),
            float(y) / float(h_frame),
            float(w) / float(w_frame),
            float(h) / float(h_frame),
        )

    @staticmethod
    def _face_box_score(x: float, y: float, w: float, h: float) -> float:
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
        center_score = max(0.0, 1.0 - min(distance / 0.5, 1.0))
        size_score = min((w * h) / 0.2, 1.0)
        return center_score * size_score

    @staticmethod
    def _normalize_sharpness(value: float) -> float:
        return min(max(value / 1000.0, 0.0), 1.0)

    @staticmethod
    def _normalize_motion(value: float) -> float:
        return min(max(value / 10.0, 0.0), 1.0)

    @staticmethod
    def _normalize_stability(value: float) -> float:
        return 1.0 - min(max(value / 10000.0, 0.0), 1.0)

    @staticmethod
    def _normalize_face_score(value: float) -> float:
        return min(max(value, 0.0), 1.0)

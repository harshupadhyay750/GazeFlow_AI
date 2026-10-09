from collections import deque
from pathlib import Path

import cv2
import numpy as np

from calibration.calibration_model import CalibrationModel
from core.face_tracker import FaceTracker
from core.gaze_estimator import EyeMeasurement, GazeEstimator
from core.gaze_smoother import GazeSmoother


class TrackingEngine:
    FACE_POINTS = (10, 152, 234, 454, 1)
    EYE_COLORS = {"left": (255, 210, 50), "right": (60, 220, 255)}
    IRIS_COLORS = {"left": (230, 80, 255), "right": (70, 255, 130)}
    IRIS_POINTS = {"left": range(473, 478), "right": range(468, 473)}

    def __init__(self, model_path: Path, settings: dict) -> None:
        self.tracker = FaceTracker(model_path)
        self.estimator = GazeEstimator(
            float(settings.get("horizontal_threshold", 0.12)),
            float(settings.get("vertical_threshold", 0.14)),
        )
        self.calibration_model: CalibrationModel | None = None
        self.gaze_smoother = GazeSmoother(
            smoothing=float(settings.get("gaze_smoothing", 0.35)),
            dead_zone=float(settings.get("center_dead_zone", 0.05)),
            hysteresis=float(settings.get("gaze_hysteresis", 0.08)),
        )
        self.minimum_confidence = float(settings.get("minimum_confidence", 0.45))
        self.gaze_history: deque[tuple[float, float]] = deque(
            maxlen=int(settings.get("stability_window", 10))
        )
        self.last_snapshot: dict | None = None

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, dict]:
        preview = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        faces = self.tracker.detect(rgb_frame, timestamp_ms)

        if len(faces) != 1:
            self.gaze_history.clear()
            multiple_faces = len(faces) > 1
            return preview, {
                "state": "NO_FACE",
                "detail": "Multiple faces detected; keep one face in view." if multiple_faces
                else "No face detected. Center your face in the camera.",
                "face": "MULTIPLE" if multiple_faces else "NOT DETECTED",
                "left_eye": "NOT DETECTED",
                "right_eye": "NOT DETECTED",
                "gaze": "--",
                "confidence": 0,
            }

        landmarks = faces[0]
        gaze, eyes = self.estimator.estimate(landmarks)
        self._draw_landmarks(preview, landmarks, eyes)
        measurable = [
            eye for eye in eyes.values()
            if eye.visible and eye.iris_available
        ]
        if len(measurable) == 2:
            self.gaze_history.append((
                sum(eye.horizontal for eye in measurable) / 2,
                sum(eye.vertical for eye in measurable) / 2,
            ))
        else:
            self.gaze_history.clear()

        gaze_position = self.estimator.estimate_position(landmarks, self.calibration_model)
        if gaze_position is not None:
            smoothed_position = self.gaze_smoother.apply(*gaze_position)
        else:
            smoothed_position = (0.0, 0.0)

        confidence = self._confidence(eyes)
        state = "LOW_CONFIDENCE" if confidence < self.minimum_confidence * 100 else "TRACKING"
        self.last_snapshot = {
            "features": self.estimator.extract_features(landmarks),
            "gaze_position": smoothed_position,
            "gaze": gaze,
            "confidence": confidence,
            "valid": gaze_position is not None,
        }
        return preview, {
            "state": state,
            "detail": "Both eyes and iris landmarks are being tracked."
            if state == "TRACKING" else "Improve lighting and face the camera directly.",
            "face": "DETECTED",
            "left_eye": self._eye_status(eyes["left"]),
            "right_eye": self._eye_status(eyes["right"]),
            "gaze": gaze or "--",
            "confidence": confidence,
            "gaze_position": smoothed_position,
            "feature_vector": self.estimator.extract_features(landmarks),
        }

    @staticmethod
    def _eye_status(eye: EyeMeasurement) -> str:
        if not eye.visible:
            return "NOT DETECTED"
        return "DETECTED" if eye.iris_available else "EYE ONLY"

    def _confidence(self, eyes: dict[str, EyeMeasurement]) -> int:
        score = 30
        score += 20 * sum(eye.visible for eye in eyes.values())
        score += 10 * sum(eye.iris_available for eye in eyes.values())
        if len(self.gaze_history) < 2:
            stability = 1.0
        else:
            positions = np.asarray(self.gaze_history, dtype=np.float32)
            jitter = float(np.std(positions, axis=0).mean())
            stability = max(0.0, 1.0 - jitter / 0.12)
        score += 10 * stability
        return max(0, min(100, round(score)))

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list[tuple[float, float]],
        eyes: dict[str, EyeMeasurement],
    ) -> None:
        height, width = frame.shape[:2]
        for index in self.FACE_POINTS:
            if index < len(landmarks):
                x, y = landmarks[index]
                cv2.circle(frame, (int(x * width), int(y * height)), 3, (70, 245, 145), -1)

        eye_indices = {
            "left": (*GazeEstimator.LEFT_EYE["corners"], *GazeEstimator.LEFT_EYE["vertical"]),
            "right": (*GazeEstimator.RIGHT_EYE["corners"], *GazeEstimator.RIGHT_EYE["vertical"]),
        }
        for side, indices in eye_indices.items():
            for index in indices:
                if index < len(landmarks):
                    x, y = landmarks[index]
                    cv2.circle(
                        frame,
                        (int(x * width), int(y * height)),
                        3,
                        self.EYE_COLORS[side],
                        -1,
                    )

            if eyes[side].iris_available:
                for index in self.IRIS_POINTS[side]:
                    if index < len(landmarks):
                        x, y = landmarks[index]
                        cv2.circle(
                            frame,
                            (int(x * width), int(y * height)),
                            4 if index == GazeEstimator.LEFT_EYE["iris"]
                            or index == GazeEstimator.RIGHT_EYE["iris"] else 2,
                            self.IRIS_COLORS[side],
                            -1,
                        )

    def close(self) -> None:
        self.tracker.close()
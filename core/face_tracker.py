from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceTrackerError(RuntimeError):
    """Raised when the MediaPipe model cannot be loaded or used."""


class FaceTracker:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FaceTrackerError(
                f"The Face Landmarker model was not found at:\n{model_path}\n\n"
                "See the README for the required face_landmarker.task download."
            )

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except (OSError, RuntimeError, ValueError) as error:
            raise FaceTrackerError(
                f"MediaPipe could not load the Face Landmarker model: {error}"
            ) from error
        self.last_timestamp_ms = -1

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> list[list[tuple[float, float]]]:
        self.last_timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(image, self.last_timestamp_ms)
        return [
            [(landmark.x, landmark.y) for landmark in face]
            for face in result.face_landmarks
        ]

    def close(self) -> None:
        self.landmarker.close()
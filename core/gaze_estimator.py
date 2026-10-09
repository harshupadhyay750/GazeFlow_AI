from dataclasses import dataclass

from calibration.calibration_model import CalibrationModel


@dataclass(frozen=True)
class EyeMeasurement:
    visible: bool
    iris_available: bool
    horizontal: float | None = None
    vertical: float | None = None


class GazeEstimator:
    RIGHT_EYE = {"corners": (33, 133), "vertical": (159, 145), "iris": 468}
    LEFT_EYE = {"corners": (362, 263), "vertical": (386, 374), "iris": 473}

    def __init__(self, horizontal_threshold: float, vertical_threshold: float) -> None:
        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold

    @staticmethod
    def _measure_eye(landmarks: list[tuple[float, float]], indices: dict) -> EyeMeasurement:
        required = (*indices["corners"], *indices["vertical"])
        if len(landmarks) <= max(required):
            return EyeMeasurement(visible=False, iris_available=False)

        first_corner, second_corner = (landmarks[index] for index in indices["corners"])
        upper, lower = (landmarks[index] for index in indices["vertical"])
        eye_width = abs(second_corner[0] - first_corner[0])
        eye_height = abs(lower[1] - upper[1])
        if eye_width < 0.008 or eye_height < 0.004:
            return EyeMeasurement(visible=False, iris_available=False)

        iris_index = indices["iris"]
        if len(landmarks) <= iris_index:
            return EyeMeasurement(visible=True, iris_available=False)

        iris_x, iris_y = landmarks[iris_index]
        horizontal = (iris_x - min(first_corner[0], second_corner[0])) / eye_width
        vertical = (iris_y - min(upper[1], lower[1])) / eye_height
        return EyeMeasurement(
            visible=True,
            iris_available=True,
            horizontal=horizontal,
            vertical=vertical,
        )

    def extract_features(self, landmarks: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
        left_eye = self._measure_eye(landmarks, self.LEFT_EYE)
        right_eye = self._measure_eye(landmarks, self.RIGHT_EYE)
        if not left_eye.visible or not left_eye.iris_available or not right_eye.visible or not right_eye.iris_available:
            return None

        left_horizontal = float(left_eye.horizontal)
        left_vertical = float(left_eye.vertical)
        right_horizontal = float(right_eye.horizontal)
        right_vertical = float(right_eye.vertical)
        return (
            (left_horizontal + right_horizontal) / 2.0,
            (left_vertical + right_vertical) / 2.0,
            left_horizontal - right_horizontal,
            left_vertical - right_vertical,
        )

    def estimate_position(
        self,
        landmarks: list[tuple[float, float]],
        calibration_model: CalibrationModel | None = None,
    ) -> tuple[float, float] | None:
        features = self.extract_features(landmarks)
        if features is None:
            return None

        if calibration_model is not None:
            return calibration_model.predict(features)

        horizontal = (features[0] + 0.5) / 1.0
        vertical = (features[1] + 0.5) / 1.0
        return (max(0.0, min(1.0, horizontal)), max(0.0, min(1.0, vertical)))

    def estimate(self, landmarks: list[tuple[float, float]]) -> tuple[str | None, dict[str, EyeMeasurement]]:
        eyes = {
            "left": self._measure_eye(landmarks, self.LEFT_EYE),
            "right": self._measure_eye(landmarks, self.RIGHT_EYE),
        }
        measurable = [
            eye for eye in eyes.values()
            if eye.visible and eye.iris_available
        ]
        if not measurable:
            return None, eyes

        horizontal = sum(eye.horizontal for eye in measurable) / len(measurable)
        vertical = sum(eye.vertical for eye in measurable) / len(measurable)

        if horizontal < 0.5 - self.horizontal_threshold:
            direction = "LEFT"
        elif horizontal > 0.5 + self.horizontal_threshold:
            direction = "RIGHT"
        elif vertical < 0.5 - self.vertical_threshold:
            direction = "UP"
        elif vertical > 0.5 + self.vertical_threshold:
            direction = "DOWN"
        else:
            direction = "CENTER"
        return direction, eyes
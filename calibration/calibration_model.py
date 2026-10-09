from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationSample:
    target: tuple[float, float]
    features: tuple[float, float, float, float]


class CalibrationModel:
    """Simple linear calibrated gaze mapper based on iris/eye features."""

    def __init__(self) -> None:
        self.samples: list[CalibrationSample] = []
        self.coefficients: np.ndarray | None = None
        self.intercept: float | None = None
        self.training_error: float | None = None

    def fit(self, samples: list[CalibrationSample]) -> None:
        if not samples:
            raise ValueError("At least one calibration sample is required.")

        self.samples = list(samples)
        feature_matrix = np.asarray([sample.features for sample in samples], dtype=np.float64)
        target_matrix = np.asarray([sample.target for sample in samples], dtype=np.float64)

        if feature_matrix.shape[0] < 2:
            raise ValueError("Calibration requires at least two accepted samples.")

        design = np.hstack([feature_matrix, np.ones((feature_matrix.shape[0], 1), dtype=np.float64)])
        coefficients, _, _, _ = np.linalg.lstsq(design, target_matrix, rcond=None)
        self.coefficients = coefficients[:-1, :]
        self.intercept = np.asarray(coefficients[-1, :], dtype=np.float64)
        self.training_error = self._compute_error(self.samples)

    def predict(self, features: tuple[float, float, float, float] | list[float]) -> tuple[float, float] | None:
        if self.coefficients is None or self.intercept is None:
            raise ValueError("The calibration model must be fitted before prediction.")

        feature_vector = np.asarray(features, dtype=np.float64)
        if feature_vector.shape != (self.coefficients.shape[0],):
            raise ValueError(
                "Expected feature count does not match the calibration model configuration."
            )

        prediction = np.dot(self.coefficients.T, feature_vector) + self.intercept
        return (float(prediction[0]), float(prediction[1]))

    def _compute_error(self, samples: list[CalibrationSample]) -> float:
        errors = []
        for sample in samples:
            prediction = self.predict(sample.features)
            if prediction is None:
                continue
            error = float(np.linalg.norm(np.asarray(prediction) - np.asarray(sample.target)))
            errors.append(error)
        return float(np.mean(errors)) if errors else 1.0

    def quality_report(self) -> dict:
        if not self.samples:
            return {"accepted_samples": 0, "mean_error": float("inf"), "usable": False}

        errors = []
        for sample in self.samples:
            prediction = self.predict(sample.features)
            if prediction is None:
                continue
            errors.append(float(np.linalg.norm(np.asarray(prediction) - np.asarray(sample.target))))

        mean_error = float(np.mean(errors)) if errors else float("inf")
        return {
            "accepted_samples": len(self.samples),
            "mean_error": mean_error,
            "max_error": float(np.max(errors)) if errors else float("inf"),
            "usable": len(self.samples) >= 6 and mean_error <= 0.30,
        }

    def to_dict(self) -> dict:
        if self.coefficients is None or self.intercept is None:
            raise ValueError("There is no fitted calibration data to save.")
        return {
            "schema_version": 1,
            "coefficients": [
                [float(value) for value in row] for row in np.asarray(self.coefficients, dtype=np.float64).tolist()
            ],
            "intercept": [float(value) for value in np.asarray(self.intercept, dtype=np.float64).tolist()],
            "training_error": float(self.training_error or 0.0),
            "samples": [
                {"target": [float(value) for value in sample.target], "features": [float(value) for value in sample.features]}
                for sample in self.samples
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CalibrationModel":
        if not isinstance(payload, dict):
            raise ValueError("Calibration payload must be a dictionary.")

        schema_version = int(payload.get("schema_version", 0))
        if schema_version != 1:
            raise ValueError(f"Unsupported calibration schema version: {schema_version}")

        coefficients = payload.get("coefficients")
        intercept = payload.get("intercept")
        if not isinstance(coefficients, list) or not coefficients:
            raise ValueError("Calibration file is missing coefficients.")
        if intercept is None:
            raise ValueError("Calibration file is missing the intercept value.")

        model = cls()
        model.coefficients = np.asarray(coefficients, dtype=np.float64)
        model.intercept = np.asarray(intercept, dtype=np.float64)
        model.training_error = float(payload.get("training_error", 0.0))
        samples = payload.get("samples", [])
        if isinstance(samples, list):
            model.samples = [
                CalibrationSample(
                    target=(float(item["target"][0]), float(item["target"][1])),
                    features=(float(item["features"][0]), float(item["features"][1]), float(item["features"][2]), float(item["features"][3])),
                )
                for item in samples
                if isinstance(item, dict) and isinstance(item.get("target"), list) and isinstance(item.get("features"), list)
            ]
        return model

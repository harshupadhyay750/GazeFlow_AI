import math

import pytest

from calibration.calibration_model import CalibrationModel, CalibrationSample


def test_model_fits_synthetic_data():
    samples = [
        CalibrationSample(target=(0.1, 0.2), features=(0.1, 0.2, 0.0, 0.0)),
        CalibrationSample(target=(0.3, 0.4), features=(0.3, 0.4, 0.1, 0.1)),
        CalibrationSample(target=(0.7, 0.8), features=(0.7, 0.8, 0.2, 0.2)),
        CalibrationSample(target=(0.9, 0.9), features=(0.9, 0.9, 0.3, 0.3)),
    ]
    model = CalibrationModel()
    model.fit(samples)
    estimate = model.predict(samples[0].features)
    assert estimate is not None
    assert estimate[0] >= 0.0 and estimate[0] <= 1.0
    assert estimate[1] >= 0.0 and estimate[1] <= 1.0


def test_model_rejects_insufficient_samples():
    model = CalibrationModel()
    with pytest.raises(ValueError):
        model.fit([])


def test_model_uses_point_distance_for_quality():
    samples = [
        CalibrationSample(target=(0.2, 0.2), features=(0.2, 0.2, 0.1, 0.1)),
        CalibrationSample(target=(0.8, 0.8), features=(0.8, 0.8, 0.2, 0.2)),
    ]
    model = CalibrationModel()
    model.fit(samples)
    quality = model.quality_report()
    assert "mean_error" in quality
    assert quality["mean_error"] >= 0.0

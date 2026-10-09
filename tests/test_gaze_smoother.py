from core.gaze_smoother import GazeSmoother


def test_smoother_applies_center_dead_zone():
    smoother = GazeSmoother(smoothing=0.45, dead_zone=0.06)
    assert smoother.apply(0.0, 0.0) == (0.0, 0.0)
    assert smoother.apply(0.09, 0.0) == (0.0, 0.0)


def test_smoother_uses_hysteresis_for_direction_changes():
    smoother = GazeSmoother(smoothing=0.0, dead_zone=0.0, hysteresis=0.08)
    smoother.apply(0.2, 0.0)
    assert smoother.apply(0.15, 0.0) == (0.15, 0.0)
    assert smoother.apply(0.0, 0.0) == (0.0, 0.0)

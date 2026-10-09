from __future__ import annotations

import math


class GazeSmoother:
    """Simple smoothing and dead-zone filter for normalized gaze coordinates."""

    def __init__(
        self,
        smoothing: float = 0.35,
        dead_zone: float = 0.05,
        hysteresis: float = 0.08,
    ) -> None:
        self.smoothing = float(smoothing)
        self.dead_zone = float(dead_zone)
        self.hysteresis = float(hysteresis)
        self.last_smoothed: tuple[float, float] | None = None

    def apply(self, x: float, y: float) -> tuple[float, float]:
        x_value = float(x)
        y_value = float(y)

        center_tolerance = self.dead_zone + self.hysteresis
        if abs(x_value) <= center_tolerance and abs(y_value) <= center_tolerance:
            self.last_smoothed = (0.0, 0.0)
            return (0.0, 0.0)

        if self.last_smoothed is None:
            self.last_smoothed = (x_value, y_value)
            return (x_value, y_value)

        dx = x_value - self.last_smoothed[0]
        dy = y_value - self.last_smoothed[1]
        if abs(dx) < self.hysteresis and abs(dy) < self.hysteresis:
            self.last_smoothed = (x_value, y_value)
            return (x_value, y_value)

        smoothed_x = self.last_smoothed[0] + self.smoothing * dx
        smoothed_y = self.last_smoothed[1] + self.smoothing * dy
        self.last_smoothed = (smoothed_x, smoothed_y)
        return self.last_smoothed

    def reset(self) -> None:
        self.last_smoothed = None

    def distance_from_center(self, x: float, y: float) -> float:
        if self.last_smoothed is None:
            return math.hypot(float(x), float(y))
        return math.hypot(float(x) - self.last_smoothed[0], float(y) - self.last_smoothed[1])

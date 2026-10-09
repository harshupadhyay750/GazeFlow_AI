from __future__ import annotations

import json
import platform
from pathlib import Path

from .calibration_model import CalibrationModel


class CalibrationManager:
    """Persist the fitted gaze calibration model to a user-scoped data directory."""

    def __init__(self, file_path: str | Path | None = None) -> None:
        self.file_path = Path(file_path) if file_path is not None else self.default_path()

    @staticmethod
    def default_path() -> Path:
        if platform.system() == "Windows":
            return Path.home() / "AppData" / "Roaming" / "GazeFlow AI" / "calibration.json"
        return Path.home() / ".gazeflow" / "calibration.json"

    def save(self, model: CalibrationModel) -> Path:
        payload = {"schema_version": 1, "model": model.to_dict()}
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.file_path

    def load(self) -> CalibrationModel | None:
        if not self.file_path.exists():
            return None

        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None

        if not isinstance(payload, dict):
            return None

        model_payload = payload.get("model")
        if not isinstance(model_payload, dict):
            return None

        try:
            return CalibrationModel.from_dict(model_payload)
        except ValueError:
            return None

    def clear(self) -> None:
        if self.file_path.exists():
            self.file_path.unlink(missing_ok=True)

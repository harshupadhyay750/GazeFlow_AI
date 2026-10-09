import time
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from calibration.calibration_model import CalibrationModel
from core.camera import CameraError, CameraSource
from core.tracking_engine import TrackingEngine


class TrackingWorker(QObject):
    frame_ready = Signal(QImage, object)
    camera_ready = Signal(int)
    error = Signal(str)
    finished = Signal()

    def __init__(self, project_dir: Path, settings: dict) -> None:
        super().__init__()
        self.project_dir = project_dir
        self.settings = settings
        self.camera: CameraSource | None = None
        self.engine: TrackingEngine | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._process_frame)
        self.stopping = False
        self.cleaned_up = False

    @Slot()
    def start_tracking(self) -> None:
        try:
            self.camera = CameraSource(
                [int(index) for index in self.settings.get("camera_indices", [0, 1, 2, 3])],
                int(self.settings.get("camera_width", 640)),
                int(self.settings.get("camera_height", 480)),
            )
            camera_index = self.camera.open()
            self.camera_ready.emit(camera_index)
            self.engine = TrackingEngine(
                self.project_dir / "assets" / "face_landmarker.task",
                self.settings,
            )
            self.timer.start()
        except Exception as error:
            self.error.emit(str(error) or f"Could not start tracking: {error.__class__.__name__}")
            self._cleanup()

    @Slot()
    def _process_frame(self) -> None:
        if self.stopping or self.camera is None or self.engine is None:
            return
        try:
            frame = self.camera.read()
            preview, result = self.engine.process_frame(frame, int(time.monotonic() * 1000))
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(
                rgb.data,
                width,
                height,
                channels * width,
                QImage.Format.Format_RGB888,
            ).copy()
            self.frame_ready.emit(image, result)
        except CameraError as error:
            self.error.emit(str(error))
            self._cleanup()
        except Exception as error:
            self.error.emit(f"Tracking stopped because frame processing failed: {error}")
            self._cleanup()

    @Slot()
    def stop_tracking(self) -> None:
        self.stopping = True
        self._cleanup()

    def set_calibration_model(self, model: CalibrationModel | None) -> None:
        if self.engine is not None:
            self.engine.calibration_model = model

    def get_latest_snapshot(self) -> dict:
        if self.engine is None or self.engine.last_snapshot is None:
            return {}
        return self.engine.last_snapshot

    def _cleanup(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        self.timer.stop()
        try:
            if self.engine is not None:
                self.engine.close()
                self.engine = None
        finally:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.finished.emit()


def create_tracking_thread(project_dir: Path, settings: dict) -> tuple[QThread, TrackingWorker]:
    thread = QThread()
    worker = TrackingWorker(project_dir, settings)
    worker.moveToThread(thread)
    thread.started.connect(worker.start_tracking)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    return thread, worker
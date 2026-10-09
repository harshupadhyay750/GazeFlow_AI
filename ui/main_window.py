from pathlib import Path

from PySide6.QtCore import QMetaObject, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from calibration.calibration_manager import CalibrationManager
from calibration.calibration_model import CalibrationModel
from calibration.calibration_window import CalibrationWindow
from core.tracking_worker import TrackingWorker, create_tracking_thread


class MainWindow(QMainWindow):
    stop_requested = Signal()

    def __init__(self, project_dir: Path, settings: dict) -> None:
        super().__init__()
        self.project_dir = project_dir
        self.settings = settings
        self.calibration_manager = CalibrationManager()
        self.calibration_model: CalibrationModel | None = self.calibration_manager.load()
        self.tracking_thread = None
        self.tracking_worker: TrackingWorker | None = None
        self.setWindowTitle("EyeControl AI")
        self.setMinimumSize(920, 610)
        self.resize(1100, 700)
        self._build_ui()
        self._refresh_calibration_status()
        self._set_state("IDLE", "Tracking is stopped. Start tracking to connect to a webcam.")

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #111719; color: #e7efec; }
            QLabel { background: transparent; }
            QLabel#brandMark { background: #b6f36a; color: #172018; border-radius: 8px;
                font-size: 16px; font-weight: 800; }
            QLabel#appName { font-size: 22px; font-weight: 700; }
            QLabel#eyebrow { color: #92a49d; font-size: 11px; }
            QFrame#previewFrame, QFrame#statusPanel { background: #192123;
                border: 1px solid #2c3a39; border-radius: 10px; }
            QLabel#preview { background: #0b1011; border-radius: 7px; color: #82938d; }
            QLabel#sectionTitle { color: #b6f36a; font-size: 12px; font-weight: 700; }
            QLabel#metricLabel { color: #92a49d; font-size: 12px; }
            QLabel#metricValue { color: #edf5f1; font-size: 13px; font-weight: 600; }
            QLabel#stateValue { font-size: 17px; font-weight: 700; }
            QLabel#gazeValue { color: #b6f36a; font-size: 30px; font-weight: 750; }
            QLabel#detail { color: #aab9b4; font-size: 12px; }
            QPushButton { min-height: 38px; padding: 0 14px; border-radius: 6px;
                border: 1px solid #3c4b48; background: #202b2c; font-weight: 600; }
            QPushButton:hover { background: #2b3938; }
            QPushButton:disabled { color: #63716d; background: #1b2425; }
            QPushButton#startButton { background: #b6f36a; color: #172018; border: none; }
            QPushButton#startButton:hover { background: #c8ff83; }
            QPushButton#stopButton { color: #ffb5a6; }
        """)

        root = QWidget()
        page = QVBoxLayout(root)
        page.setContentsMargins(26, 22, 26, 24)
        page.setSpacing(18)

        header = QHBoxLayout()
        mark = QLabel("EC")
        mark.setObjectName("brandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(42, 42)
        name = QLabel("EyeControl AI")
        name.setObjectName("appName")
        eyebrow = QLabel("EYE TRACKING  /  PHASE 1")
        eyebrow.setObjectName("eyebrow")
        brand_text = QVBoxLayout()
        brand_text.setSpacing(2)
        brand_text.addWidget(name)
        brand_text.addWidget(eyebrow)
        header.addWidget(mark)
        header.addLayout(brand_text)
        header.addStretch()
        self.header_status = QLabel("IDLE")
        self.header_status.setObjectName("eyebrow")
        header.addWidget(self.header_status)
        page.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(18)
        preview_frame = QFrame()
        preview_frame.setObjectName("previewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        preview_layout.setSpacing(10)
        preview_heading = QLabel("LIVE CAMERA")
        preview_heading.setObjectName("sectionTitle")
        self.preview = QLabel("Camera preview will appear here")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(560, 390)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(preview_heading)
        preview_layout.addWidget(self.preview, 1)
        content.addWidget(preview_frame, 3)

        status_panel = QFrame()
        status_panel.setObjectName("statusPanel")
        status_panel.setFixedWidth(280)
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(18, 18, 18, 18)
        status_layout.setSpacing(12)
        status_title = QLabel("TRACKING STATUS")
        status_title.setObjectName("sectionTitle")
        self.state_value = QLabel("IDLE")
        self.state_value.setObjectName("stateValue")
        self.state_detail = QLabel()
        self.state_detail.setObjectName("detail")
        self.state_detail.setWordWrap(True)
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.state_value)
        status_layout.addWidget(self.state_detail)
        status_layout.addSpacing(9)
        self.face_value = self._add_metric(status_layout, "Face")
        self.left_eye_value = self._add_metric(status_layout, "Left eye")
        self.right_eye_value = self._add_metric(status_layout, "Right eye")
        self.calibration_status_value = self._add_metric(status_layout, "Calibration")
        self.gaze_value = self._add_metric(status_layout, "Gaze", prominent=True)
        self.gaze_position_value = self._add_metric(status_layout, "Smoothed gaze")
        self.confidence_value = self._add_metric(status_layout, "Confidence")
        status_layout.addStretch()
        self.camera_value = QLabel("Camera: disconnected")
        self.camera_value.setObjectName("eyebrow")
        status_layout.addWidget(self.camera_value)
        content.addWidget(status_panel)
        page.addLayout(content, 1)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Start Tracking")
        self.start_button.setObjectName("startButton")
        self.stop_button = QPushButton("Stop Tracking")
        self.stop_button.setObjectName("stopButton")
        self.calibration_button = QPushButton("Calibration")
        self.settings_button = QPushButton("Settings")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_tracking)
        self.stop_button.clicked.connect(self.stop_tracking)
        self.calibration_button.clicked.connect(self.show_calibration_dialog)
        self.settings_button.clicked.connect(self.show_settings_info)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch()
        controls.addWidget(self.calibration_button)
        controls.addWidget(self.settings_button)
        page.addLayout(controls)
        self.setCentralWidget(root)

    @staticmethod
    def _add_metric(layout: QVBoxLayout, title: str, prominent: bool = False) -> QLabel:
        row = QHBoxLayout()
        label = QLabel(title)
        label.setObjectName("metricLabel")
        value = QLabel("--")
        value.setObjectName("gazeValue" if prominent else "metricValue")
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(label)
        row.addWidget(value)
        layout.addLayout(row)
        return value

    @Slot()
    def start_tracking(self) -> None:
        if self.tracking_thread is not None:
            return
        self.preview.setText("Connecting to webcam…")
        self._set_state("IDLE", "Starting camera and loading the face-tracking model.")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        thread, worker = create_tracking_thread(self.project_dir, self.settings)
        self.tracking_thread = thread
        self.tracking_worker = worker
        if self.calibration_model is not None:
            worker.set_calibration_model(self.calibration_model)
        self.stop_requested.connect(worker.stop_tracking)
        worker.frame_ready.connect(self._update_frame)
        worker.camera_ready.connect(self._camera_connected)
        worker.error.connect(self._show_tracking_error)
        thread.finished.connect(self._tracking_finished)
        thread.start()

    @Slot()
    def stop_tracking(self) -> None:
        if self.tracking_worker is not None:
            self.stop_requested.emit()

    @Slot(object, object)
    def _update_frame(self, image, result: dict) -> None:
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        state = result["state"]
        self._set_state(state, result["detail"])
        self.face_value.setText(result["face"])
        self.left_eye_value.setText(result["left_eye"])
        self.right_eye_value.setText(result["right_eye"])
        self.gaze_value.setText(result["gaze"])
        gaze_position = result.get("gaze_position") or (0.0, 0.0)
        self.gaze_position_value.setText(
            f"({gaze_position[0]:.2f}, {gaze_position[1]:.2f})"
        )
        self.confidence_value.setText(f"{result['confidence']}%")

    @Slot(int)
    def _camera_connected(self, index: int) -> None:
        self.camera_value.setText(f"Camera: connected (device {index})")

    @Slot(str)
    def _show_tracking_error(self, message: str) -> None:
        self.preview.setText("Camera preview unavailable")
        self._set_state("IDLE", message)
        QMessageBox.warning(self, "Tracking unavailable", message)

    @Slot()
    def _tracking_finished(self) -> None:
        self.tracking_worker = None
        self.tracking_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_value.setText("Camera: disconnected")
        self._set_state("IDLE", "Tracking is stopped. Start tracking to connect to a webcam.")
        self.face_value.setText("--")
        self.left_eye_value.setText("--")
        self.right_eye_value.setText("--")
        self.calibration_status_value.setText("Not Calibrated" if self.calibration_model is None else "Calibrated")
        self.gaze_value.setText("--")
        self.gaze_position_value.setText("--")
        self.confidence_value.setText("--")

    def _refresh_calibration_status(self) -> None:
        if self.calibration_model is None:
            text = "Not Calibrated"
        else:
            report = self.calibration_model.quality_report()
            text = "Calibrated" if report.get("usable") else "Calibration Needed"
        self.calibration_status_value.setText(text)

    def _set_state(self, state: str, detail: str) -> None:
        labels = {
            "IDLE": "IDLE",
            "TRACKING": "ACTIVE",
            "NO_FACE": "NO FACE",
            "LOW_CONFIDENCE": "LOW CONFIDENCE",
            "CALIBRATING": "CALIBRATING",
            "CALIBRATION_NEEDED": "CALIBRATION NEEDED",
        }
        self.state_value.setText(labels.get(state, state))
        self.header_status.setText(labels.get(state, state))
        self.state_detail.setText(detail)

    def show_calibration_dialog(self) -> None:
        if self.tracking_worker is None:
            QMessageBox.information(
                self,
                "Calibration unavailable",
                "Start tracking before running calibration so the webcam feed can be sampled.",
            )
            return

        self.calibration_status_value.setText("Calibrating")
        self._set_state("CALIBRATING", "Collecting calibration samples from the live webcam feed.")
        dialog = CalibrationWindow(
            self,
            self.tracking_worker,
            self.calibration_manager,
            self.settings,
        )
        dialog.exec()
        loaded = self.calibration_manager.load()
        if loaded is not None:
            self.calibration_model = loaded
            if self.tracking_worker is not None:
                self.tracking_worker.set_calibration_model(loaded)
        self._refresh_calibration_status()
        self._set_state("TRACKING", "Calibration complete. Tracking continues with the saved personalized model.")

    def show_settings_info(self) -> None:
        QMessageBox.information(
            self,
            "Tracking settings",
            "Camera and gaze thresholds are configured in config/settings.json. "
            "Restart tracking after changing the file.",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        worker = self.tracking_worker
        thread = self.tracking_thread
        if worker is not None and thread is not None and thread.isRunning():
            QMetaObject.invokeMethod(
                worker,
                "stop_tracking",
                Qt.ConnectionType.BlockingQueuedConnection,
            )
            thread.wait(2000)
        event.accept()
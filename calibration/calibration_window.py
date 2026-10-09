from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from calibration.calibration_manager import CalibrationManager
from calibration.calibration_model import CalibrationModel, CalibrationSample


class CalibrationWindow(QDialog):
    """Nine-point calibration workflow for the current active tracking session."""

    POINTS = [
        (0.18, 0.18),
        (0.50, 0.18),
        (0.82, 0.18),
        (0.18, 0.50),
        (0.50, 0.50),
        (0.82, 0.50),
        (0.18, 0.82),
        (0.50, 0.82),
        (0.82, 0.82),
    ]

    def __init__(
        self,
        parent: QWidget | None,
        tracking_worker: Any,
        calibration_manager: CalibrationManager,
        settings: dict,
    ) -> None:
        super().__init__(parent)
        self.tracking_worker = tracking_worker
        self.calibration_manager = calibration_manager
        self.settings = settings
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.current_index = 0
        self.samples: list[CalibrationSample] = []
        self.rejected_targets = 0
        self.target_buffer: list[tuple[float, float]] = []
        self.target_duration = float(settings.get("calibration_target_duration", 2.0))
        self.required_valid = int(settings.get("minimum_valid_samples", 8))
        self._setup_ui()
        self._reset_state()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Eye Calibration")
        self.resize(760, 520)
        self.setModal(True)
        self.setStyleSheet(
            """
            QDialog { background: #121a1b; color: #edf6f4; }
            QLabel { color: #edf6f4; }
            QPushButton { min-height: 36px; border-radius: 7px; background: #1f2d2e; border: 1px solid #3d4c4d; }
            QPushButton#primary { background: #b6f36a; color: #102010; border: none; }
            QPushButton#danger { background: #3a1a1a; border: 1px solid #5a2c2c; color: #ffd5d1; }
            """
        )

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(20, 20, 20, 20)
        self.root.setSpacing(12)

        self.title_label = QLabel("Eye Calibration")
        self.title_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.root.addWidget(self.title_label)

        self.instructions = QLabel(
            "Keep your face centered in the camera and look directly at the target. "
            "Each point will be sampled until the tracker has enough stable data."
        )
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("color: #a9bdb7; font-size: 13px;")
        self.root.addWidget(self.instructions)

        self.progress_label = QLabel("Point 1 of 9")
        self.progress_label.setStyleSheet("color: #b6f36a; font-weight: 700;")
        self.root.addWidget(self.progress_label)

        self.target_holder = QLabel("●")
        self.target_holder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_holder.setStyleSheet(
            "QLabel { background: rgba(255, 255, 255, 0.08); color: #f7f6ff; border-radius: 999px; font-size: 32px; }"
        )
        self.target_holder.setFixedSize(46, 46)
        self.target_holder.hide()
        self.root.addWidget(self.target_holder, 1)

        self.status_label = QLabel("Ready to begin calibration.")
        self.status_label.setWordWrap(True)
        self.root.addWidget(self.status_label)

        button_row = QWidget()
        button_layout = QVBoxLayout(button_row)
        button_layout.setSpacing(10)
        start_button = QPushButton("Start Calibration")
        start_button.setObjectName("primary")
        start_button.clicked.connect(self.start_calibration)
        restart_button = QPushButton("Restart Calibration")
        restart_button.clicked.connect(self.restart_calibration)
        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("danger")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(start_button)
        button_layout.addWidget(restart_button)
        button_layout.addWidget(cancel_button)
        self.root.addWidget(button_row)

        self.finish_message = QLabel("")
        self.finish_message.setWordWrap(True)
        self.finish_message.hide()
        self.root.addWidget(self.finish_message)

        self.target_holder.setVisible(False)
        self._set_target_position(0)

    def _set_target_position(self, index: int) -> None:
        point = self.POINTS[index]
        x = int(self.width() * point[0]) - 23
        y = int(self.height() * point[1]) - 23
        self.target_holder.move(x, y)

    def _reset_state(self) -> None:
        self.current_index = 0
        self.samples = []
        self.target_buffer = []
        self.rejected_targets = 0
        self.progress_label.setText("Point 1 of 9")
        self.status_label.setText("Ready to begin calibration.")
        self.finish_message.hide()
        self.target_holder.hide()

    def start_calibration(self) -> None:
        self._reset_state()
        self._advance_to_current_target()

    def restart_calibration(self) -> None:
        self.start_calibration()

    def _advance_to_current_target(self) -> None:
        if self.current_index >= len(self.POINTS):
            self._finish_calibration()
            return

        self.target_holder.show()
        self._set_target_position(self.current_index)
        self.progress_label.setText(f"Point {self.current_index + 1} of {len(self.POINTS)}")
        self.status_label.setText(
            "Look directly at the target and keep your head still. "
            "A stable sample is being collected."
        )
        self.target_buffer = []
        self.timer.start(int(self.target_duration * 1000))

    def _tick(self) -> None:
        snapshot = self.tracking_worker.get_latest_snapshot() if self.tracking_worker else {}
        features = snapshot.get("features")
        if not features:
            self.rejected_targets += 1
            self.status_label.setText(
                "Face or iris landmarks are missing. Improve lighting or reposition your face."
            )
            self.timer.stop()
            self._consume_target()
            return

        self.target_buffer.append(tuple(float(v) for v in features))
        if len(self.target_buffer) >= self.required_valid:
            median_features = tuple(
                float(value)
                for value in map(
                    lambda index: sorted([item[index] for item in self.target_buffer])[len(self.target_buffer) // 2],
                    range(len(self.target_buffer[0])),
                )
            )
            sample = CalibrationSample(
                target=self.POINTS[self.current_index],
                features=median_features[:4],
            )
            self.samples.append(sample)
            self.timer.stop()
            self.status_label.setText("Target accepted. Moving to the next calibration point.")
            self._consume_target()
            return

        self.status_label.setText(
            f"Collecting stable samples for this target ({len(self.target_buffer)}/{self.required_valid})..."
        )

    def _consume_target(self) -> None:
        self.current_index += 1
        if self.current_index >= len(self.POINTS):
            self._finish_calibration()
            return
        self._advance_to_current_target()

    def _finish_calibration(self) -> None:
        self.timer.stop()
        self.target_holder.hide()

        if len(self.samples) < 4:
            suggestions = [
                "Improve room lighting so your face is clearly visible.",
                "Keep your face comfortably in view and avoid large head movements.",
                "Repeat calibration and try to hold a steady gaze at each target."
            ]
            self.finish_message.setText(
                "Calibration did not collect enough valid data. " + " ".join(suggestions)
            )
            self.finish_message.show()
            self.status_label.setText("Calibration quality is insufficient for a reliable model.")
            return

        model = CalibrationModel()
        model.fit(self.samples)
        report = model.quality_report()
        self.calibration_manager.save(model)

        suggestions = []
        if report["mean_error"] > 0.25:
            suggestions.append("Repeat calibration if gaze estimates still feel unstable.")
        if len(self.samples) < 8:
            suggestions.append("Collect a few more stable samples to improve the model.")
        suggestions.extend([
            "Improve room lighting.",
            "Keep the face comfortably visible to the camera.",
            "Avoid moving too far away from the webcam."
        ])

        quality_note = "This calibration is usable." if report["usable"] else "The model may need a retry for better stability."
        message = (
            f"Calibration complete. {len(self.samples)} valid points were accepted. "
            f"Mean validation error: {report['mean_error']:.3f}. {quality_note} "
            + " ".join(suggestions[:3])
        )
        self.finish_message.setText(message)
        self.finish_message.show()
        self.status_label.setText("Calibration saved successfully.")

    def reject(self) -> None:
        self.timer.stop()
        self.target_holder.hide()
        self.status_label.setText("Calibration cancelled.")
        self.finish_message.setText("The current calibration session was cancelled. The app remains in a valid state.")
        self.finish_message.show()
        self.hide()

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.timer.stop()
        event.accept()

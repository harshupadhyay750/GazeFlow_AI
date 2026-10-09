import platform

import cv2


class CameraError(RuntimeError):
    """Raised when a webcam cannot be opened or stops returning frames."""


class CameraSource:
    def __init__(self, indices: list[int], width: int = 640, height: int = 480) -> None:
        self.indices = indices
        self.width = width
        self.height = height
        self.capture: cv2.VideoCapture | None = None
        self.index: int | None = None

    @staticmethod
    def _open_capture(index: int) -> cv2.VideoCapture:
        if platform.system() == "Windows":
            return cv2.VideoCapture(index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(index)

    @classmethod
    def available_indices(cls, indices: list[int]) -> list[int]:
        available = []
        for index in indices:
            capture = cls._open_capture(index)
            if capture.isOpened():
                available.append(index)
            capture.release()
        return available

    def open(self) -> int:
        for index in self.indices:
            capture = self._open_capture(index)
            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture = capture
            self.index = index
            return index

        raise CameraError(
            "No webcam could be opened. Check that it is connected, close other apps "
            "using it, and allow camera access in Windows Privacy settings."
        )

    def read(self):
        if self.capture is None:
            raise CameraError("The webcam is not open.")

        success, frame = self.capture.read()
        if not success or frame is None:
            self.release()
            raise CameraError(
                "The webcam stopped sending video. Reconnect it and start tracking again."
            )
        return frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.index = None
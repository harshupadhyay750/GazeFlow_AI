# GazeFlow_AI

An AI-powered, webcam-based eye-control project using Python, OpenCV, and MediaPipe. Phase 1 is a Windows desktop foundation for real-time webcam, face, eye, iris, and basic gaze tracking. It is for observation only: it does not control the mouse or keyboard, scroll, click, automate a browser, or connect to online services.

## Requirements

- Windows 10 or 11
- Python 3.11 or 3.12 (64-bit x64). On Windows ARM64, use x64 Python under Windows emulation because OpenCV does not publish Windows ARM64 wheels.
- A webcam and permission to use it in Windows Settings > Privacy & security > Camera
- The MediaPipe Face Landmarker model described below

## Installation

Open PowerShell in this `EyeControlAI` folder. On Windows x64, if Python 3.12 is installed, create and activate a virtual environment with:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows ARM64, use the path to an installed x64 Python 3.12 executable to create `.venv` instead of `py -3.12`:

```powershell
& "C:\Path\To\Python312-x64\python.exe" -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate the environment again.

## Required model file

The official MediaPipe Face Landmarker task model is included here:

`assets/face_landmarker.task`

Official source: [Face Landmarker, float16](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

The model is loaded locally at startup; tracking does not require a network connection. If the file is removed, download it from the official source above and restore it to the listed path. The application explains how to fix a missing model and releases the webcam if it cannot be loaded.

## Run

From the project folder with the virtual environment active:

```powershell
python main.py
```

Press **Start Tracking** to open the first available webcam at indices 0-3. Press **Stop Tracking** to release it. Closing the window also stops tracking and releases camera/model resources. The preview is mirrored and draws face reference points, eye landmarks, and iris centers in distinct colors.

## Modules

- `main.py`: creates the Qt application and loads JSON settings.
- `core/camera.py`: probes configured device indices, opens the webcam, reads frames, and releases the device.
- `core/face_tracker.py`: runs MediaPipe Face Landmarker in video mode and returns actual normalized face landmarks.
- `core/gaze_estimator.py`: derives approximate horizontal and vertical gaze from eye corners, eyelids, and iris centers.
- `core/tracking_engine.py`: combines detection, confidence/stability scoring, status selection, and preview annotations.
- `core/tracking_worker.py`: owns camera/model processing on a Qt worker thread and transfers copied frames to the UI.
- `ui/main_window.py`: desktop layout, live preview, tracking metrics, controls, and user-facing error messages.
- `config/settings.json`: webcam indices, resolution, gaze thresholds, and confidence/stability settings.
- `assets/`: location for `face_landmarker.task`.

## Configuration

Edit `config/settings.json` before starting tracking:

- `camera_indices`: device indexes to probe in order.
- `camera_width` and `camera_height`: requested webcam resolution (the camera may choose a different supported mode).
- `horizontal_threshold` and `vertical_threshold`: normalized iris displacement from the eye center required to report a direction instead of `CENTER`.
- `minimum_confidence`: threshold from 0.0 to 1.0 for the `LOW_CONFIDENCE` state.
- `stability_window`: number of recent gaze samples used for the stability portion of confidence.

Gaze directions are approximate, based on visible eye geometry, and are not personalized. Head pose, camera placement, lighting, glasses, and individual anatomy affect the result. Calibration is intentionally UI-only in Phase 1.

## Troubleshooting

- **No webcam / permission denied:** connect the camera, close video-call apps that may own it, then allow desktop apps to access the camera in Windows privacy settings. Adjust `camera_indices` if the device is not index 0-3.
- **Camera disconnects:** reconnect it and press Start Tracking again. The current session stops and releases the device.
- **Model unavailable:** verify the exact path and filename `assets/face_landmarker.task`, and ensure the download completed (it should not be an HTML error page).
- **No face detected:** improve front lighting, move into the camera view, and keep only one face in frame.
- **Low confidence / missing eyes:** face the camera, reduce glare, remove obstructions where possible, and improve lighting. Iris dots are only drawn when the model provides iris landmarks.
- **Dependency installation fails:** use 64-bit Python 3.11 or 3.12 in a fresh virtual environment, then reinstall `requirements.txt`.
- **Gaze direction is reversed or drifts:** adjust the geometric thresholds in `config/settings.json`. This basic estimator is not a substitute for per-user calibration.

## Phase 1 scope

The application only captures and analyzes webcam frames for local face/eye/iris detection and gaze display. It performs no mouse or keyboard actions, scrolling, clicking, social-media automation, voice processing, account management, or cloud processing.

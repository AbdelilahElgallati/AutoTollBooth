# Vehicle Detector and Counter using YOLOv8
This project is built to detect and count vehicles in a given video file or live video stream of a highway using YOLOv8. The program outputs a video file with the detected vehicles and the total count of vehicle Inflow and Outflow from the Tolling Booth.

# Dependencies

* streamlit
* ultralytics
* opencv-python-headless
* numpy
* pandas
* tqdm

## Quick start (recommended)

1. Create and activate a Python 3.11 virtual environment (or use the provided `.venv311`):

```powershell
# AutoTollBooth — Automatic Toll Booth Vehicle Detector and Counter

AutoTollBooth detects and counts vehicles from uploaded traffic videos using YOLOv8. It runs as a Streamlit web app and provides a live annotated preview while maintaining simple inflow/outflow counts for vehicles crossing a configurable detection line.

This repository contains the Streamlit UI (`app.py`), the detection/tracking helpers in `src/`, and a small helper to ensure YOLO weights are available under the project's `models/` folder.

## Features

- Streamlit UI for uploading video files and viewing live annotated output
- Vehicle detection + tracking (YOLOv8 via Ultralytics)
- Inflow / outflow counters using a configurable horizontal line
- Automatic model management: app ensures weights are available in `models/` (uses Ultralytics cache as fallback)
- Lightweight PowerShell helper for Windows users (`run_app.ps1`)

## Quick overview

- Source: `app.py` (Streamlit UI)
- Detection: `src/detector.py` (loads YOLO, helper `ensure_model()`)
- Processing: `src/processor.py` (frame processing, annotation)
- Tracking/counting: `src/tracker.py`

## Requirements

- Python 3.11 (recommended). Pre-built wheels for numpy, torch and other scientific packages are abundant for 3.11 which avoids heavy local builds.
- Windows (tested), but app runs on Linux/macOS where supported wheels exist.

Core dependencies (see `requirements.txt`):

- streamlit, ultralytics, opencv-python(-headless), numpy, pandas, torch, torchvision, matplotlib, tqdm

## Installation (recommended)

1. Create and activate a Python 3.11 venv (the repository includes a `.venv311` in examples):

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate
```

2. Install project dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Running the app

Start Streamlit from the project root while the venv is active:

```powershell
streamlit run app.py
```

## Model handling

- The app looks for YOLO weights under `models/` (e.g. `models/yolov8n.pt`).
- On first access, the app will attempt to ensure the model exists under `models/` silently (a spinner is shown). If that fails it will fall back to loading via the Ultralytics cache/name (so the app can continue).
- The helper function `src.detector.ensure_model(model_name)` is available for programmatic control.

## Configuration

- Pick model size in the sidebar (`yolov8n`, `yolov8s`, `yolov8m`). Larger models give better accuracy at higher latency.
- Confidence threshold and detection line position are adjustable from the sidebar.

## Project structure

- `app.py` — Streamlit app and main UI
- `requirements.txt` — Python dependencies
- `src/` — application logic
    - `detector.py` — YOLO loading and model helper
    - `processor.py` — frame processing and annotation
    - `tracker.py` — counting logic
- `models/` — where weights are stored (created automatically)


Enjoy — if you want, I can also add automatic prefetching of all model sizes from the UI or a small test harness.

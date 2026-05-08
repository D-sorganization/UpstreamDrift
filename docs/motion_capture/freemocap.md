# FreeMoCap Integration

## Overview

UpstreamDrift uses [FreeMoCap](https://github.com/freemocap/freemocap) as a sidecar motion-capture pipeline. FreeMoCap is an open-source markerless motion capture system that uses multiple webcams to triangulate 3D landmarks.

Because FreeMoCap is AGPL-licensed and carries heavy dependencies (PyQt6, OpenCV, PyTorch, MediaPipe), it is strictly run as a completely isolated subprocess. **No FreeMoCap packages should be installed in the main UpstreamDrift environment.**

## Architecture / AGPL Boundary

We maintain a strict process boundary:

1. The user records a session with FreeMoCap in its own isolated environment.
2. The `src/tools/freemocap_sidecar/run_freemocap.py` script spawns the FreeMoCap process via `subprocess`.
3. FreeMoCap writes the triangulated 3D landmark data to a designated folder.
4. UpstreamDrift reads the generated `landmarks.csv` (or `.npy`) file.

This file hand-off ensures that UpstreamDrift does not link against or import AGPL code, preserving the separation between the AGPL FreeMoCap system and our internal environment.

## Installation

Do not install FreeMoCap via pip into your main development environment.

Instead, create a separate virtual environment:

```bash
python -m venv venv_freemocap
source venv_freemocap/bin/activate  # or venv_freemocap\Scripts\activate on Windows
pip install freemocap
```

## Calibration and Usage

1. Use a standard checkerboard for multi-camera calibration as outlined in the [FreeMoCap documentation](https://docs.freemocap.org).
2. Record your session.
3. Note that MediaPipe Holistic was not trained on high-speed sports motion (like a 100mph clubhead), so tracking dropouts on the downswing are expected.
4. Process the data using the sidecar script:

```bash
python src/tools/freemocap_sidecar/run_freemocap.py --input <path_to_videos> --output <path_to_landmarks>
```

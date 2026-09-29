# FreeMoCap Integration for UpstreamDrift

This module provides a **sidecar integration** with [FreeMoCap](https://github.com/freemocap/freemocap) for markerless 3D motion capture using multiple webcams.

## Overview

FreeMoCap is integrated as an out-of-process subprocess using a filesystem-based communication pattern. This design:

1. **Maintains license compliance**: FreeMoCap is AGPL-3.0 licensed. Running it as a separate process keeps the boundary clean.
2. **Avoids dependency conflicts**: FreeMoCap pins specific versions of OpenCV, PySide6, and other packages that may conflict with UpstreamDrift's dependencies.
3. **Prevents GUI toolkit clashes**: FreeMoCap uses PySide6 while UpstreamDrift uses PyQt6.

## Architecture

```
UpstreamDrift main process (PyQt6, FastAPI, our engines)
        |
        | spawn subprocess
        v
[isolated venv: freemocap-env, Python 3.12, PySide6]
        |
        | runs `python -m freemocap` headless or `freemocap_cli`
        | reads videos from <session>/videos/
        v
[FreeMoCap pipeline: skellycam -> skellytracker -> aniposelib -> skellyforge]
        |
        | writes outputs to <session>/freemocap_output/
        v
[3D landmarks CSV/npy + calibration JSON + diagnostic plots]
        |
        | UpstreamDrift adapter reads + validates + remaps to our schema
        v
[Existing OpenSim / Pinocchio / MuJoCo pipeline]
```

## Installation

### Step 1: Create the FreeMoCap environment

Run the setup script to create an isolated Python environment:

```bash
cd src/motion_capture/freemocap_ingest
chmod +x setup_freemocap_env.sh
./setup_freemocap_env.sh
```

This will:

- Create a virtual environment at `~/freemocap-env`
- Install FreeMoCap and its dependencies
- Verify the installation

### Step 2: Verify installation

```bash
source ~/freemocap-env/bin/activate
python -m freemocap --help
deactivate
```

## Usage

### Quick Start

```bash
# Run FreeMoCap capture on a session directory
python -m src.motion_capture.freemocap_ingest /path/to/session --video-dir /path/to/videos

# Parse existing FreeMoCap output
python -m src.motion_capture.freemocap_ingest --parse /path/to/freemocap_output

# Full pipeline with export
python -m src.motion_capture.freemocap_ingest /path/to/session \
    --video-dir /path/to/videos \
    --parse-output \
    --export-npy output.npy
```

### Command Reference

```
Usage: python -m src.motion_capture.freemocap_ingest [OPTIONS] [SESSION_DIR]

FreeMoCap motion capture integration for UpstreamDrift

Positional Arguments:
  session_dir          Session directory for capture data

Options:
  --parse              Parse mode: parse existing FreeMoCap output
  --video-dir PATH     Directory containing video files
  --output-dir PATH    Output directory for processed data
  --freemocap-env PATH Path to FreeMoCap Python environment
  --gui                Run with GUI (not headless)
  --timeout SECONDS    Timeout in seconds (default: 3600)
  --parse-output       After capture, parse the output files
  --export-npy PATH    Export parsed data to numpy file
  --export-csv PATH    Export parsed data to CSV file
  -v, --verbose        Verbose output
  --dry-run            Show what would be done without executing
```

### Examples

#### Capture from video files

```bash
python -m src.motion_capture.freemocap_ingest ~/sessions/golf_swing_001 \
    --video-dir ~/sessions/golf_swing_001/videos \
    --parse-output \
    --export-csv ~/sessions/golf_swing_001/landmarks.csv
```

#### Parse existing output

```bash
python -m src.motion_capture.freemocap_ingest --parse \
    ~/sessions/golf_swing_001/freemocap_output \
    --export-npy ~/sessions/golf_swing_001/landmarks.npy
```

#### Using the launcher directly in Python

```python
from src.motion_capture.freemocap_ingest import FreeMoCapLauncher, LaunchConfig
from pathlib import Path

launcher = FreeMoCapLauncher()
config = LaunchConfig(
    session_dir=Path("~/sessions/golf_swing_001").expanduser(),
    video_dir=Path("~/sessions/golf_swing_001/videos").expanduser(),
    headless=True,
)
result = launcher.launch(config)

if result.success:
    print(f"Output: {result.output_dir}")
```

#### Using the output adapter

```python
from src.motion_capture.freemocap_ingest import FreeMoCapOutputAdapter
from pathlib import Path

adapter = FreeMoCapOutputAdapter(Path("~/sessions/golf_swing_001/freemocap_output").expanduser())
session = adapter.parse()

print(f"Session: {session.session_id}")
print(f"Frames: {len(session.frames)}")
print(f"Landmarks: {len(session.frames[0].points)}")

# Export to numpy
data = adapter.export_to_numpy(Path("landmarks.npy"))
print(f"Data shape: {data.shape}")  # (frames, landmarks, 4)
```

## Output Format

### Landmark Data

The parsed session contains 3D landmark data from MediaPipe Holistic:

- **Body**: 33 landmarks (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.)
- **Hands**: Key points (wrist, thumb, index, pinky)
- **Face**: Simplified key points

Each landmark has:

- `x, y, z`: 3D coordinates in meters (origin at camera array center)
- `confidence`: Tracking confidence (0.0 - 1.0)
- `visible`: Boolean indicating if landmark is visible

### Calibration Data

FreeMoCap outputs camera calibration data including:

- Camera intrinsics (focal length, principal point)
- Camera extrinsics (position and orientation)
- Triangulation parameters

### File Structure

```
session/
├── videos/
│   ├── cam0.mp4
│   ├── cam1.mp4
│   └── ...
├── freemocap_output/
│   ├── freemocap_3d_landmarks_body.csv
│   ├── camera_calibration.json
│   ├── recording_metadata.json
│   └── plots/
└── logs/
    └── freemocap_20260508-120000.log
```

## Multi-Camera Calibration

FreeMoCap requires calibrated cameras for accurate 3D reconstruction.

### Calibration Procedure

1. Print a checkerboard pattern (available in FreeMoCap docs)
2. Record each camera individually waving the checkerboard
3. Run FreeMoCap calibration:
   ```bash
   source ~/freemocap-env/bin/activate
   python -m freemocap calibrate --video-dir ~/calibration_videos
   ```
4. Use the calibration for subsequent captures

### Accuracy Expectations

- **Positional accuracy**: ~1-2 cm with good calibration
- **Temporal accuracy**: Limited by camera frame rate (typically 30-60 Hz)
- **Failure modes**: High-speed motion (golf swing) may cause tracking dropouts

## Known Limitations

1. **MediaPipe accuracy on golf-speed motion**: MediaPipe Holistic was not trained on high-speed sports. Tracking dropouts on the downswing are likely.

2. **Clubhead tracking**: MediaPipe is body-only. Clubhead tracking requires a separate detector.

3. **Calibration UX**: Multi-camera calibration with a checkerboard is a friction point for users.

4. **Real-time processing**: This integration is for post-processing only. Real-time streaming is not supported.

## AGPL License Boundary

FreeMoCap is licensed under AGPL-3.0. This integration maintains a clean boundary:

- FreeMoCap runs in a separate process
- Communication is via filesystem only
- No Python imports across the boundary
- Main UpstreamDrift environment has no FreeMoCap packages installed

Verify the boundary:

```bash
# In main UpstreamDrift env
pip list | grep -E "freemocap|skelly|anipose|PySide"
# Should return nothing

# In FreeMoCap env
source ~/freemocap-env/bin/activate
pip list | grep -E "freemocap|skelly|anipose|PySide"
# Should show FreeMoCap packages
```

## Troubleshooting

### FreeMoCap environment not found

```
Error: FreeMoCap environment not found.
```

Run the setup script to create the environment:

```bash
./setup_freemocap_env.sh
```

### Tracking quality is poor

- Ensure good lighting
- Use high-contrast clothing
- Position cameras for full body coverage
- Calibrate cameras carefully
- Consider higher frame rate cameras

### Import errors

If you see import errors when running FreeMoCap:

```bash
source ~/freemocap-env/bin/activate
pip install --upgrade freemocap
```

## References

- [FreeMoCap Repository](https://github.com/freemocap/freemocap)
- [FreeMoCap Documentation](https://docs.freemocap.org)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [Aniposelib](https://github.com/lambdaloop/aniposelib) (multi-camera triangulation)

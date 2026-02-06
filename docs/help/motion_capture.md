# Motion Capture Integration

Import, process, and analyze motion capture data for biomechanical analysis.

## Overview

UpstreamDrift supports multiple motion capture formats and pose estimation systems, allowing you to import real-world movement data for analysis and simulation.

## Supported Formats

### C3D Files (.c3d)

C3D is the industry-standard format for biomechanics motion capture.

**Features:**
- 3D marker positions
- Analog data (force plates, EMG)
- Frame rate and timing information
- Metadata and labels

**Use for:** Professional motion capture lab data (Vicon, OptiTrack, Qualisys)

### CSV Files (.csv)

Flexible format for custom marker data.

**Expected columns:**
- Time or frame number
- X, Y, Z coordinates per marker
- Optional: velocity, acceleration data

**Example format:**
```csv
time,marker1_x,marker1_y,marker1_z,marker2_x,marker2_y,marker2_z
0.000,0.123,0.456,0.789,0.321,0.654,0.987
0.001,0.124,0.457,0.790,0.322,0.655,0.988
```

### JSON Files (.json)

Structured format with hierarchical data.

**Use for:** Pose estimation output, custom applications

**Example structure:**
```json
{
  "frame_rate": 120,
  "frames": [
    {
      "time": 0.0,
      "markers": {
        "left_shoulder": [0.1, 0.5, 1.2],
        "right_shoulder": [0.3, 0.5, 1.2]
      }
    }
  ]
}
```

## Importing Motion Capture Data

### Using the Import Dialog

1. **Open Import Dialog**
   - File menu > Import Motion Capture
   - Or press Ctrl+I

2. **Select File**
   - Browse to your data file
   - Supported formats shown in filter

3. **Configure Import Settings**
   - Set coordinate system (if needed)
   - Map markers to standard names
   - Specify frame range (optional)

4. **Preview Data**
   - View marker trajectories
   - Check for gaps or errors
   - Verify coordinate system

5. **Import**
   - Click Import to load data
   - Data appears in Motion Capture panel

### Using Python API

```python
from shared.python.motion_capture import MotionCaptureLoader

# Load C3D file
mocap = MotionCaptureLoader.load_c3d("swing.c3d")

# Load CSV with custom mapping
mocap = MotionCaptureLoader.load_csv(
    "data.csv",
    marker_columns={
        "left_shoulder": ["LSH_X", "LSH_Y", "LSH_Z"],
        "right_shoulder": ["RSH_X", "RSH_Y", "RSH_Z"],
    }
)
```

## C3D Viewer

The built-in C3D Viewer provides specialized visualization for motion capture data.

### Features

- **3D Marker Visualization**
  - Animated marker positions
  - Trajectory trails
  - Color-coded marker groups

- **Analog Data Plotting**
  - Force plate data
  - EMG signals
  - Custom analog channels

- **Navigation**
  - Frame-by-frame stepping
  - Playback speed control
  - Jump to specific frame/time

### Controls

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Left/Right | Step frame |
| Home | Go to start |
| End | Go to end |
| +/- | Adjust speed |

## Pose Estimation from Video

UpstreamDrift can extract motion data from video using pose estimation.

### Supported Systems

| System | Keypoints | Speed | Accuracy |
|--------|-----------|-------|----------|
| MediaPipe | 33 | Fast | Good |
| OpenPose | 25 | Medium | Excellent |
| MoveNet | 17 | Very Fast | Good |

### Processing Video

1. **Import Video**
   - File > Import Video for Pose Estimation
   - Supported formats: MP4, AVI, MOV

2. **Select Pose Estimator**
   - MediaPipe (recommended for local processing)
   - OpenPose (requires separate installation)
   - MoveNet (fast, lower accuracy)

3. **Configure Settings**
   - Frame rate for extraction
   - Confidence threshold
   - Smoothing options

4. **Process**
   - Click "Process Video"
   - Progress bar shows completion
   - Results saved automatically

5. **Review Results**
   - Preview detected poses
   - Adjust confidence threshold if needed
   - Export or use directly

### Tips for Video Capture

- Use high frame rate (60fps+) for fast movements
- Ensure good lighting
- Minimize background clutter
- Keep camera stable
- Capture from multiple angles if possible

## Motion Retargeting

Map motion capture data to your simulation model.

### Retargeting Process

1. **Load Motion Data**
   - Import your motion capture file

2. **Select Target Model**
   - Choose the simulation model to drive

3. **Define Marker Mapping**
   - Map mocap markers to model joints/bodies
   - Example: "LSHO" marker -> "left_shoulder" joint

4. **Configure Options**
   - Scaling: Match model proportions
   - Filtering: Smooth noisy data
   - Gap filling: Interpolate missing data

5. **Execute Retargeting**
   - Click "Retarget"
   - Review results in 3D view

6. **Fine-tune**
   - Adjust individual joint mappings
   - Apply offsets if needed
   - Re-run retargeting

### Marker Mapping

Standard marker set mappings are provided for common configurations:

| Marker Set | Markers | Use Case |
|------------|---------|----------|
| Plug-in Gait | 39 | Full body clinical |
| Helen Hayes | 15 | Lower body |
| Cleveland Clinic | 12 | Upper body golf |
| Custom | Variable | User-defined |

### Python API for Retargeting

```python
from shared.python.motion_capture import MotionRetargeting

# Create retargeter
retargeter = MotionRetargeting(
    source_skeleton="vicon_full_body",
    target_model=my_model
)

# Define mapping
marker_mapping = {
    "LSHO": "left_shoulder",
    "RSHO": "right_shoulder",
    "LELB": "left_elbow",
    # ... more markers
}
retargeter.set_marker_mapping(marker_mapping)

# Perform retargeting
joint_trajectory = retargeter.retarget(mocap_data)
```

## Data Quality

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Gaps | Occlusion | Gap filling interpolation |
| Noise | Marker vibration | Low-pass filtering |
| Spikes | Marker swap | Manual correction |
| Drift | Calibration | Re-calibrate or correct |

### Quality Metrics

The system reports:
- Gap percentage per marker
- RMS noise estimate
- Outlier detection
- Coordinate system verification

## Troubleshooting

### Import Fails

- Verify file format is correct
- Check file is not corrupted
- Ensure all required columns present (CSV)
- Check encoding (UTF-8 recommended)

### Wrong Coordinate System

- Use coordinate system transform tool
- Specify axis mapping in import dialog
- Apply rotation/translation after import

### Missing Markers

- Enable gap filling
- Interpolation options: Linear, Cubic, Pattern-based
- Mark critical frames for manual review

---

*See also: [Full User Manual](../USER_MANUAL.md) | [Visualization](visualization.md) | [Analysis Tools](analysis_tools.md)*

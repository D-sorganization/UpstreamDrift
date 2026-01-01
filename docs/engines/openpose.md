# OpenPose Integration

## Overview
OpenPose is the state-of-the-art library for real-time multi-person keypoint detection. The Golf Modeling Suite integrates OpenPose to extract 2D kinematic data from video inputs, which can then be lifted to 3D for biomechanical analysis.

## Key Features in Suite
- **2D Keypoint Detection**: Bodies, hands, feet, and face landmarks.
- **Video Processing Pipeline**: frame-by-frame extraction and JSON serialization.
- **Integration**: Feeds directly into the Kinematic Reconstruction pipeline.

## Usage
Located in `shared/python/pose_estimation/`.

### Python Access
```python
from shared.python.pose_estimation.openpose_estimator import OpenPoseEstimator

estimator = OpenPoseEstimator()
keypoints = estimator.process_image(image_array)
```

### GUI
Launch the Pose Estimation GUI for video processing:
```bash
python -m shared.python.pose_estimation.openpose_gui
```

## Dependencies
Requires the `openpose` binary or python bindings to be installed and accessible in the system path or configured in `config/models.yaml`.

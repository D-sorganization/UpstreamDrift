# Pose2Sim Ingestion

`Pose2SimAdapter` loads local Pose2Sim session outputs into the current
motion-pipeline CIR:

- one `KeypointSequence` per camera for 2-D detections,
- one `Calibration` shared by all camera streams,
- an optional triangulated 3-D `KeypointSequence` converted from the Pose2Sim
  TRC output.

The default detector layout is MediaPipe, which keeps the local path permissive.
OpenPose is supported only when explicitly requested:

```python
from pathlib import Path

from src.shared.python.motion_pipeline.sources import (
    Pose2SimDetector,
    load_pose2sim_observations,
)

observations = load_pose2sim_observations(Path("Session/Trial"))
openpose_observations = load_pose2sim_observations(
    Path("Session/Trial"),
    detector=Pose2SimDetector.OPENPOSE,
)
```

Expected session files:

- `calibration.json`, `camera_calibration.json`, or `pose2sim_calibration.json`
  at the session root.
- Per-camera detection JSON files under `detections/`, `pose-2d/`, or `pose2d/`.
- Optional triangulated `.trc` under `pose-3d/`, `pose3d/`, or `triangulated/`.

Camera detection confidence is preserved on every 2-D keypoint. When a
triangulated TRC is present, each 3-D keypoint receives the average confidence
from matching camera keypoints at the same frame. This keeps residual weighting
information available for downstream estimators instead of collapsing the
session to an unweighted skeleton.

On `origin/main`, the public observation contract is `KeypointSequence` plus
`Calibration` in `src.shared.python.motion_pipeline.contracts`. The adapter
therefore returns a `Pose2SimObservations` bundle over those public types. When
CC-12's `CanonicalObservations` lands, this bundle can be mapped without
changing the loaded confidence or calibration fields.

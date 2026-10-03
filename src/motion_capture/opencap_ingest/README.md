# OpenCap Sidecar Ingestion Runner

This module implements the isolated sidecar runner for `opencap-core` (Stanford NMBL), adhering to **ADR-0053**.

## Architecture & Boundary Invariants

- **Subprocess / Container Only:** OpenCap is executed solely as an external process (via `managed_popen` with guaranteed cleanup) or as a Docker container.
- **No Core Dependency:** OpenCap source code is never vendored, bundled, or imported into UpstreamDrift product core.
- **No TensorFlow in Core:** The marker augmenter's TensorFlow dependencies stay entirely on the other side of the sidecar boundary.
- **Commercial Default:** By default, detector is set to `hrnet` / `mmpose` (Apache-2.0). OpenPose is non-commercial and requires explicit user opt-in (`allow_non_commercial=True`).
- **Canonical Observations & Kinematics:** Outputs are collected via `load_opencap_session` into canonical `OpenCapSession` objects.

## Usage

```python
from pathlib import Path
from src.motion_capture.opencap_ingest import OpenCapLaunchConfig, OpenCapLauncher

launcher = OpenCapLauncher()
config = OpenCapLaunchConfig(
    session_dir=Path("./my_session"),
    video_dir=Path("./videos"),
    calibration_dir=Path("./calibration"),
    detector="hrnet",
)
result = launcher.launch(config)
if result.success:
    session = result.session
    print("Loaded trial:", session.trial)
```

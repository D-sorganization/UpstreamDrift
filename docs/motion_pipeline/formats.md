# Motion Pipeline — Format Matrix

> Auto-generated format support matrix for motion capture sources. Hand-written notes on each source's quirks.

## Format Support Matrix

| Format              | Extension      | Adapter                | 3D Support                  | Confidence | Temporal | Notes                               |
| ------------------- | -------------- | ---------------------- | --------------------------- | ---------- | -------- | ----------------------------------- |
| **BVH**             | `.bvh`         | `BVHAdapter`           | ✅ Yes                      | ❌ No      | ✅ Yes   | Euler order varies (XYZ vs ZXY)     |
| **TRC**             | `.trc`         | `TRCAdapter`           | ✅ Yes                      | ❌ No      | ✅ Yes   | OpenSim / Vicon Nexus / Theia, Y-up |
| **OpenSim STO/MOT** | `.sto`, `.mot` | `STOMotAdapter`        | n/a (joint angles)          | ❌ No      | ✅ Yes   | `inDegrees` flag honored            |
| **OpenPose**        | `.json`        | `OpenPoseJSONAdapter`  | ❌ 2D only                  | ✅ Yes     | ✅ Yes   | BODY_25 or COCO_18 schema           |
| **AlphaPose**       | `.json`        | `AlphaPoseJSONAdapter` | ❌ 2D only                  | ✅ Yes     | ✅ Yes   | COCO-17 multi-frame                 |
| **HRNet**           | `.json`        | `HRNetJSONAdapter`     | ❌ 2D only                  | ✅ Yes     | ✅ Yes   | COCO-17 single-person               |
| **MediaPipe**       | `.json`        | `MediaPipeJSONAdapter` | partial 3D (relative depth) | ✅ Yes     | ✅ Yes   | 33 landmarks, normalized coords     |
| **CSV**             | `.csv`         | `CSVAdapter`           | ✅ Yes                      | ❌ No      | ✅ Yes   | columns: `frame, time, x_*/y_*/z_*` |
| **C3D**             | `.c3d`         | `C3DAdapter`           | ✅ Yes                      | ✅ Yes     | ✅ Yes   | Binary, requires `ezc3d`            |
| **FBX**             | `.fbx`         | _planned_              | ✅ Yes                      | ❌ No      | ✅ Yes   | Proprietary, Blender conversion     |
| **Qualisys**        | `.qtm`         | _planned_              | ✅ Yes                      | ✅ Yes     | ✅ Yes   | Native QTM format                   |

> The full canonical list of shipped adapters lives in
> `src/shared/python/motion_pipeline/sources/` and is exercised by
> [`tests/integration/motion_pipeline/test_loader_golden_roundtrip.py`](../../tests/integration/motion_pipeline/test_loader_golden_roundtrip.py)
> against the golden fixtures in
> [`tests/data/motion_pipeline/golden/`](../../tests/data/motion_pipeline/README.md).

---

## Source-Specific Quirks

### Theia

- **World Axis**: Y-up, Z-forward (differs from MuJoCo's Z-up)
- **Unit**: Millimeters (convert to meters before IK)
- **Marker Names**: CamelCase (e.g., `RASI`, `LASI`)
- **Conversion**:
  ```python
  from src.shared.python.motion_pipeline.converters import TheiaConverter
  converter = TheiaConverter()
  motion = converter.load("theia_capture.trc")
  motion.to_meters()  # Convert from mm
  motion.reorient_axes(up="Z")  # Convert to Z-up
  ```

### OpenPose

- **Confidence Semantics**:
  - `0`: Not detected
  - `1`: Low confidence (occluded)
  - `2`: High confidence
- **Keypoint Order**: COCO (18 keypoints) or Body25 (25 keypoints)
- **2D Only**: Requires lifting to 3D via triangulation or learned methods
- **Example**:
  ```json
  {
    "people": [{
      "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...],
      "hand_right_keypoints_2d": [...],
      "hand_left_keypoints_2d": [...]
    }]
  }
  ```

### MediaPipe

- **Confidence**: Two values per keypoint:
  - `visibility`: 0.0-1.0 (how visible in frame)
  - `presence`: 0.0-1.0 (likelihood of being present)
- **3D Output**: MediaPipe Pose can output 3D coordinates (relative depth)
- **Coordinate System**: Normalized (0-1) image coordinates
- **Example**:
  ```python
  from src.shared.python.motion_pipeline.converters import MediaPipeConverter
  converter = MediaPipeConverter()
  motion = converter.load("mediapipe_output.json")
  motion.denormalize(image_width=1920, image_height=1080)
  ```

### BVH

- **Euler Order**: Varies by source (XYZ, XZY, YXZ, YZX, ZXY, ZYX)
- **Root Motion**: First 3 channels = translation, next 3 = rotation
- **Frame Time**: Specified in header (e.g., `FRAME TIME: 0.0333` for 30fps)
- **Conversion**:
  ```python
  from src.shared.python.motion_pipeline.converters import BVHConverter
  converter = BVHConverter(euler_order="ZXY")
  motion = converter.load("motion.bvh")
  ```

### C3D

- **Binary Format**: Requires `c3d` Python package
- **Contains**: Analog data (force plates) + point data (markers)
- **Confidence**: Stored as residuals (lower = better)
- **Example**:
  ```python
  import c3d
  with open("capture.c3d", "rb") as f:
      reader = c3d.Reader(f)
      for points, analog in reader.read_frames():
          # points.shape = (markers, 4, frames) - 4th dim is residual
          pass
  ```

---

## Auto-Generation Script

The format matrix is auto-generated by CI test #4571:

```bash
python3 tests/unit/motion_pipeline/test_format_matrix.py --generate
```

This produces `docs/motion_pipeline/formats.md` from the canonical format definitions in `src/shared/python/motion_pipeline/formats/`.

---

## Related Documents

- [User Workflow Guide](README.md) — How to run the pipeline
- [Troubleshooting](troubleshooting.md) — Common failure modes
- [ADR 0019](../adr/0019-motion-pipeline-architecture.md) - Architecture decisions

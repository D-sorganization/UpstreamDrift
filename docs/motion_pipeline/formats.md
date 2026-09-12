# Motion Pipeline — Format Matrix

> Auto-generated format support matrix for motion capture sources. Hand-written notes on each source's quirks.

## Format Support Matrix

| Format              | Extension      | Adapter                 | 3D Support                  | Confidence | Temporal | Notes                                          |
| ------------------- | -------------- | ----------------------- | --------------------------- | ---------- | -------- | ---------------------------------------------- |
| **BVH**             | `.bvh`         | `BVHAdapter`            | ✅ Yes                      | ❌ No      | ✅ Yes   | Euler order varies (XYZ vs ZXY)                |
| **TRC**             | `.trc`         | `TRCAdapter`            | ✅ Yes                      | ❌ No      | ✅ Yes   | OpenSim / Vicon Nexus / Theia, Y-up            |
| **OpenCap Session** | directory      | `OpenCapSessionAdapter` | ✅ Yes                      | ❌ No      | ✅ Yes   | Augmented-marker TRC to canonical observations |
| **OpenSim STO/MOT** | `.sto`, `.mot` | `STOMotAdapter`         | n/a (joint angles)          | ❌ No      | ✅ Yes   | `inDegrees` flag honored                       |
| **OpenPose**        | `.json`        | `OpenPoseJSONAdapter`   | ❌ 2D only                  | ✅ Yes     | ✅ Yes   | BODY_25 or COCO_18 schema                      |
| **AlphaPose**       | `.json`        | `AlphaPoseJSONAdapter`  | ❌ 2D only                  | ✅ Yes     | ✅ Yes   | COCO-17 multi-frame                            |
| **HRNet**           | `.json`        | `HRNetJSONAdapter`      | ❌ 2D only                  | ✅ Yes     | ✅ Yes   | COCO-17 single-person                          |
| **MediaPipe**       | `.json`        | `MediaPipeJSONAdapter`  | partial 3D (relative depth) | ✅ Yes     | ✅ Yes   | 33 landmarks, normalized coords                |
| **DeepLabCut**      | `.h5`, `.hdf5`, `.csv` | `DeepLabCutAdapter` | ❌ 2D only              | ✅ Yes     | ✅ Yes   | Custom bodyparts (`schema="custom"`), `fps` option (default 30) |
| **4D-Humans/HMR2**  | `.csv` (sidecar) | `HMR2Adapter`         | ✅ Yes                      | ❌ No      | ✅ Yes   | 22 SMPL body joints in meters (`schema="custom"`), sidecar `joints3d.csv` |
| **CSV**             | `.csv`         | `CSVAdapter`            | ✅ Yes                      | ❌ No      | ✅ Yes   | columns: `frame, time, x_*/y_*/z_*`            |
| **C3D**             | `.c3d`         | `C3DAdapter`            | ✅ Yes                      | ✅ Yes     | ✅ Yes   | Binary, requires `ezc3d`                       |
| **FBX**             | `.fbx`         | _planned_               | ✅ Yes                      | ❌ No      | ✅ Yes   | Proprietary, Blender conversion                |
| **Qualisys**        | `.qtm`         | _planned_               | ✅ Yes                      | ✅ Yes     | ✅ Yes   | Native QTM format                              |

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

### OpenCap

- **Session Layout**: accepts an OpenCap session directory and finds an
  augmented-marker TRC file under the session tree.
- **Output Contract**: returns `CanonicalObservations`, preserving session
  metadata and source provenance.
- **Marker Names**: normalizes common OpenCap labels such as `R_ASIS` and
  `R_Shoulder` to OpenSim marker-site names such as `R.ASIS` and `R.Acromium`.
- **Units**: delegates TRC parsing to `TRCAdapter`, so millimeters are converted
  to meters before observations reach downstream IK.

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

### DeepLabCut

- **Column Layout**: pandas DataFrame with a 3-level column MultiIndex
  `(scorer, bodyparts, coords)` where `coords` is `x` / `y` / `likelihood`.
  Stored as HDF5 (`DataFrame.to_hdf`, key usually `df_with_missing`, either
  pandas "fixed" or "table" layout) or as CSV with 3 header rows.
- **Custom Keypoints**: bodyparts are arbitrary user-defined names (e.g.
  `clubhead`, `hosel`, `ball`); the adapter emits `schema_name="custom"` with
  names preserved verbatim and `likelihood` mapped to keypoint confidence.
- **Timestamps**: DLC files are frame-indexed with no time column; timestamps
  are synthesized from the adapter's `fps` option (default 30.0).
- **Dependencies**: the HDF5 reader runs on `h5py` directly — neither
  PyTables nor the `deeplabcut` package is required.
- **Not Supported**: multi-animal DLC output (extra `individuals` column
  level) is rejected with a descriptive error.
- **Example**:
  ```python
  from src.shared.python.motion_pipeline.sources import DeepLabCutAdapter
  sequence = DeepLabCutAdapter(fps=120.0).load_checked("videoDLC_resnet50.h5")
  ```

### 4D-Humans / HMR 2.0 (monocular 3D)

- **Sidecar Output**: the adapter reads `joints3d.csv` written by the
  HMR2 sidecar (`src/tools/hmr2_sidecar/run_hmr2.py`) — columns
  `frame,time` then `<joint>_x,<joint>_y,<joint>_z` for the 22 SMPL body
  joints, positions in **meters**, timestamps from the `time` column.
- **Conservative Sniffing**: a CSV is claimed only when its header matches
  the sidecar column contract exactly, or when it has the joint-triplet
  shape *and* a sibling `metadata.json` names the 4D-Humans tool; generic
  `frame,timestamp,x_*` trajectory CSVs stay with `CSVAdapter`.
- **Schema**: SMPL is not a `SchemaName` literal member, so the sequence
  uses `schema_name="custom"` with joint names preserved verbatim.
- **Shape Betas**: the sidecar also writes `betas.json` (10 SMPL shape
  coefficients + gender); `src.tools.hmr2_sidecar.betas_bridge` converts
  it into the character builder's `BodyParameters(smplx_betas=...)`.
- **Licensing**: 4D-Humans is CC-BY-NC and SMPL model files are
  research-restricted — the sidecar only ever invokes a user-installed
  external tool (`HMR2_COMMAND` env var) as a subprocess; no CC-BY-NC
  code is imported or vendored, and stub artifacts keep contract tests
  stable when the tool is absent.
- **Example**:
  ```python
  from src.shared.python.motion_pipeline.sources import load_any
  sequence = load_any("out/joints3d.csv")  # routed to HMR2Adapter
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

- **Binary Format**: Parsed by the Rust `upstream_mocap_io` wheel when the
  `mocap-io` extra is installed, with an `ezc3d` fallback (the `c3d` extra).
  The unrelated `py-c3d` package is **not** used.
- **Contains**: Analog data (force plates) + point data (markers)
- **Confidence**: Stored as residuals (lower = better)
- **Example**:
  ```python
  from src.shared.python.motion_pipeline.sources import load_any
  sequence = load_any("capture.c3d")  # Rust parser first, ezc3d fallback
  ```
- **Writing**: `src.motion_capture.canonical_c3d_exporter` (ezc3d-backed) is
  the only supported C3D writer; C3D is an output format, never a lossy
  intermediate inside the pipeline.

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
- [ADR 0007](../adr/0007-motion-pipeline-architecture.md) — Architecture decisions

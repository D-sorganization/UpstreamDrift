# Versioned Matched Swing Candidate Specification (`matched-swing-candidate-v1`)

Governs the unified, versioned, engine-independent trajectory artifact format (`MatchedSwingCandidate`) shared across all six physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite, Simscape) and tools (tour matching viewer, cross-engine replay, setup parity, export packages, and ledger).

Established under **MS-15 (#10334)** in accordance with the Matched Swing Program contract (**MS-100 / #10374**, **MS-104 / #10378**).

---

## 1. Profiles

Every candidate declares one of two explicit profiles in its metadata:

| Profile     | Purpose                                                                | Required State / Data                                                                                | Acceptance Eligibility                                                       |
| ----------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `kinematic` | Pose & IK trajectory tracking (e.g., marker IK solves, OpenSim `.mot`) | Monotone $t$, coordinates $q$, optional markers $m$                                                  | Kinematic validation only; **rejected for dynamic full-swing qualification** |
| `dynamic`   | Fully forward-replayable simulation trajectory                         | Monotone $t$, coordinates $q$, velocities $v$, actuator controls $\tau$, solver & contact parameters | Eligible for dynamic ladder qualification (G1 $\to$ G2 $\to$ G3)             |

> [!IMPORTANT]
> Converters strictly preserve missing-field annotations. A kinematic artifact (e.g. from historical IK runs or OpenSim `.mot` files without torque data) will never synthesize or invent artificial actuator dynamics, ensuring fail-closed integrity.

---

## 2. Package Format (`.npz`)

A candidate package is packaged as a self-contained `.npz` archive without Python pickle dependency (`allow_pickle=False`):

- **`manifest_json`**: String array encoding canonical JSON metadata (`CandidateMetadata`).
- **`time_s`**: 1D `float64` array of shape $(N,)$ with strictly monotonically increasing physical timestamps ($t_0 \ge 0$).
- **`q`**: 2D `float64` array of shape $(N, n_q)$ representing generalized coordinates in document order.
- **`v`**: 2D `float64` array of shape $(N, n_v)$ representing tangent-space generalized velocities (required for `dynamic`).
- **`tau`**: 2D `float64` array of shape $(N, n_u)$ representing actuator efforts/torques (required for `dynamic`, prohibited in `kinematic`).
- **`model_markers_m`**: Optional 3D `float64` array of shape $(N, n_m, 3)$ representing model marker spatial coordinates.
- **`target_markers_m`**: Optional 3D `float64` array of shape $(N, n_m, 3)$ representing optical target marker coordinates.
- **`marker_validity`**: Optional 2D `bool` array of shape $(N, n_m)$ indicating tracked marker visibility.
- **`actuator_states`**: Optional 2D `float64` array of internal actuator/muscle activation states.
- **`external_forces`**: Optional 2D `float64` array of external ground reaction forces / contact wrench measurements.

---

## 3. Metadata Schema (`CandidateMetadata`)

```json
{
  "schema_version": "matched-swing-candidate-v1",
  "profile": "dynamic",
  "engine": "mujoco",
  "model_name": "full_body_anthro_driver",
  "model_sha256": "4e72...b91a",
  "source_c3d_sha256": "545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba",
  "document_sha256": "9b12...c44d",
  "coordinate_names": ["root_tx", "root_ty", "root_tz", "..."],
  "velocity_names": ["root_vx", "root_vy", "root_vz", "..."],
  "actuator_names": ["tau_pelvis_tilt", "tau_lumbar_extension", "..."],
  "marker_names": ["WaistLeft", "WaistRight", "..."],
  "units": {
    "time": "s",
    "position": "m",
    "angle": "rad",
    "linear_velocity": "m/s",
    "angular_velocity": "rad/s",
    "force": "N",
    "torque": "N*m"
  },
  "frame_convention": "z_up_y_forward",
  "interpolation": "cubic_spline",
  "event_indices": {
    "address": 0,
    "top_of_backswing": 298,
    "impact": 510,
    "finish": 653
  },
  "coverage_mask": {},
  "solver_settings": {
    "integrator": "implicitfast",
    "dt": 0.002
  },
  "contact_parameters": {},
  "closure_parameters": {},
  "checksums": {
    "time_s": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "q": "8f42...a901",
    "v": "3b12...55cd",
    "tau": "0a12...99ef"
  },
  "missing_fields": [],
  "extra": {}
}
```

---

## 4. Immutability & Tamper Detection

Upon instantiation of `MatchedSwingCandidate`:

1. All arrays have `flags.writeable = False` enforced. In-place modification attempts will raise `ValueError: assignment destination is read-only`.
2. Array contents are hashed using SHA-256 (`arr.tobytes()`).
3. `load_candidate(..., validate_checksums=True)` compares real-time byte hashes against `metadata.checksums`. Any tampering or corruption raises `ValueError: Tampering detected: checksum mismatch`.

---

## 5. Generalized Coordinates vs. Tangent Space ($n_q \ne n_v$)

For articulated models where the configuration topology differs from tangent velocity space (such as a 7-parameter quaternion floating base with 6-DOF spatial velocity vector):

- `coordinate_names` has length $n_q$ (e.g. 7).
- `velocity_names` has length $n_v$ (e.g. 6).
- The consistency of actuator efforts $\tau_{act}$ and generalized forces $\tau_q$ across coordinate representations is verified via the virtual work principle:
  $$\tau_q^T v = \tau_{act}^T \dot{q}_{act} = \tau_{act}^T (B^T v)$$
  validated by `check_virtual_work_consistency()`.

---

## 6. Legacy Format Converters

Provided by `src.shared.python.motion_matching.candidate_convert`:

- `convert_returned81_replay(npz_path, spec=None, engine="unknown")`: Losslessly translates `*_returned81_replay.npz` archives to kinematic candidate packages.
- `convert_opensim_mot(mot_path, spec=None, engine="opensim")`: Translates OpenSim `.mot` and `.sto` motion files to kinematic candidate packages with degrees-to-radians conversion.
- `convert_ground_support_ik(npz_path, spec=None, engine="mujoco")`: Translates `ik_trajectory.npz`.
- `convert_ground_support_dynamics(npz_path, spec=None, engine="mujoco")`: Translates `dynamics_record.npz` to dynamic candidate packages.

---

## 7. Python API Quickstart

### Saving a Candidate:

```python
from src.shared.python.motion_matching.candidate import (
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_io import save_candidate

metadata = CandidateMetadata(
    profile=CandidateProfile.DYNAMIC,
    engine="mujoco",
    model_name="full_body_anthro_driver",
    coordinate_names=tuple(coords),
    actuator_names=tuple(actuators),
)
candidate = MatchedSwingCandidate(
    metadata=metadata,
    time_s=time_s,
    q=q,
    v=v,
    tau=tau,
    markers=CandidateMarkers(model_markers_m=model_markers),
)
save_candidate(candidate, "candidate.npz")
```

### Loading a Candidate:

```python
from src.shared.python.motion_matching.candidate_io import load_candidate

candidate = load_candidate("candidate.npz", validate_checksums=True)
print(candidate.metadata.profile)  # CandidateProfile.DYNAMIC
print(candidate.q.shape)           # (N, nq)
```

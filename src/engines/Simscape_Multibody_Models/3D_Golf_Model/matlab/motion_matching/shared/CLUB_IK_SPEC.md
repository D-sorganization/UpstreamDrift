# Club Trajectory Loader Specification

The "club IK" is misnamed for Phase 1 — there is no actual joint-angle inverse kinematics yet. It is a **trajectory ingest pipeline** that converts a measured swing (from any source format) into the canonical schema consumed by the cost function.

When body markers come online, this same module gains a real IK stage that solves shoulder/elbow/wrist joint angles. Until then, only the club is observed.

## Output schema (canonical `target` struct)

```matlab
target = struct( ...
    "time",         (N×1) double, simulation timegrid in seconds, monotonic, dt = 1/sample_rate, ...
    "butt",         (N×3) double, butt position in metres, world frame, ...
    "clubhead",     (N×3) double, clubhead position in metres, world frame, ...
    "club_quat",    (N×4) double, club orientation as unit quaternion [w x y z], ...
    "impact_idx",   scalar uint32, index of impact frame (max clubhead speed), ...
    "source",       struct with provenance: filename, format, subject_id, trial_id, sha256 ...
);
```

Python mirror is a `dataclass`:

```python
@dataclass(frozen=True)
class ClubTarget:
    time: np.ndarray            # (N,) float64
    butt: np.ndarray            # (N,3) float64
    clubhead: np.ndarray        # (N,3) float64
    club_quat: np.ndarray       # (N,4) float64, [w,x,y,z]
    impact_idx: int
    source: SourceProvenance
```

## Source formats supported

### 1. Excel (priority for Phase 1)

[Wiffle_ProV1_club_3D_data.xlsx](../../src/apps/golf_gui/Motion%20Capture%20Plotter/) — already parsed by [mocap_data_loader.py](../../src/apps/golf_gui/Motion%20Capture%20Plotter/mocap_data_loader.py).

- **Units in source:** inches, frames at the file's native rate.
- **Conversion:** inches → metres via `× 0.0254`.
- **Orientation:** the file stores 3×3 rotation matrices per frame; convert to unit quaternion with sign normalised so `q[0] >= 0` to suppress the `q ↔ -q` ambiguity at the source.
- **Sheets:** `TW_wiffle`, `TW_ProV1`, `GW_wiffle`, `GW_ProV11`. Each is one swing.

### 2. C3D (priority for Phase 1, validation)

[Data/Optical C3D Files/](../../Data/Optical%20C3D%20Files/). One file is known to exist; **it has not been verified to parse.** Issue #013 is the validation pass.

- Use the existing Python reader at [c3d_reader.py](../../python/src/c3d_reader.py) via the MATLAB `pyrunfile` interface, or use BTK if installed.
- Marker names will need to be mapped to butt/clubhead — the convention in the file is unknown; #013 will document it.

### 3. Synthetic (for testing)

The TDD harness needs a target that **provably** can be matched. The synthetic source runs the Simscape model with a known coefficient vector `θ_truth`, records the resulting club trajectory, and emits it as a `target` struct.

Issue #014 builds this. Every option's tests use it as the trivial-fit oracle: if `fit(synthesize_swing(θ)) ≉ θ` (or at least produces RMSE < 1 mm), the option is broken.

## Time alignment

The measured swing and the simulation must share the same timegrid before the cost function can subtract them.

Default policy:

1. **Detect impact.** Index of maximum `‖d r_clubhead/dt‖`. Use central differences with a 5-point stencil over the raw mocap.
2. **Define window.** Take `[t_impact − 0.25, t_impact + 0.05]` seconds (or the full swing if shorter).
3. **Resample.** Linear interpolation in position, SLERP in orientation, onto the simulation timegrid `0 : 1/sample_rate : T`.
4. **Anchor.** Re-time so the measured impact lines up with the simulation's expected impact (default: `t = 0.25 s` into the simulation).

Alternative policies:

- `time_alignment = "address"` — align the address (start) frame instead of impact. Useful if the measurement starts cleanly.
- `time_alignment = "none"` — caller has already aligned and resampled. Loader passes through.

## Function signatures

MATLAB (issue #013):

```matlab
function target = load_club_target_excel(xlsx_path, sheet_name, opts)
%LOAD_CLUB_TARGET_EXCEL  Read club 6-DOF trajectory from a Wiffle-style xlsx file.
%
%   target = LOAD_CLUB_TARGET_EXCEL(XLSX_PATH, SHEET_NAME, OPTS) returns a
%   target struct as specified in CLUB_IK_SPEC.md.

function target = load_club_target_c3d(c3d_path, opts)
%LOAD_CLUB_TARGET_C3D  Read club 6-DOF trajectory from a C3D file.

function target = synthesize_target_from_coefficients(theta, opts)
%SYNTHESIZE_TARGET_FROM_COEFFICIENTS  Build a target struct by running the
%   Simscape model with known coefficients. Used as the oracle for tests.
```

Python mirror (issue #017):

```python
def load_club_target_excel(path: Path, sheet: str, opts: AlignOptions) -> ClubTarget: ...
def load_club_target_c3d(path: Path, opts: AlignOptions) -> ClubTarget: ...
def synthesize_target_from_coefficients(theta: np.ndarray, opts: AlignOptions) -> ClubTarget: ...
```

## Validation rules (DbC postconditions)

Every loader function must produce a target that satisfies:

1. `target.time` is strictly increasing, `time(1) == 0`, `time(end) ≤ T_max + eps`.
2. All trajectory arrays have the same number of rows as `time`.
3. `target.butt`, `target.clubhead` contain no NaN/Inf; magnitudes plausible (`‖r‖ < 5 m`).
4. `target.club_quat` rows are unit-norm to within `1e-6`.
5. `1 ≤ target.impact_idx ≤ N`.
6. `target.source.sha256` matches a freshly computed sha256 of the source file.

## Implementation notes

- The Python implementation should reuse [c3d_reader.py](../../python/src/c3d_reader.py) and [mocap_data_loader.py](../../src/apps/golf_gui/Motion%20Capture%20Plotter/mocap_data_loader.py) where practical — do not re-implement the parsers.
- The MATLAB Excel loader can call the Python loader via `pyrunfile` to avoid duplicating the sheet-parsing logic. Issue #013 picks the approach.
- The synthesizer must use the **exact same** Simscape callback that the optimizer will use (`simulate_with_coefficients` from issue #018). That guarantees `fit(synthesize(θ)) → θ` is achievable; if not, the optimizer is broken — not the data.

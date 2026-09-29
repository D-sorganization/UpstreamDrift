# Club Trajectory Loader Specification

The "club IK" is misnamed for Phase 1 — there is no actual joint-angle inverse kinematics yet. It is a **trajectory ingest pipeline** that converts a measured swing (from any source format) into the canonical schema consumed by the cost function.

When body markers come online, this same module gains a real IK stage that solves shoulder/elbow/wrist joint angles. Until then, only the club is observed.

> **Test-trial inventory:** The canonical list of measured swings used by the test suite — sheet names, event markers, expected CHS_mph values, per-trial data-quality notes, and instructions for adding a new trial — lives in [TEST_TRIALS.md](TEST_TRIALS.md).

## Output schema (canonical `target` struct)

```matlab
target = struct( ...
    "time",         (N×1) double, simulation timegrid in seconds, monotonic, dt = 1/sample_rate, ...
    % --- PRIMARY matching anchors (mid-hands frame on the club shaft) ---
    "grip",         (N×3) double, grip / mid-hands position in metres, world frame, ...
    "grip_quat",    (N×4) double, grip / mid-hands orientation as unit quaternion [w x y z], ...
    % --- SECONDARY signals (clubhead / club orientation; non-rigid via shaft flex,
    %     and the player's club may differ in length from the modeled club) ---
    "clubhead",     (N×3) double, clubhead position in metres, world frame, ...
    "club_quat",    (N×4) double, club orientation as unit quaternion [w x y z], ...
    "impact_idx",   scalar uint32, index of impact frame (taken from documented event marker when present), ...
    "events",       struct with A_sample, T_sample, I_sample, F_sample, CHS_mph from row-1 header (Wiffle xlsx), ...
    % --- Backward-compat alias of `grip` for older callers that read `butt` ---
    "butt",         (N×3) double, == target.grip (deprecated; will be removed once all callers migrate), ...
    "source",       struct with provenance: filename, format, subject_id, trial_id, sha256 ...
);
```

### Why grip-primary?

The body→club interface is **rigid at the grip** (the player's hands holding the club).
The clubhead is a non-rigid extension because of (a) shaft flex during the swing and
(b) the player's actual club length almost never matches the modeled club length to the
millimetre. Matching on the grip position + grip orientation gives an exact (sub-mm),
club-length-independent target for the body kinematics; the modeled club's clubhead is
then a deterministic rigid extension of that grip pose. See COST_FUNCTION_SPEC.md
for the corresponding cost-term reweighting.

Python mirror is a `dataclass`:

```python
@dataclass(frozen=True)
class ClubTarget:
    time: np.ndarray            # (N,) float64
    grip: np.ndarray            # (N,3) float64  PRIMARY anchor
    grip_quat: np.ndarray       # (N,4) float64  PRIMARY anchor [w,x,y,z]
    clubhead: np.ndarray        # (N,3) float64  secondary
    club_quat: np.ndarray       # (N,4) float64  secondary  [w,x,y,z]
    impact_idx: int
    events: dict | None         # A/T/I/F samples, CHS_mph
    source: SourceProvenance
```

## Source formats supported

### 1. Excel (priority for Phase 1)

[Wiffle_ProV1_club_3D_data.xlsx](../../src/apps/golf_gui/Motion%20Capture%20Plotter/) — already parsed by [mocap_data_loader.py](../../src/apps/golf_gui/Motion%20Capture%20Plotter/mocap_data_loader.py).

- **Units in source:** **centimetres** (the Definitions tab claims "inches" but the actual values are cm; mid-hands→clubhead distance is constant 106.93 across every frame, which is 1.07 m in cm and 2.71 m if treated as inches). Frames at the file's native rate (240 Hz).
- **Conversion:** cm → metres via `× 0.01`.
- **Event markers:** row 1 of each sheet is `<trial> | A | <addr#> | T | <top#> | I | <impact#> | F | <finish#> | CHS | <mph>`. The loader reads these into `target.events` and uses the documented `I_sample` for `impact_idx` (the speed-argmax heuristic is not authoritative — it can latch onto the wrong local maximum).
- **Orientation:** the file stores 3×3 rotation matrices per frame; convert to unit quaternion with sign normalised so `q[0] >= 0` to suppress the `q ↔ -q` ambiguity at the source.
- **Sheets:** `TW_wiffle`, `TW_ProV1`, `GW_wiffle`, `GW_ProV11`. Each is one swing. See [TEST_TRIALS.md](TEST_TRIALS.md) for the canonical inventory of trials, their event markers, expected CHS values, and per-trial data-quality notes.

### 2. C3D (priority for Phase 1, validation)

[Data/Mocap C3D Files/](../../Data/Mocap%20C3D%20Files/). One file is known to exist; **it has not been verified to parse.** Issue #013 is the validation pass.

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

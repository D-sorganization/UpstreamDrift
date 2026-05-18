# feat(motion-matching): `.mat` club-target loader (TW/GW × ProV1/Wiffle)

## Why

The motion-matching dispatcher `src/shared/python/motion_matching/load_club_target.py` currently routes `.xlsx` → `loaders/excel.py` and `.c3d` → `loaders/c3d.py`. We also have eight `.mat` files (`TW_ProV1.mat`, `TW_wiffle.mat`, `GW_ProV1.mat`, `GW_wiffle.mat`, plus `_targetKinematics.mat` siblings) under `src/engines/physics_engines/pinocchio/data/<dataset>/` that encode the same swings as the xlsx but with a cleaner schema:

```
data.time            (Nx1)         seconds
data.midhands_xyz    (Nx3)         metres
data.midhands_dircos (Nx3x3)       rotation
data.clubface_xyz    (Nx3)         metres
data.clubface_dircos (Nx3x3)       rotation
params.Address       int           1-based frame index
params.TopOfBackswing int          1-based frame index
params.Impact        int           1-based frame index
params.Finish        int           1-based frame index
params.impact_frame  int           same as Impact
```

These files give us a `ClubTarget` with **stamped impact** rather than a kinematic-peak heuristic, AND with **real 3-DOF clubface orientation** — both improvements over the xlsx path. They should be a first-class loader.

## What to build

`src/shared/python/motion_matching/loaders/matlab_dataset.py`:

```python
def load_club_target_mat(path: Path | str, opts: AlignOptions) -> ClubTarget: ...
```

### Behaviour

1. Open with `scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)`.
2. Read `data.time`, the four kinematic arrays, and `params.Impact` (1-based → 0-based).
3. Convert `data.midhands_xyz` → `butt`, `data.clubface_xyz` → `clubhead`. (Verify and document the convention; the matcher already uses "Mid-hands" and "Center of club face" — these correspond to butt-end and head respectively for our purposes.)
4. Build per-frame quaternion from `data.clubface_dircos` (rotation matrix → quaternion via existing `loaders/_quaternion.py: rotmat_to_quat`).
5. Sanity-check rotation matrices: `det ≈ +1`, columns orthonormal within 1e-3. Reject otherwise with a clear `ValueError`.
6. Resample onto the simulation grid using the existing `loaders/_align.py: resample_target`. Pass the stamped impact frame as `impact_raw` instead of running `detect_impact_index`.
7. Build `SourceProvenance(format="mat_dataset", filename=path.name, subject_id=path.stem.split("_")[0], trial_id=path.stem, sha256=...)`.

### Dispatcher wiring

Update `load_club_target` in `load_club_target.py`:

- Add `_MAT_SUFFIXES = frozenset({".mat"})` and route accordingly.
- `load_club_target_mat` becomes part of the `__all__` re-exports.

## Generic naming

Module file is `matlab_dataset.py` — the `.mat` extension is the MATLAB binary format and is fine to name. Function name `load_club_target_mat`. Do not embed any subject initials, lab name, or vendor in the module-level code. The only `*_data/*` directory references are kept in tests/fixtures and use generic dir names per the rename issue.

## Acceptance criteria

- [ ] `load_club_target_mat(path, opts)` returns a validated `ClubTarget`.
- [ ] `load_club_target(path)` routes `.mat` to the new loader (with the same `opts`/`sheet=` signature shape as today; `sheet=` is silently ignored for `.mat`).
- [ ] Stamped impact (from `params.Impact`) is preserved end-to-end and matches the resampled `impact_idx` to within 1 sample at 1 kHz.
- [ ] Rotation-matrix validity is enforced with a precise error message.
- [ ] DbC pre/postconditions on the public function (path exists, opts.sample_rate_hz > 0, return type is `ClubTarget`).
- [ ] Mypy + ruff clean.

## Tests

`tests/unit/motion_matching/test_loaders_matlab.py`:

- All four canonical files (`TW_ProV1`, `TW_wiffle`, `GW_ProV1`, `GW_wiffle`) load to a `ClubTarget` whose impact-time clubhead speed is in `[35, 55] m/s` (driver) or `[30, 45] m/s` (iron — `GW` files).
- Stamped vs. heuristic impact: assert `impact_idx_from_mat == impact_idx_from_xlsx ± 2` for the same trial loaded both ways.
- Reflect-rejection test: synthetic `dircos` with `det = -1` raises with a message mentioning "rotation".
- Bad path: nonexistent file → `FileNotFoundError`.

## Files touched

- New: `src/shared/python/motion_matching/loaders/matlab_dataset.py`
- Edit: `src/shared/python/motion_matching/load_club_target.py` (dispatch)
- Edit: `src/shared/python/motion_matching/__init__.py`
- Edit: `src/shared/python/motion_matching/loaders/__init__.py`
- New: `tests/unit/motion_matching/test_loaders_matlab.py`

## References

- Probe of the eight files (already verified):
  - `data` struct fields: `time, midhands_xyz, midhands_dircos, clubface_xyz, clubface_dircos`
  - `params` struct fields: `impact_frame, backswing_start, swing_start, Address, TopOfBackswing, Impact, Finish`
- Sister xlsx loader: `src/shared/python/motion_matching/loaders/excel.py`
- Quat utility: `src/shared/python/motion_matching/loaders/_quaternion.py`

# test(motion-matching): integration tests + golden fixtures for body / club / ball pipeline

## Why

The new `BodyTarget`, `ClubBallTarget`, `.mat` loader, body-segments helper, and animated preview each ship with focused unit tests. We also need **end-to-end integration tests** that stitch the whole pipeline together against the four canonical C3D files and the eight `.mat` files in the repo, plus a small set of **golden snapshots** so accidental regressions show up as diff failures rather than silent kinematic drift.

## What to build

### A. End-to-end integration test

`tests/integration/test_multi_source_pipeline.py` (marker `integration`):

For each of the eight `.mat` files and the four `.c3d` files:

- Load the matching `ClubTarget` (or `ClubBallTarget`) and, where the C3D body markers apply (the four C3D files), the `BodyTarget` sharing the same timegrid.
- Validate: 1 kHz × 0.300 s grid (301 samples) by default, `impact_idx == 251`, `time[0] == 0`.
- Sanity-check kinematic ranges against the verified-by-hand baseline:
  | File | Expected impact clubhead speed (m/s) | Tol |
  |---|---|---|
  | `data/C3D_TA_Driver.c3d` | 51.0 | ±0.5 |
  | `data/C3D_TA_Iron.c3d` | 39.6 | ±0.5 |
  | `TW_ProV1.mat` | computed at first run, pinned thereafter | ±0.5 |
  | `TW_wiffle.mat` | computed at first run, pinned thereafter | ±0.5 |
  | `GW_ProV1.mat` | computed at first run, pinned thereafter | ±0.5 |
  | `GW_wiffle.mat` | computed at first run, pinned thereafter | ±0.5 |
- Verify quaternion unit-norm to `1e-6` for every frame.

### B. Golden-snapshot tests

Add a small `tests/fixtures/motion_matching/` directory with **JSON snapshots** (not the raw .npy — text-diffable) of:

- Last 10 frames of `time`, `butt`, `clubhead`, `club_quat` for the driver C3D.
- Last 10 frames of `time`, marker_xyz` for a stable subset (Mid-hands/clubface/HeadTop/WaistLeft/WaistRight).

Tolerance: 1e-6 absolute, 1e-9 relative; rounding to 1e-9 in the snapshot file. Use `numpy.testing.assert_allclose`.

A regenerator command: `python -m tests.fixtures.motion_matching.regenerate` — bumps every snapshot and prints a diff summary. Document the regenerate procedure in the test docstring; never auto-regenerate in CI.

### C. Headless GUI smoke test

`tests/ui/test_motion_target_preview_headless.py`:

- `QT_QPA_PLATFORM=offscreen` plus matplotlib `Agg`.
- Open the matcher, load `data/C3D_TA_Driver.c3d` for both club and body, scrub the timeline from 0 → end, assert the bottom-right artist count matches the layer-visibility settings.

## Generic naming

Test module names, fixture filenames, fixture key names — generic. The C3D file paths inside the test reference the files by their repo-relative paths (which keep their existing names per the rename issue's scope rule).

## Acceptance criteria

- [ ] Integration test passes locally and in CI on Linux + Windows.
- [ ] Golden snapshots committed and stable (zero diff on rerun).
- [ ] Snapshot regenerator command documented and idempotent.
- [ ] Headless GUI smoke test runs in <30 s.
- [ ] CI matrix marker `integration` triggers these tests; `not slow and not live_simulation` default still skips them when the developer runs the fast suite.

## Files touched

- New: `tests/integration/test_multi_source_pipeline.py`
- New: `tests/fixtures/motion_matching/*.json`
- New: `tests/fixtures/motion_matching/regenerate.py`
- New: `tests/ui/test_motion_target_preview_headless.py`
- Edit: `pyproject.toml` (only if any new pytest marker needed — none expected; `integration` already registered).

## Sequencing

Lands after the loaders (BodyTarget, body C3D, `.mat`, ClubBallTarget, body skeleton) and the matcher animated preview / source-toggle work, since it tests the whole stack together.

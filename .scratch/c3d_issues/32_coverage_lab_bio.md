# test(upstream_drift_tools/lab/bio): bring C3D reader internals to ≥90% coverage

## Goal

Raise coverage of `src/shared/python/upstream_drift_tools/lab/bio/` (5 files: `c3d_reader.py`, `_c3d_io.py`, `_c3d_models.py`, `_c3d_markers.py`, `_c3d_analog.py`) to **≥90% line, ≥80% branch**.

Higher bar than other areas because this is the foundational C3D reader — every loader and viewer depends on it.

## Current state

`tests/unit/upstream_drift_tools/lab/bio/test_c3d_markers.py` exists (3 tests from PR #4616 covering the cp1252 + heuristic fix). Other modules have spot coverage from `tests/heavy_integration/test_c3d_data_pipeline.py` and `tests/integration/test_c3d_workflow.py`.

Likely uncovered surface:
- `c3d_reader.py:C3DDataReader` — `export_points` / `export_analog` JSON / NPZ paths, `force_plate_dataframe` with synthetic plate data, `points_dataframe` with `markers` filter / `residual_nan_threshold`
- `_c3d_io.py` — `validate_export_path`, `unit_scale` (every unit pair), `sanitize_for_csv` (every prefix), `write_export` per format, `build_metadata` corner cases (missing analog, missing events)
- `_c3d_analog.py` — `detect_force_plate_channels` (each pattern: plain, prefixed, Vicon-style), `build_force_plate_dataframe` with explicit plate filter, COP computation with synthetic forces
- `_c3d_markers.py` — `validate_marker_positions` (already 3 tests; add: max-position, all-NaN, partial NaN), `build_points_dataframe` (with explicit markers filter, residual threshold, target_units conversion)
- `_c3d_models.py` — `C3DMetadata` / `C3DEvent` dataclass validation, `SCHEMA_VERSION` invariants

## Process

For each module, write a `tests/unit/upstream_drift_tools/lab/bio/test_<module>.py`. Use synthetic ezc3d-shaped dicts (no real C3D file required for unit tests; the real file is exercised in the existing integration tests).

Build a small helper `_synthetic_c3d_dict(n_frames, n_markers, marker_names, n_analog=0)` returning the dict shape ezc3d's `c3d()` produces. This lets us test edge cases (missing groups, empty analog, malformed labels) without committing test fixtures.

## Acceptance

- [ ] `pytest tests/unit/upstream_drift_tools/lab/bio/ --cov=src/shared/python/upstream_drift_tools/lab/bio --cov-report=term-missing --cov-branch` reports **≥90% line, ≥80% branch**.
- [ ] No production changes.
- [ ] PR body lists per-file delta.
- [ ] mypy + ruff + file-size budget clean.

## Files touched

- New: `tests/unit/upstream_drift_tools/lab/bio/test_c3d_reader.py`
- New: `tests/unit/upstream_drift_tools/lab/bio/test_c3d_io.py`
- New: `tests/unit/upstream_drift_tools/lab/bio/test_c3d_analog.py`
- Extend: `tests/unit/upstream_drift_tools/lab/bio/test_c3d_markers.py`
- New: `tests/unit/upstream_drift_tools/lab/bio/test_c3d_models.py`

# feat(motion-matching): C3D body-marker loader producing `BodyTarget`

Depends on the `BodyTarget` dataclass issue.

## Why

Every C3D file in the repo contains the full set of anatomical markers needed to drive a body skeleton, but `src/shared/python/motion_matching/loaders/c3d.py` currently discards them — it only emits a `ClubTarget`. We need a sibling loader that pulls the body markers out, fills short occlusion gaps, converts coordinate frames, resamples onto the simulation grid, and returns a validated `BodyTarget`.

## What to build

Add `src/shared/python/motion_matching/loaders/c3d_body.py`:

```python
def load_body_target_c3d(
    path: Path | str,
    opts: AlignOptions,
    *,
    marker_set: Sequence[str] | None = None,
    impact_source: ClubTarget | None = None,
) -> BodyTarget: ...
```

### Behaviour

1. **Parse**: reuse the canonical `C3DDataReader` from `src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py`. (Do NOT use the legacy duplicate at `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py` — that path is being deprecated by a separate issue.)
2. **Marker selection**: when `marker_set` is `None`, default to the canonical anatomical-marker subset (28 Plug-in-Gait markers verified to exist in our four reference files). Exclude sentinels and known-occluded markers via a configurable exclusion list (see the existing `EXCLUDED_MARKERS` set in `loaders/_optical.py` for the pattern, but rename it source-neutrally per the rename issue).
3. **Gap-fill**: use the same NaN-spline-fill that `loaders/_optical.py` already provides (`fill_short_gaps`, ≤5 frames default). Mark longer gaps as NaN — do NOT extrapolate.
4. **Coordinate frame**: source data is metres, Y-up. Apply the existing `y_up_to_z_up` swap so the result is right-handed Z-up (consistent with `ClubTarget`).
5. **Time alignment + resample**:
   - If `impact_source` is provided, use its `impact_idx` and `time` grid (so body and club share a clock).
   - Otherwise, detect impact via the same kinematic peak heuristic the club loader uses, applied to the wrist or clubhead-cluster centroid. Document the chosen heuristic in the docstring.
   - Resample marker trajectories onto `opts.simulation_time_s` / `opts.sample_rate_hz` using cubic interpolation per coordinate, NaN-preserving (do not interpolate across long gaps).
6. **Events**: extract any C3D event annotations (`EVENT.LABELS` / `EVENT.TIMES`) and convert to `BodyEvent` instances with frame indices on the resampled grid. Our four reference files have no events; loader must handle the empty case gracefully.
7. **Provenance**: populate `SourceProvenance(format="c3d", ...)` with the file's basename, sha256, and stem-derived subject/trial ids.

### Public API

Add a top-level `load_body_target` dispatcher in `src/shared/python/motion_matching/load_body_target.py` mirroring `load_club_target.py`. Currently it only routes `.c3d`, but the dispatch shape is in place for future formats.

Re-export both from `src/shared/python/motion_matching/__init__.py`.

## Generic naming

Module name `c3d_body.py`. Function names `load_body_target_c3d` / `load_body_target`. No reference to a specific lab, vendor, or study in code, docstrings, error messages, log messages, or test names. Source-specific marker-set helpers (e.g. the existing 28-marker Plug-in-Gait subset) must live behind generic accessors like `default_anatomical_marker_set()` — never `optical_marker_set()`.

## Acceptance criteria

- [ ] `load_body_target_c3d(path, opts, *, marker_set=None, impact_source=None)` returns a validated `BodyTarget`.
- [ ] `load_body_target(path, *, opts=None)` dispatcher routes by file extension.
- [ ] Works on all four C3D files in the repo with default settings — verified by integration tests.
- [ ] When `impact_source` is provided, body and club targets share a timegrid (same `time[0]`, `time[-1]`, length).
- [ ] All public functions carry preconditions/postconditions per CLAUDE.md DbC standard (use the existing `precondition`/`postcondition` decorators from `src.shared.python.core.contracts`).
- [ ] Logger (not print) reports: marker count loaded, occlusion-gap stats, sample count after resample, impact frame.
- [ ] Mypy + ruff clean, file-size budget respected.

## Tests

`tests/unit/motion_matching/test_load_body_target_c3d.py`:
- happy path on `data/C3D_TA_Driver.c3d`: 28 markers, sample count matches `opts`, impact_idx in expected window.
- occlusion handling: `RShoulderTop` (~14% coverage) is not silently filled; remains mostly NaN.
- shared-clock test: load body + club from same file with same `AlignOptions`; body's `time` == club's `time` exactly when `impact_source=club_target`.
- bad path: nonexistent file → `FileNotFoundError`.
- explicit `marker_set` honoured (subset returned in given order).

`tests/integration/test_c3d_body_pipeline.py` (heavy; marker `slow`):
- All four C3D files load, validate, return the expected marker set.

## Files touched

- New: `src/shared/python/motion_matching/loaders/c3d_body.py`
- New: `src/shared/python/motion_matching/load_body_target.py`
- Edit: `src/shared/python/motion_matching/__init__.py`
- Edit: `src/shared/python/motion_matching/loaders/__init__.py`
- New: `tests/unit/motion_matching/test_load_body_target_c3d.py`
- New: `tests/integration/test_c3d_body_pipeline.py`

## References

- Parent contract: `BodyTarget` issue (must merge first).
- Existing club loader: `src/shared/python/motion_matching/loaders/c3d.py`
- Cluster-pose helpers: `src/shared/python/motion_matching/loaders/_optical.py` (renamed by separate issue) — re-use `fill_short_gaps`, `y_up_to_z_up` after the rename.
- C3D parameter probe (verified): 38 markers, 360 Hz, units metres, no analog, no events.

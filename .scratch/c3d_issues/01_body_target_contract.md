# feat(motion-matching): introduce `BodyTarget` canonical contract for full-body marker trajectories

## Tracking issue

This is the first of a 12-issue effort to bring full-body C3D mocap into the canonical motion-matching pipeline. See the tracking issue (filed alongside this one as the "Multi-source motion targets" tracker) for the full plan.

## Why

The motion-matching pipeline today exposes only `ClubTarget` (`src/shared/python/motion_matching/club_target.py`) — a 6-DOF club trajectory (butt + clubhead + quaternion). The four C3D files in the repo also carry **28 anatomical body markers** (Plug-in-Gait subset: pelvis/back/head/shoulder/elbow/wrist/knee/ankle/toe + occluded `RShoulderTop`), but the existing C3D loader (`src/shared/python/motion_matching/loaders/c3d.py`) discards them and only returns club state derived from the two 3-marker rigid clusters on the club.

To match a full-body skeleton model against measured swing data we need a **first-class, validated, immutable target dataclass** for the body markers — parallel to `ClubTarget` — that downstream cost functions, visualisers, and exporters can rely on without each one re-parsing the raw C3D.

## What to build

Add a new module `src/shared/python/motion_matching/body_target.py` that exposes:

```python
@dataclass(frozen=True)
class BodyTarget:
    """Canonical full-body marker trajectory.

    Validated at construction; any violation of the validation rules raises
    ``ValueError``.  Frozen so loaders are forced to produce a fully-formed,
    validated artifact rather than mutating one in place.
    """
    time:         np.ndarray  # (N,) seconds, strictly increasing, time[0] == 0
    marker_xyz:   np.ndarray  # (N, M, 3) metres, NaN allowed for occluded
    marker_names: tuple[str, ...]  # length M; canonical Plug-in-Gait names
    impact_idx:   int         # frame index of impact on the resampled grid
    events:       tuple[BodyEvent, ...]   # named events with frame indices
    source:       SourceProvenance        # reuse existing dataclass
    coordinate_frame: Literal["z_up_right_handed"] = "z_up_right_handed"
```

with a companion `BodyEvent(label: str, frame: int, time_s: float)` dataclass.

### Validation rules (`__post_init__`)

| Rule | Failure |
|---|---|
| `time` is 1-D, length `N >= 2`, strictly increasing, `time[0] == 0` (within `1e-9`) | ValueError |
| `marker_xyz.shape == (N, M, 3)`, M == `len(marker_names)`, M >= 3 | ValueError |
| `marker_names` are strings, unique, non-empty | ValueError |
| Per-marker `|xyz|` < `MAX_BODY_POSITION_NORM_M = 3.0` (sanity bound for human-scale motion) for finite samples | ValueError |
| At least one frame is fully finite for at least 50% of markers (loader is responsible for cropping leading/trailing all-NaN windows) | ValueError |
| `0 <= impact_idx < N` | ValueError |
| `events` frame indices in `[0, N)`; labels non-empty unique | ValueError |
| `source` is a `SourceProvenance` instance | TypeError |

Reuse `SourceProvenance` from `club_target.py`. Add `MAX_BODY_POSITION_NORM_M = 3.0` and a `BODY_TARGET_SCHEMA_VERSION = 1` constant.

### Re-exports

* `target.py` — re-export `BodyTarget`, `BodyEvent`, `MAX_BODY_POSITION_NORM_M`, `BODY_TARGET_SCHEMA_VERSION` so external callers can keep using the `from src.shared.python.motion_matching import target` entry point.
* `__init__.py` — extend `__all__` with the new symbols.

## Generic naming requirement

This dataclass and its validation must be **source-agnostic**. The module/class names, docstrings, and error messages must not reference any specific motion-capture vendor, study, lab, or person. The only hint at provenance lives in the `SourceProvenance.format` field (free-form string set by the loader, e.g. `"c3d"`).

## Acceptance criteria

- [ ] `BodyTarget` dataclass added in `src/shared/python/motion_matching/body_target.py`, frozen, validated in `__post_init__`.
- [ ] `BodyEvent` companion dataclass added.
- [ ] Validation constants + `BODY_TARGET_SCHEMA_VERSION` exposed.
- [ ] Re-exported via `target.py` and `__init__.py`.
- [ ] Unit tests in `tests/unit/motion_matching/test_body_target.py` covering every validation rule (one happy path + one failure case per rule). Use `pytest.raises(ValueError, match=...)` so the messages are pinned.
- [ ] Mypy clean (existing mypy gate must continue to pass).
- [ ] No print() in src/, no TODO/FIXME without a tracked issue.
- [ ] File-size budget respected (< 1200 lines per file).

## Out of scope

- Loader implementation — covered by the C3D body loader issue.
- Skeleton segment connectivity (head→neck→spine etc.) — covered by the body-skeleton segments issue.
- UI integration — covered by the matcher animated preview issue.

## Files touched

- New: `src/shared/python/motion_matching/body_target.py`
- New: `tests/unit/motion_matching/test_body_target.py`
- Edit: `src/shared/python/motion_matching/target.py` (re-exports)
- Edit: `src/shared/python/motion_matching/__init__.py` (re-exports)

## References

- Existing parallel: `src/shared/python/motion_matching/club_target.py`
- C3D files in repo: `data/C3D_TA_*.c3d`, `src/engines/physics_engines/pinocchio/data/optical_tour_average/CaptureExport*.c3d` (4 files, 38 markers each, 360 Hz, ~1.8 s)
- Marker names verified by `ezc3d.c3d()` probing; see the parent investigation report.

# feat(motion-matching): `ClubBallTarget` — club kinematics + ball impact boundary condition

## Why

The user wants the motion-matching tool to support two club-data modes:

1. **Club only** — current behaviour, using `ClubTarget`.
2. **Club + ball** — adds a ball boundary condition: ball position at impact, launch direction, ball speed.

Today there is no canonical container for the ball state, and no loader extracts it. The ball is treated implicitly (via a stamped impact frame) but motion matching can produce visibly better fits when the ball boundary state is part of the cost (e.g. ensuring the clubface aligns with launch direction at impact).

## What to build

`src/shared/python/motion_matching/club_ball_target.py`:

```python
@dataclass(frozen=True)
class BallImpactState:
    """Ball state at the moment of impact (boundary condition)."""
    position_at_impact_m: np.ndarray    # (3,) world-frame Z-up metres
    launch_direction:    np.ndarray     # (3,) unit vector; NaN if unknown
    launch_speed_mps:    float          # NaN if unknown
    spin_rpm:            float          # NaN if unknown (placeholder)


@dataclass(frozen=True)
class ClubBallTarget:
    """Composite target combining a `ClubTarget` with a `BallImpactState`."""
    club:        ClubTarget
    ball_impact: BallImpactState

    # delegates for convenience
    @property
    def time(self) -> np.ndarray: return self.club.time
    @property
    def impact_idx(self) -> int:  return self.club.impact_idx
```

### Validation

`ClubBallTarget.__post_init__`:

- `club` is a `ClubTarget`.
- `ball_impact` is a `BallImpactState`.
- `ball_impact.position_at_impact_m` shape `(3,)`, all finite, `|r| < MAX_POSITION_NORM_M`.
- `ball_impact.launch_direction` shape `(3,)`; if any component is NaN, the whole vector must be NaN (no partial unknowns); when finite, unit-norm to `1e-6`.
- `ball_impact.launch_speed_mps` is finite-or-NaN; if finite, in `[0, 100]`.
- `ball_impact.spin_rpm` is finite-or-NaN; if finite, in `[0, 15000]`.

### Default ball-impact extraction

For our existing club datasets we don't have a Trackman/launch-monitor feed. The default ball-impact extractor must therefore approximate from the club state:

`extract_ball_impact_from_clubtarget(target: ClubTarget) -> BallImpactState`

- `position_at_impact_m` := `target.clubhead[target.impact_idx]` (within ball-radius, fine for visualisation; flag in docstring this is an approximation).
- `launch_direction` := unit vector of clubhead velocity at impact (numerical gradient on the resampled grid).
- `launch_speed_mps` := `|v_clubhead at impact|` × elasticity factor `e=1.5` clamped to 100 (documented as a stand-in pending real launch-monitor data).
- `spin_rpm` := `NaN`.

A separate loader path can later inject real launch-monitor data via a sibling function.

## Generic naming

Module: `club_ball_target.py`. No reference to specific launch monitors, brand names, ball makes, or studies.

## Acceptance criteria

- [ ] `BallImpactState` and `ClubBallTarget` dataclasses, frozen, validated.
- [ ] `extract_ball_impact_from_clubtarget` extractor with documented approximation behaviour.
- [ ] Re-exported from `target.py` and `__init__.py`.
- [ ] Unit tests pin every validation rule (one happy + one fail per rule).
- [ ] Mypy + ruff clean, file size budget respected.

## Out of scope

- Real launch-monitor data ingest (Trackman / FlightScope / GCQuad): future issue once we have data files in the repo.
- Cost-function term that uses the ball boundary: future issue under the cost module.

## Files touched

- New: `src/shared/python/motion_matching/club_ball_target.py`
- Edit: `src/shared/python/motion_matching/target.py`
- Edit: `src/shared/python/motion_matching/__init__.py`
- New: `tests/unit/motion_matching/test_club_ball_target.py`

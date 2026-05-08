# feat(pinocchio): adapt motion-matching to `MultiSourceTarget`; bring forward existing `club_target_adapter`

Per-engine child of cross-engine parity (#4513). Depends on #4514 (canonical fit_swing API).

## Why

Pinocchio's motion-matching is the most mature path — `club_target_adapter.py` already consumes `ClubTarget`, the surrogate model exists, and several CI tests live around it. Wrap it in the new `FitSwingProvider` contract for symmetry with the other engines.

## What to build

`src/engines/physics_engines/pinocchio/python/motion_matching/provider.py`:

```python
class PinocchioFitSwingProvider:
    engine_name = "pinocchio"

    def fit_swing(self, target, opts: FitOptions) -> FitResult: ...
    def supports_body_target(self) -> bool: return False
    def supports_ball_target(self) -> bool: return False
```

Implementation thinly wraps the existing `club_target_adapter.fit_*` callables. Reuse `simulate_with_coefficients` (#4118) and the existing CVAE/regressor where present.

## Acceptance criteria

- [ ] Provider registered at import time.
- [ ] Reproduces existing Pinocchio motion-matching numbers (within 1% impact-clubhead-speed RMSE) on `TW_ProV1.mat` — pin numerical regression test.
- [ ] Mypy + ruff + file-size budget clean.

## Files touched

- New: `src/engines/physics_engines/pinocchio/python/motion_matching/provider.py`
- Edit: `.../pinocchio/python/motion_matching/__init__.py`
- New: `tests/unit/motion_matching/pinocchio/test_provider.py`

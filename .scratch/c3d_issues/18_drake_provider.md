# feat(drake): adapt motion-matching to `MultiSourceTarget`; expose `fit_swing` provider

Per-engine child of cross-engine parity (#4513). Depends on #4514 (canonical fit_swing API).

## Why

Drake's `motion_matching/` package has working forward simulation but its motion-matching surface predates `MultiSourceTarget`. Bring it forward.

## What to build

`src/engines/physics_engines/drake/python/motion_matching/provider.py`:

```python
class DrakeFitSwingProvider:
    engine_name = "drake"

    def fit_swing(self, target, opts: FitOptions) -> FitResult: ...
    def supports_body_target(self) -> bool: return False  # initial pass
    def supports_ball_target(self) -> bool: return False
```

Initial implementation:

1. Accept `MultiSourceTarget`; pull out `target.club` for the club fit.
2. Optimiser: keep the existing scipy.optimize / pydrake gradient path; just rewire input/output.
3. Return `FitResult` with `theta` `(N, n_joints)`, `simulated_clubhead`, `simulated_butt`, basic RMSE metrics.
4. Register the provider at module-import time.

A second pass can later wire body-target cost terms once #4520 (cost terms) lands.

## Acceptance criteria

- [ ] `DrakeFitSwingProvider` registered automatically when `src.engines.physics_engines.drake.python.motion_matching` imports.
- [ ] `fit_swing(target, opts)` returns a valid `FitResult` for the four `.mat` and four `.c3d` reference files when `target.club` is present.
- [ ] Reproduces existing Drake motion-matching numbers (within 1% impact-clubhead-speed RMSE) on the canonical Wiffle ProV1 trace — pin a numerical regression test.
- [ ] Mypy + ruff + file-size budget clean.

## Files touched

- New: `src/engines/physics_engines/drake/python/motion_matching/provider.py`
- Edit: `src/engines/physics_engines/drake/python/motion_matching/__init__.py` (registration)
- Edit: any existing `fit_swing.py` in the drake tree to use the new types or be deprecated
- New: `tests/unit/motion_matching/drake/test_provider.py`

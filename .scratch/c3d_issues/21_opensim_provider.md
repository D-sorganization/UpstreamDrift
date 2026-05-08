# feat(opensim): adapt motion-matching to `MultiSourceTarget`; refresh prescribed-controller path

Per-engine child of cross-engine parity (#4513). Depends on #4514 (canonical fit_swing API).

## Why

OpenSim's motion-matching uses a prescribed-controller path that's seen multiple recent fixes (#4322, #4327, etc.). The provider needs to land on the new contract while keeping the controller behaviour stable.

## What to build

`src/engines/physics_engines/opensim/python/motion_matching/provider.py`:

```python
class OpenSimFitSwingProvider:
    engine_name = "opensim"

    def fit_swing(self, target, opts: FitOptions) -> FitResult: ...
    def supports_body_target(self) -> bool: return False
    def supports_ball_target(self) -> bool: return False
```

Wraps the existing prescribed-controller fit pipeline. Convert `MultiSourceTarget`/`ClubTarget` -> existing inputs, convert outputs -> `FitResult`.

## Acceptance criteria

- [ ] Provider registered at import time.
- [ ] No regression in the prescribed-controller fixes pinned by recent issues — existing OpenSim tests in `tests/unit/engines/opensim/` continue to pass.
- [ ] Mypy + ruff + file-size budget clean.

## Files touched

- New: `src/engines/physics_engines/opensim/python/motion_matching/provider.py`
- Edit: `.../opensim/python/motion_matching/__init__.py`
- New: `tests/unit/motion_matching/opensim/test_provider.py`

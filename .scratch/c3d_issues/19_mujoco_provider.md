# feat(mujoco): adapt motion-matching to `MultiSourceTarget`; expose `fit_swing` provider

Per-engine child of cross-engine parity (#4513). Depends on #4514 (canonical fit_swing API).

## Why

MuJoCo's motion-matching has the most recent improvements (recent `fit_swing.py` and `synthesize.py` work, including the einsum optimisations). Bring it under the new `FitSwingProvider` contract so the matcher can drive it through the same registry as every other engine.

## What to build

`src/engines/physics_engines/mujoco/python/motion_matching/provider.py`:

```python
class MujocoFitSwingProvider:
    engine_name = "mujoco"

    def fit_swing(self, target, opts: FitOptions) -> FitResult: ...
    def supports_body_target(self) -> bool: return False
    def supports_ball_target(self) -> bool: return False
```

Wire it on top of the existing `fit_swing.py` and `synthesize.py` in the mujoco tree; the new provider is a thin adapter that converts `MultiSourceTarget`/`ClubTarget` -> the engine's existing input shape and converts the engine's output -> `FitResult`.

## Acceptance criteria

- [ ] Provider registered at import time.
- [ ] Accepts `MultiSourceTarget` and `ClubTarget`; reads from `target.club` for the swing.
- [ ] Returns a valid `FitResult` for the four `.mat` and four `.c3d` reference files when `target.club` is present.
- [ ] Reproduces existing MuJoCo motion-matching numbers (within 1% impact-clubhead-speed RMSE) — pin numerical regression test.
- [ ] Mypy + ruff + file-size budget clean.

## Files touched

- New: `src/engines/physics_engines/mujoco/python/motion_matching/provider.py`
- Edit: `.../mujoco/python/motion_matching/__init__.py`
- New: `tests/unit/motion_matching/mujoco/test_provider.py`

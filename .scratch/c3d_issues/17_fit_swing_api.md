# feat(motion-matching): canonical engine-side `fit_swing(target, opts) -> FitResult` API + provider registry

Foundation issue for the cross-engine parity tracker (#4513).

## Why

Each physics engine currently has its own ad-hoc motion-matching entry point. Drake, MuJoCo, Pinocchio, OpenSim, MyoSim, and Simscape all have a `motion_matching/` directory with overlapping but inconsistent APIs. To deliver a unified user experience (one matcher view, one leaderboard, one set of diagnostics), every engine needs to expose the **same** `fit_swing` callable. This issue defines that contract and ships the registry that lets the matcher discover engines at runtime.

## What to build

`src/shared/python/motion_matching/fit_swing.py`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class FitSwingProvider(Protocol):
    """Engine-side motion-matching entry point."""

    engine_name: str   # "drake" | "mujoco" | "pinocchio" | "opensim" | ...

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget | ClubBallTarget,
        opts: FitOptions,
    ) -> FitResult: ...

    def supports_body_target(self) -> bool: ...
    def supports_ball_target(self) -> bool: ...
```

`FitOptions` extends `AlignOptions` with optimisation knobs (`max_iters`, `tol`, `seed`, `regulariser`, `cost_terms` enum set, `initial_theta`).

`FitResult` (frozen dataclass) carries:

- `theta` — fitted joint-angle trajectory `(N, n_joints)`
- `target` — the input target (so consumers can re-render)
- `simulated_clubhead` `(N, 3)` — engine-rendered clubhead trace
- `simulated_butt` `(N, 3)` — engine-rendered mid-hands trace
- `cost_breakdown` — `dict[str, np.ndarray]` per-frame cost terms
- `metrics` — `FitMetrics` with summary RMSE / max-error / time-of-impact-error
- `engine_name`, `engine_version`, `wall_time_s`, `n_iters`, `converged: bool`

`src/shared/python/motion_matching/provider_registry.py`:

```python
def register_provider(provider: FitSwingProvider) -> None: ...
def get_provider(engine_name: str) -> FitSwingProvider: ...
def available_engines() -> list[str]: ...
```

Auto-discovery via entry points: each engine module's `__init__.py` calls `register_provider(MyEngineProvider())` at import time. Discovery walks `src.engines.physics_engines.*.python.motion_matching` packages.

## Acceptance criteria

- [ ] `FitSwingProvider`, `FitOptions`, `FitResult`, `FitMetrics` defined in `src/shared/python/motion_matching/fit_swing.py`.
- [ ] Provider registry with `register_provider`, `get_provider`, `available_engines`.
- [ ] All four dataclasses frozen + validated in `__post_init__`.
- [ ] Mypy clean (Protocol with `runtime_checkable`).
- [ ] Unit tests: synthetic provider that returns canned `FitResult`; registry round-trip; idempotent registration.
- [ ] Re-exported via `src/shared/python/motion_matching/__init__.py`.
- [ ] No print, no TODO without an issue, file-size budget.

## Out of scope

- Per-engine adapter implementations (separate child issues B–F).
- Cost-function terms (separate issue J).

## Files touched

- New: `src/shared/python/motion_matching/fit_swing.py`
- New: `src/shared/python/motion_matching/provider_registry.py`
- Edit: `src/shared/python/motion_matching/__init__.py`
- New: `tests/unit/motion_matching/test_fit_swing_api.py`
- New: `tests/unit/motion_matching/test_provider_registry.py`

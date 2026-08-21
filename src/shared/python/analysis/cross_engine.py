"""Cross-engine robustness comparison service (issue #7455).

UI-independent compute path for perturbation-based cross-engine comparison.
This module is the single home of the logic previously embedded in
``src/launchers/cross_engine_dashboard.py`` and is consumed by:

- the PyQt6 dashboard (GUI and ``--no-gui`` CLI), and
- the web API route ``POST /analysis/cross-engine``.

Design by Contract
------------------
- ``run_comparison_with_results`` requires a non-empty engine-name list.
- ``run_cross_engine_study`` additionally requires every engine name to be
  one of :data:`ENGINE_NAMES` (the API surface must not silently fall back
  to a stub for an unknown engine).
- ``robustness_score`` post: result is clamped to [0, 1].

DRY
---
Reuses ``CrossEnginePerturbationRunner`` / ``CrossEngineSimConfig`` from
``pendulum_simulator.cross_engine_perturbation`` — no duplicated stepping
or statistics code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEnginePerturbationRunner,
    CrossEngineRunResult,
    CrossEngineSimConfig,
)

logger = get_logger(__name__)

#: Engine names selectable for a cross-engine comparison. ``pendulum_stub``
#: is always available; the real engines fall back to a deterministic stub
#: in the desktop dashboard but are rejected-or-real in the API study.
ENGINE_NAMES: tuple[str, ...] = ("mujoco", "drake", "pinocchio", "pendulum_stub")

#: Backend identity tags (#8817): every comparison result declares whether an
#: engine actually ran real physics or the deterministic 2-DOF stub.
BACKEND_REAL: str = "real"
BACKEND_STUB: str = "stub_2dof"

#: Cross-engine CV summary keys, in stable chart order.
CV_METRIC_KEYS: tuple[str, ...] = (
    "cv_total_energy_final",
    "cv_end_effector_speed_final",
    "cv_peak_end_effector_speed",
)

#: Per-engine metric names, in stable chart order.
METRIC_KEYS: tuple[str, ...] = (
    "total_energy_final",
    "end_effector_speed_final",
    "peak_end_effector_speed",
)


class StubEngine:
    """Minimal steppable engine for unavailable physics packages.

    Design by Contract
    ------------------
    Pre:  name must be a non-empty string
    Post: get_state() returns two 1-D arrays of equal length
    """

    def __init__(self, name: str, n_dof: int = 2) -> None:
        if not name:
            raise ValueError("Engine stub name must be non-empty")
        self._name = name
        self._n_dof = n_dof
        self._q = np.zeros(n_dof)
        self._v = np.zeros(n_dof)

    def reset(self) -> None:
        """Reset state to zero."""
        self._q = np.zeros(self._n_dof)
        self._v = np.zeros(self._n_dof)

    def set_control(self, u: np.ndarray) -> None:
        """Apply control as an impulse to velocity."""
        u_arr = np.asarray(u, dtype=float)
        n = min(len(u_arr), self._n_dof)
        self._v[:n] += u_arr[:n] * 0.01

    def step(self, dt: float | None = None) -> None:
        """Integrate with Euler plus damping."""
        effective_dt = dt if dt is not None else 0.01
        damping = 0.95
        self._q = self._q + self._v * effective_dt  # type: ignore[assignment]
        self._v = self._v * damping  # type: ignore[assignment]

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return positions and velocities."""
        return self._q.copy(), self._v.copy()


def try_build_real_engine(name: str) -> Any | None:
    """Attempt to instantiate a real physics engine by name.

    Returns the engine instance on success, or None if the package is
    unavailable.  All import errors are caught and logged as warnings.

    Parameters
    ----------
    name : str
        One of 'mujoco', 'drake', 'pinocchio'.

    Returns
    -------
    SteppableEngine instance or None
    """
    try:
        if name == "mujoco":
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (  # noqa: PLC0415
                MuJoCoPhysicsEngine,
            )

            return MuJoCoPhysicsEngine()  # type: ignore[abstract]
        if name == "drake":
            from src.engines.physics_engines.drake.python.drake_physics_engine import (  # noqa: PLC0415
                DrakePhysicsEngine,
            )

            return DrakePhysicsEngine()  # type: ignore[abstract]
        if name == "pinocchio":
            from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: PLC0415
                PinocchioPhysicsEngine,
            )

            return PinocchioPhysicsEngine()

    except (ImportError, ValueError, RuntimeError):  # noqa: BLE001
        logger.warning("Engine '%s' unavailable — will use stub", name, exc_info=False)
    return None


def build_engine(
    name: str,
    *,
    try_real: Callable[[str], Any | None] | None = None,
) -> Any:
    """Return a real engine instance or a stub if the real one is unavailable.

    Parameters
    ----------
    name : str
        Engine name.
    try_real : callable, optional
        Real-engine builder, injectable for testing. Defaults to
        :func:`try_build_real_engine`.

    Returns
    -------
    SteppableEngine instance (real or stub)
    """
    engine, _backend = build_engine_with_backend(name, try_real=try_real)
    return engine


def build_engine_with_backend(
    name: str,
    *,
    try_real: Callable[[str], Any | None] | None = None,
) -> tuple[Any, str]:
    """Build an engine and report the actual backend identity (#8817).

    Design by Contract
    ------------------
    Post: the returned backend tag is :data:`BACKEND_REAL` iff the real
    engine was instantiated; a stub fallback is always tagged
    :data:`BACKEND_STUB` so callers can disclose the substitution.

    Returns
    -------
    tuple of (engine instance, backend tag)
    """
    if name == "pendulum_stub":
        return StubEngine("pendulum_stub"), BACKEND_STUB
    builder = try_real if try_real is not None else try_build_real_engine
    real = builder(name)
    if real is not None:
        return real, BACKEND_REAL
    return StubEngine(name), BACKEND_STUB


def substituted_engines(backends: dict[str, str]) -> list[str]:
    """Return engines that silently degraded to a stub.

    ``pendulum_stub`` is requested *as* a stub, so it never counts as a
    substitution.
    """
    return [
        name
        for name, backend in backends.items()
        if backend == BACKEND_STUB and name != "pendulum_stub"
    ]


def run_comparison_with_results(
    engine_names: list[str],
    config: CrossEngineSimConfig,
) -> tuple[dict[str, CrossEngineRunResult], dict[str, float]]:
    """Execute the comparison and return per-engine results and CV summary.

    Single compute path shared by the desktop dashboard (GUI worker and
    ``--no-gui`` CLI) and the web API study.

    Parameters
    ----------
    engine_names : list of str
        Names of engines to include (at least one).
    config : CrossEngineSimConfig
        Simulation configuration.

    Returns
    -------
    tuple of (results dict keyed by engine name, CV summary dict)
    """
    results, cv_summary, _backends = run_comparison_with_provenance(
        engine_names, config
    )
    return results, cv_summary


def run_comparison_with_provenance(
    engine_names: list[str],
    config: CrossEngineSimConfig,
) -> tuple[dict[str, CrossEngineRunResult], dict[str, float], dict[str, str]]:
    """Execute the comparison and additionally report backend identity (#8817).

    Design by Contract
    ------------------
    Pre:  engine_names is non-empty
    Post: the backends dict has exactly one entry per requested engine, each
          :data:`BACKEND_REAL` or :data:`BACKEND_STUB`.

    Returns
    -------
    tuple of (results dict, CV summary dict, backends dict mapping engine
    name to backend tag)
    """
    if not engine_names:
        raise ValueError("At least one engine name must be provided")
    runner = CrossEnginePerturbationRunner(config)
    backends: dict[str, str] = {}
    for name in engine_names:
        engine, backend = build_engine_with_backend(name)
        backends[name] = backend
        runner.register_engine(name, engine)  # type: ignore[arg-type]
    n_steps = round(config.t_end / config.dt)
    base_profile = np.zeros(n_steps)
    results = runner.run_comparison(base_profile)
    cv_summary = runner.compute_cv_summary(results)
    return results, cv_summary, backends


def cv_values(cv_summary: dict[str, float]) -> list[float]:
    """Return CV values in stable chart order (see :data:`CV_METRIC_KEYS`)."""
    return [cv_summary.get(key, 0.0) for key in CV_METRIC_KEYS]


def robustness_score(values: list[float]) -> float:
    """Convert aggregate CV values into the displayed robustness score.

    Post: result is clamped to [0, 1].
    """
    mean_cv = float(np.mean(values)) if values else 0.0
    return max(0.0, min(1.0, 1.0 - mean_cv))


def _safe_cv(mean: float, std: float) -> float:
    """Coefficient of variation, 0.0 when the mean is near zero."""
    if abs(mean) < 1e-12:
        return 0.0
    return std / abs(mean)


def _engine_metric_stats(result: CrossEngineRunResult) -> dict[str, dict[str, float]]:
    """Per-metric {mean, std, cv, robustness_score} for one engine result."""
    pairs: dict[str, tuple[float, float]] = {
        "total_energy_final": (
            result.mean_total_energy_final,
            result.std_total_energy_final,
        ),
        "end_effector_speed_final": (
            result.mean_end_effector_speed_final,
            result.std_end_effector_speed_final,
        ),
        "peak_end_effector_speed": (
            result.mean_peak_end_effector_speed,
            result.std_peak_end_effector_speed,
        ),
    }
    stats: dict[str, dict[str, float]] = {}
    for metric, (mean, std) in pairs.items():
        cv = _safe_cv(mean, std)
        stats[metric] = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "robustness_score": robustness_score([cv]),
        }
    return stats


def per_engine_robustness(
    results: dict[str, CrossEngineRunResult],
) -> dict[str, float]:
    """Genuine per-engine robustness: 1 − mean per-metric CV across trials.

    Post: every value is clamped to [0, 1]; one entry per engine result.
    """
    return {
        name: robustness_score(
            [stats["cv"] for stats in _engine_metric_stats(result).values()]
        )
        for name, result in results.items()
    }


def run_cross_engine_study(
    engine_names: list[str],
    config: CrossEngineSimConfig,
    *,
    allow_stub_substitution: bool = True,
) -> dict[str, Any]:
    """Run a perturbation study and return a JSON-serialisable summary.

    Parameters
    ----------
    engine_names : list of str
        Engines to compare. Every name must be in :data:`ENGINE_NAMES`;
        the API must not silently substitute a stub for a typo.
    config : CrossEngineSimConfig
        Simulation configuration (validated on construction).
    allow_stub_substitution : bool, keyword-only
        When ``False``, refuse to degrade a requested real engine to the
        2-DOF stub — raise instead of returning stub-backed numbers
        (#8817). Defaults to ``True`` (substitution is disclosed in the
        payload either way).

    Returns
    -------
    dict with keys:
        ``engines``  — per-engine ``{backend, robustness_score,
        metrics: {name: {mean, std, cv, robustness_score}}}`` where
        ``backend`` is :data:`BACKEND_REAL` or :data:`BACKEND_STUB`
        ``cv_summary`` — cross-engine CV per metric (identical to the
        desktop ``--no-gui`` CLI output)
        ``robustness_overall`` — 1 − mean(CV), clamped to [0, 1]
        (aggregate across engines — NOT a per-engine value)
        ``stubbed_engines`` — requested real engines that silently fell
        back to the 2-DOF stub (empty when everything ran real)
        ``config`` — echo of the effective configuration

    Raises
    ------
    ValueError : if engine_names is empty, contains an unknown engine, or
        a stub substitution occurred while ``allow_stub_substitution`` is
        ``False``
    """
    unknown = [name for name in engine_names if name not in ENGINE_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown engine name(s) {unknown}; supported: {list(ENGINE_NAMES)}"
        )
    results, cv_summary, backends = run_comparison_with_provenance(engine_names, config)
    substituted = substituted_engines(backends)
    if substituted and not allow_stub_substitution:
        raise ValueError(
            f"Engine(s) {substituted} are unavailable and stub substitution "
            "is disallowed (allow_stub_substitution=False); install the "
            "engine(s) or drop them from the request"
        )
    robustness_by_engine = per_engine_robustness(results)
    overall = robustness_score(cv_values(cv_summary))
    return {
        "engines": {
            name: {
                "backend": backends[name],
                "robustness_score": robustness_by_engine[name],
                "metrics": _engine_metric_stats(result),
            }
            for name, result in results.items()
        },
        "cv_summary": dict(cv_summary),
        "robustness_overall": overall,
        "stubbed_engines": substituted,
        "config": {
            "t_end": config.t_end,
            "dt": config.dt,
            "noise_amplitude": config.noise_amplitude,
            "n_trials": config.n_trials,
            "seed": config.seed,
        },
    }

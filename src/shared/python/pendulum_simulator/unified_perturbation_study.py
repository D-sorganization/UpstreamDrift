"""Unified cross-engine perturbation study on the canonical triple pendulum (#13795).

Re-runs the #2021 perturbation study with one setup for every engine: the
canonical :data:`TRIPLE_PENDULUM_SPEC` (one gravity constant), the RK4 step
that SAP-3 (#13792) shows converged at 1 ms, one noise seed and zero-order-hold
control. The in-house dynamics and MuJoCo each compute their own state, tip
velocity and energy; the receipt is assembled by
:func:`cross_engine_receipt.build_receipt`.

Design by Contract
------------------
- Pre: ``step(dt)`` needs a finite positive ``dt``; controls have three entries.
- Post: :func:`run_unified_study` returns a receipt that passes
  :func:`validate_receipt`, plus the within-engine trial CV of every metric
  (the noise-induced sensitivity band the cross-engine CV is read against).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.shared.python.analysis.cross_engine import BACKEND_REAL, METRIC_KEYS, _safe_cv
from src.shared.python.analysis.cross_engine_receipt import (
    SharedSetup,
    build_receipt,
    validate_receipt,
)
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEnginePerturbationRunner,
    CrossEngineRunResult,
    CrossEngineSimConfig,
)
from src.shared.python.pendulum_simulator.step_convergence import (
    InHouseTripleAdapter,
    MuJoCoTripleAdapter,
)
from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    TRIPLE_PENDULUM_SPEC,
)

N_JOINTS = 3


class ObservableTripleAdapter(Protocol):
    """A step-convergence adapter that also reports state, tip velocity and energy."""

    @property
    def engine_name(self) -> str: ...

    def reset(self) -> None: ...

    def configure(self, integrator: str, dt: float) -> None: ...

    def step(
        self, t: float, dt: float, control: tuple[float, float, float]
    ) -> None: ...

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]: ...

    def tip_velocity(self) -> np.ndarray: ...

    def total_energy(self) -> float: ...


@dataclass(frozen=True)
class UnifiedStudy:
    """The one setup every engine runs (#2021 duration, noise and trial count)."""

    dt_s: float = 1e-3
    integrator: str = "rk4"
    seed: int = 42
    interpolation_mode: str = "zoh"
    t_end_s: float = 1.5
    n_trials: int = 10
    noise_amplitude_nm: float = 0.1


STUDY = UnifiedStudy()


class SteppableTripleEngine:
    """Expose an adapter through the perturbation runner's ``SteppableEngine`` API."""

    def __init__(self, adapter: ObservableTripleAdapter, study: UnifiedStudy = STUDY):
        self._adapter = adapter
        self._study = study
        self._control = (0.0, 0.0, 0.0)
        self._steps = 0

    def reset(self) -> None:
        self._adapter.reset()
        self._adapter.configure(self._study.integrator, self._study.dt_s)
        self._control = (0.0, 0.0, 0.0)
        self._steps = 0

    def set_control(self, u: np.ndarray) -> None:
        values = np.asarray(u, dtype=float).ravel()
        if values.shape != (N_JOINTS,) or not np.all(np.isfinite(values)):
            raise ValueError(f"control must be {N_JOINTS} finite torques, got {u!r}")
        self._control = (float(values[0]), float(values[1]), float(values[2]))

    def step(self, dt: float | None = None) -> None:
        dt = self._study.dt_s if dt is None else dt
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}")
        self._adapter.step(self._steps * dt, dt, self._control)
        self._steps += 1

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._adapter.joint_state()

    def get_end_effector_velocity(self) -> np.ndarray:
        return self._adapter.tip_velocity()

    def get_total_energy(self) -> float:
        return self._adapter.total_energy()


def _trial_cv(result: CrossEngineRunResult) -> dict[str, float | None]:
    band: dict[str, float | None] = {}
    for metric in METRIC_KEYS:
        cv = _safe_cv(
            getattr(result, f"mean_{metric}"), getattr(result, f"std_{metric}")
        )
        band[metric] = None if math.isnan(cv) else cv
    return band


def run_unified_study(
    adapters: list[ObservableTripleAdapter] | None = None,
    study: UnifiedStudy = STUDY,
) -> dict[str, Any]:
    """Run every adapter on the shared setup and return the validated receipt."""
    adapters = adapters or [InHouseTripleAdapter(), MuJoCoTripleAdapter()]
    config = CrossEngineSimConfig(
        t_end=study.t_end_s,
        dt=study.dt_s,
        noise_amplitude=study.noise_amplitude_nm,
        n_trials=study.n_trials,
        seed=study.seed,
    )
    runner = CrossEnginePerturbationRunner(config)
    for adapter in adapters:
        runner.register_engine(
            adapter.engine_name, SteppableTripleEngine(adapter, study)
        )
    n_steps = round(study.t_end_s / study.dt_s)
    results = runner.run_comparison(np.zeros((n_steps, N_JOINTS)))
    setup = SharedSetup.from_spec(
        TRIPLE_PENDULUM_SPEC,
        dt_s=study.dt_s,
        integrator=study.integrator,
        seed=study.seed,
        interpolation_mode=study.interpolation_mode,
    )
    receipt = build_receipt(
        setup,
        dict.fromkeys(results, BACKEND_REAL),
        results,
        n_trials=study.n_trials,
        t_end_s=study.t_end_s,
        noise_amplitude_nm=study.noise_amplitude_nm,
    )
    receipt["sensitivity_context"] = {
        "within_engine_trial_cv": {n: _trial_cv(r) for n, r in results.items()},
        "drive_mode_receipt": "docs/validation/drive_mode_sensitivity_receipt.json",
    }
    validate_receipt(receipt)
    return receipt


def receipt_cvs_match(fresh: dict[str, Any], committed: dict[str, Any]) -> bool:
    """True when both receipts share a setup and agree on every cross-engine CV."""
    if fresh["setup"] != committed["setup"]:
        return False
    for metric, cv in committed["cv_summary"].items():
        got, want = fresh["cv_summary"][metric]["cv"], cv["cv"]
        if (got is None) != (want is None):
            return False
        if got is not None and not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-15):
            return False
    return True


__all__ = [
    "STUDY",
    "SteppableTripleEngine",
    "UnifiedStudy",
    "receipt_cvs_match",
    "run_unified_study",
]

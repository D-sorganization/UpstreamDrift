"""Unified cross-engine perturbation study on the canonical triple pendulum (#13795).

Re-runs the #2021 study with one setup for every engine (canonical spec, one g,
1 ms RK4 per SAP-3 #13792, one seed, zero-order hold). In-house dynamics and
MuJoCo each compute their own state, tip velocity and energy.

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
from typing import Any

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
TripleAdapter = InHouseTripleAdapter | MuJoCoTripleAdapter


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

    def __init__(self, adapter: TripleAdapter, study: UnifiedStudy = STUDY):
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
    adapters: list[TripleAdapter] | None = None,
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
        "within_engine_trial_cv": {n: _trial_cv(r) for n, r in results.items()}
    }
    validate_receipt(receipt)
    return receipt

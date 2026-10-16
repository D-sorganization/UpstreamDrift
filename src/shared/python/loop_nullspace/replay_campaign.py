"""Paired Monte Carlo over the NSX-31 replay (NSX-91, #12720; rules R10, R13).

:func:`replay_strategy` turns one :class:`ReplayCase` (an allocation replayed in
open loop or with declared feedback) into a :class:`~paired_monte_carlo.Strategy`
whose runner

1. realises the campaign's torque disturbance for the trial draw,
2. adds it to the replay as a *known external generalised load* held constant
   over each allocation interval (zero-order hold, covariance ``S / dt`` per
   interval, the discretisation the NSX-90 prediction assumes, V6),
3. runs :func:`replay.replay` and maps the report to the outcome vector with the
   caller's ``outcome_fn`` (for example the NSX-83 impact outcome map, which may
   raise :class:`~paired_monte_carlo.TrialMiss`),
4. turns a terminated replay (non-finite or blown-up state) into a retained
   failure.

The mode of the strategy is the replay mode (open loop vs feedback-assisted,
R10); the integrator signature is the replay's fixed RK4 step ``dt / substeps``
(R13).  Because RK4 evaluates the load at the end of an interval, the held noise
is right-continuous: the stage at the interval end takes the next interval's
value, which smooths but preserves the integral of the noise over the window.

Non-claim: a replay is not physical acceptance (see :data:`replay.NON_CLAIM`);
the dispersion it yields is conditional on the declared allocation and noise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.engines.feedback_comparison import ReplayPolicy
from src.shared.python.loop_nullspace.allocation_spec import AllocationResult
from src.shared.python.loop_nullspace.evidence import Array, readonly_float
from src.shared.python.loop_nullspace.paired_mc_spec import (
    CampaignSpec,
    IntegratorSignature,
    TrialDraw,
)
from src.shared.python.loop_nullspace.paired_monte_carlo import Strategy
from src.shared.python.loop_nullspace.feedback_runtime import AnyFeedbackLaw
from src.shared.python.loop_nullspace.replay import (
    ReplayReport,
    ReplaySettings,
    ReplaySetup,
    mode_for_policy,
    replay,
)
from src.shared.python.perturbation.disturbance import Spectrum

__all__ = ["ReplayCase", "TorqueNoise", "replay_integrator", "replay_strategy"]

_GRID_RTOL = 1e-9
OutcomeFn = Callable[[ReplayReport], ArrayLike]


@dataclass(frozen=True)
class ReplayCase:
    """One replay to be repeated under noise: everything ``replay()`` takes."""

    allocation: AllocationResult
    provider: Any
    setup: ReplaySetup
    policy: ReplayPolicy
    feedback: AnyFeedbackLaw | None = None
    settings: ReplaySettings | None = None

    @property
    def effective_settings(self) -> ReplaySettings:
        return self.settings if self.settings is not None else ReplaySettings()

    def frame_times(self) -> Array:
        """Allocation frame times (validated by ``replay`` itself)."""
        return np.asarray(self.allocation.times_s, dtype=np.float64)


@dataclass(frozen=True)
class TorqueNoise:
    """Which campaign disturbance drives the replay and how it loads the model.

    ``load_map`` is ``(n, n_channels)``: the generalised load per unit of each
    channel (an identity block for noise added to joint torques).
    ``signal_from_torque`` feeds the allocation's actuated torque at the frame
    opening each interval (zero-order hold) to a signal-dependent (M1)
    disturbance, so the noise level follows the allocation under test (NSX-92);
    it must be set exactly when the disturbance declares a signal law.  ``scale``
    ``(n_steps, n_channels)``, when given, multiplies each trial's draw
    interval by interval: the declared noise-inflation penalty of a stiffened
    joint (NSX-93), the same factor the linear prediction applies to its
    covariance (``scale^2``).
    """

    disturbance: str
    load_map: Array
    signal_from_torque: bool = False
    scale: Array | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "load_map", readonly_float(self.load_map, "load_map"))
        if self.load_map.ndim != 2:
            raise ValueError(f"load_map must be 2-D, got shape {self.load_map.shape}")
        if self.scale is not None:
            scale = readonly_float(self.scale, "scale")
            if scale.ndim != 2 or np.any(scale < 0.0):
                raise ValueError("scale must be a non-negative (n_steps, n_channels)")
            object.__setattr__(self, "scale", scale)


def replay_integrator(case: ReplayCase, dt: float) -> IntegratorSignature:
    """The replay's integrator: fixed-step RK4 with step ``dt / substeps``."""
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be positive and finite, got {dt}")
    return IntegratorSignature(
        "RK4 (counterfactual.rk4_step_with_load)",
        dt / case.effective_settings.substeps,
    )


def _check_grid(spec: CampaignSpec, case: ReplayCase, noise: TorqueNoise) -> None:
    dist = spec.disturbance(noise.disturbance)
    if dist.persistent or dist.spectrum is not Spectrum.WHITE:
        raise ValueError(
            "torque noise must be a white per-step disturbance (a persistent "
            "or coloured one has no zero-order-hold interval covariance S / dt)"
        )
    if dist.has_signal_law != noise.signal_from_torque:
        raise ValueError(
            "signal_from_torque must be set exactly when the disturbance "
            f"declares a signal-dependent law (law={dist.has_signal_law}, "
            f"signal_from_torque={noise.signal_from_torque})"
        )
    times = case.frame_times()
    steps = np.diff(times)
    if steps.size != spec.n_steps:
        raise ValueError(
            f"spec.n_steps={spec.n_steps} but the allocation has {steps.size} intervals"
        )
    if not np.allclose(steps, spec.dt, rtol=_GRID_RTOL, atol=0.0):
        raise ValueError(
            f"spec.dt={spec.dt} must equal every allocation interval "
            f"(got {steps.min()}..{steps.max()})"
        )
    if noise.scale is not None and noise.scale.shape != (steps.size, dist.n_channels):
        raise ValueError(
            f"scale must be ({steps.size}, {dist.n_channels}), got {noise.scale.shape}"
        )
    n = case.setup.n
    if noise.load_map.shape != (n, dist.n_channels):
        raise ValueError(
            f"load_map must be ({n}, {dist.n_channels}) for {n} coordinates and "
            f"{dist.n_channels} channels, got {noise.load_map.shape}"
        )


def _held_load(
    times: Array, noise_series: Array, load_map: Array, base: Any
) -> Callable[[float, Array], Array]:
    """``(t, q) -> base(t, q) + load_map @ w[k(t)]`` with ``w`` held per interval."""
    series = noise_series @ load_map.T  # (n_steps, n)

    def load(t: float, q: Array) -> Array:
        k = int(
            np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(series) - 1)
        )
        extra = series[k]
        return extra if base is None else np.asarray(base(t, q)) + extra

    return load


def _interval_signal(case: ReplayCase) -> Array:
    """Actuated torque at the frame opening each interval, ``(n_steps, n_a)``."""
    torque = case.allocation.torque
    if torque is None:
        raise ValueError("the allocation carries no torque to drive the signal law")
    return np.asarray(torque, dtype=np.float64)[:-1]


def replay_strategy(
    spec: CampaignSpec,
    name: str,
    case: ReplayCase,
    noise: TorqueNoise,
    outcome_fn: OutcomeFn,
) -> Strategy:
    """A campaign :class:`Strategy` that replays ``case`` under the trial noise.

    Pre (``ValueError``/``TypeError``): ``outcome_fn`` callable; the noise
    disturbance exists, is white, and its grid (``n_steps``, ``dt``) equals the
    allocation intervals; ``load_map`` conforms to the model.  Post: the runner
    never resets state, labels the strategy with the replay's own mode, and
    raises ``RuntimeError`` for a terminated replay (retained as a failure).
    """
    if not callable(outcome_fn):
        raise TypeError("outcome_fn must be callable ReplayReport -> outcomes")
    _check_grid(spec, case, noise)
    times = case.frame_times()
    mode = mode_for_policy(case.policy)
    record = case.allocation.record
    hypotheses = tuple(record.hypotheses) if record else ()

    signal = _interval_signal(case) if noise.signal_from_torque else None

    def runner(draw: TrialDraw) -> ArrayLike:
        w = draw.realise(noise.disturbance, signal=signal)
        if noise.scale is not None:
            w = w * noise.scale
        setup = replace(
            case.setup,
            external_load=_held_load(
                times, w, noise.load_map, case.setup.external_load
            ),
        )
        report = replay(
            case.allocation,
            case.provider,
            policy=case.policy,
            setup=setup,
            feedback=case.feedback,
            settings=case.settings,
        )
        if report.mode is not mode:
            raise TypeError("replay returned a report of another mode")
        if report.terminated_reason is not None:
            raise RuntimeError(f"replay terminated: {report.terminated_reason}")
        return outcome_fn(report)

    return Strategy(
        name=name,
        runner=runner,
        mode=mode,
        integrator=replay_integrator(case, spec.dt),
        hypotheses=hypotheses,
    )

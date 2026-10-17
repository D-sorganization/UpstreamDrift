"""Co-contraction sweep at fixed net torque, typed with NSX-81 evidence (NSX-94).

A *declared* experiment, never an inference: the caller names the
:class:`CocontractionIndex` and the levels (or their count); the net torque is
the experiment's input, and the sweep reports, level by level, what the null
direction of the activation fibre changes and what it leaves alone.

* Invariant -- the joint torque (to ``(m + 2) eps sum|G_k|``, ~1e-13 N m for the
  elbow fixture) and the reserve fraction.  Both are ``INVARIANT`` records.
* Changed -- the signal-dependent muscle-noise covariance pushed to the joint
  (``H-NOISE-SIGNAL-DEPENDENT``, ``FAMILY_DEPENDENT``: it depends on the
  allocation, here the declared minimum-norm representative per level) and the
  short-range joint stiffness (``H-MUSCLE-SHORT-RANGE-STIFFNESS``,
  ``FAMILY_DEPENDENT``).  When the noise spec declares no signal law the noise
  is additive, level-independent and ``INVARIANT`` conditioned on
  ``H-NOISE-ADDITIVE``.  A stiffness the model cannot supply is an
  ``UNAVAILABLE`` record with a reason while the noise record is still reported.
* Fail closed -- a frame whose reserves carry the load or whose force law is
  not linear to tolerance is ``NOT_QUALIFIED``: every value record is
  ``UNAVAILABLE`` with the reasons; no number is produced.

Non-claims (also carried in :attr:`CocontractionSweep.non_claims`): the sweep
says nothing about the golfer's actual co-contraction (an EMG study is a
deferred validation); stiffness and noise parameters are declared hypotheses;
the short-range-stiffness evidence is feline and force-dependent, not human.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.muscle_fiber import (
    LEVEL_INSET,
    CocontractionIndex,
    FiberInfeasibleError,
    activation_fiber,
)
from src.shared.python.loop_nullspace.muscle_frame import (
    RESERVE_FRACTION_LIMIT,
    MuscleFrame,
    QualificationStatus,
    qualify_frame,
)
from src.shared.python.loop_nullspace.muscle_noise import joint_torque_covariance
from src.shared.python.loop_nullspace.muscle_stiffness import (
    StiffnessModel,
    StiffnessNotQualified,
    muscle_stiffness,
)
from src.shared.python.perturbation.disturbance import DisturbanceSpec

__all__ = [
    "NON_CLAIMS",
    "CocontractionSweep",
    "SweepConfig",
    "SweepStatus",
    "cocontraction_sweep",
]

NON_CLAIMS = (
    "Declared experiment, not an estimate of the golfer's co-contraction (EMG is "
    "a deferred validation). Noise gain and stiffness are declared hypotheses; "
    "short-range-stiffness evidence is feline and force-dependent (CIT-CUI-2008). "
    "Representatives are minimum-norm per level, not physiology."
)

_DEFAULT_LEVELS = 9
_NOISE_UNITS = "N^2 m^2"
_STIFFNESS_UNITS = "N m/rad"


class SweepStatus(str, Enum):
    """Whether the sweep produced values."""

    COMPUTED = "COMPUTED"
    NOT_QUALIFIED = "NOT_QUALIFIED"


@dataclass(frozen=True)
class SweepConfig:
    """Declared inputs of the sweep.

    ``levels`` (explicit co-contraction levels) and ``n_levels`` (an evenly
    spaced grid inside the fibre's range, inset by ``LEVEL_INSET``) are mutually
    exclusive; neither means ``_DEFAULT_LEVELS`` points.  ``stiffness_model``
    ``None`` declares no stiffness model (records ``UNAVAILABLE``).
    """

    noise: DisturbanceSpec
    dt: float
    stiffness_model: StiffnessModel | None
    force_gain: float | None = None
    levels: tuple[float, ...] | None = None
    n_levels: int | None = None
    reserve_limit: float = RESERVE_FRACTION_LIMIT
    linearity_limit: float = RESERVE_FRACTION_LIMIT

    def __post_init__(self) -> None:
        if self.levels is not None and self.n_levels is not None:
            raise ValueError("give levels or n_levels, not both")
        if self.levels is not None:
            if len(self.levels) == 0 or not all(math.isfinite(x) for x in self.levels):
                raise ValueError("levels must be a non-empty finite sequence")
        if self.n_levels is not None and self.n_levels < 2:
            raise ValueError("n_levels must be at least 2")
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError(f"dt must be positive and finite, got {self.dt}")


@dataclass(frozen=True)
class CocontractionSweep:
    """Result of :func:`cocontraction_sweep` (all records are NSX-81 typed)."""

    status: SweepStatus
    reasons: tuple[str, ...]
    index_name: str
    levels: Array
    activations: Array
    torque_residual: Array
    engine_torque_gap: Array
    level_range: tuple[float, float] | None
    fiber_nullity: int | None
    net_torque: QuantityRecord
    reserve_fraction: QuantityRecord
    noise_trace: QuantityRecord
    stiffness_trace: QuantityRecord
    stiffness_min_eigenvalue: QuantityRecord
    non_claims: str = NON_CLAIMS


def _record(
    frame: MuscleFrame,
    values: ArrayLike,
    units: str,
    dependence: AllocationDependence,
    *,
    hypotheses: tuple[str, ...] = (),
    conditioning: tuple[str, ...] = (),
    reason: str | None = None,
) -> QuantityRecord:
    return QuantityRecord(
        values=np.asarray(values, dtype=np.float64),
        units=units,
        allocation_dependence=dependence,
        model_provenance=frame.model_provenance,
        input_provenance=frame.input_provenance,
        hypotheses=hypotheses,
        conditioning=conditioning,
        reason=reason,
    )


def _unavailable(frame: MuscleFrame, units: str, reason: str) -> QuantityRecord:
    return _record(
        frame, [np.nan], units, AllocationDependence.UNAVAILABLE, reason=reason
    )


def _not_qualified(
    frame: MuscleFrame, index: CocontractionIndex, reasons: tuple[str, ...]
) -> CocontractionSweep:
    why = "NOT_QUALIFIED: " + "; ".join(reasons)
    n_mus = frame.n_muscles
    return CocontractionSweep(
        status=SweepStatus.NOT_QUALIFIED,
        reasons=reasons,
        index_name=index.name,
        levels=np.zeros(0),
        activations=np.zeros((0, n_mus)),
        torque_residual=np.zeros(0),
        engine_torque_gap=np.zeros(0),
        level_range=None,
        fiber_nullity=None,
        net_torque=_unavailable(frame, "N m", why),
        reserve_fraction=_unavailable(frame, "1", why),
        noise_trace=_unavailable(frame, _NOISE_UNITS, why),
        stiffness_trace=_unavailable(frame, _STIFFNESS_UNITS, why),
        stiffness_min_eigenvalue=_unavailable(frame, _STIFFNESS_UNITS, why),
    )


def _resolve_levels(
    config: SweepConfig, lo: float, hi: float, index_name: str
) -> Array:
    span = hi - lo
    if config.levels is not None:
        lev = np.asarray(config.levels, dtype=np.float64)
        slack = 1e-9 * max(span, 1.0)
        if np.any(lev < lo - slack) or np.any(lev > hi + slack):
            raise FiberInfeasibleError(
                f"level outside the fibre's range [{lo:.6g}, {hi:.6g}] of "
                f"{index_name!r}: {lev.tolist()}"
            )
        return np.clip(lev, lo + 0.0, hi)
    n_lev = config.n_levels or _DEFAULT_LEVELS
    frac = LEVEL_INSET + (1.0 - 2.0 * LEVEL_INSET) * np.linspace(0.0, 1.0, n_lev)
    return np.asarray(lo + span * frac)


def _stiffness_records(
    frame: MuscleFrame, acts: Array, config: SweepConfig
) -> tuple[QuantityRecord, QuantityRecord]:
    if config.stiffness_model is None:
        why = "no stiffness model declared"
    else:
        try:
            traces, eigs = [], []
            for act in acts:
                joint = muscle_stiffness(
                    frame,
                    act,
                    config.stiffness_model,
                    force_gain=config.force_gain,
                ).joint
                traces.append(float(np.trace(joint)))
                eigs.append(float(np.linalg.eigvalsh(joint)[0]))
            hyp = ("H-MUSCLE-SHORT-RANGE-STIFFNESS",)
            dep = AllocationDependence.FAMILY_DEPENDENT
            return (
                _record(frame, traces, _STIFFNESS_UNITS, dep, hypotheses=hyp),
                _record(frame, eigs, _STIFFNESS_UNITS, dep, hypotheses=hyp),
            )
        except StiffnessNotQualified as exc:
            why = str(exc)
    return (
        _unavailable(frame, _STIFFNESS_UNITS, why),
        _unavailable(frame, _STIFFNESS_UNITS, why),
    )


def _noise_record(
    frame: MuscleFrame, acts: Array, config: SweepConfig
) -> QuantityRecord:
    traces = [
        float(np.trace(joint_torque_covariance(frame, config.noise, act, dt=config.dt)))
        for act in acts
    ]
    if config.noise.has_signal_law:
        return _record(
            frame,
            traces,
            _NOISE_UNITS,
            AllocationDependence.FAMILY_DEPENDENT,
            hypotheses=("H-NOISE-SIGNAL-DEPENDENT",),
        )
    return _record(
        frame,
        traces,
        _NOISE_UNITS,
        AllocationDependence.INVARIANT,
        conditioning=("H-NOISE-ADDITIVE",),
    )


def cocontraction_sweep(
    frame: MuscleFrame,
    torque: ArrayLike,
    index: CocontractionIndex,
    config: SweepConfig,
) -> CocontractionSweep:
    """Sweep the declared co-contraction ``index`` at the fixed net ``torque``.

    Preconditions (``ValueError``): ``torque`` finite, non-zero, length ``n``;
    ``index`` sized for the frame; explicit ``config.levels`` inside the fibre's
    range (:class:`FiberInfeasibleError` otherwise).  Postconditions: for a
    ``COMPUTED`` sweep every level's activation lies in the box and reproduces
    the net torque to ``torque_residual`` (machine precision); the ``net_torque``
    record is ``INVARIANT``.  A frame that fails the reserve or linearity gate
    returns ``NOT_QUALIFIED`` with no numerical values.
    """
    qual = qualify_frame(
        frame,
        torque,
        reserve_limit=config.reserve_limit,
        linearity_limit=config.linearity_limit,
    )
    if qual.status is not QualificationStatus.QUALIFIED:
        return _not_qualified(frame, index, qual.reasons)
    fiber = activation_fiber(
        frame,
        torque,
        reserve_limit=config.reserve_limit,
        linearity_limit=config.linearity_limit,
    )
    lo, hi = fiber.index_range(index)
    levels = _resolve_levels(config, lo, hi, index.name)
    acts = np.array([fiber.point_at_level(index, float(c)) for c in levels])
    resid = np.array([fiber.residual(a) for a in acts])
    gap = np.array(
        [
            float(np.max(np.abs(frame.engine_torque(a) - fiber.target_torque)))
            for a in acts
        ]
    )
    stiff_trace, stiff_eig = _stiffness_records(frame, acts, config)
    return CocontractionSweep(
        status=SweepStatus.COMPUTED,
        reasons=(),
        index_name=index.name,
        levels=levels,
        activations=acts,
        torque_residual=resid,
        engine_torque_gap=gap,
        level_range=(lo, hi),
        fiber_nullity=fiber.nullity,
        net_torque=_record(
            frame, fiber.target_torque, "N m", AllocationDependence.INVARIANT
        ),
        reserve_fraction=_record(
            frame, [fiber.reserve_fraction], "1", AllocationDependence.INVARIANT
        ),
        noise_trace=_noise_record(frame, acts, config),
        stiffness_trace=stiff_trace,
        stiffness_min_eigenvalue=stiff_eig,
    )

"""Separated constitutive models for five meanings of mechanical ``slack``.

The models are deliberately scalar and synthetic. They provide auditable
states and energy ledgers for perturbation design; they do not identify a
human tissue, grip behavior, or control strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
SlackKind = Literal[
    "contact_disengagement",
    "transmission_backlash",
    "structural_preload",
    "biological_series_compliance",
    "control_deadband",
]


@dataclass(frozen=True, slots=True)
class SlackParameters:
    """Parameters for one declared scalar slack class."""

    kind: SlackKind
    threshold: float
    stiffness: float
    damping: float = 0.0
    preload: float = 0.0

    def __post_init__(self) -> None:
        values = np.array([self.threshold, self.stiffness, self.damping, self.preload])
        if not np.all(np.isfinite(values)):
            raise ValueError("slack parameters must be finite")
        if self.threshold < 0.0 or self.stiffness <= 0.0 or self.damping < 0.0:
            raise ValueError(
                "threshold/damping must be nonnegative and stiffness positive"
            )
        if self.kind != "structural_preload" and self.preload != 0.0:
            raise ValueError("preload is defined only for structural_preload")


@dataclass(frozen=True, slots=True)
class SlackTrace:
    """Constitutive response and reference-explicit energy ledger."""

    transmitted: FloatArray
    elastic: FloatArray
    dissipative: FloatArray
    stored_energy: FloatArray
    engaged: npt.NDArray[np.bool_]


def _outside_dead_zone(displacement: FloatArray, threshold: float) -> FloatArray:
    return np.sign(displacement) * np.maximum(np.abs(displacement) - threshold, 0.0)


def evaluate_slack(
    displacement: npt.ArrayLike,
    rate: npt.ArrayLike,
    parameters: SlackParameters,
) -> SlackTrace:
    """Evaluate exactly one slack class without borrowing another class's state."""

    x = np.asarray(displacement, dtype=np.float64)
    xd = np.asarray(rate, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or xd.shape != x.shape:
        raise ValueError(
            "displacement and rate must be matching one-dimensional arrays"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(xd)):
        raise ValueError("displacement and rate must be finite")

    kind = parameters.kind
    if kind == "contact_disengagement":
        deformation = np.maximum(x - parameters.threshold, 0.0)
        engaged = deformation > 0.0
        elastic = parameters.stiffness * deformation
        dissipative = parameters.damping * np.maximum(xd, 0.0) * engaged
    elif kind == "transmission_backlash":
        deformation = _outside_dead_zone(x, parameters.threshold)
        engaged = deformation != 0.0
        elastic = parameters.stiffness * deformation
        dissipative = parameters.damping * xd * engaged
    elif kind == "structural_preload":
        deformation = x + parameters.preload
        engaged = np.ones_like(x, dtype=np.bool_)
        elastic = parameters.stiffness * deformation
        dissipative = parameters.damping * xd
    elif kind == "biological_series_compliance":
        deformation = np.maximum(x - parameters.threshold, 0.0)
        engaged = deformation > 0.0
        elastic = parameters.stiffness * deformation
        dissipative = parameters.damping * xd * engaged
    elif kind == "control_deadband":
        deformation = np.zeros_like(x)
        engaged = np.abs(x) > parameters.threshold
        elastic = np.zeros_like(x)
        dissipative = parameters.stiffness * _outside_dead_zone(x, parameters.threshold)
    else:  # pragma: no cover - Literal plus runtime fail-closed guard
        raise ValueError(f"unsupported slack kind: {kind}")

    stored = 0.5 * parameters.stiffness * deformation**2
    return SlackTrace(
        transmitted=elastic + dissipative,
        elastic=elastic,
        dissipative=dissipative,
        stored_energy=stored,
        engaged=engaged,
    )


def energy_residual(
    time_s: npt.ArrayLike,
    rate: npt.ArrayLike,
    trace: SlackTrace,
) -> float:
    """Return work minus stored-energy change and viscous/control work."""

    time = np.asarray(time_s, dtype=np.float64)
    velocity = np.asarray(rate, dtype=np.float64)
    if time.shape != velocity.shape or time.shape != trace.transmitted.shape:
        raise ValueError("time, rate, and trace must have matching shapes")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing")
    input_work = float(np.trapezoid(trace.transmitted * velocity, x=time))
    dissipative_work = float(np.trapezoid(trace.dissipative * velocity, x=time))
    energy_change = float(trace.stored_energy[-1] - trace.stored_energy[0])
    return input_work - dissipative_work - energy_change

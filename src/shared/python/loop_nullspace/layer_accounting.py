"""Effort and energy accounting of the impedance and feedback layer (NSX-93).

The allocated torque, the declared impedance torque and the delayed-feedback
torque are three different things and are accounted on three lines, never
summed into one "torque used" (rule R10; R6 for the power terms).  Each line
carries the replay's :class:`~replay_spec.FeedbackBudget` (peak, energy and
absolute-work ratios against the allocated torque, computed by
:func:`replay_spec.feedback_budget`) and an :class:`EffortCost`:

* ``l2_effort = int sum_i tau_i^2 dt`` [N^2 m^2 s], the quadratic effort;
* ``positive_work_j`` / ``negative_work_j`` = ``int sum_i (tau_i w_i)_{+/-} dt``
  [J], the joint power the layer injects or absorbs (a *joint* power, not a
  muscle power, and not metabolic cost -- rule R6).  All integrals are
  trapezoids over the replay nodes.

For the impedance the energy bookkeeping is also kept apart: the energy
dissipated by the damping, ``int de^T D de dt >= 0`` (with ``de = S(v_ref -
v)``), and the peak elastic energy ``1/2 e^T K e`` stored by the stiffness.

Postcondition: with no declared saturation ``impedance + feedback`` equals the
replay's feedback series to rounding; with a saturation the difference is the
clipped part and ``saturated`` says so.  Non-claim: effort here is an
accounting of the declared model, not a physiological cost.  Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.feedback_layer import FeedbackLayer
from src.shared.python.loop_nullspace.replay import ReplayReport
from src.shared.python.loop_nullspace.replay_spec import (
    FeedbackBudget,
    ReplaySetup,
    feedback_budget,
)

__all__ = [
    "ComponentAccount",
    "EffortCost",
    "ImpedanceEnergy",
    "LayerAccounting",
    "account_layer",
    "effort_cost",
]


@dataclass(frozen=True)
class EffortCost:
    """Quadratic effort and signed joint work of one torque series."""

    l2_effort: float
    positive_work_j: float
    negative_work_j: float

    @property
    def net_work_j(self) -> float:
        """Positive minus negative work."""
        return self.positive_work_j - self.negative_work_j

    @property
    def abs_work_j(self) -> float:
        """Positive plus negative work."""
        return self.positive_work_j + self.negative_work_j


def effort_cost(times_s: Array, torque: Array, velocity: Array) -> EffortCost:
    """Trapezoid effort and work of ``torque`` ``(T, n_a)`` against ``velocity``.

    Raises:
        ValueError: Shapes that do not conform or fewer than two nodes.
    """
    t = np.asarray(times_s, dtype=np.float64)
    tau = np.asarray(torque, dtype=np.float64)
    omega = np.asarray(velocity, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or tau.ndim != 2 or tau.shape != omega.shape:
        raise ValueError("times_s (T,), torque and velocity (T, n_a) must conform")
    if tau.shape[0] != t.size:
        raise ValueError(f"torque needs {t.size} rows, got {tau.shape[0]}")
    power = tau * omega
    return EffortCost(
        l2_effort=float(np.trapezoid(np.sum(tau**2, axis=1), t)),
        positive_work_j=float(np.trapezoid(np.sum(np.maximum(power, 0.0), axis=1), t)),
        negative_work_j=float(np.trapezoid(np.sum(np.maximum(-power, 0.0), axis=1), t)),
    )


@dataclass(frozen=True)
class ComponentAccount:
    """One torque line: its series, replay budget and effort cost."""

    torque: Array
    budget: FeedbackBudget
    cost: EffortCost


@dataclass(frozen=True)
class ImpedanceEnergy:
    """Energy bookkeeping of the declared impedance (J)."""

    dissipated_j: float
    peak_elastic_j: float


@dataclass(frozen=True)
class LayerAccounting:
    """The three torque lines of a layer replay, kept separate.

    ``allocated`` is the allocation alone; ``impedance`` and ``feedback`` are
    ``None`` when the layer does not declare them; ``total`` is their sum (the
    replay's own feedback line).  ``saturated`` is ``True`` if a declared
    saturation clipped the sum, so ``impedance + feedback != total.torque``.
    """

    allocated: EffortCost
    total: ComponentAccount
    impedance: ComponentAccount | None
    feedback: ComponentAccount | None
    impedance_energy: ImpedanceEnergy | None
    saturated: bool
    hypotheses: tuple[str, ...]


def _component(
    report: ReplayReport, series: Array, velocity: Array
) -> ComponentAccount:
    budget = feedback_budget(report.times_s, report.allocated_torque, series, velocity)
    return ComponentAccount(
        series, budget, effort_cost(report.times_s, series, velocity)
    )


def _impedance_energy(
    report: ReplayReport, layer: FeedbackLayer, setup: ReplaySetup
) -> ImpedanceEnergy:
    imp = layer.impedance
    assert imp is not None and report.reference_q is not None
    assert report.reference_v is not None
    sel = setup.actuation
    e = (report.reference_q - report.q) @ sel.T
    de = (report.reference_v - report.v) @ sel.T
    k_all, d_all = imp.gain_series(report.times_s)
    dissipation = np.einsum("ti,tij,tj->t", de, d_all, de)
    elastic = 0.5 * np.einsum("ti,tij,tj->t", e, k_all, e)
    return ImpedanceEnergy(
        dissipated_j=float(np.trapezoid(dissipation, report.times_s)),
        peak_elastic_j=float(np.max(elastic)),
    )


def account_layer(
    report: ReplayReport, layer: FeedbackLayer, setup: ReplaySetup
) -> LayerAccounting:
    """Split a layer replay into allocated, impedance and feedback lines.

    ``report`` must be the feedback-mode replay of ``layer`` with ``setup``.

    Raises:
        TypeError / ValueError: ``report`` is not a feedback replay of ``layer``.
    """
    if not isinstance(layer, FeedbackLayer):
        raise TypeError("layer must be a FeedbackLayer")
    if report.feedback_torque is None or report.feedback_law is not layer:
        raise ValueError("report must be the feedback-mode replay of this layer")
    imp, fb = layer.component_torques(setup, report.times_s, report.q, report.v)
    velocity = report.v @ setup.actuation.T
    total = _component(report, report.feedback_torque, velocity)
    unsat = imp + fb
    return LayerAccounting(
        allocated=effort_cost(report.times_s, report.allocated_torque, velocity),
        total=total,
        impedance=(
            None if layer.impedance is None else _component(report, imp, velocity)
        ),
        feedback=None if layer.feedback is None else _component(report, fb, velocity),
        impedance_energy=(
            None if layer.impedance is None else _impedance_energy(report, layer, setup)
        ),
        saturated=bool(not np.array_equal(unsat, report.feedback_torque)),
        hypotheses=layer.hypotheses,
    )

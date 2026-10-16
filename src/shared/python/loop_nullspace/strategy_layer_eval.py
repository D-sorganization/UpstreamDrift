"""Layer and muscle evaluators of the strategy catalogue (NSX-95, #12724).

Thin adapters that *compose* the NSX-93 layer
(:func:`~.layer_linear.layer_variance`, :func:`~.layer_linear.impedance_state_gain`)
and the NSX-94 co-contraction curve
(:func:`~.muscle_variance.cocontraction_variance_curve`); they add only what the
comparison needs on top: the strategy-off baseline, a failure verdict from the
delay margin, the expected quadratic effort of the declared torque line and the
variance budget in canonical input classes.

Expected layer effort.  With the closed-loop covariance ``P_t`` of the augmented
state ``z_t`` and the declared torque ``u_t = -G_t z_t`` (impedance on the current
block, feedback on the delayed block), ``E[u_t^T u_t] = tr(G_t P_t G_t^T)`` and the
effort is ``sum_t dt tr(G_t P_t G_t^T)`` [torque^2 s].  It is a separate line from
the allocated torque (rule R10) and is a property of the declared model, not a
physiological cost.

Delay verdict.  A feedback arm is ``UNSTABLE`` when the frozen delay margin of any
step is at or below the declared delay (conservative: for a delay system stability
can switch more than once, but the first loss is the one reported by NSX-93).
"""

from __future__ import annotations

import math
from typing import Any
from dataclasses import replace

import numpy as np

from src.shared.python.loop_nullspace.evidence import AllocationDependence, Array
from src.shared.python.loop_nullspace.feedback_layer import DELAY_HYPOTHESIS_ID
from src.shared.python.loop_nullspace.layer_linear import (
    LayerLinearization,
    LayerOptions,
    LayerVariance,
    impedance_state_gain,
    inflate_step_covariances,
    layer_variance,
)
from src.shared.python.loop_nullspace.muscle_variance import (
    cocontraction_variance_curve,
)
from src.shared.python.loop_nullspace.strategy_arms import (
    ArmResult,
    ArmSpec,
    ArmStatus,
    ComparisonContext,
    LayerContext,
    skipped_arm,
)
from src.shared.python.loop_nullspace.strategy_spec import (
    EFFORT_ACTIVATION,
    EFFORT_LAYER,
    StrategyId,
)
from src.shared.python.loop_nullspace.trajectory_system import unverified_ids
from src.shared.python.loop_nullspace.variance_propagation import (
    VarianceBudget,
    propagate_covariance,
)

__all__ = [
    "canonical_budget",
    "evaluate_cocontraction_arm",
    "evaluate_layer_arm",
    "expected_layer_effort",
]

_BUDGET_NAMES = {
    "torque_noise": "additive_noise",
    "initial_state": "initial_state",
    "state": "initial_state",
    "muscle_noise": "muscle_noise",
    "external": "external_disturbance",
}


def canonical_budget(budget: VarianceBudget) -> dict[str, float]:
    """Variance parts of the first outcome in the canonical input-class names."""
    out: dict[str, float] = {}
    for name, part in budget.contributions.items():
        key = _BUDGET_NAMES.get(name, name)
        out[key] = out.get(key, 0.0) + float(part[0])
    return out


def expected_layer_effort(lin: LayerLinearization, p0: Array, sigma: Array) -> float:
    """``sum_t dt tr(G_t P_t G_t^T)`` of the declared layer torque (module doc).

    Pre: ``sigma`` is the per-step noise covariance the variance used.  Post:
    non-negative; zero without a declared gain.
    """
    if lin.impedance_gain is None and lin.feedback_gain is None:
        return 0.0
    n_x, blocks = lin.n_x, lin.augmented_blocks
    gain = np.zeros((lin.n_steps, lin.g_u.shape[2], blocks * n_x))
    if lin.impedance_gain is not None:
        gain[:, :, :n_x] += lin.impedance_gain
    if lin.feedback_gain is not None:
        gain[:, :, (blocks - 1) * n_x :] += lin.feedback_gain
    traj = propagate_covariance(
        lin.closed_loop(), lin.augment_covariance(p0), np.asarray(sigma)
    )
    quad = np.einsum("tij,tjk,tik->t", gain, traj[:-1], gain)
    return float(lin.dt * np.sum(quad))


# ------------------------------------------------------------------ layer arms
def _check_layer_spec(spec: ArmSpec, layer: LayerContext) -> None:
    sid = spec.strategy
    if sid is StrategyId.S7:
        if spec.feedback_gain is None or spec.impedance is not None:
            raise ValueError("S7 needs a feedback_gain and takes no impedance")
        return
    if spec.impedance is None:
        raise ValueError(
            f"{sid.value} needs a declared impedance (or cocontraction=True for S2 "
            "on a muscle model)"
        )
    if spec.feedback_gain is not None:
        raise ValueError(f"{sid.value} takes no feedback_gain (that is S7)")
    last = float(np.asarray(layer.times_s)[-1])
    if sid is StrategyId.S3 and any(w.t_end_s < last for w in spec.impedance.windows):
        raise ValueError(
            "S3 stabilizes near impact: every impedance window must end at or "
            f"after the last node ({last:.4g} s)"
        )


def _controlled(
    spec: ArmSpec, layer: LayerContext
) -> tuple[LayerLinearization, Array, tuple[str, ...], bool]:
    """Controller-carrying chain, inflated noise, layer hypotheses, is_zero."""
    base, sigma = layer.linearization, np.asarray(layer.sigma)
    hyps: tuple[str, ...] = ()
    imp_gain = None
    zero = True
    if spec.impedance is not None:
        imp_gain = impedance_state_gain(
            spec.impedance, layer.times_s, layer.selection, layer.basis, layer.offset
        )
        scale = spec.impedance.noise_scale_series(layer.times_s)
        sigma = inflate_step_covariances(sigma, scale)
        hyps += spec.impedance.hypotheses
        zero = spec.impedance.is_zero
    fb = None
    if spec.feedback_gain is not None:
        fb = np.asarray(spec.feedback_gain, dtype=np.float64)
        hyps += (DELAY_HYPOTHESIS_ID,)
        zero = not bool(np.any(fb))
    lin = replace(
        base,
        impedance_gain=imp_gain,
        feedback_gain=fb,
        delay_steps=spec.delay_steps if fb is not None else 0,
    )
    return lin, sigma, hyps, zero


def _assess(
    lin: LayerLinearization,
    layer: LayerContext,
    sigma: Array,
    hyps: tuple[str, ...],
    margin_bound: int,
) -> LayerVariance:
    return layer_variance(
        lin,
        layer.p0,
        sigma,
        layer.outcome_rows,
        layer.declaration,
        layer_hypotheses=hyps,
        outcome_units=layer.outcome_units,
        options=LayerOptions(max_margin_steps=margin_bound),
    )


def _delay_verdict(result: LayerVariance) -> tuple[float, str]:
    """``(smallest margin in s or inf, failure reason or "")``."""
    found = [m for m in result.delay_margins if m.margin_steps is not None]
    worst = min(
        (m.margin_steps for m in found if m.margin_steps is not None), default=None
    )
    if worst is not None and worst <= result.delay_steps:
        return math.inf, (
            f"closed loop at or beyond the frozen delay margin ({worst} steps <= "
            f"delay {result.delay_steps} steps)"
        )
    margins = [m.margin_s for m in found if m.margin_s is not None]
    return (min(margins) if margins else math.inf), ""


def evaluate_layer_arm(spec: ArmSpec, context: ComparisonContext) -> ArmResult:
    """S2 (impedance), S3 (impedance window ending at impact) or S7 (feedback).

    Preconditions (``ValueError``): the spec carries what its strategy needs and
    nothing else; S3 windows end at the last node.  Postconditions: the result is
    ``UNAVAILABLE`` without a layer context; ``UNSTABLE`` without numbers when the
    frozen delay margin is reached; ``NO_MECHANISM`` with ``ratio == 1`` when the
    declared gain is zero; otherwise ``OK`` with the NSX-93 variance, the
    expected layer effort and its input-class budget.
    """
    layer = context.layer
    if layer is None:
        return skipped_arm(
            spec, ArmStatus.UNAVAILABLE, "the context has no layer representation"
        )
    _check_layer_spec(spec, layer)
    lin, sigma, hyps, zero = _controlled(spec, layer)
    bound = max(20, spec.delay_steps + 5)
    base = _assess(layer.linearization, layer, layer.sigma, (), bound)
    ctrl = _assess(lin, layer, sigma, hyps, bound)
    margin_s, why = _delay_verdict(ctrl)
    common: dict[str, Any] = {
        "strategy": spec.strategy,
        "label": spec.label,
        "chs_mps": context.chs_mps,
        "baseline_variance": float(base.assessment.variance.values[0]),
        "baseline_effort": 0.0,
        "effort_basis": EFFORT_LAYER,
        "hypotheses": tuple(dict.fromkeys(hyps)),
        "unverified_hypothesis_ids": unverified_ids(hyps),
    }
    if why:
        return ArmResult(status=ArmStatus.UNSTABLE, reason=why, **common)
    var = float(ctrl.assessment.variance.values[0])
    diagnostics = {} if spec.feedback_gain is None else {"delay_margin_s": margin_s}
    return ArmResult(
        status=ArmStatus.NO_MECHANISM if zero else ArmStatus.OK,
        reason="the declared gain is zero: no effect" if zero else "",
        face_variance=common["baseline_variance"] if zero else var,
        effort=expected_layer_effort(lin, layer.p0, sigma),
        certificate="linear-closed-form",
        budget=canonical_budget(ctrl.assessment.budget),
        diagnostics=diagnostics,
        record=ctrl.assessment.variance,
        **common,
    )


# ------------------------------------------------------------- co-contraction
def evaluate_cocontraction_arm(spec: ArmSpec, context: ComparisonContext) -> ArmResult:
    """S2 on a muscle model: variance at one level of the NSX-94 sweep.

    The baseline is the *lowest* level of the sweep (the least co-contraction that
    still produces the net torque), not zero co-contraction.  ``cocontraction_level
    = None`` selects the variance-minimising level of the computed curve.
    Postconditions: ``UNAVAILABLE`` (with the sweep's reason) when the frame is not
    qualified or a declared stiffness is missing; ``NO_MECHANISM`` when neither
    signal-dependent noise nor stiffness is declared (the curve is then flat).
    """
    muscle = context.muscle
    if muscle is None:
        return skipped_arm(
            spec, ArmStatus.UNAVAILABLE, "the context has no muscle representation"
        )
    curve = cocontraction_variance_curve(
        muscle.frame, muscle.torque, muscle.index, muscle.config, muscle.model
    )
    record = curve.variance
    if record.allocation_dependence is AllocationDependence.UNAVAILABLE:
        return skipped_arm(spec, ArmStatus.UNAVAILABLE, record.reason or "unavailable")
    levels, var = curve.sweep.levels, record.values[:, 0]
    if spec.cocontraction_level is None:
        k = int(np.argmin(var))
    else:
        k = int(np.argmin(np.abs(levels - spec.cocontraction_level)))
        if not levels[0] <= spec.cocontraction_level <= levels[-1]:
            raise ValueError(
                f"cocontraction_level {spec.cocontraction_level} is outside the "
                f"sweep range [{levels[0]:.4g}, {levels[-1]:.4g}]"
            )
    flat = record.allocation_dependence is AllocationDependence.INVARIANT
    acts = curve.sweep.activations
    return ArmResult(
        strategy=spec.strategy,
        label=spec.label,
        status=ArmStatus.NO_MECHANISM if flat else ArmStatus.OK,
        reason="no declared mechanism makes the variance depend on co-contraction"
        if flat
        else "",
        face_variance=float(var[0] if flat else var[k]),
        baseline_variance=float(var[0]),
        chs_mps=context.chs_mps,
        effort=float(np.sum(acts[0 if flat else k] ** 2)),
        baseline_effort=float(np.sum(acts[0] ** 2)),
        effort_basis=EFFORT_ACTIVATION,
        certificate="linear-closed-form",
        budget=canonical_budget(curve.assessments[0 if flat else k].budget),
        mechanisms=record.hypotheses,
        hypotheses=tuple(dict.fromkeys((*record.hypotheses, *record.conditioning))),
        unverified_hypothesis_ids=unverified_ids(record.hypotheses),
        diagnostics={
            "level": float(levels[0 if flat else k]),
            "lowest_level": float(levels[0]),
        },
        record=record,
    )

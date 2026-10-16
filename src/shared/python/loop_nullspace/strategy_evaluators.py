"""Evaluators of the strategy catalogue: composition of the NSX dependencies.

:func:`evaluate_arms` is the entry point.  It looks the strategy up in the
catalogue and either runs the real evaluator the entry points to or returns an
arm that says ``NOT_IMPLEMENTED`` / ``UNAVAILABLE`` with the entry's reason and no
numbers (a strategy is never given a made-up effect).

Allocation strategies (this module) call the NSX-92 solves; the layer and
muscle strategies live in :mod:`.strategy_layer_eval`.

One mechanism at a time.  S1 and S5 both act through the allocation but through
different channels, so each is evaluated with the *other* channel switched off
and re-declared: S1 keeps the signal-dependent noise (M1) and freezes the
geometric preload at the nominal motion; S5 keeps the preload (M2) with additive
noise.  Their effects are therefore attributable, and are not additive when both
channels act together (that combined arm is the NSX-92 problem itself).
"""

from __future__ import annotations

import dataclasses
from typing import Any
from collections.abc import Sequence

from src.shared.python.loop_nullspace.allocation_spec import NotStrictlyConvexError
from src.shared.python.loop_nullspace.noise_aware_allocation import (
    minimise_face_variance,
)
from src.shared.python.loop_nullspace.noise_aware_front import (
    ChsCase,
    ChsPoint,
    chs_variance_front,
)
from src.shared.python.loop_nullspace.noise_aware_spec import (
    NoiseAwareResult,
    NoiseAwareSettings,
    NoiseAwareStatus,
)
from src.shared.python.loop_nullspace.strategy_arms import (
    AllocationContext,
    ArmResult,
    ArmSpec,
    ArmStatus,
    ComparisonContext,
    skipped_arm,
    typed_variance_record,
)
from src.shared.python.loop_nullspace.strategy_layer_eval import (
    evaluate_cocontraction_arm,
    evaluate_layer_arm,
)
from src.shared.python.loop_nullspace.strategy_spec import (
    EFFORT_ALLOCATION,
    Availability,
    StrategyId,
    entry_for,
)
from src.shared.python.loop_nullspace.trajectory_system import unverified_ids
from src.shared.python.perturbation.disturbance import DisturbanceSpec
from src.shared.python.loop_nullspace.variance_records import (
    Mechanism,
    VarianceDeclaration,
    declared_mechanisms,
    noise_hypotheses,
)

__all__ = [
    "evaluate_allocation_arm",
    "evaluate_arms",
    "evaluate_speed_accuracy",
]

_PART_NAMES = {
    "initial": "initial_state",
    "additive": "additive_noise",
    "signal_dependent": "signal_dependent",
}
_FAILURE = {
    NoiseAwareStatus.INFEASIBLE: ArmStatus.INFEASIBLE,
    NoiseAwareStatus.NOT_CONVERGED: ArmStatus.NOT_CONVERGED,
    NoiseAwareStatus.SOLVER_FAILURE: ArmStatus.NOT_CONVERGED,
}
_CERTIFICATE = {
    NoiseAwareStatus.OPTIMAL: "global-convex",
    NoiseAwareStatus.LOCAL_STATIONARY: "local",
    NoiseAwareStatus.OBJECTIVE_FLAT_INVARIANT: "flat-invariant",
}


def evaluate_arms(
    specs: Sequence[ArmSpec], context: ComparisonContext
) -> tuple[ArmResult, ...]:
    """Evaluate every arm in ``context`` (S8 expands to one arm per speed case).

    Postconditions: one result per spec (S8: per case) in order; a strategy
    without an evaluator comes back ``NOT_IMPLEMENTED``/``UNAVAILABLE`` with its
    catalogue reason; nothing is ever estimated for it.
    """
    if not isinstance(context, ComparisonContext):
        raise TypeError("context must be a ComparisonContext")
    out: list[ArmResult] = []
    for spec in specs:
        if not isinstance(spec, ArmSpec):
            raise TypeError("every spec must be an ArmSpec")
        out.extend(_dispatch(spec, context))
    return tuple(out)


def _dispatch(spec: ArmSpec, context: ComparisonContext) -> tuple[ArmResult, ...]:
    entry = entry_for(spec.strategy)
    if not entry.availability.implemented:
        reasons = "; ".join(r.reason for r in entry.representations)
        status = (
            ArmStatus.UNAVAILABLE
            if entry.availability is Availability.UNAVAILABLE
            else ArmStatus.NOT_IMPLEMENTED
        )
        return (skipped_arm(spec, status, f"{entry.id.value}: {reasons}"),)
    sid = spec.strategy
    if sid in (StrategyId.S1, StrategyId.S5):
        return (evaluate_allocation_arm(spec, context),)
    if sid is StrategyId.S8:
        return evaluate_speed_accuracy(spec, context)
    if sid is StrategyId.S2 and spec.cocontraction:
        return (evaluate_cocontraction_arm(spec, context),)
    return (evaluate_layer_arm(spec, context),)


# ----------------------------------------------------------- allocation arms
def _declaration(
    base: VarianceDeclaration, model_spec: DisturbanceSpec, *, preload: bool
) -> VarianceDeclaration:
    specs = [model_spec]
    return dataclasses.replace(
        base,
        mechanisms=declared_mechanisms(specs, geometric_preload=preload),
        noise_hypotheses=noise_hypotheses(specs),
    )


def _isolate(
    alloc: AllocationContext, strategy: StrategyId
) -> tuple[AllocationContext, str]:
    """The context with the other strategy's channel switched off, and its id."""
    model = alloc.model
    if strategy is StrategyId.S1:  # keep M1, freeze the preload
        model = dataclasses.replace(model, a_slope=None)
        decl = _declaration(alloc.declaration, model.spec, preload=False)
        return dataclasses.replace(alloc, model=model, declaration=decl), (
            Mechanism.SIGNAL_DEPENDENT_NOISE.value
        )
    plain = dataclasses.replace(model.spec, signal_gain=None, signal_bounds=None)
    model = dataclasses.replace(model, spec=plain)  # keep M2, additive noise
    decl = _declaration(alloc.declaration, plain, preload=model.preload_active)
    return dataclasses.replace(alloc, model=model, declaration=decl), (
        Mechanism.GEOMETRIC_PRELOAD.value
    )


def _solve(alloc: AllocationContext, cap: float | None) -> NoiseAwareResult | str:
    settings = NoiseAwareSettings(effort_cap=cap) if cap is not None else None
    try:
        return minimise_face_variance(
            alloc.feasible_sets,
            alloc.times_s,
            alloc.torque_scale,
            alloc.model,
            alloc.declaration,
            settings,
        )
    except NotStrictlyConvexError as exc:
        return f"variance not strictly convex on the retained torque space: {exc}"


def _allocation_result(
    spec: ArmSpec,
    context: ComparisonContext,
    res: NoiseAwareResult,
    alloc: AllocationContext,
    channel: str,
) -> ArmResult:
    common: dict[str, Any] = {
        "strategy": spec.strategy,
        "label": spec.label,
        "chs_mps": context.chs_mps,
        "effort_basis": EFFORT_ALLOCATION,
        "mechanisms": res.mechanisms,
        "hypotheses": res.hypotheses,
        "unverified_hypothesis_ids": res.unverified_hypothesis_ids,
    }
    if res.status in _FAILURE or res.variance is None:
        status = _FAILURE.get(res.status, ArmStatus.NOT_CONVERGED)
        return ArmResult(status=status, reason=res.reason or res.status.value, **common)
    flat = res.status is NoiseAwareStatus.OBJECTIVE_FLAT_INVARIANT
    baseline = float(res.reference_variance or res.variance)
    return ArmResult(
        status=ArmStatus.NO_MECHANISM if flat else ArmStatus.OK,
        reason=(
            f"no declared mechanism ({channel}) couples the allocation to the "
            f"variance; {res.reason}"
            if flat
            else ""
        ),
        face_variance=baseline if flat else float(res.variance),
        baseline_variance=baseline,
        effort=res.effort,
        baseline_effort=res.reference_effort,
        certificate=_CERTIFICATE[res.status],
        budget={_PART_NAMES[k]: v for k, v in res.variance_parts.items()},
        record=res.record,
        **common,
    )


def evaluate_allocation_arm(spec: ArmSpec, context: ComparisonContext) -> ArmResult:
    """S1 (null-space redistribution via M1) or S5 (preload channel, M2).

    Preconditions: an allocation context (else ``UNAVAILABLE``).  Postconditions:
    ``OK`` carries the NSX-92 optimum under ``spec.effort_cap`` (matched effort:
    effort never exceeds the cap, an unreachable cap is ``INFEASIBLE``, never
    relaxed); ``NO_MECHANISM`` (flat objective, rule V1) reports ``ratio == 1``;
    the certificate is ``global-convex`` (M1) or ``local`` (M2).
    """
    if context.allocation is None:
        return skipped_arm(
            spec, ArmStatus.UNAVAILABLE, "the context has no allocation representation"
        )
    alloc, channel = _isolate(context.allocation, spec.strategy)
    out = _solve(alloc, spec.effort_cap)
    if isinstance(out, str):
        return ArmResult(
            spec.strategy,
            spec.label,
            ArmStatus.NOT_CONVERGED,
            out,
            chs_mps=context.chs_mps,
            effort_basis=EFFORT_ALLOCATION,
        )
    return _allocation_result(spec, context, out, alloc, channel)


# ------------------------------------------------------------ speed-accuracy
def _case_arm(
    spec: ArmSpec,
    point: ChsPoint,
    baseline: tuple[float, float],
    alloc: AllocationContext,
) -> ArmResult:
    decl = alloc.declaration
    common: dict[str, Any] = {
        "strategy": spec.strategy,
        "label": f"{spec.label}:{point.label}",
        "chs_mps": point.chs_mps,
        "effort_basis": EFFORT_ALLOCATION,
        "hypotheses": tuple(decl.ids()) + tuple(decl.noise_hypotheses),
    }
    if point.variance is None:
        status = _FAILURE.get(point.status, ArmStatus.NOT_CONVERGED)
        return ArmResult(status=status, reason=point.reason, **common)
    hyps = common["hypotheses"]
    return ArmResult(
        status=ArmStatus.OK,
        face_variance=float(point.variance),
        baseline_variance=baseline[0],
        effort=point.effort,
        baseline_effort=baseline[1],
        certificate=point.certificate,
        mechanisms=tuple(decl.ids()),
        unverified_hypothesis_ids=unverified_ids(hyps),
        record=typed_variance_record(point.variance, "rad^2", decl),
        **common,
    )


def evaluate_speed_accuracy(
    spec: ArmSpec, context: ComparisonContext
) -> tuple[ArmResult, ...]:
    """S8: one arm per speed case from the NSX-92 CHS front at a common effort cap.

    Every arm is judged against the *reference case's* strategy-off baseline, so a
    slower swing that is quieter shows a ratio below one: that is the tempting
    shortcut the matched-speed comparison exists to refuse.

    Preconditions (``ValueError``): an allocation context (else ``UNAVAILABLE``),
    at least two cases, ``effort_cap`` and a ``reference_case`` that names a case.
    """
    alloc = context.allocation
    if alloc is None:
        return (
            skipped_arm(
                spec,
                ArmStatus.UNAVAILABLE,
                "the context has no allocation representation",
            ),
        )
    cases: tuple[ChsCase, ...] = spec.speed_cases
    if len(cases) < 2:
        raise ValueError("S8 needs at least two speed_cases")
    if spec.effort_cap is None:
        raise ValueError("S8 needs an effort_cap (the matched effort of the front)")
    ref = next((c for c in cases if c.label == spec.reference_case), None)
    if ref is None:
        raise ValueError(
            f"reference_case {spec.reference_case!r} is not one of "
            f"{[c.label for c in cases]}"
        )
    ref_ctx = dataclasses.replace(
        alloc, model=ref.model, feasible_sets=ref.feasible_sets
    )
    base = _solve(ref_ctx, None)
    if isinstance(base, str) or base.reference_variance is None:
        reason = base if isinstance(base, str) else base.reason
        return tuple(
            ArmResult(
                spec.strategy,
                f"{spec.label}:{c.label}",
                ArmStatus.INFEASIBLE,
                f"reference case has no baseline: {reason}",
                chs_mps=c.chs_mps,
                effort_basis=EFFORT_ALLOCATION,
            )
            for c in cases
        )
    front = chs_variance_front(
        cases, alloc.times_s, alloc.torque_scale, alloc.declaration, spec.effort_cap
    )
    baseline = (float(base.reference_variance), float(base.reference_effort or 0.0))
    return tuple(_case_arm(spec, p, baseline, alloc) for p in front.points)

"""The sensitivity stage of ``facade.analyze`` (NSX-99, #12728).

One computation path for every surface: the PyQt tabs and the React panel draw
the ``sensitivity``, ``variance_budget`` and ``strategies`` sections this module
builds, through the plot model, and compute nothing themselves.

There is **no new mathematics** here.  The stage composes

* NSX-90 ``assess_variance`` (adjoint kernels, variance, budget, nonlinearity
  gate) for the primary outcome's kernel heatmap, variance and budget;
* NSX-95 ``evaluate_arms`` / ``run_comparison`` for the strategy comparison; and
* optionally an NSX-91 paired Monte Carlo summary, compared with the linear
  prediction by ``cross_check_linear``.

What the engine knows (the linearised dynamics, noise covariances built from the
declared disturbances, comparison contexts) enters through a
:class:`SensitivityProvider` on the ``EngineContext``; the stage never invents
those inputs.  The stage declares which strategies and outcome it was asked for
and reports, with its reason, every one it could not evaluate.

Cancellation: ``check()`` (the facade's cancel check) runs between sub-steps; a
cancelled stage raises before any result is returned, so nothing partial reaches
the cache.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.paired_mc_spec import CampaignSpec
from src.shared.python.loop_nullspace.paired_statistics import (
    RecordProvenance,
    StrategySummary,
    cross_check_linear,
    sd_records,
)
from src.shared.python.loop_nullspace.report_serialization import (
    json_float,
    quantity_dict,
)
from src.shared.python.loop_nullspace.sensitivity_identity import assumption_digest
from src.shared.python.loop_nullspace.sensitivity_options import SensitivitySpec
from src.shared.python.loop_nullspace.sensitivity_sections import (
    arm_sd_dict,
    budget_records,
    comparison_dict,
    primary_row,
    sd_record,
)
from src.shared.python.loop_nullspace.strategy_arms import (
    ArmResult,
    ArmSpec,
    ComparisonContext,
)
from src.shared.python.loop_nullspace.strategy_campaign import (
    ComparisonMode,
    MatchSettings,
    run_comparison,
)
from src.shared.python.loop_nullspace.strategy_evaluators import evaluate_arms
from src.shared.python.loop_nullspace.strategy_spec import (
    NON_CLAIM as STRATEGY_NON_CLAIM,
)
from src.shared.python.loop_nullspace.strategy_spec import StrategyId, entry_for
from src.shared.python.loop_nullspace.variance_propagation import VarianceProblem
from src.shared.python.loop_nullspace.variance_records import (
    NON_CLAIMS as VARIANCE_NON_CLAIMS,
)
from src.shared.python.loop_nullspace.variance_records import (
    GateSpec,
    VarianceAssessment,
    VarianceDeclaration,
    assess_variance,
    noise_hypotheses,
)

__all__ = [
    "McEvidence",
    "SensitivityInputs",
    "SensitivityProvider",
    "SensitivityRequest",
    "SensitivitySections",
    "run_sensitivity",
]

CheckFn = Callable[[], None]
STRATEGY_OUTCOME = "face_angle"
NON_CLAIMS = (
    VARIANCE_NON_CLAIMS,
    STRATEGY_NON_CLAIM,
    "Disturbance amplitudes, mechanisms and impedance values are declared "
    "hypotheses, not measurements of any golfer.",
)


@dataclass(frozen=True)
class SensitivityRequest:
    """What a provider is told: the declaration and the analysed frames.

    ``feasible_sets`` is empty when the envelope stage did not run.
    """

    spec: SensitivitySpec
    preset: str
    engine_name: str
    snapshots: tuple[ClosedLoopSnapshot, ...]
    times_s: tuple[float, ...]
    frame_indices: tuple[int, ...]
    feasible_sets: tuple[FeasibleSet, ...] = ()


@dataclass(frozen=True)
class McEvidence:
    """A paired-Monte-Carlo summary to set beside the linear prediction."""

    summary: StrategySummary
    spec: CampaignSpec
    provenance: RecordProvenance

    def outcome_names(self) -> tuple[str, ...]:
        """Names of the campaign's outcome channels, in order."""
        return tuple(channel.name for channel in self.spec.outcomes)


@dataclass(frozen=True)
class SensitivityInputs:
    """Everything the engine supplies for one sensitivity run.

    ``provenance`` must identify the data and settings behind the payload that
    is too bulky to hash (feasible sets, muscle models): it enters the cache
    identity.  ``contexts`` need ``match``; ``arm_specs`` declare the setting of
    each implemented strategy (a requested implemented strategy without one is
    reported ``not attempted``).

    Raises:
        TypeError: A field of the wrong kind.
        ValueError: Inconsistent shapes or names, empty units or provenance,
            contexts without comparison settings, or duplicate context labels.
    """

    problem: VarianceProblem
    declaration: VarianceDeclaration
    outcome_names: tuple[str, ...]
    outcome_units: str
    input_labels: tuple[str, ...]
    input_units: str
    step_times_s: tuple[float, ...]
    provenance: Mapping[str, Any]
    gate: GateSpec | None = None
    contexts: tuple[ComparisonContext, ...] = ()
    arm_specs: tuple[ArmSpec, ...] = ()
    match: MatchSettings | None = None
    mode: ComparisonMode = ComparisonMode.UNCONSTRAINED
    mc: McEvidence | None = None
    _n: tuple[int, int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, VarianceProblem):
            raise TypeError("problem must be a VarianceProblem")
        if not isinstance(self.declaration, VarianceDeclaration):
            raise TypeError("declaration must be a VarianceDeclaration")
        if not isinstance(self.mode, ComparisonMode):
            raise TypeError("mode must be a ComparisonMode")
        n_out = int(np.shape(self.problem.outcome_rows)[0])
        n_steps, n_w = (int(x) for x in np.shape(self.problem.step_covariance)[:2])
        for name, got, want in (
            ("outcome_names", len(self.outcome_names), n_out),
            ("input_labels", len(self.input_labels), n_w),
            ("step_times_s", len(self.step_times_s), n_steps),
        ):
            if got != want:
                raise ValueError(f"{name} needs {want} entries, got {got}")
        if len(set(self.outcome_names)) != n_out or not all(self.outcome_names):
            raise ValueError("outcome_names must be unique and non-empty")
        if not (self.outcome_units and self.input_units):
            raise ValueError("outcome_units and input_units must be non-empty")
        if not isinstance(self.provenance, Mapping) or len(self.provenance) == 0:
            raise ValueError("provenance must name the data behind the inputs")
        if (self.contexts or self.arm_specs) and self.match is None:
            raise ValueError("contexts and arm specs need comparison MatchSettings")
        labels = [c.label for c in self.contexts]
        if len(set(labels)) != len(labels):
            raise ValueError(f"context labels must be unique, got {labels}")
        if self.mc is not None and self.mc.outcome_names() != tuple(self.outcome_names):
            raise ValueError("the Monte Carlo outcomes must match outcome_names")
        object.__setattr__(self, "_n", (n_out, n_steps, n_w))
        object.__setattr__(self, "contexts", tuple(self.contexts))
        object.__setattr__(self, "arm_specs", tuple(self.arm_specs))


SensitivityProvider = Callable[[SensitivityRequest], SensitivityInputs]


@dataclass(frozen=True)
class SensitivitySections:
    """The three report sections; the last two are ``None`` if not computed."""

    sensitivity: dict[str, Any]
    variance_budget: dict[str, Any] | None
    strategies: dict[str, Any] | None


# ---------------------------------------------------------------- validation
def _require_consistent(spec: SensitivitySpec, inputs: SensitivityInputs) -> None:
    """The engine's variance model must be the one that was declared."""
    if inputs.declaration.mechanisms != spec.mechanisms:
        raise ValueError(
            "the engine's variance declaration disagrees with the declared "
            f"mechanisms: {sorted(m.value for m in inputs.declaration.mechanisms)} "
            f"versus {sorted(m.value for m in spec.mechanisms)}"
        )
    declared = noise_hypotheses(spec.disturbances)
    if tuple(sorted(inputs.declaration.noise_hypotheses)) != declared:
        raise ValueError(
            "the engine's noise hypotheses disagree with the declared "
            f"disturbances: {inputs.declaration.noise_hypotheses} versus {declared}"
        )


def _identity_parts(spec: SensitivitySpec, inputs: SensitivityInputs) -> dict[str, Any]:
    return {
        "options": spec.as_dict(),
        "problem": inputs.problem,
        "declaration": inputs.declaration,
        "outcomes": [inputs.outcome_names, inputs.outcome_units],
        "inputs": [inputs.input_labels, inputs.input_units, inputs.step_times_s],
        "gate": inputs.gate,
        "match": inputs.match,
        "mode": inputs.mode,
        "arm_specs": inputs.arm_specs,
        "contexts": [
            [c.label, c.chs_mps, c.allocation is not None, c.layer is not None]
            + [c.muscle is not None]
            for c in inputs.contexts
        ],
        "mc": None if inputs.mc is None else [inputs.mc.spec, inputs.mc.provenance],
        "provenance": inputs.provenance,
    }


def _primary(
    spec: SensitivitySpec, inputs: SensitivityInputs
) -> tuple[str, int, list[str]] | None:
    """First prioritised outcome the engine produces, plus those it skipped."""
    skipped: list[str] = []
    for name in spec.outcome_priority:
        if name in inputs.outcome_names:
            return name, inputs.outcome_names.index(name), skipped
        skipped.append(name)
    return None


# ------------------------------------------------------------------- pieces
def _variance_blocks(
    inputs: SensitivityInputs, index: int, name: str
) -> tuple[dict[str, Any], dict[str, Any], VarianceAssessment]:
    """(sensitivity base, variance_budget section, assessment) for the primary."""
    assessment = assess_variance(
        inputs.problem,
        inputs.declaration,
        outcome_units=inputs.outcome_units,
        input_units=inputs.input_units,
        gate=inputs.gate,
    )
    variance = primary_row(assessment.variance, 0, index)
    kernel = primary_row(assessment.kernel_input, 1, index)
    budget = assessment.budget
    classes = list(budget.contributions)
    noise = [c for c in classes if c in budget.by_time]
    parts = np.array([budget.contributions[c][index] for c in classes])
    by_time = np.array([budget.by_time[c][:, index] for c in noise])
    records = budget_records(
        variance, classes, parts, by_time.reshape(len(noise), -1), float(parts.sum())
    )
    section = {
        "status": "complete",
        "reason": None,
        "outcome": name,
        "units": variance.units,
        "classes": classes,
        "noise_classes": noise,
        "step_times_s": [json_float(t) for t in inputs.step_times_s],
        "parts": quantity_dict(records["parts"]),
        "shares": quantity_dict(records["shares"]),
        "by_time": quantity_dict(records["by_time"]),
    }
    sd = sd_record(variance, inputs.outcome_units)
    base = {
        "kernel": {"outcome": name, "quantity": quantity_dict(kernel)},
        "variance": quantity_dict(variance),
        "sd": quantity_dict(sd),
    }
    return base, section, assessment


def _mc_block(
    inputs: SensitivityInputs, assessment: VarianceAssessment
) -> list[dict[str, Any]]:
    mc = inputs.mc
    if mc is None:
        return []
    checks = cross_check_linear(
        mc.summary,
        mc.spec,
        inputs.problem.forward_outcome_covariance(),
        gate=assessment.nonlinearity,
    )
    records = sd_records(mc.summary, mc.spec, mc.provenance)
    out = []
    for check in checks:
        rec = records.get(check.outcome)
        out.append(
            {
                "outcome": check.outcome,
                "status": check.status.value,
                "predicted_sd": json_float(check.predicted_sd),
                "mc_sd": json_float(check.mc_sd),
                "mc_sd_ci": [json_float(x) for x in check.mc_sd_ci],
                "gap": json_float(check.gap),
                "failure_rate": json_float(check.failure_rate),
                "mc_sd_quantity": None if rec is None else quantity_dict(rec),
            }
        )
    return out


def _stub(strategy: StrategyId) -> ArmSpec:
    return ArmSpec(strategy=strategy, label=strategy.value)


def _evaluate(
    spec: SensitivitySpec, inputs: SensitivityInputs, check: CheckFn
) -> tuple[list[ArmResult], list[dict[str, str]]]:
    """Every requested arm in every context; the rest with a reason."""
    wanted = set(spec.strategies)
    declared = [a for a in inputs.arm_specs if a.strategy in wanted]
    have = {a.strategy for a in declared}
    not_run: list[dict[str, str]] = []
    skipped_specs: list[ArmSpec] = []
    for sid in spec.strategies:
        if entry_for(sid).availability.implemented and sid not in have:
            not_run.append(
                {
                    "label": sid.value,
                    "reason": "the engine's sensitivity provider declared no arm "
                    "setting for this strategy",
                }
            )
        elif not entry_for(sid).availability.implemented and sid not in have:
            skipped_specs.append(_stub(sid))
    arms: list[ArmResult] = []
    first = inputs.contexts[0]
    for ctx in inputs.contexts:
        check()
        specs = [a for a in declared if entry_for(a.strategy).availability.implemented]
        arms += _prefixed(ctx, evaluate_arms(specs, ctx))
    check()
    dormant = [
        a for a in declared if not entry_for(a.strategy).availability.implemented
    ]
    arms += _prefixed(first, evaluate_arms([*dormant, *skipped_specs], first))
    return arms, not_run


def _prefixed(ctx: ComparisonContext, arms: Sequence[ArmResult]) -> list[ArmResult]:
    return [dataclasses.replace(a, label=f"{ctx.label}/{a.label}") for a in arms]


def _strategies_section(
    spec: SensitivitySpec,
    inputs: SensitivityInputs,
    primary: str,
    check: CheckFn,
) -> tuple[dict[str, Any], list[ArmResult]]:
    head = {"strategy_set": [s.value for s in spec.strategies]}
    why = ""
    if len(spec.strategies) == 0:
        why = "no strategy was requested"
    elif not inputs.contexts:
        why = "the engine supplied no comparison context"
    elif primary != STRATEGY_OUTCOME:
        why = (
            "strategy comparisons rank the face-angle variance only; the primary "
            f"outcome is {primary!r}"
        )
    if why:
        return {"status": "not_run", "reason": why, **head}, []
    arms, not_run = _evaluate(spec, inputs, check)
    assert inputs.match is not None
    report = run_comparison(arms, inputs.mode, inputs.match)
    body = comparison_dict(report)
    body["contexts"] = [c.label for c in inputs.contexts]
    body["not_attempted"] = [
        *(
            {"label": lab, "reason": _reason(arms, lab)}
            for lab in body["not_attempted"]
        ),
        *not_run,
    ]
    return {"status": "complete", "reason": None, **head, **body}, arms


def _reason(arms: Sequence[ArmResult], label: str) -> str:
    for arm in arms:
        if arm.label == label:
            return arm.reason
    return ""


# -------------------------------------------------------------------- entry
def run_sensitivity(
    spec: SensitivitySpec, inputs: SensitivityInputs, check: CheckFn
) -> SensitivitySections:
    """Variance, budget and strategy comparison for the declared sensitivity.

    Preconditions (``TypeError``/``ValueError``): typed arguments; the engine's
    variance declaration has exactly the declared mechanisms and noise
    hypotheses.  Postconditions: every number is a ``QuantityRecord`` form with
    its evidence axes; the primary outcome is the first prioritised outcome the
    engine produces (face angle first), and if none is produced the section is
    ``unavailable`` with the reason and no number; strategies that cannot be
    evaluated appear with their reason and no number; ``check()`` ran between
    sub-steps, so a cancel raises before anything is returned.
    """
    if not isinstance(spec, SensitivitySpec):
        raise TypeError("spec must be a SensitivitySpec")
    if not isinstance(inputs, SensitivityInputs):
        raise TypeError("inputs must be a SensitivityInputs")
    _require_consistent(spec, inputs)
    digest = assumption_digest(_identity_parts(spec, inputs))
    check()
    head: dict[str, Any] = {
        "options": spec.as_dict(),
        "cache_identity": digest,
        "outcomes": list(inputs.outcome_names),
        "outcome_priority": list(spec.outcome_priority),
        "provenance": dict(inputs.provenance),
        "non_claims": list(NON_CLAIMS),
    }
    chosen = _primary(spec, inputs)
    if chosen is None:
        reason = (
            f"none of the prioritised outcomes {list(spec.outcome_priority)} is "
            f"produced by the engine ({list(inputs.outcome_names)})"
        )
        return SensitivitySections(
            {"status": "unavailable", "reason": reason, **head}, None, None
        )
    name, index, skipped = chosen
    base, budget, assessment = _variance_blocks(inputs, index, name)
    check()
    strategies, arms = _strategies_section(spec, inputs, name, check)
    dispersion = {
        "outcome": name,
        "linear": {"variance": base["variance"], "sd": base["sd"]},
        "arms": [arm_sd_dict(a) for a in arms if a.record is not None],
        "mc": _mc_block(inputs, assessment),
    }
    check()
    sensitivity = {
        "status": "complete",
        "reason": None,
        "primary_outcome": name,
        "skipped_priority": skipped,
        "outcome_units": inputs.outcome_units,
        "input_labels": list(inputs.input_labels),
        "input_units": inputs.input_units,
        "step_times_s": [json_float(t) for t in inputs.step_times_s],
        "declared_mechanisms": [m.value for m in sorted(spec.mechanisms, key=str)],
        "noise_hypotheses": list(inputs.declaration.noise_hypotheses),
        "linear_validity": assessment.linear_validity.value,
        "gate": _gate_dict(inputs.gate),
        "kernel": base["kernel"],
        "face_dispersion": dispersion,
        **head,
    }
    return SensitivitySections(sensitivity, budget, strategies)


def _gate_dict(gate: GateSpec | None) -> dict[str, Any] | None:
    if gate is None:
        return None
    return {
        "amplitude": json_float(gate.amplitude),
        "gate_threshold": json_float(gate.gate_threshold),
        "response_floor": json_float(gate.response_floor),
        "include_axes": bool(gate.include_axes),
    }

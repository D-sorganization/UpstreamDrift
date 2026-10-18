"""Sensitivity panels of the NSX plot model (NSX-99, #12728).

Four panels beside the NSX-42 ones, drawn by both UIs from the same JSON and
computing nothing:

``kernel_heatmap``
    Kernel of the primary outcome (face angle first), input (joint) by time.
    Band first: ``lower``/``upper`` per cell.  An allocation-invariant kernel is
    a degenerate band (``is_band`` false); a family-dependent one (geometric
    preload or muscle impedance declared) is shown only if the report carries a
    family band for it -- a single allocation is never presented as the family.
``variance_budget``
    Variance of the primary outcome by input class, with shares and per-step
    parts for the noise classes.
``strategy_pareto``
    Admitted arms as points (variance ratio to the strategy-off baseline, clubhead
    speed, effort in its own basis, rank), the non-dominated fronts of the shown
    points, and every arm that has no point -- excluded, failed, unavailable or
    not attempted -- with its reason.
``face_dispersion``
    The linear face-angle standard deviation, the per-arm ones and, if the engine
    supplied a paired Monte Carlo, its cross-check against the linear prediction.

Display decisions are obeyed, not remade.  :func:`sensitivity_quantities` lists
the report quantities and the keys under which ``display_decision`` is asked
(``plot_model_decisions`` asks it once per key).  A quantity whose decision does
not admit what the panel draws is withheld with the decision's reason and no
number; a missing decision withholds by name.  An ``UNAVAILABLE`` or
``FAMILY_DEPENDENT`` series with an unverified hypothesis is therefore never
drawn, because ``display_decision`` answers ``WITHHOLD`` for it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.shared.python.loop_nullspace.output_policy_integration import (
    DisplayAction,
    DisplayDecision,
)
from src.shared.python.loop_nullspace.plot_model import (
    _SHOW_BAND,
    DecisionView,
    _Decisions,
    _num,
    decision_key,
)

__all__ = [
    "BudgetPart",
    "DispersionEntry",
    "DispersionPanel",
    "KernelHeatmap",
    "McCheck",
    "ParetoFront",
    "ParetoPoint",
    "SensitivityPanels",
    "SkippedArm",
    "StrategyParetoPanel",
    "VarianceBudgetPanel",
    "build_sensitivity_panels",
    "sensitivity_quantities",
]


def _shown_state(
    shown: bool, decision: DecisionView | None, reason: str, numbers: bool, what: str
) -> None:
    """Shared invariant: numbers only when shown, under an admitting decision."""
    if shown and not numbers:
        raise ValueError(f"{what}: a shown item needs numbers")
    if not shown and (numbers or not reason):
        raise ValueError(f"{what}: an unshown item carries a reason and no numbers")
    if shown and not (
        decision and decision.admits(DisplayAction.SHOW_VALUE, *_SHOW_BAND)
    ):
        raise ValueError(f"{what}: shown without an admitting decision")


@dataclass(frozen=True)
class KernelHeatmap:
    """Kernel of the primary outcome, input (joint) by time, band first.

    ``lower`` and ``upper`` are ``(n_steps, n_inputs)``; they coincide
    (``is_band`` false) for an allocation-invariant kernel.  A family-dependent
    kernel is shown only when the report carries its band, never as one point.
    """

    outcome: str | None
    units: str
    shown: bool
    input_labels: tuple[str, ...]
    step_times_s: tuple[float | None, ...]
    lower: tuple[tuple[float | None, ...], ...]
    upper: tuple[tuple[float | None, ...], ...]
    is_band: bool
    decision: DecisionView | None
    reason: str

    def __post_init__(self) -> None:
        _shown_state(self.shown, self.decision, self.reason, bool(self.lower), "kernel")
        if self.shown and (
            len(self.lower) != len(self.step_times_s)
            or len(self.upper) != len(self.lower)
            or any(len(r) != len(self.input_labels) for r in (*self.lower, *self.upper))
        ):
            raise ValueError("kernel: lower/upper must be (steps, inputs)")


@dataclass(frozen=True)
class BudgetPart:
    """One input class: its variance, its share and (noise classes) per-step parts."""

    input_class: str
    variance: float | None
    share: float | None
    by_time: tuple[float | None, ...]


@dataclass(frozen=True)
class VarianceBudgetPanel:
    """Variance of the primary outcome by input class."""

    outcome: str | None
    units: str
    shown: bool
    step_times_s: tuple[float | None, ...]
    parts: tuple[BudgetPart, ...]
    decision: DecisionView | None
    reason: str

    def __post_init__(self) -> None:
        _shown_state(self.shown, self.decision, self.reason, bool(self.parts), "budget")
        if (
            self.shown
            and self.decision
            and not self.decision.admits(DisplayAction.SHOW_VALUE)
        ):
            raise ValueError("budget: parts are shown only under a SHOW_VALUE decision")


@dataclass(frozen=True)
class ParetoPoint:
    """An admitted arm; numbers (ratio, speed, effort, rank) only when shown."""

    label: str
    strategy: str
    status: str
    shown: bool
    rank: int | None
    beats_baseline: bool
    variance_ratio: float | None
    chs_mps: float | None
    effort: float | None
    effort_basis: str
    mechanisms: tuple[str, ...]
    decision: DecisionView | None
    reason: str

    def __post_init__(self) -> None:
        numbers = self.variance_ratio is not None
        if self.shown and not numbers:
            raise ValueError(f"point {self.label!r}: a shown point needs its ratio")
        if not self.shown and (
            numbers or self.rank is not None or self.chs_mps is not None
            or self.effort is not None or not self.reason
        ):  # fmt: skip
            raise ValueError(
                f"point {self.label!r}: withheld means a reason, no numbers"
            )
        if self.shown and not (
            self.decision and self.decision.admits(DisplayAction.SHOW_VALUE)
        ):
            raise ValueError(f"point {self.label!r}: shown without SHOW_VALUE")


@dataclass(frozen=True)
class ParetoFront:
    """Non-dominated shown arms; withheld arms are dropped and only counted."""

    name: str
    labels: tuple[str, ...]
    withheld_count: int


@dataclass(frozen=True)
class SkippedArm:
    """An arm with no point: excluded, failed, unavailable or not attempted."""

    label: str
    status: str
    exclusion: str | None
    reason: str


@dataclass(frozen=True)
class StrategyParetoPanel:
    """Strategy comparison: ranked shown points, fronts and every skipped arm."""

    mode: str | None
    ranking_basis: str | None
    points: tuple[ParetoPoint, ...]
    fronts: tuple[ParetoFront, ...]
    skipped: tuple[SkippedArm, ...]
    failure_rate: float | None
    winner: str | None
    assumptions: tuple[str, ...]
    non_claim: str | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        shown = {p.label for p in self.points if p.shown}
        if self.winner is not None and self.winner not in shown:
            raise ValueError("the winner must be a shown point")
        for front in self.fronts:
            if not set(front.labels) <= shown:
                raise ValueError(
                    f"front {front.name!r} names a point that is not shown"
                )


@dataclass(frozen=True)
class DispersionEntry:
    """A face-angle standard deviation (``label`` names its source), band first."""

    label: str
    units: str
    shown: bool
    lower: float | None
    upper: float | None
    is_band: bool
    decision: DecisionView | None
    reason: str

    def __post_init__(self) -> None:
        both = self.lower is not None and self.upper is not None
        _shown_state(self.shown, self.decision, self.reason, both, f"sd {self.label!r}")


@dataclass(frozen=True)
class McCheck:
    """Monte Carlo SD set beside the linear prediction (both must be shown)."""

    outcome: str
    status: str
    shown: bool
    predicted_sd: float | None
    mc_sd: float | None
    mc_sd_ci: tuple[float | None, ...]
    gap: float | None
    failure_rate: float | None
    decision: DecisionView | None
    reason: str

    def __post_init__(self) -> None:
        numbers = self.predicted_sd is not None and self.mc_sd is not None
        _shown_state(self.shown, self.decision, self.reason, numbers, "mc check")


@dataclass(frozen=True)
class DispersionPanel:
    """Linear face-angle SD, per-arm SDs and the Monte Carlo cross-check."""

    outcome: str | None
    units: str
    linear: DispersionEntry | None
    arms: tuple[DispersionEntry, ...]
    mc: tuple[McCheck, ...]
    unavailable_reason: str | None


@dataclass(frozen=True)
class SensitivityPanels:
    """The four sensitivity panels with the stage's declarations (report 2.0.0)."""

    status: str
    reason: str | None
    primary_outcome: str | None
    skipped_priority: tuple[str, ...]
    outcomes: tuple[str, ...]
    declared_mechanisms: tuple[str, ...]
    noise_hypotheses: tuple[str, ...]
    linear_validity: str | None
    cache_identity: str | None
    non_claims: tuple[str, ...]
    kernel_heatmap: KernelHeatmap
    variance_budget: VarianceBudgetPanel
    strategy_pareto: StrategyParetoPanel
    face_dispersion: DispersionPanel


# ---------------------------------------------------------------- the asks
def sensitivity_quantities(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Decision key -> report quantity, for every quantity the panels may draw."""
    sens = report.get("sensitivity")
    if not sens or sens.get("status") != "complete":
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    kernel = sens["kernel"]
    out[decision_key("kernel_heatmap", kernel["outcome"])] = kernel["quantity"]
    budget = report.get("variance_budget")
    if budget and budget.get("status") == "complete":
        out[decision_key("variance_budget", "variance")] = budget["parts"]
    strat = report.get("strategies")
    if strat and strat.get("status") == "complete":
        for arm in strat["arms"]:
            if arm["admitted"] and arm["ratio"] is not None:
                out[decision_key("strategy_pareto", arm["label"])] = arm["ratio"]
    disp = sens["face_dispersion"]
    out[decision_key("face_dispersion", "linear")] = disp["linear"]["sd"]
    for item in disp["arms"]:
        if item["sd"] is not None:
            out[decision_key("face_dispersion", item["label"])] = item["sd"]
    for check in disp["mc"]:
        if check["mc_sd_quantity"] is not None:
            key = decision_key("face_dispersion", f"mc:{check['outcome']}")
            out[key] = check["mc_sd_quantity"]
    return out


# ------------------------------------------------------------------ admit
@dataclass(frozen=True)
class _Got:
    """What a decision lets a panel draw (or why nothing)."""

    view: DecisionView | None
    lower: Any
    upper: Any
    is_band: bool
    reason: str

    @property
    def shown(self) -> bool:
        return not self.reason


def _admit(
    quantity: Mapping[str, Any],
    decision: DisplayDecision | None,
    key: str,
    *,
    allow_band: bool = True,
) -> _Got:
    """Apply ``decision`` to ``quantity``: values, a family band, or a reason."""
    if decision is None:
        return _Got(None, None, None, False, f"no display decision for {key}")
    view = DecisionView.of(decision)
    values = quantity["values"]
    if decision.action is DisplayAction.SHOW_VALUE:
        if values is None:
            return _Got(view, None, None, False, "the report carries no values")
        return _Got(view, values, values, False, "")
    if decision.action in _SHOW_BAND:
        band = quantity.get("band")
        if not allow_band:
            why = "this panel shows values only, not a family band"
        elif band is None:
            why = "the report carries no family band (one allocation is not a band)"
        else:
            return _Got(view, band["lower"], band["upper"], True, "")
        return _Got(view, None, None, False, f"{decision.reason}; {why}")
    return _Got(view, None, None, False, decision.reason)


def _grid(rows: Sequence[Sequence[Any]]) -> tuple[tuple[float | None, ...], ...]:
    return tuple(tuple(_num(x) for x in row) for row in rows)


def _first(value: Any) -> float | None:
    while isinstance(value, list | tuple):
        value = value[0]
    return _num(value)


def _times(raw: Sequence[Any]) -> tuple[float | None, ...]:
    return tuple(_num(t) for t in raw)


# ----------------------------------------------------------------- panels
def _kernel(sens: Mapping[str, Any], decs: _Decisions) -> KernelHeatmap:
    kernel = sens["kernel"]
    key = decision_key("kernel_heatmap", kernel["outcome"])
    got = _admit(kernel["quantity"], decs.get(key), key)
    units = str(kernel["quantity"]["units"])
    base: dict[str, Any] = {
        "outcome": kernel["outcome"],
        "units": units,
        "input_labels": tuple(sens["input_labels"]),
        "step_times_s": _times(sens["step_times_s"]),
        "decision": got.view,
    }
    if not got.shown:
        return KernelHeatmap(
            shown=False, lower=(), upper=(), is_band=False, reason=got.reason, **base
        )
    return KernelHeatmap(
        shown=True,
        lower=_grid(got.lower),
        upper=_grid(got.upper),
        is_band=got.is_band,
        reason=got.reason or "shown",
        **base,
    )


def _no_budget(outcome: str | None, why: str) -> VarianceBudgetPanel:
    return VarianceBudgetPanel(outcome, "", False, (), (), None, why)


def _budget(budget: Mapping[str, Any] | None, decs: _Decisions) -> VarianceBudgetPanel:
    if not budget or budget.get("status") != "complete":
        why = "the variance budget was not computed"
        return _no_budget(None, (budget or {}).get("reason") or why)
    key = decision_key("variance_budget", "variance")
    got = _admit(budget["parts"], decs.get(key), key, allow_band=False)
    outcome, units = str(budget["outcome"]), str(budget["units"])
    times = _times(budget["step_times_s"])
    if not got.shown:
        return VarianceBudgetPanel(
            outcome, units, False, times, (), got.view, got.reason
        )
    parts = [_num(v) for v in budget["parts"]["values"]]
    shares = budget["shares"]["values"]
    series = dict(
        zip(budget["noise_classes"], budget["by_time"]["values"], strict=True)
    )
    return VarianceBudgetPanel(
        outcome,
        units,
        True,
        times,
        tuple(
            BudgetPart(
                name,
                parts[i],
                None if shares is None else _num(shares[i]),
                tuple(_num(x) for x in series.get(name, ())),
            )
            for i, name in enumerate(budget["classes"])
        ),
        got.view,
        got.reason or "shown",
    )


def _point(arm: Mapping[str, Any], decs: _Decisions) -> ParetoPoint:
    key = decision_key("strategy_pareto", arm["label"])
    got = _admit(arm["ratio"], decs.get(key), key, allow_band=False)
    common: dict[str, Any] = {
        "label": arm["label"],
        "strategy": arm["strategy"],
        "status": arm["status"],
        "beats_baseline": bool(arm["beats_baseline"]),
        "effort_basis": arm["effort_basis"],
        "mechanisms": tuple(arm["mechanisms"]),
        "decision": got.view,
    }
    if not got.shown:
        return ParetoPoint(
            shown=False,
            rank=None,
            variance_ratio=None,
            chs_mps=None,
            effort=None,
            reason=got.reason,
            **common,
        )
    return ParetoPoint(
        shown=True,
        rank=arm["rank"],
        variance_ratio=_first(got.lower),
        chs_mps=_num(arm["chs_mps"]),
        effort=_num(arm["effort"]),
        reason=got.reason or "shown",
        **common,
    )


def _skipped(
    strat: Mapping[str, Any], points: Sequence[ParetoPoint]
) -> list[SkippedArm]:
    seen = {p.label for p in points}
    out = []
    for arm in strat["arms"]:
        if arm["label"] in seen:
            continue
        why = arm["reason"] or f"excluded: {arm['exclusion'] or 'no variance ratio'}"
        out.append(SkippedArm(arm["label"], arm["status"], arm["exclusion"], why))
        seen.add(arm["label"])
    for item in strat["not_attempted"]:
        if item["label"] not in seen:
            out.append(SkippedArm(item["label"], "not_attempted", None, item["reason"]))
    return out


def _empty_pareto(why: str) -> StrategyParetoPanel:
    return StrategyParetoPanel(None, None, (), (), (), None, None, (), None, why)


def _pareto(strat: Mapping[str, Any] | None, decs: _Decisions) -> StrategyParetoPanel:
    if not strat or strat.get("status") != "complete":
        why = (strat or {}).get("reason") or "the strategy comparison was not run"
        return _empty_pareto(why)
    points = [
        _point(arm, decs)
        for arm in strat["arms"]
        if arm["admitted"] and arm["ratio"] is not None
    ]
    shown = {p.label for p in points if p.shown}
    fronts = []
    for name, labels in strat["fronts"].items():
        keep = tuple(lab for lab in labels if lab in shown)
        fronts.append(ParetoFront(name, keep, len(labels) - len(keep)))
    winner = strat["winner"]
    return StrategyParetoPanel(
        mode=strat["mode"],
        ranking_basis=strat["ranking_basis"],
        points=tuple(points),
        fronts=tuple(fronts),
        skipped=tuple(_skipped(strat, points)),
        failure_rate=_num(strat["failure_rate"]),
        winner=winner if winner in shown else None,
        assumptions=tuple(strat["assumptions"]),
        non_claim=strat["non_claim"],
        unavailable_reason=None,
    )


def _entry(
    label: str,
    quantity: Mapping[str, Any] | None,
    reason: str,
    key: str,
    decs: _Decisions,
) -> DispersionEntry:
    if quantity is None:
        return DispersionEntry(label, "", False, None, None, False, None, reason)
    got = _admit(quantity, decs.get(key), key)
    units = str(quantity["units"])
    if not got.shown:
        return DispersionEntry(
            label, units, False, None, None, False, got.view, got.reason
        )
    return DispersionEntry(
        label,
        units,
        True,
        _first(got.lower),
        _first(got.upper),
        got.is_band,
        got.view,
        got.reason or "shown",
    )


def _mc(check: Mapping[str, Any], linear: DispersionEntry, decs: _Decisions) -> McCheck:
    name = check["outcome"]
    quantity = check["mc_sd_quantity"]
    base: dict[str, Any] = {"outcome": name, "status": check["status"]}
    none: dict[str, Any] = {
        "predicted_sd": None,
        "mc_sd": None,
        "mc_sd_ci": (),
        "gap": None,
        "failure_rate": None,
    }
    if quantity is None:
        return McCheck(
            shown=False,
            decision=None,
            reason="no Monte Carlo trial succeeded for this outcome",
            **base,
            **none,
        )
    key = decision_key("face_dispersion", f"mc:{name}")
    got = _admit(quantity, decs.get(key), key, allow_band=False)
    if not got.shown:
        return McCheck(
            shown=False, decision=got.view, reason=got.reason, **base, **none
        )
    if not linear.shown or linear.is_band:
        why = "the linear prediction is not shown as a single value; no comparison"
        return McCheck(shown=False, decision=got.view, reason=why, **base, **none)
    return McCheck(
        shown=True,
        decision=got.view,
        reason=got.reason or "shown",
        predicted_sd=linear.lower,
        mc_sd=_first(got.lower),
        mc_sd_ci=tuple(_num(x) for x in check["mc_sd_ci"]),
        gap=_num(check["gap"]),
        failure_rate=_num(check["failure_rate"]),
        **base,
    )


def _dispersion(sens: Mapping[str, Any], decs: _Decisions) -> DispersionPanel:
    disp = sens["face_dispersion"]
    linear = _entry(
        "linear",
        disp["linear"]["sd"],
        "",
        decision_key("face_dispersion", "linear"),
        decs,
    )
    arms = tuple(
        _entry(
            item["label"],
            item["sd"],
            item["reason"] or "no variance",
            decision_key("face_dispersion", item["label"]),
            decs,
        )
        for item in disp["arms"]
    )
    checks = tuple(_mc(c, linear, decs) for c in disp["mc"])
    return DispersionPanel(disp["outcome"], linear.units, linear, arms, checks, None)


def _unavailable(status: str, reason: str) -> SensitivityPanels:
    return SensitivityPanels(
        status=status,
        reason=reason,
        primary_outcome=None,
        skipped_priority=(),
        outcomes=(),
        declared_mechanisms=(),
        noise_hypotheses=(),
        linear_validity=None,
        cache_identity=None,
        non_claims=(),
        kernel_heatmap=KernelHeatmap(
            None, "", False, (), (), (), (), False, None, reason
        ),
        variance_budget=_no_budget(None, reason),
        strategy_pareto=_empty_pareto(reason),
        face_dispersion=DispersionPanel(None, "", None, (), (), reason),
    )


def build_sensitivity_panels(
    report: Mapping[str, Any], decs: _Decisions
) -> SensitivityPanels:
    """The four panels of a 2.0.0 report, obeying ``decs`` (see module docstring).

    Postconditions: every number drawn has an admitting decision; a missing
    decision withholds by name; a report whose stage did not run (rejected,
    insufficient frames, unavailable primary outcome) gives empty panels that
    say why.
    """
    sens = report.get("sensitivity")
    if sens is None:
        why = f"sensitivity not computed (report status {report['status']!r})"
        return _unavailable("not_run", report.get("reason") or why)
    if sens["status"] != "complete":
        return _unavailable(str(sens["status"]), str(sens["reason"]))
    return SensitivityPanels(
        status="complete",
        reason=None,
        primary_outcome=sens["primary_outcome"],
        skipped_priority=tuple(sens["skipped_priority"]),
        outcomes=tuple(sens["outcomes"]),
        declared_mechanisms=tuple(sens["declared_mechanisms"]),
        noise_hypotheses=tuple(sens["noise_hypotheses"]),
        linear_validity=sens["linear_validity"],
        cache_identity=sens["cache_identity"],
        non_claims=tuple(sens["non_claims"]),
        kernel_heatmap=_kernel(sens, decs),
        variance_budget=_budget(report.get("variance_budget"), decs),
        strategy_pareto=_pareto(report.get("strategies"), decs),
        face_dispersion=_dispersion(sens, decs),
    )

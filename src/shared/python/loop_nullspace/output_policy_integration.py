"""Display decisions for NSX quantities (NSX-41, ADR-0054 D7 Option A).

:func:`display_decision` is the ONE gate both UIs call; no UI may implement its
own.  It maps a :class:`QuantityRecord`'s evidence axes onto a surface.

Surfaces
--------
``research_view``
    ``INVARIANT`` -> value with a model-provenance badge.  ``FAMILY_DEPENDENT``
    -> a band, or (``representative=True``) a labelled representative overlaid
    on its band under a *named preset that exists in the registry*; a missing
    or unknown preset withholds.  ``UNAVAILABLE`` -> its reason.
``swing_report``
    Nothing enters unless ``output_policy.py`` already admits the output name.
    The policy is *delegated to*, never restated: the caller passes the
    ``dict[str, OutputStatus]`` returned by
    :func:`output_policy.evaluate_output_policy` as ``policy_statuses`` plus the
    ``output_name`` the record stands for.  A QuantityRecord carries no output
    name or receipt, so there is no honest implicit mapping; with either
    argument absent the decision is WITHHOLD.  Admission is necessary, not
    sufficient: the record must also be ``VALIDATED``, be available, and pass
    every research-view rule (verified ids, preset), so the report is never
    looser than the research view.  Calibration requirements are untouched.

Conditioning (addendum 2026-10-15)
----------------------------------
``conditioning`` ids are chips on every record, including ``INVARIANT``, kept
apart from allocation-hypothesis chips.  They pass the same
``Registry.require_verified_for_display`` gate as ``hypotheses``: an unknown or
unverified id withholds.  A record with conditioning is labelled "conditioned
on ..."; it is never labelled unconditioned.

Fail closed
-----------
Well-typed but unrecognised input (unknown surface string, unknown
``allocation_dependence`` or ``validation_status`` value, output name absent
from the statuses) yields WITHHOLD.  Wrongly *typed* arguments raise
``TypeError``; empty strings or half-supplied policy context raise
``ValueError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    QuantityRecord,
    ValidationStatus,
)
from src.shared.python.loop_nullspace.registry import Registry, load_registry
from src.shared.python.motion_matching.output_policy import OutputStatus

__all__ = [
    "DisplayAction",
    "DisplayDecision",
    "DisplaySurface",
    "display_decision",
]


class DisplaySurface(str, Enum):
    """Where a quantity is about to be shown."""

    RESEARCH_VIEW = "research_view"
    SWING_REPORT = "swing_report"


class DisplayAction(str, Enum):
    """What a UI may render."""

    SHOW_VALUE = "SHOW_VALUE"
    SHOW_BAND = "SHOW_BAND"
    SHOW_REPRESENTATIVE_ON_BAND = "SHOW_REPRESENTATIVE_ON_BAND"
    SHOW_REASON = "SHOW_REASON"
    WITHHOLD = "WITHHOLD"


@dataclass(frozen=True)
class DisplayDecision:
    """Outcome of :func:`display_decision`.

    Invariants: ``reason`` is non-empty; ``conditioned`` equals
    ``bool(conditioning_chips)``; a WITHHOLD carries no chips or badges.
    """

    surface: str
    action: DisplayAction
    reason: str
    badges: tuple[str, ...] = ()
    hypothesis_chips: tuple[str, ...] = ()
    conditioning_chips: tuple[str, ...] = ()
    conditioned: bool = False
    preset: str | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action, DisplayAction):
            raise TypeError("DisplayDecision.action must be a DisplayAction")
        if not self.reason:
            raise ValueError("DisplayDecision requires a reason")
        if self.conditioned != bool(self.conditioning_chips):
            raise ValueError("conditioned must match conditioning_chips")
        shown = self.badges or self.hypothesis_chips or self.conditioning_chips
        if self.action is DisplayAction.WITHHOLD and shown:
            raise ValueError("a WITHHOLD decision must not carry chips or badges")


def _withhold(surface: str, reason: str) -> DisplayDecision:
    return DisplayDecision(surface, DisplayAction.WITHHOLD, reason)


def _check_args(
    record: object,
    surface: object,
    preset: object,
    representative: object,
    registry: object,
) -> str:
    """Validate argument types; return the surface as a plain string."""
    if not isinstance(record, QuantityRecord):
        raise TypeError("record must be a QuantityRecord")
    if not isinstance(surface, str):
        raise TypeError("surface must be a DisplaySurface or str")
    if not isinstance(representative, bool):
        raise TypeError("representative must be a bool")
    if preset is not None:
        if not isinstance(preset, str):
            raise TypeError("preset must be a str or None")
        if not preset:
            raise ValueError("preset must be non-empty")
    if registry is not None and not isinstance(registry, Registry):
        raise TypeError("registry must be a Registry or None")
    return surface.value if isinstance(surface, DisplaySurface) else surface


def _check_policy_args(
    output_name: object, policy_statuses: object
) -> Mapping[str, OutputStatus] | None:
    if (output_name is None) != (policy_statuses is None):
        raise ValueError("output_name and policy_statuses must be given together")
    if policy_statuses is None:
        return None
    if not isinstance(output_name, str):
        raise TypeError("output_name must be a str")
    if not output_name:
        raise ValueError("output_name must be non-empty")
    if not isinstance(policy_statuses, Mapping):
        raise TypeError("policy_statuses must be a mapping of OutputStatus")
    if not all(isinstance(v, OutputStatus) for v in policy_statuses.values()):
        raise TypeError("policy_statuses values must be OutputStatus")
    return policy_statuses


def _policy_gate(
    output_name: str | None, statuses: Mapping[str, OutputStatus] | None
) -> str:
    """Empty string when ``output_policy`` admits ``output_name``, else why not."""
    if statuses is None or output_name is None:
        return "no output-policy context supplied; swing report withholds"
    status = statuses.get(output_name)
    if status is None:
        return f"output {output_name!r} is unknown to the output policy"
    if not status.admitted:
        return f"output policy: {output_name!r} is {status.status} ({status.reason})"
    return ""


def _unverified(registry: Registry, ids: tuple[str, ...]) -> tuple[str, ...]:
    """Ids failing ``Registry.require_verified_for_display`` (unknown included)."""
    bad = []
    for ident in ids:
        try:
            registry.require_verified_for_display([ident])
        except ValueError:
            bad.append(ident)
    return tuple(bad)


def _label(record: QuantityRecord, cond: tuple[str, ...], preset: str | None) -> str:
    dep = record.allocation_dependence
    base = dep.value
    if preset is not None and dep is AllocationDependence.FAMILY_DEPENDENT:
        base = f"representative under preset {preset} (overlaid on band)"
    if cond:
        return f"{base}, conditioned on {', '.join(cond)}"
    if dep is AllocationDependence.INVARIANT:
        return f"{base}, unconditioned (given the model)"
    return base


def _badges(record: QuantityRecord) -> tuple[str, ...]:
    prov = record.model_provenance
    validated = record.validation_status is ValidationStatus.VALIDATED
    return (
        f"model: {prov.name} {prov.version}",
        "VALIDATED" if validated else "UNVALIDATED",
    )


def _shown(
    record: QuantityRecord,
    surface: str,
    action: DisplayAction,
    preset: str | None,
    reason: str,
) -> DisplayDecision:
    cond = record.conditioning
    return DisplayDecision(
        surface=surface,
        action=action,
        reason=reason,
        badges=_badges(record),
        hypothesis_chips=record.hypotheses,
        conditioning_chips=cond,
        conditioned=bool(cond),
        preset=preset,
        label=_label(record, cond, preset),
    )


def _family_action(
    registry: Registry, representative: bool, preset: str | None
) -> tuple[DisplayAction, str] | str:
    """Band / representative action, or a withhold reason string."""
    if preset is not None:
        try:
            registry.preset(preset)
        except KeyError:
            return f"preset {preset!r} is not in the registry"
    if not representative:
        return DisplayAction.SHOW_BAND, "family band"
    if preset is None:
        return "a representative requires a named preset"
    return DisplayAction.SHOW_REPRESENTATIVE_ON_BAND, "labelled representative"


def _research_decision(
    record: QuantityRecord,
    surface: str,
    registry: Registry,
    representative: bool,
    preset: str | None,
) -> DisplayDecision:
    dep = record.allocation_dependence
    if dep is AllocationDependence.UNAVAILABLE:
        return DisplayDecision(
            surface, DisplayAction.SHOW_REASON, f"unavailable: {record.reason}"
        )
    if dep not in (
        AllocationDependence.INVARIANT,
        AllocationDependence.FAMILY_DEPENDENT,
    ):
        return _withhold(surface, f"unknown allocation_dependence {dep!r}")
    bad = _unverified(registry, (*record.hypotheses, *record.conditioning))
    if bad:
        return _withhold(surface, f"not verified for display: {', '.join(bad)}")
    if dep is AllocationDependence.INVARIANT:
        return _shown(
            record, surface, DisplayAction.SHOW_VALUE, None, "invariant given model"
        )
    outcome = _family_action(registry, representative, preset)
    if isinstance(outcome, str):
        return _withhold(surface, outcome)
    action, why = outcome
    return _shown(record, surface, action, preset, why)


def display_decision(
    record: QuantityRecord,
    surface: DisplaySurface | str,
    *,
    preset: str | None = None,
    representative: bool = False,
    output_name: str | None = None,
    policy_statuses: Mapping[str, OutputStatus] | None = None,
    registry: Registry | None = None,
) -> DisplayDecision:
    """Decide how ``record`` may be shown on ``surface`` (see module docstring).

    Args:
        record: The quantity and its evidence axes.
        surface: A :class:`DisplaySurface` or its string value.
        preset: Named preset for a FAMILY_DEPENDENT representative.
        representative: Request a labelled representative overlaid on the band.
        output_name: Policy output the record stands for (swing report only).
        policy_statuses: Result of ``evaluate_output_policy`` (swing report).
        registry: Hypothesis registry; defaults to the shipped one.

    Returns:
        A frozen :class:`DisplayDecision`.  Postcondition: WITHHOLD unless every
        applicable rule passes; the swing report never shows what the output
        policy withholds.

    Raises:
        TypeError: Wrongly typed argument.
        ValueError: Empty ``preset``/``output_name``, or only one of
            ``output_name`` / ``policy_statuses`` given.
    """
    name = _check_args(record, surface, preset, representative, registry)
    statuses = _check_policy_args(output_name, policy_statuses)
    reg = registry if registry is not None else load_registry()
    if name == DisplaySurface.RESEARCH_VIEW.value:
        return _research_decision(record, name, reg, representative, preset)
    if name != DisplaySurface.SWING_REPORT.value:
        return _withhold(name, f"unknown surface {name!r}")
    why_not = _policy_gate(output_name, statuses)
    if not why_not and record.validation_status is not ValidationStatus.VALIDATED:
        why_not = "record is not VALIDATED"
    if not why_not and record.allocation_dependence is AllocationDependence.UNAVAILABLE:
        why_not = f"unavailable: {record.reason}"
    if why_not:
        return _withhold(name, why_not)
    return _research_decision(record, name, reg, representative, preset)

"""Display decisions for every quantity of a report (NSX-42, #12574).

:func:`~.plot_model.build_plot_model` consumes decisions and never makes them.
This module is the adapter that *asks* :func:`~.output_policy_integration.
display_decision` -- the one gate -- once per quantity of an ``analyze()``
report, under the keys :func:`~.plot_model.decision_key` defines.  It restates
no display rule.

A report carries the six evidence axes of every quantity, so each report
quantity is rebuilt as a :class:`QuantityRecord` (:func:`quantity_record_from_dict`)
and handed to ``display_decision`` unchanged.  Envelope bands ask for a band,
the allocation asks for a labelled representative under the report's preset.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
    ValidationStatus,
)
from src.shared.python.loop_nullspace.grip_split import SPLIT_COLUMN_NAMES
from src.shared.python.loop_nullspace.output_policy_integration import (
    DisplayDecision,
    display_decision,
)
from src.shared.python.loop_nullspace.plot_model import (
    build_plot_model,
    decision_key,
)
from src.shared.python.loop_nullspace.registry import Registry
from src.shared.python.motion_matching.output_policy import OutputStatus

__all__ = ["quantity_record_from_dict", "report_decisions"]

_LEVELS = ("phi", "phi_dot", "phi_ddot")


def _nan_nested(raw: Any) -> Any:
    if isinstance(raw, list):
        return [_nan_nested(x) for x in raw]
    return np.nan if raw is None else raw


def _values(raw: Any) -> np.ndarray:
    """The report's nested lists (``None`` for non-finite) as a float array."""
    if raw is None:
        return np.full((1,), np.nan)
    return np.array(_nan_nested(raw), dtype=float)


def quantity_record_from_dict(data: Mapping[str, Any]) -> QuantityRecord:
    """Rebuild a :class:`QuantityRecord` from its report form (six axes).

    Raises:
        TypeError: ``data`` is not a mapping.
        ValueError: A key is missing or an enum value is unknown.
    """
    if not isinstance(data, Mapping):
        raise TypeError("data must be a report quantity mapping")
    try:
        return QuantityRecord(
            values=_values(data["values"]),
            units=data["units"],
            allocation_dependence=AllocationDependence(data["allocation_dependence"]),
            model_provenance=ModelProvenance(**data["model_provenance"]),
            input_provenance=InputProvenance(**data["input_provenance"]),
            validation_status=ValidationStatus(data["validation_status"]),
            conventions=tuple(data["conventions"]),
            hypotheses=tuple(data["hypotheses"]),
            reason=data["reason"],
        )
    except KeyError as exc:
        raise ValueError(f"report quantity lacks {exc.args[0]!r}") from exc


class _Asker:
    """Calls ``display_decision`` for one surface and collects the answers."""

    def __init__(
        self,
        surface: str,
        registry: Registry | None,
        policy: tuple[Mapping[str, OutputStatus], Mapping[str, str]] | None,
    ) -> None:
        self.surface = surface
        self.registry = registry
        self.policy = policy
        self.out: dict[str, DisplayDecision] = {}

    def ask(self, key: str, data: Mapping[str, Any], **kwargs: Any) -> None:
        if self.policy is not None and key in self.policy[1]:
            kwargs["output_name"] = self.policy[1][key]
            kwargs["policy_statuses"] = self.policy[0]
        self.out[key] = display_decision(
            quantity_record_from_dict(data),
            self.surface,
            registry=self.registry,
            **kwargs,
        )


def _ask_envelopes(asker: _Asker, env: Mapping[str, Any] | None) -> None:
    if env is None:
        return
    for name, data in env["quantities"].items():
        panel = "hand_split" if name in SPLIT_COLUMN_NAMES else "envelopes"
        asker.ask(decision_key(panel, name), data)


def _ask_invariants(asker: _Asker, inv: Mapping[str, Any] | None) -> None:
    if inv is None:
        return
    for frame in inv["frames"]:
        for name, data in frame["quantities"].items():
            asker.ask(decision_key("invariants", name, int(frame["index"])), data)


def report_decisions(
    report: Mapping[str, Any],
    surface: str = "research_view",
    *,
    registry: Registry | None = None,
    policy_statuses: Mapping[str, OutputStatus] | None = None,
    output_names: Mapping[str, str] | None = None,
) -> dict[str, DisplayDecision]:
    """One ``display_decision`` per quantity of ``report``, keyed for the model.

    ``policy_statuses`` and ``output_names`` (``decision_key`` -> policy output
    name) are passed through to ``display_decision`` for the swing report; a
    key without an output name is asked without policy context and therefore
    withheld there.

    Raises:
        TypeError: ``report`` is not a mapping.
        ValueError: Only one of ``policy_statuses`` / ``output_names`` given.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be the mapping returned by analyze()")
    if (policy_statuses is None) != (output_names is None):
        raise ValueError("policy_statuses and output_names must be given together")
    policy = None if policy_statuses is None else (policy_statuses, output_names or {})
    asker = _Asker(surface, registry, policy)
    _ask_invariants(asker, report["invariants"])
    _ask_envelopes(asker, report["envelopes"])
    alloc = report["allocation"]
    if alloc is not None and alloc.get("torque") is not None:
        asker.ask(
            decision_key("allocation", "torque"),
            alloc["torque"],
            representative=True,
            preset=str(report["preset"]),
        )
    adm = report["admission"]
    if adm is not None:
        for level in _LEVELS:
            asker.ask(decision_key("diagnostics", level), adm["residuals"][level])
    return asker.out


def plot_model_payload(
    report: Mapping[str, Any],
    *,
    preset: str | None = None,
    surface: str = "research_view",
    decision_overrides: Mapping[str, DisplayDecision] | None = None,
) -> dict[str, Any]:
    """The JSON-safe plot model of ``report`` (the cross-UI ``to_dict`` form).

    Runs :func:`report_decisions` (one ``display_decision`` per quantity) and
    :func:`~.plot_model.build_plot_model`; this helper decides nothing.
    ``decision_overrides`` replaces the answer for a decision key and exists
    for fixtures that exercise a renderer path (the golden); production callers
    pass none.  ``preset``, when given, must be the report's own preset.

    Raises:
        TypeError: ``report`` is not a mapping.
        ValueError: ``report`` is malformed (missing sections or keys, a preset
            that differs from ``preset``), or an override key is unused.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be the mapping returned by analyze()")
    if preset is not None and report.get("preset") != preset:
        raise ValueError(
            f"report preset {report.get('preset')!r} differs from requested {preset!r}"
        )
    try:
        decisions = report_decisions(report, surface)
    except (KeyError, AttributeError) as error:
        raise ValueError(f"malformed report: {error!r}") from error
    decisions.update(decision_overrides or {})
    try:
        return build_plot_model(report, decisions).to_dict()
    except (KeyError, AttributeError) as error:
        raise ValueError(f"malformed report: {error!r}") from error

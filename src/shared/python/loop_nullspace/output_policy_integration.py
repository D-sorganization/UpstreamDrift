"""Output-policy integration for Closed-Loop Null-Space Exploration (NSX-41, #12573).

Implements the ADR-0054 §5 owner decision governing how NSX quantities interact
with the fail-closed output policy across product surfaces:

1. ``swing_report``: fail-closed (``allowed=False, render_mode="WITHHELD"``)
   unless ``calibration_qualified=True`` AND ``allocation_dependence == "INVARIANT"``
   AND ``gate_status == "PASS"``. ``FAMILY_DEPENDENT`` or uncalibrated NSX
   quantities never enter the swing report.
2. ``research_view``:
   - ``INVARIANT`` with ``gate_status == "PASS"`` ->
     ``allowed=True, render_mode="INVARIANT_LINE_WITH_BADGE"`` with the
     mandatory ``MODEL_CONDITIONED_INVARIANT`` provenance badge;
   - ``FAMILY_DEPENDENT`` with ``gate_status == "PASS"``, a named preset in the
     NSX registry, and verified hypotheses ->
     ``allowed=True, render_mode="BAND_WITH_REPRESENTATIVE"`` (or
     ``"ENVELOPE_BAND"`` when ``preset_name == "envelope_only"``),
     ``requires_band_underlay=True``; missing preset or unverified hypothesis
     fails closed (``allowed=False, render_mode="WITHHELD"``);
   - ``UNAVAILABLE`` or ``gate_status != "PASS"`` ->
     ``allowed=False, render_mode="UNAVAILABLE_REASON"`` rendering the explicit
     reason.
3. Unknown ``surface`` or ``allocation_dependence`` fails closed
   (``allowed=False, render_mode="WITHHELD"``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "DisplayDecision",
    "RenderMode",
    "SurfaceKind",
    "display_decision",
]

SurfaceKind = Literal["research_view", "swing_report"]
RenderMode = Literal[
    "INVARIANT_LINE_WITH_BADGE",
    "ENVELOPE_BAND",
    "BAND_WITH_REPRESENTATIVE",
    "UNAVAILABLE_REASON",
    "WITHHELD",
]

_VALID_SURFACES: frozenset[str] = frozenset({"research_view", "swing_report"})
_VALID_DEPENDENCE: frozenset[str] = frozenset(
    {"INVARIANT", "FAMILY_DEPENDENT", "UNAVAILABLE"}
)
_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "loop_nullspace_registry.json"
)


@dataclass(frozen=True)
class DisplayDecision:
    """Immutable display decision for one NSX quantity on a target surface."""

    allowed: bool
    surface: SurfaceKind
    render_mode: RenderMode
    badge_text: str
    reason: str
    preset_name: str | None
    requires_band_underlay: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "allowed": self.allowed,
            "surface": self.surface,
            "render_mode": self.render_mode,
            "badge_text": self.badge_text,
            "reason": self.reason,
            "preset_name": self.preset_name,
            "requires_band_underlay": self.requires_band_underlay,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DisplayDecision:
        """Rebuild a :class:`DisplayDecision` from its dictionary representation."""
        surface_val = str(payload.get("surface", "research_view"))
        surface_typed: SurfaceKind = (
            "swing_report" if surface_val == "swing_report" else "research_view"
        )
        mode_val = str(payload.get("render_mode", "WITHHELD"))
        mode_typed: RenderMode = "WITHHELD"
        if mode_val in {
            "INVARIANT_LINE_WITH_BADGE",
            "ENVELOPE_BAND",
            "BAND_WITH_REPRESENTATIVE",
            "UNAVAILABLE_REASON",
            "WITHHELD",
        }:
            mode_typed = mode_val  # type: ignore[assignment]
        preset_raw = payload.get("preset_name")
        return cls(
            allowed=bool(payload.get("allowed", False)),
            surface=surface_typed,
            render_mode=mode_typed,
            badge_text=str(payload.get("badge_text", "")),
            reason=str(payload.get("reason", "")),
            preset_name=str(preset_raw) if preset_raw is not None else None,
            requires_band_underlay=bool(payload.get("requires_band_underlay", False)),
        )


@lru_cache(maxsize=1)
def _registry_sets() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(verified_hypothesis_ids, valid_preset_names)`` from registry."""
    raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    verified_hyps = frozenset(
        str(h["id"])
        for h in raw.get("hypotheses", ())
        if h.get("verification_status") == "VERIFIED"
    )
    presets = frozenset(str(p["name"]) for p in raw.get("presets", ()))
    return verified_hyps, presets


def _extract_field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _enum_str(val: Any) -> str:
    if hasattr(val, "value"):
        return str(val.value)
    return str(val) if val is not None else ""


def _invariant_badge(record: Any) -> str:
    prov = _extract_field(record, "model_provenance")
    engine = "unknown"
    partition = "bare_club"
    if isinstance(prov, Mapping):
        engine = str(prov.get("engine", "unknown"))
        partition = str(prov.get("bare_club_partition_id", "bare_club"))
    elif prov is not None:
        engine = str(getattr(prov, "engine", "unknown"))
        partition = str(getattr(prov, "bare_club_partition_id", "bare_club"))
    convs = tuple(_extract_field(record, "conventions", ()) or ())
    conv_text = ",".join(str(c) for c in convs) or "C-FRAME-GRIP"
    return f"MODEL_CONDITIONED_INVARIANT [{engine} | {partition} | {conv_text}]"


def _withheld(
    surface: SurfaceKind, reason: str, preset_name: str | None = None
) -> DisplayDecision:
    return DisplayDecision(
        allowed=False,
        surface=surface,
        render_mode="WITHHELD",
        badge_text="NOT VALIDATED",
        reason=reason,
        preset_name=preset_name,
        requires_band_underlay=False,
    )


def _decide_swing_report(
    record: Any,
    dependence: str,
    gate_status: str,
    calibration_qualified: bool,
    preset_name: str | None,
) -> DisplayDecision:
    if calibration_qualified and dependence == "INVARIANT" and gate_status == "PASS":
        return DisplayDecision(
            allowed=True,
            surface="swing_report",
            render_mode="INVARIANT_LINE_WITH_BADGE",
            badge_text=_invariant_badge(record),
            reason="calibration and validity gates satisfied",
            preset_name=preset_name,
            requires_band_underlay=False,
        )
    if dependence == "FAMILY_DEPENDENT":
        return _withheld(
            "swing_report",
            "family-dependent NSX quantities are never admitted to swing_report",
            preset_name,
        )
    return _withheld(
        "swing_report",
        "withheld under fail-closed output policy without calibration and passing gate",
        preset_name,
    )


def _decide_research_family(record: Any, preset_name: str | None) -> DisplayDecision:
    verified_hyps, valid_presets = _registry_sets()
    if not preset_name or preset_name not in valid_presets:
        return _withheld(
            "research_view",
            "family-dependent quantity requires a named registry preset",
            preset_name,
        )
    hyps = tuple(str(h) for h in (_extract_field(record, "hypotheses", ()) or ()))
    if not hyps or any(h not in verified_hyps for h in hyps):
        return _withheld(
            "research_view",
            "family-dependent quantity carries missing or unverified hypotheses",
            preset_name,
        )
    mode: RenderMode = (
        "ENVELOPE_BAND"
        if preset_name == "envelope_only"
        else "BAND_WITH_REPRESENTATIVE"
    )
    badge = f"PRIOR_CONDITIONED [{preset_name}: {', '.join(hyps)}]"
    return DisplayDecision(
        allowed=True,
        surface="research_view",
        render_mode=mode,
        badge_text=badge,
        reason=f"admitted on research_view under preset '{preset_name}'",
        preset_name=preset_name,
        requires_band_underlay=True,
    )


def display_decision(
    record: Any,
    surface: SurfaceKind,
    *,
    calibration_qualified: bool = False,
    preset_name: str | None = None,
) -> DisplayDecision:
    """Evaluate fail-closed output-policy display rules for an NSX quantity.

    Args:
        record: A ``QuantityRecord``, ``EnvelopeRecord``, or mapping carrying
            ``allocation_dependence``, ``gate_status``, ``reason``,
            ``hypotheses``, ``conventions``, and ``model_provenance``.
        surface: Target product surface (``"research_view"`` or ``"swing_report"``).
        calibration_qualified: Whether subject-specific calibration and external
            validation gates are satisfied.
        preset_name: Active hypothesis preset name when displaying
            family-dependent quantities.

    Returns:
        Frozen :class:`DisplayDecision`. Unknown surfaces or allocation
        dependences always fail closed with ``allowed=False`` and
        ``render_mode="WITHHELD"``.
    """
    if record is None:
        raise ValueError("record must not be None")
    surface_str = str(surface)
    if surface_str not in _VALID_SURFACES:
        return _withheld("research_view", f"unknown surface: {surface_str!r}")
    typed_surface: SurfaceKind = (
        "swing_report" if surface_str == "swing_report" else "research_view"
    )
    dependence = _enum_str(_extract_field(record, "allocation_dependence"))
    if dependence not in _VALID_DEPENDENCE:
        return _withheld(
            typed_surface,
            f"unknown allocation_dependence: {dependence!r}",
            preset_name,
        )
    gate_status = _enum_str(_extract_field(record, "gate_status", "PASS")) or "PASS"
    raw_reason = _extract_field(record, "reason")
    reason_str = str(raw_reason) if raw_reason else "quantity is unavailable"

    if typed_surface == "swing_report":
        return _decide_swing_report(
            record, dependence, gate_status, calibration_qualified, preset_name
        )

    if dependence == "UNAVAILABLE" or gate_status != "PASS":
        return DisplayDecision(
            allowed=False,
            surface="research_view",
            render_mode="UNAVAILABLE_REASON",
            badge_text="UNAVAILABLE",
            reason=reason_str,
            preset_name=preset_name,
            requires_band_underlay=False,
        )
    if dependence == "INVARIANT":
        return DisplayDecision(
            allowed=True,
            surface="research_view",
            render_mode="INVARIANT_LINE_WITH_BADGE",
            badge_text=_invariant_badge(record),
            reason="invariant quantity admitted with model-provenance badge",
            preset_name=preset_name,
            requires_band_underlay=False,
        )
    return _decide_research_family(record, preset_name)

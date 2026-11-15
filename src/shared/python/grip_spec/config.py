"""Load and override authoring :class:`GripSpec` fields from CLI or config."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from src.shared.python.contracts import require
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec


def apply_shaft_roll_overrides(
    spec: GripSpec,
    overrides: Mapping[str, Any],
) -> GripSpec:
    """Return ``spec`` with optional ``lead_shaft_roll`` / ``trail_shaft_roll`` keys."""
    require(isinstance(spec, GripSpec), "spec must be a GripSpec")
    lead = overrides.get("lead_shaft_roll")
    trail = overrides.get("trail_shaft_roll")
    if lead is None and trail is None:
        return spec
    lead_hand = spec.lead
    trail_hand = spec.trail
    if lead is not None:
        lead_hand = replace(lead_hand, shaft_roll_deg=float(lead))
    if trail is not None:
        trail_hand = replace(trail_hand, shaft_roll_deg=float(trail))
    return replace(spec, lead=lead_hand, trail=trail_hand)


__all__ = ["apply_shaft_roll_overrides"]

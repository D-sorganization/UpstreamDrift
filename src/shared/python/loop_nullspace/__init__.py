"""Closed-Loop Null-Space Exploration (NSX) package."""

from __future__ import annotations

from src.shared.python.loop_nullspace.capacity import (
    CapacitySlotEntry,
    CapacityVector,
    capacity_for,
)
from src.shared.python.loop_nullspace.registry import (
    ConventionRecord,
    HypothesisRecord,
    HypothesisSet,
    LoopNullspaceRegistry,
    PresetRecord,
    RegistryCatalog,
    SourceRecord,
    convention,
    hypothesis,
    load_registry,
    preset,
    require_verified_for_display,
)

__all__ = [
    "CapacitySlotEntry",
    "CapacityVector",
    "ConventionRecord",
    "HypothesisRecord",
    "HypothesisSet",
    "LoopNullspaceRegistry",
    "PresetRecord",
    "RegistryCatalog",
    "SourceRecord",
    "capacity_for",
    "convention",
    "hypothesis",
    "load_registry",
    "preset",
    "require_verified_for_display",
]

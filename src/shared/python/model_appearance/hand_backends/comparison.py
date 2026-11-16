"""Backends shown side by side by the comparison harness (#15146).

Columns are the capsule baseline followed by every default-enabled backend in
the shared registry (ADR-0057 A3), in registry order.
"""

from __future__ import annotations

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends import capsule
from src.shared.python.model_appearance.hand_backends.registry import (
    build_hand_visual,
    list_hand_visual_backends,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    HandVisualSettings,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandSide,
    HandVisualResult,
)


def comparison_backend_names() -> tuple[str, ...]:
    """``("capsule", <registered backends>...)``: stable harness column order."""
    return (capsule.BACKEND_ID, *list_hand_visual_backends())


def build_for_comparison(
    name: str,
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> HandVisualResult:
    """One hand from the capsule baseline or a registered backend, by name."""
    if name == capsule.BACKEND_ID:
        return capsule.build_hand_visual(
            grip_spec, side, grip_radius_m=grip_radius_m, settings=settings
        )
    return build_hand_visual(grip_spec, side, grip_radius_m=grip_radius_m, backend=name)


__all__ = ["build_for_comparison", "comparison_backend_names"]

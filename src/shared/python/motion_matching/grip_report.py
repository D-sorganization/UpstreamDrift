"""Golf-language description of a grip (HRG-GR3, #13204).

Angles are reported as ``(alpha, psi, gamma)`` with the nearest product preset
in golf words: strength weak, neutral or strong (``alpha``) and palm fingers,
neutral or palm (``psi``). Presets are coaching labels, not anatomical
thresholds, and a fitted value is a set-identified calibration, not a
measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.shared.python.core.handedness import (
    DEFAULT_HANDEDNESS,
    Handedness,
    lead_side,
    trail_side,
)
from src.shared.python.motion_matching.grip_contract import (
    GRIP_STRENGTH_PRESETS_DEG,
    PALM_PRESETS_DEG,
    HandAngles,
    hand_from_euler,
)


def _nearest(value: float, presets: Mapping[str, float]) -> str:
    return min(presets, key=lambda name: abs(presets[name] - value))


def describe_hand(angles: HandAngles) -> str:
    """Nearest presets in golf words, for example ``"strong, palm"``."""
    if not isinstance(angles, HandAngles):
        raise TypeError("angles must be a HandAngles")
    strength = _nearest(angles.alpha_deg, GRIP_STRENGTH_PRESETS_DEG)
    palm = _nearest(angles.psi_deg, PALM_PRESETS_DEG)
    return f"{strength}, {palm}"


def describe_angles(angles: HandAngles) -> str:
    """``alpha=.. psi=.. gamma=.. deg (weak, fingers)`` for one hand."""
    return (
        f"alpha={angles.alpha_deg:.1f} psi={angles.psi_deg:.1f} "
        f"gamma={angles.gamma_deg:.1f} deg ({describe_hand(angles)})"
    )


def describe_rotation_deg(
    rotation_deg: Mapping[str, Sequence[float]],
    handedness: Handedness = DEFAULT_HANDEDNESS,
) -> list[str]:
    """One golf-language line per hand of a ``grip_rotation_deg`` mapping.

    Keys are the anatomical sides ``"L"``/``"R"``. A triple that has no golf
    form (singular or out of the angle ranges) is reported as unavailable with
    the reason, never as zero.
    """
    lines = []
    for role, side in (
        ("lead", lead_side(handedness)),
        ("trail", trail_side(handedness)),
    ):
        if side not in rotation_deg:
            lines.append(f"{role} ({side}): unavailable (no rotation given)")
            continue
        try:
            text = describe_angles(hand_from_euler(rotation_deg[side]))
        except ValueError as exc:
            text = f"unavailable ({exc})"
        lines.append(f"{role} ({side}): {text}")
    return lines

"""Authoritative palette and RGBA conversion for force and torque wrench kinds (ADR-0052, #11288).

Pure Python module with zero rendering or GUI dependencies.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from src.shared.python.force_overlay.contracts import WrenchKind

__all__ = [
    "FORCE_KIND_PALETTE",
    "get_kind_rgba",
    "label_variant_hex",
    "hex_to_rgba",
]

# Official palette registered for force overlay wrench categories.
# Categorical hex values stay distinct from tension (#0000ff) and compression (#ff0000).
FORCE_KIND_PALETTE: Final[dict[str, str]] = {
    "joint_actuator": "#E69F00",
    "joint_reaction": "#CC79A7",
    "contact": "#009E73",
    "grip": "#56B4E9",
    "external": "#000000",
    "gravity": "#999999",
    "muscle": "#D55E00",
}

_HEX_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
)


def hex_to_rgba(hex_str: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert hex color string to normalized (r, g, b, a) tuple in [0.0, 1.0].

    Accepts #RGB, #RRGGBB, or #RRGGBBAA with or without leading '#'.
    """
    if not isinstance(hex_str, str):
        raise TypeError(f"hex_str must be str, got {type(hex_str).__name__}")

    cleaned = hex_str.strip()
    match = _HEX_REGEX.match(cleaned)
    if not match:
        raise ValueError(f"Invalid hex color: {hex_str!r}")

    hex_digits = match.group(1)
    if len(hex_digits) == 3:
        r = int(hex_digits[0] * 2, 16) / 255.0
        g = int(hex_digits[1] * 2, 16) / 255.0
        b = int(hex_digits[2] * 2, 16) / 255.0
        a = float(alpha)
    elif len(hex_digits) == 6:
        r = int(hex_digits[0:2], 16) / 255.0
        g = int(hex_digits[2:4], 16) / 255.0
        b = int(hex_digits[4:6], 16) / 255.0
        a = float(alpha)
    else:  # 8 digits
        r = int(hex_digits[0:2], 16) / 255.0
        g = int(hex_digits[2:4], 16) / 255.0
        b = int(hex_digits[4:6], 16) / 255.0
        a = int(hex_digits[6:8], 16) / 255.0

    return (
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
        max(0.0, min(1.0, a)),
    )


def get_kind_rgba(
    kind: WrenchKind | str, alpha: float = 1.0
) -> tuple[float, float, float, float]:
    """Retrieve the normalized RGBA tuple for a given wrench kind."""
    key = kind.value if hasattr(kind, "value") else str(kind)
    hex_color = FORCE_KIND_PALETTE.get(key, "#000000")
    return hex_to_rgba(hex_color, alpha=alpha)


#: Lighten (+) or darken (-) the GRIP colour per label so left and right hands
#: and the moments of force stay distinguishable from the net (#11716).
_LABEL_SHADE: Final[tuple[tuple[str, float], ...]] = (
    ("grip:hand_left", 0.45),
    ("grip:hand_right", -0.35),
    ("grip:mof_left", 0.7),
    ("grip:mof_right", -0.6),
)


def label_variant_hex(label: str, base_hex: str) -> str:
    """Shade ``base_hex`` for per-hand grip labels; other labels are unchanged.

    A positive shade mixes toward white, a negative one toward black.
    """
    for prefix, shade in _LABEL_SHADE:
        if label.startswith(prefix):
            digits = base_hex.strip().lstrip("#")
            if len(digits) != 6:
                return base_hex
            target = 255 if shade > 0 else 0
            mix = abs(shade)
            chans = [
                round(int(digits[i : i + 2], 16) * (1 - mix) + target * mix)
                for i in (0, 2, 4)
            ]
            return "#" + "".join(f"{c:02X}" for c in chans)
    return base_hex

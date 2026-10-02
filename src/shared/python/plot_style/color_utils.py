"""Color utility functions for hex and RGBA conversions.

Provides canonical, validated conversions between hexadecimal color strings
and normalized RGBA float tuples in [0.0, 1.0].
"""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import Final

__all__ = ["hex_to_rgba", "rgba_to_hex"]

_HEX_COLOR_RE: Final[re.Pattern[str]] = re.compile(
    r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
)


def hex_to_rgba(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert a hex color string to a normalized RGBA tuple.

    Parameters
    ----------
    value:
        A hex color string starting with '#' formatted as '#rgb',
        '#rrggbb', or '#rrggbbaa' (case-insensitive).
    alpha:
        Default alpha multiplier in [0.0, 1.0], applied to the color.
        Defaults to 1.0.

    Returns
    -------
    tuple[float, float, float, float]
        Tuple of (r, g, b, a) values in [0.0, 1.0].

    Raises
    ------
    TypeError
        If ``value`` is not a string or ``alpha`` is not numeric.
    ValueError
        If ``value`` does not match '#rgb', '#rrggbb', or '#rrggbbaa',
        or if ``alpha`` is not finite or outside [0.0, 1.0].
    """
    if not isinstance(value, str):
        raise TypeError(f"value must be a string; got {type(value).__name__}")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"alpha must be numeric; got {type(alpha).__name__}")
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or not 0.0 <= alpha_f <= 1.0:
        raise ValueError(f"alpha must be finite in [0.0, 1.0]; got {alpha!r}")

    if not _HEX_COLOR_RE.fullmatch(value):
        raise ValueError(
            f"Invalid hex color string: {value!r}. "
            "Expected format '#rgb', '#rrggbb', or '#rrggbbaa'."
        )

    hex_digits = value[1:]
    if len(hex_digits) == 3:
        r = int(hex_digits[0] * 2, 16) / 255.0
        g = int(hex_digits[1] * 2, 16) / 255.0
        b = int(hex_digits[2] * 2, 16) / 255.0
        a = alpha_f
    elif len(hex_digits) == 6:
        r = int(hex_digits[0:2], 16) / 255.0
        g = int(hex_digits[2:4], 16) / 255.0
        b = int(hex_digits[4:6], 16) / 255.0
        a = alpha_f
    else:  # len == 8
        r = int(hex_digits[0:2], 16) / 255.0
        g = int(hex_digits[2:4], 16) / 255.0
        b = int(hex_digits[4:6], 16) / 255.0
        parsed_alpha = int(hex_digits[6:8], 16) / 255.0
        a = parsed_alpha * alpha_f

    return (r, g, b, a)


def rgba_to_hex(
    rgba: tuple[float, float, float, float] | tuple[float, float, float],
    include_alpha: bool = True,
) -> str:
    """Convert a normalized RGB or RGBA tuple to a lowercase hex string.

    Parameters
    ----------
    rgba:
        A 3-tuple (r, g, b) or 4-tuple (r, g, b, a) with components in [0.0, 1.0].
    include_alpha:
        Whether to include the alpha channel in the output hex string.
        Ignored if ``rgba`` is a 3-tuple. Defaults to True.

    Returns
    -------
    str
        Lowercase hex string formatted as '#rrggbb' or '#rrggbbaa'.

    Raises
    ------
    TypeError
        If ``rgba`` is not a tuple/sequence or any component is non-numeric.
    ValueError
        If ``rgba`` does not contain 3 or 4 components, or if any component
        is non-finite or outside [0.0, 1.0].
    """
    if not isinstance(rgba, (tuple, list)):
        raise TypeError(f"rgba must be a tuple or list; got {type(rgba).__name__}")
    if len(rgba) not in (3, 4):
        raise ValueError(f"rgba must contain 3 or 4 components; got {len(rgba)}")

    int_components: list[int] = []
    for idx, c in enumerate(rgba):
        if isinstance(c, bool) or not isinstance(c, Real):
            raise TypeError(
                f"component at index {idx} must be numeric; got {type(c).__name__}"
            )
        val = float(c)
        if not math.isfinite(val):
            raise ValueError(f"component at index {idx} must be finite; got {val!r}")
        if not 0.0 <= val <= 1.0:
            raise ValueError(
                f"component at index {idx} must be in [0.0, 1.0]; got {val!r}"
            )
        quantized = int(round(val * 255.0))
        quantized = max(0, min(255, quantized))
        int_components.append(quantized)

    r, g, b = int_components[0], int_components[1], int_components[2]
    if len(int_components) == 4 and include_alpha:
        a = int_components[3]
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    return f"#{r:02x}{g:02x}{b:02x}"

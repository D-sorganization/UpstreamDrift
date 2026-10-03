"""Pure-Python color representation utilities (hex/RGBA conversions).

DRY single-source color parser and serializer conforming to ADR-0052 and #11289.
Headless import safe: this module does not import GUI or physics engine SDKs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from numbers import Real
from typing import Final

from ._types import RGBATuple

__all__ = [
    "hex_to_rgba",
    "rgba_to_hex",
]

_HEX_3_PATTERN: Final[re.Pattern[str]] = re.compile(r"^#[0-9a-fA-F]{3}$")
_HEX_6_PATTERN: Final[re.Pattern[str]] = re.compile(r"^#[0-9a-fA-F]{6}$")
_HEX_8_PATTERN: Final[re.Pattern[str]] = re.compile(r"^#[0-9a-fA-F]{8}$")


def hex_to_rgba(value: str, alpha: float = 1.0) -> RGBATuple:
    """Convert a hex color string to an ``(r, g, b, a)`` float tuple in ``[0, 1]``.

    Accepts ``#rgb``, ``#rrggbb``, and ``#rrggbbaa``. Raises ``ValueError`` for
    any other string, including named colors or malformed hex.

    Parameters
    ----------
    value:
        A hex color string matching ``#rgb``, ``#rrggbb``, or ``#rrggbbaa``.
    alpha:
        Opacity multiplier in ``[0.0, 1.0]``, default 1.0. For ``#rrggbbaa``,
        the parsed alpha is multiplied by this parameter.

    Returns
    -------
    RGBATuple
        Tuple of ``(r, g, b, a)`` floats in ``[0.0, 1.0]``.

    Raises
    ------
    TypeError
        If ``value`` is not a string or ``alpha`` is not a real number.
    ValueError
        If ``value`` is not a recognized hex pattern, or ``alpha`` is non-finite
        or outside ``[0.0, 1.0]``.
    """
    if not isinstance(value, str):
        raise TypeError(f"hex color must be str; got {type(value).__name__}")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"alpha must be a real number; got {alpha!r}")
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f):
        raise ValueError(f"alpha must be finite; got {alpha_f}")
    if not 0.0 <= alpha_f <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0]; got {alpha_f}")

    if _HEX_6_PATTERN.match(value) is not None:
        r = int(value[1:3], 16) / 255.0
        g = int(value[3:5], 16) / 255.0
        b = int(value[5:7], 16) / 255.0
        return (r, g, b, alpha_f)

    if _HEX_3_PATTERN.match(value) is not None:
        r = int(value[1] * 2, 16) / 255.0
        g = int(value[2] * 2, 16) / 255.0
        b = int(value[3] * 2, 16) / 255.0
        return (r, g, b, alpha_f)

    if _HEX_8_PATTERN.match(value) is not None:
        r = int(value[1:3], 16) / 255.0
        g = int(value[3:5], 16) / 255.0
        b = int(value[5:7], 16) / 255.0
        hex_alpha = int(value[7:9], 16) / 255.0
        return (r, g, b, hex_alpha * alpha_f)

    raise ValueError(f"Invalid hex color string: {value!r}")


def rgba_to_hex(
    rgba: Sequence[float],
    *,
    include_alpha: bool | None = None,
) -> str:
    """Convert an RGBA or RGB float sequence in ``[0, 1]`` to a lowercase hex string.

    Parameters
    ----------
    rgba:
        A sequence of 3 (RGB) or 4 (RGBA) numbers in ``[0.0, 1.0]``.
    include_alpha:
        Whether to include the alpha channel in the output hex string:
        - ``True``: always format as ``#rrggbbaa`` (requires len 4).
        - ``False``: always format as ``#rrggbb``.
        - ``None`` (default): format as ``#rrggbbaa`` if len is 4 and alpha != 1.0,
          otherwise format as ``#rrggbb``.

    Returns
    -------
    str
        Lowercase hex string starting with ``#`` (e.g. ``"#0000ff"``).

    Raises
    ------
    TypeError
        If ``rgba`` is not a sequence or contains non-real values.
    ValueError
        If ``len(rgba)`` is not 3 or 4, or any component is non-finite or outside ``[0.0, 1.0]``.
    """
    if isinstance(rgba, (str, bytes)) or not isinstance(rgba, Sequence):
        raise TypeError(f"rgba must be a sequence of floats; got {type(rgba).__name__}")
    if len(rgba) not in (3, 4):
        raise ValueError(f"rgba must have length 3 or 4; got length {len(rgba)}")

    clamped_ints: list[int] = []
    for idx, component in enumerate(rgba):
        if isinstance(component, bool) or not isinstance(component, Real):
            raise TypeError(
                f"component at index {idx} must be a real number; got {component!r}"
            )
        val = float(component)
        if not math.isfinite(val):
            raise ValueError(f"component at index {idx} must be finite; got {val}")
        if not 0.0 <= val <= 1.0:
            raise ValueError(
                f"component at index {idx} must be in [0.0, 1.0]; got {val}"
            )
        # Half-up rounding matches standard sRGB 8-bit quantization
        byte_val = int(math.floor(val * 255.0 + 0.5))
        clamped_ints.append(min(255, max(0, byte_val)))

    r_byte, g_byte, b_byte = clamped_ints[0], clamped_ints[1], clamped_ints[2]

    should_include_alpha: bool
    if include_alpha is True:
        if len(rgba) < 4:
            raise ValueError("cannot include alpha when rgba has length 3")
        should_include_alpha = True
    elif include_alpha is False:
        should_include_alpha = False
    else:
        # None: include alpha only if len 4 and not fully opaque
        should_include_alpha = len(rgba) == 4 and clamped_ints[3] != 255

    if should_include_alpha:
        return f"#{r_byte:02x}{g_byte:02x}{b_byte:02x}{clamped_ints[3]:02x}"
    return f"#{r_byte:02x}{g_byte:02x}{b_byte:02x}"

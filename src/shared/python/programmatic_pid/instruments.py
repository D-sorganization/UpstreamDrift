"""Instrument (bubble tag) rendering.

Instruments are drawn as circles with tag text inside, following ISA 5.1
conventions for P&ID instrument identification.

Preconditions:
    - ``instrument`` dict must have at least ``x`` and ``y`` keys.

Postconditions:
    - ``add_instrument`` adds a circle and tag text to the modelspace.
"""

from __future__ import annotations

from typing import Any

from programmatic_pid.geometry import to_float
from programmatic_pid.rendering import add_text


def add_instrument(
    msp: Any,
    instrument: dict[str, Any],
    text_h: float,
    text_layer: str,
    default_layer: str,
    radius: float,
    show_number_suffix: bool = False,
    label_placer: Any | None = None,
) -> None:
    """Draw an instrument bubble (circle + tag text) on the modelspace.

    Postconditions:
        - A circle entity is added on *layer*.
        - Tag text is centred inside the bubble.
        - If *label_placer* is provided, the bubble rect is reserved.
    """
    if instrument is None:
        raise ValueError("instrument must be provided")
    layer = instrument.get("layer", default_layer)
    x = to_float(instrument.get("x", 0.0))
    y = to_float(instrument.get("y", 0.0))
    bubble = str(instrument.get("tag") or instrument.get("id") or "").strip()
    number = str(instrument.get("id", "")).split("-", 1)[-1]

    r = max(to_float(radius, 1.8), 0.4)
    msp.add_circle((x, y), radius=r, dxfattribs={"layer": layer})
    if label_placer is not None:
        label_placer.reserve_rect((x - r, y - r, x + r, y + r))
    add_text(msp, bubble, x, y, max(text_h * 0.45, 0.5), layer=text_layer)
    if show_number_suffix and number:
        add_text(
            msp,
            number,
            x + max(to_float(radius, 1.8), 0.4) + 0.5,
            y,
            max(text_h * 0.5, 0.5),
            layer=text_layer,
            align="MIDDLE_LEFT",
        )

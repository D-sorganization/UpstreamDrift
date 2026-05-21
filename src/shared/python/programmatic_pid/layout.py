"""Layout computation and label placement.

This module handles spatial layout of the P&ID canvas: computing panel
positions, spreading overlapping instruments, and collision-free label
placement.

Preconditions:
    - Specs passed to layout functions must already be validated.

Postconditions:
    - ``compute_layout_regions`` returns a dict with ``layout_cfg``,
      ``equipment_bbox``, ``canvas_bbox``, and ``panels`` keys.
    - ``LabelPlacer.find_position`` always returns a valid (x, y, align) tuple.
"""

from __future__ import annotations  # noqa: E402, F404

from collections.abc import Sequence  # noqa: E402
from typing import Any  # noqa: E402

from programmatic_pid.equipment import equipment_dims  # noqa: E402
from programmatic_pid.geometry import rects_overlap, text_box, to_float  # noqa: E402
from programmatic_pid.spec_loader import get_drawing, get_layout_config  # noqa: E402

# ---------------------------------------------------------------------------
# Label collision avoidance
# ---------------------------------------------------------------------------


class LabelPlacer:
    """Track occupied rectangles and find collision-free label positions.

    Invariant: ``self.occupied`` contains all rectangles reserved so far
    as ``(x_min, y_min, x_max, y_max)`` tuples.
    """

    def __init__(self) -> None:
        self.occupied: list[tuple[float, float, float, float]] = []

    def reserve_rect(self, rect: tuple[float, float, float, float]) -> None:
        """Mark a rectangular area as occupied."""
        self.occupied.append(rect)

    def reserve_text(
        self,
        text: str,
        x: float,
        y: float,
        h: float,
        align: str = "MIDDLE_CENTER",
    ) -> None:
        """Reserve the bounding box of a text string."""
        self.reserve_rect(text_box(text, x, y, h, align=align))

    def find_position(
        self,
        text: str,
        anchor: Sequence[float],
        h: float,
        preferred: list[tuple[float, float, str]],
    ) -> tuple[float, float, str]:
        """Find the first non-colliding position from *preferred* offsets.

        Preconditions:
            - *preferred* is non-empty.

        Postconditions:
            - The returned position's bounding box is reserved.
            - Always returns a valid (x, y, align) — falls back to first
              preferred offset if all collide.
        """
        if not (preferred):
            raise ValueError("preferred positions list must not be empty")

        ax, ay = to_float(anchor[0]), to_float(anchor[1])
        for dx, dy, align in preferred:
            x = ax + dx
            y = ay + dy
            candidate = text_box(text, x, y, h, align=align)
            if not any(
                rects_overlap(candidate, r, pad=h * 0.20) for r in self.occupied
            ):
                self.reserve_rect(candidate)
                return x, y, align
        fallback = preferred[0]
        x = ax + fallback[0]
        y = ay + fallback[1]
        align = fallback[2]
        self.reserve_rect(text_box(text, x, y, h, align=align))
        return x, y, align


# ---------------------------------------------------------------------------
# Instrument spreading
# ---------------------------------------------------------------------------


def spread_instrument_positions(
    instruments: list[dict[str, Any]],
    min_spacing: float = 3.5,
) -> list[dict[str, Any]]:
    """Adjust instrument positions to avoid overlapping bubbles.

    Preconditions:
        - Each instrument dict has optional ``x`` and ``y`` keys.

    Postconditions:
        - Returned list has the same length as *instruments*.
        - Each returned dict is a shallow copy with updated ``x`` / ``y``.
    """
    if instruments is None:
        raise ValueError("instruments must be provided")
    placed: list[tuple[float, float]] = []
    output: list[dict[str, Any]] = []
    ring = [
        (0.0, 0.0),
        (2.0, 0.0),
        (-2.0, 0.0),
        (0.0, 2.0),
        (0.0, -2.0),
        (2.0, 2.0),
        (-2.0, 2.0),
        (2.0, -2.0),
        (-2.0, -2.0),
    ]
    spacing = max(to_float(min_spacing, 3.5), 1.2)
    for ins in instruments:
        base_x = to_float(ins.get("x", 0.0))
        base_y = to_float(ins.get("y", 0.0))
        chosen = (base_x, base_y)
        for radius in (1.0, 1.8, 2.6, 3.6):
            found = None
            for ox, oy in ring:
                cand = (base_x + ox * radius, base_y + oy * radius)
                if all(
                    (cand[0] - px) ** 2 + (cand[1] - py) ** 2 >= spacing**2
                    for px, py in placed
                ):
                    found = cand
                    break
            if found is not None:
                chosen = found
                break
        placed.append(chosen)
        copy = dict(ins)
        copy["x"] = chosen[0]
        copy["y"] = chosen[1]
        output.append(copy)
    return output


# ---------------------------------------------------------------------------
# Canvas / panel layout
# ---------------------------------------------------------------------------


def get_equipment_bounds(
    spec: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) bounding all equipment."""
    equipment = spec.get("equipment", [])
    if not equipment:
        return 0.0, 0.0, 240.0, 160.0
    x_min = min(to_float(eq.get("x", 0.0)) for eq in equipment)
    y_min = min(to_float(eq.get("y", 0.0)) for eq in equipment)
    x_max = max(to_float(eq.get("x", 0.0)) + equipment_dims(eq)[0] for eq in equipment)
    y_max = max(to_float(eq.get("y", 0.0)) + equipment_dims(eq)[1] for eq in equipment)
    return x_min, y_min, x_max, y_max


def compute_layout_regions(spec: dict[str, Any]) -> dict[str, Any]:
    """Compute canvas dimensions and panel positions for the P&ID.

    Returns a dict with keys:
        ``layout_cfg``, ``equipment_bbox``, ``canvas_bbox``, ``panels``
    """
    layout_cfg = get_layout_config(spec)
    eq_min_x, eq_min_y, eq_max_x, eq_max_y = get_equipment_bounds(spec)

    gap = layout_cfg["gap"]
    right_w = layout_cfg["right_panel_width"]
    bottom_h = layout_cfg["bottom_panel_height"]
    title_h = layout_cfg["title_block_height"]

    process_w = max(eq_max_x - eq_min_x, 60.0)
    process_h = max(eq_max_y - eq_min_y, 50.0)
    left_pad = max(gap * 0.75, 4.0)
    top_pad = max(gap * 1.35, 10.0)

    canvas_x_min = eq_min_x - left_pad
    canvas_x_max = eq_max_x + gap + right_w + left_pad
    canvas_y_min = eq_min_y - (bottom_h + title_h + gap * 2.0)
    canvas_y_max = eq_max_y + top_pad

    control_w = max(process_w * 0.58, 56.0)
    mass_w = max(process_w - control_w - gap, 38.0)
    if control_w + mass_w + gap > process_w:
        scale = process_w / (control_w + mass_w + gap)
        control_w *= scale
        mass_w *= scale

    bottom_y = eq_min_y - (bottom_h + gap)
    panels = {
        "control": (eq_min_x, bottom_y, control_w, bottom_h),
        "mass": (eq_min_x + control_w + gap, bottom_y, mass_w, bottom_h),
        "right": (eq_max_x + gap, eq_min_y, right_w, process_h + top_pad * 0.85),
        "title": (canvas_x_min, canvas_y_min, canvas_x_max - canvas_x_min, title_h),
    }

    return {
        "layout_cfg": layout_cfg,
        "equipment_bbox": (eq_min_x, eq_min_y, eq_max_x, eq_max_y),
        "canvas_bbox": (canvas_x_min, canvas_y_min, canvas_x_max, canvas_y_max),
        "panels": panels,
    }


def get_modelspace_extent(
    spec: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Get the modelspace extent from spec or compute from equipment bounds."""
    drawing = get_drawing(spec)
    extent = drawing.get("modelspace_extent", {})
    if all(k in extent for k in ("x_min", "y_min", "x_max", "y_max")):
        return (
            to_float(extent["x_min"]),
            to_float(extent["y_min"]),
            to_float(extent["x_max"]),
            to_float(extent["y_max"]),
        )

    equipment = spec.get("equipment", [])
    if equipment:
        x_min = min(to_float(eq.get("x", 0.0)) for eq in equipment)
        y_min = min(to_float(eq.get("y", 0.0)) for eq in equipment)
        x_max = max(
            to_float(eq.get("x", 0.0)) + equipment_dims(eq)[0] for eq in equipment
        )
        y_max = max(
            to_float(eq.get("y", 0.0)) + equipment_dims(eq)[1] for eq in equipment
        )
        margin = max((x_max - x_min) * 0.08, 5.0)
        return x_min - margin, y_min - margin, x_max + margin, y_max + margin

    return 0.0, 0.0, 240.0, 160.0

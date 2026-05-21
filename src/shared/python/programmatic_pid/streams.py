"""Stream (pipe / flow path) rendering.

Streams connect equipment via directed polylines with arrowheads and labels.

Preconditions:
    - Equipment referenced via ``from`` / ``to`` must exist in *equipment_by_id*.

Postconditions:
    - ``add_stream`` returns ``(lx, ly)`` label centre or ``None`` if nothing drawn.
"""

from __future__ import annotations

from typing import Any

from programmatic_pid.equipment import equipment_anchor
from programmatic_pid.geometry import closest_point_on_rect, text_box, to_float
from programmatic_pid.rendering import (
    add_arrow,
    add_poly_arrow,
    add_text,
    ensure_layer,
)


def resolve_endpoint(
    endpoint: dict[str, Any] | None,
    equipment_by_id: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    """Resolve an endpoint dict to (x, y) coordinates.

    Preconditions:
        - If the endpoint references equipment, that ID must be in *equipment_by_id*.

    Raises:
        KeyError: if the referenced equipment is missing.
    """
    endpoint = endpoint or {}
    if "point" in endpoint:
        px, py = endpoint["point"]
        return to_float(px), to_float(py)

    eq_id = endpoint.get("equipment")
    if not eq_id or eq_id not in equipment_by_id:
        raise KeyError(f"Unknown equipment endpoint: {eq_id}")
    return equipment_anchor(  # type: ignore[no-any-return]
        equipment_by_id[eq_id],
        endpoint.get("side", "right"),
        endpoint.get("offset", 0.0),
    )


def add_stream(
    msp: Any,
    stream: dict[str, Any],
    text_h: float,
    text_layer: str,
    equipment_by_id: dict[str, dict[str, Any]],
    arrow_size: float,
    label_scale: float = 0.82,
    label_placer: Any | None = None,
    draw_label_leader: bool = False,
    leader_layer: str = "LEADERS",
) -> tuple[float, float] | None:
    """Draw a stream (pipe) on the modelspace and return its label centre.

    Postconditions:
        - Returns ``(lx, ly)`` if a stream was drawn, ``None`` otherwise.
        - If a label is present and *label_placer* is provided, label position
          is collision-checked.
    """
    if stream is None:
        raise ValueError("stream must be provided")
    layer = stream.get("layer", "PROCESS")
    color = stream.get("color")
    lx = 0.0
    ly = 0.0

    if "vertices" in stream:
        verts = [tuple(v) for v in stream.get("vertices", [])]
        if len(verts) < 2:
            return None
        add_poly_arrow(msp, verts, layer, color=color, arrow_size=arrow_size)
        lx = sum(v[0] for v in verts) / len(verts)
        ly = sum(v[1] for v in verts) / len(verts)
    elif "start" in stream and "end" in stream:
        start = tuple(stream["start"])
        end = tuple(stream["end"])
        add_arrow(msp, start, end, layer, color=color, arrow_size=arrow_size)
        lx = (to_float(start[0]) + to_float(end[0])) / 2
        ly = (to_float(start[1]) + to_float(end[1])) / 2
    elif "from" in stream and "to" in stream:
        start = resolve_endpoint(stream.get("from"), equipment_by_id)
        end = resolve_endpoint(stream.get("to"), equipment_by_id)
        waypoints = [tuple(wp) for wp in stream.get("waypoints", [])]
        verts = [start, *waypoints, end]
        if len(verts) > 2:
            add_poly_arrow(msp, verts, layer, color=color, arrow_size=arrow_size)
            lx = sum(to_float(v[0]) for v in verts) / len(verts)
            ly = sum(to_float(v[1]) for v in verts) / len(verts)
        else:
            add_arrow(msp, start, end, layer, color=color, arrow_size=arrow_size)
            lx = (start[0] + end[0]) / 2
            ly = (start[1] + end[1]) / 2
    else:
        return None

    # --- Label ---
    label = stream.get("label", "")
    if isinstance(label, dict):
        text = str(label.get("text", "")).strip()
        lx = to_float(label.get("x", lx))
        ly = to_float(label.get("y", ly))
    else:
        text = str(label).strip()

    if not text and stream.get("name"):
        text = str(stream["name"])
    if text:
        h = max(text_h * label_scale, 0.5)
        default_x = lx
        default_y = ly + text_h * 0.72
        if label_placer is not None:
            x, y, align = label_placer.find_position(
                text,
                (lx, ly),
                h,
                preferred=[
                    (0.0, text_h * 1.1, "BOTTOM_CENTER"),
                    (0.0, -text_h * 1.1, "TOP_CENTER"),
                    (text_h * 1.5, text_h * 0.6, "BOTTOM_LEFT"),
                    (-text_h * 1.5, text_h * 0.6, "BOTTOM_RIGHT"),
                ],
            )
            add_text(msp, text, x, y, h, layer=text_layer, align=align)
            displaced = abs(x - default_x) > h * 0.35 or abs(y - default_y) > h * 0.35
            if draw_label_leader and displaced:
                if leader_layer not in msp.doc.layers:
                    ensure_layer(msp.doc, leader_layer, color=8, linetype="DASHED")
                target = closest_point_on_rect(
                    (lx, ly), text_box(text, x, y, h, align=align)
                )
                msp.add_line((lx, ly), target, dxfattribs={"layer": leader_layer})
        else:
            add_text(msp, text, default_x, default_y, h, layer=text_layer)
    return to_float(lx), to_float(ly)

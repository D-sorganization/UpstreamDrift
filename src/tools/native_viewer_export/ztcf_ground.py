"""Ground reaction and force kinds by ZTCF/ZVCF case (#15150 Track Z, Z4c).

``--force-part`` shows one case: its planted-foot ground reaction (ZTCF:
drift including gravity, ``tau = 0``; ZVCF: applied torques alone, no
gravity) at its centre of pressure in the hand arrows' case colour, and only
its force kinds. The per-foot split of welded feet is one null-space
solution; the net wrench of both feet is unique, so it is what is drawn.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.body_part_viz.mujoco_axial_loads import BodyWrench
from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.force_overlay.glyphs import ArrowGlyph, ForceGlyphStyle
from src.shared.python.force_overlay.palette import hex_to_rgba
from src.shared.python.force_overlay.tip_to_tail import part_arrows
from src.shared.python.simulation_backends.hand_force_split import HANDS
from src.tools.native_viewer_export.ztcf_feed import hand_styles

SETTINGS_PATH = Path(__file__).with_name("ztcf_ground_settings.json")
CASES = ("total", "ztcf", "zvcf")
FORCE_PARTS = ("recorded", *CASES)
_TORQUES = {WrenchKind.JOINT_ACTUATOR, WrenchKind.MUSCLE}
#: force kinds a case keeps: ZTCF has ``tau = 0``, ZVCF has no gravity
PART_KINDS = {
    "ztcf": frozenset({WrenchKind.GRAVITY}),
    "zvcf": frozenset(_TORQUES),
    "total": frozenset({WrenchKind.GRAVITY, *_TORQUES}),
}
_CASE_HEX = {"ztcf": "drift_hex", "zvcf": "control_hex", "total": "total_hex"}
_ZERO = (0.0, 0.0, 0.0)
Vec3 = tuple[float, float, float]


def load_ground_settings(path: Path | None = None) -> dict[str, Any]:
    """Settings from ``path`` (default: beside this module); ``ValueError``
    for a bad ``case_colours_hand`` or ``min_vertical_n``."""
    doc = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    if doc.get("case_colours_hand") not in HANDS:
        raise ValueError(f"case_colours_hand must be one of {HANDS}")
    limit = doc.get("min_vertical_n")
    if not (isinstance(limit, int | float) and math.isfinite(limit) and limit > 0):
        raise ValueError("min_vertical_n must be finite and positive")
    return dict(doc)


def _case(part: str) -> str:
    if part not in CASES:
        raise ValueError(f"the ground split draws a case in {CASES}, got {part!r}")
    return part


def centre_of_pressure(
    point: Sequence[float],
    force: Sequence[float],
    moment: Sequence[float],
    min_vertical_n: float = 1.0,
) -> Vec3 | None:
    """Point at the height of ``point`` where ``force`` acts with no tilting
    moment (``moment`` about ``point``, z up); ``None`` (undefined) when
    ``|F_z| <= min_vertical_n``. ``ValueError`` for a vector not finite
    length 3 or a negative ``min_vertical_n``."""
    p, f, m = (np.asarray(x, dtype=float).reshape(-1) for x in (point, force, moment))
    if any(x.shape != (3,) or not np.all(np.isfinite(x)) for x in (p, f, m)):
        raise ValueError("point, force and moment must be finite length-3 vectors")
    if not min_vertical_n >= 0.0:
        raise ValueError(f"min_vertical_n must not be negative, got {min_vertical_n}")
    if abs(f[2]) <= min_vertical_n:
        return None
    return (float(p[0] - m[1] / f[2]), float(p[1] + m[0] / f[2]), float(p[2]))


def _vec3(x: Any) -> Vec3:
    return (float(x[0]), float(x[1]), float(x[2]))


def part_ground(
    drift: Sequence[BodyWrench], control: Sequence[BodyWrench]
) -> dict[str, tuple[BodyWrench, ...]]:
    """``{"ztcf", "zvcf", "total"}`` ground wrenches, ``total`` their sum;
    ``ValueError`` when the parts are not on the same bodies and points."""
    total = []
    for d, c in zip(drift, control, strict=True):
        if d.body != c.body or not np.allclose(d.point_m, c.point_m, atol=1e-12):
            raise ValueError(f"ground parts of {d.body!r} are not at the same point")
        force = np.add(d.force_n or _ZERO, c.force_n or _ZERO)
        torque = np.add(d.torque_nm or _ZERO, c.torque_nm or _ZERO)
        total.append(BodyWrench(d.body, d.point_m, _vec3(force), _vec3(torque)))
    return {"ztcf": tuple(drift), "zvcf": tuple(control), "total": tuple(total)}


def net_ground(wrenches: Sequence[BodyWrench]) -> BodyWrench | None:
    """Net wrench of all planted feet about their mean point (``None``
    without feet)."""
    if not wrenches:
        return None
    points = np.array([w.point_m for w in wrenches], dtype=float)
    forces = np.array([w.force_n or _ZERO for w in wrenches], dtype=float)
    torques = np.array([w.torque_nm or _ZERO for w in wrenches], dtype=float)
    ref = points.mean(axis=0)
    moment = torques.sum(axis=0) + np.cross(points - ref, forces).sum(axis=0)
    return BodyWrench("net_ground", _vec3(ref), _vec3(forces.sum(0)), _vec3(moment))


def ground_arrows(
    ground_at: Callable[[int], Mapping[str, Sequence[BodyWrench]]],
    part: str,
    style: ForceGlyphStyle,
    look: Mapping[str, Any],
) -> Callable[[int], tuple[ArrowGlyph, ...]]:
    """``part``'s net ground arrow by state from its centre of pressure, in
    the hand arrows' case colour; none where that centre is undefined.
    ``ValueError`` unless ``part`` is one of :data:`CASES`."""
    colours = hand_styles()[look["case_colours_hand"]]
    _case(part)

    def at(index: int) -> tuple[ArrowGlyph, ...]:
        net = net_ground(ground_at(index)[part])
        if net is None:
            return ()
        force, torque = net.force_n or _ZERO, net.torque_nm or _ZERO
        cop = centre_of_pressure(net.point_m, force, torque, look["min_vertical_n"])
        if cop is None:
            return ()
        drift, control = (_ZERO, force) if part == "zvcf" else (force, _ZERO)
        label = look["arrow_label"]
        return part_arrows(cop, drift, control, style, part, "force", colours, label)

    return at


def case_force_key(
    part: str, look: Mapping[str, Any]
) -> tuple[str, tuple[int, int, int]]:
    """Colour-key row of ``part``'s ground arrow, in the hand case colour."""
    hand = hand_styles()[look["case_colours_hand"]]
    r, g, b, _a = hex_to_rgba(getattr(hand, _CASE_HEX[_case(part)]))
    rgb = (round(255 * r), round(255 * g), round(255 * b))
    return look["key_ground"].format(part=part.upper()), rgb


def with_part_wrenches(
    frame_at: Callable[[int], Any], part: str
) -> Callable[[int], Any]:
    """``frame_at`` keeping only ``part``'s force kinds (:data:`PART_KINDS`):
    the recorded contact arrows give way to :func:`ground_arrows`;
    ``recorded`` returns ``frame_at``. ``ValueError`` for another part."""
    if part not in FORCE_PARTS:
        raise ValueError(f"force part must be one of {FORCE_PARTS}, got {part!r}")
    if part == "recorded":
        return frame_at
    kinds = PART_KINDS[part]

    def at(index: int) -> Any:
        frame = frame_at(index)
        return replace(
            frame, wrenches=tuple(w for w in frame.wrenches if w.kind in kinds)
        )

    return at

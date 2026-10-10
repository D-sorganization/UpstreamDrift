"""Static-hold per-hand bar wrench for the Pinocchio lift pack (LIFT-4, #11744).

Reuses the shared GCV-7 grip analysis
(:mod:`src.shared.python.biomechanics.grip_wrench`) -- no second
wrench-transport routine.

Method: a rigid two-hand hold of a rigid bar is statically indeterminate,
and the Pinocchio pack URDF fuses the bar into the left hand's body
(``hand_l -> barbell_hand_l_pivot -> barbell_shaft``, both fixed joints)
with ``barbell_shaft -> barbell_grip_r`` a zero-mass virtual anchor at the
right grip, so Pinocchio merges every one of those links into ``hand_l``'s
rigid body and the engine cannot supply the hand/hand split directly.

Grip-frame note (load-bearing, verified against the real pack): despite its
name, ``barbell_hand_l_pivot`` is *not* the left hand's grip point -- the
weld ``barbell_to_hand_l`` offsets it from ``hand_l`` by the full grip
half-width so that it lands on the bar's own centre (``barbell_shaft``'s
local origin is welded to it with zero further offset). The left hand's
actual grip point is therefore the ``hand_l`` frame itself. ``barbell_grip_r``
has no such indirection: it is built from ``barbell_shaft``'s centre offset
by ``-grip_offset`` along Y, which places it (to FK precision, ~1e-6 m on
the real pack) at ``hand_r``'s own location -- it genuinely is the bar-side
right-hand grip frame.

Instead this module evaluates the bar's own mass and world centre of mass
from the URDF ``<inertial>`` of every ``barbell``-prefixed link, transported
by that link's *frame* placement at the pack start pose
(``forwardKinematics`` + ``updateFramePlacements`` -- frames exist for every
link regardless of joint type, unlike ``model.inertias`` which Pinocchio
merges upward through fixed joints). It then forms the net wrench the two
hands must exert on the bar for static equilibrium (``R = -m*g``, with the
moment of ``R`` about the grip midpoint balancing the weight acting at the
bar's centre of mass -- gravity has no moment about the bar's own centre of
mass) and splits that wrench with the shared ``allocate_min_norm`` minimum
norm allocator (through the engine-free :mod:`.bar_hold_static`).

This is a kinematic static-equilibrium calculation, not a simulation: there
is no bar acceleration to report (``bar_linear_accel_mps2`` is always
``None``, never a fabricated ``0.0``) and the force balance
(``relative_error``) is trivially ~0 by construction; the meaningful checks
are the bar mass against ``Anthropometry.bar_total_mass_kg`` and the
symmetry of the split.

Grip points are therefore ``hand_l`` (the left hand's own frame, since no
distinct bar-side left-grip frame exists) and ``barbell_grip_r`` (the
bar-side right-grip anchor). ``hand_grip_residual_m`` reports, per side, the
distance between the hand frame and the grip point actually used: always
``0.0`` for L (``hand_l`` *is* the point used, by construction -- not a
measurement that could fail) and the small FK closure residual between
``hand_r`` and ``barbell_grip_r`` for R.

Precondition: the pack must weld the bar to both hands. Squat welds the bar
to the torso instead (no ``barbell_grip_r`` anchor exists) and is reported
``unavailable``, never as a wrong or zero number.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from defusedxml import ElementTree as ET

from ..model import unavailable_bar_hold
from .bar_hold_static import static_hold_split

pin: Any = importlib.import_module("pinocchio")

_BAR_PREFIX = "barbell"
_RIGHT_GRIP_LINK = "barbell_grip_r"
#: The left hand's grip point is its own frame -- ``barbell_hand_l_pivot``
#: is welded to the *bar's centre*, not to the hand (see module docstring).
_LEFT_GRIP_FRAME = "hand_l"
_METHOD = (
    "kinematic static equilibrium (no simulation): bar mass/COM from the "
    "URDF <inertial> of every barbell-prefixed link transported by its "
    "frame placement at the pack start pose; required hand wrench "
    "R = -m*g with moment (r_com - r_mid) x R about the grip midpoint, "
    "split by the shared allocate_min_norm (GCV-7)"
)


def _vec3(text: str) -> np.ndarray:
    parts = [float(v) for v in text.split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 components, got {text!r}")
    return np.array(parts, dtype=float)


def _barbell_link_elements(root: ET.Element) -> dict[str, ET.Element]:
    return {
        link.get("name", ""): link
        for link in root.findall("link")
        if link.get("name", "").startswith(_BAR_PREFIX)
    }


def _link_mass_and_local_com(link: ET.Element) -> tuple[float, np.ndarray]:
    """Mass and local-frame centre of mass from one URDF ``<link>``."""
    inertial = link.find("inertial")
    if inertial is None:
        return 0.0, np.zeros(3)
    mass_el = inertial.find("mass")
    mass = float(mass_el.get("value", "0.0")) if mass_el is not None else 0.0
    origin = inertial.find("origin")
    xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    return mass, _vec3(xyz)


def _frame_translation(model: Any, data: Any, name: str) -> np.ndarray:
    frame_id = model.getFrameId(name, pin.BODY)
    return np.array(data.oMf[frame_id].translation, dtype=float)


def _frame_placement(model: Any, data: Any, name: str) -> Any:
    return data.oMf[model.getFrameId(name, pin.BODY)]


def _bar_mass_and_world_com(
    model: Any, data: Any, link_elements: dict[str, ET.Element]
) -> tuple[float, np.ndarray]:
    """Bar mass and world centre of mass from URDF inertials + frame placements.

    Pinocchio merges every fixed-joint link into its parent joint's rigid
    body, so ``model.inertias`` cannot be read per barbell link; the mass
    and local centre of mass instead come from the URDF ``<inertial>`` text,
    transported to world coordinates by that link's *frame* placement.

    Raises:
        ValueError: if the barbell links carry no positive mass.
    """
    total_mass = 0.0
    weighted_com = np.zeros(3)
    for name, link in link_elements.items():
        mass, local_com = _link_mass_and_local_com(link)
        if mass <= 0.0:
            continue
        placement = _frame_placement(model, data, name)
        world_com = placement.rotation @ local_com + placement.translation
        total_mass += mass
        weighted_com += mass * np.asarray(world_com, dtype=float)
    if total_mass <= 0.0:
        raise ValueError("barbell links carry no positive mass")
    return total_mass, weighted_com / total_mass


def bar_hold_wrench(
    model: Any, data: Any, root: ET.Element, q: np.ndarray
) -> dict[str, Any]:
    """Per-hand static-hold wrench on the bar for one Pinocchio lift pack model.

    Args:
        model: the pack's ``pin.Model`` (free-flyer root).
        data: ``model.createData()``.
        root: the parsed URDF ``<robot>`` element (the adapter's ``_root``).
        q: the pack's start configuration (``adapter._initial_q()``).

    Returns:
        A JSON-serialisable dict; see ``EngineAdapter.bar_hold_wrench`` for
        the field contract. Additionally carries ``hand_grip_residual_m``
        (``{"L": ..., "R": ...}``), the distance between each hand frame and
        the grip frame it holds, as a diagnostic only.

    Raises:
        TypeError: if ``root`` has no ``findall`` method, or ``q`` is not a
            1-D array matching ``model.nq``.
        ValueError: if the barbell links carry no mass, the grip points
            coincide, or the net vertical hand force is ~0.
    """
    if not hasattr(root, "findall"):
        raise TypeError("root must be a parsed URDF <robot> element")
    q_arr = np.asarray(q, dtype=float)
    if q_arr.ndim != 1 or q_arr.shape[0] != int(model.nq):
        raise TypeError(f"q must be a 1-D array of length model.nq ({model.nq})")

    links = {link.get("name", ""): link for link in root.findall("link")}
    if _RIGHT_GRIP_LINK not in links:
        return unavailable_bar_hold(
            f"no {_RIGHT_GRIP_LINK!r} anchor: the bar is not hand-held in "
            "this pack (e.g. squat welds the bar to the torso)"
        )

    pin.forwardKinematics(model, data, q_arr)
    pin.updateFramePlacements(model, data)

    bar_link_elements = _barbell_link_elements(root)
    bar_mass_kg, bar_com_world = _bar_mass_and_world_com(model, data, bar_link_elements)
    gravity = np.asarray(model.gravity.linear, dtype=float)

    grip_l = _frame_translation(model, data, _LEFT_GRIP_FRAME)  # == hand_l
    grip_r = _frame_translation(model, data, _RIGHT_GRIP_LINK)

    split = static_hold_split(bar_mass_kg, gravity, bar_com_world, grip_l, grip_r)
    left_f = split["hand_force_n"]["L"]
    right_f = split["hand_force_n"]["R"]
    sum_vertical_n = float(left_f[2] + right_f[2])
    if abs(sum_vertical_n) < 1e-9:
        raise ValueError("sum of vertical hand forces is ~0; cannot form a split")

    bar_weight_n = bar_mass_kg * split["gravity_mag_mps2"]
    n_welds = sum(
        1
        for joint in root.findall("joint")
        if joint.get("name", "").startswith("barbell_to_hand_")
    )
    hand_r = _frame_translation(model, data, "hand_r")

    return {
        "available": True,
        "reason": "",
        "split_method": "allocation",
        "bar_mass_kg": bar_mass_kg,
        "bar_weight_n": bar_weight_n,
        "hand_force_n": {"L": left_f.tolist(), "R": right_f.tolist()},
        "sum_vertical_n": sum_vertical_n,
        "relative_error": abs(sum_vertical_n - bar_weight_n) / bar_weight_n,
        "split_left_fraction": float(left_f[2] / sum_vertical_n),
        "couple_at_midpoint_nm": split["couple_at_midpoint_nm"].tolist(),
        "bar_linear_accel_mps2": None,
        "method": _METHOD,
        "n_welds": n_welds,
        "hand_grip_residual_m": {
            # grip_l IS the hand_l frame (see module docstring): the
            # residual is exactly 0 by construction, not a measurement.
            "L": 0.0,
            "R": float(np.linalg.norm(hand_r - grip_r)),
        },
    }

"""Contact wrenches from MuJoCo's own contact solution (GCV-2, #11708).

One extraction serves the MuJoCo and MyoSuite force sources (MyoSuite is
MuJoCo), so the contact-frame to world conversion, the contact torque of
``condim >= 4`` contacts and the ground-partner test live in one place.
"""

from __future__ import annotations

import mujoco
import numpy as np

from src.shared.python.biomechanics.ground_reaction_wrenches import (
    ground_reaction_overlay,
)
from src.shared.python.force_overlay.contracts import OverlayWrench, WrenchKind

__all__ = ["contact_wrenches", "ground_reaction_wrenches"]

_WORLD_BODY = 0


def _vec3(seq) -> tuple[float, float, float]:
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
    )


def contact_wrenches(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    engine: str,
    ground_only: bool = False,
) -> list[OverlayWrench]:
    """One world-frame ``CONTACT`` wrench per contact side on a non-world body.

    The force on ``geom2``'s body is ``frame.T @ f`` (MuJoCo convention) and
    the reaction on ``geom1``'s body its negative; the contact torque (friction
    spin and rolling) is included for ``condim >= 4`` and otherwise
    unavailable (``None``).  With ``ground_only`` only contacts whose partner
    is the world body are returned (ground reaction, not foot-on-club).

    Preconditions: ``data`` belongs to ``model`` and its contact list is current
    (after ``mj_forward``).
    """
    wrenches: list[OverlayWrench] = []
    c_force = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 < 0 or con.geom2 < 0:
            continue
        mujoco.mj_contactForce(model, data, i, c_force)
        frame = con.frame.reshape(3, 3)
        f_world = frame.T @ c_force[:3]
        t_world = frame.T @ c_force[3:6] if con.dim >= 4 else None
        point = _vec3(con.pos)
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        # (body, other body, sign): geom1's body feels -f, geom2's body +f.
        for body, other, sign in ((b1, b2, -1.0), (b2, b1, 1.0)):
            if body == _WORLD_BODY or (ground_only and other != _WORLD_BODY):
                continue
            name = _body_name(model, body)
            wrenches.append(
                OverlayWrench(
                    kind=WrenchKind.CONTACT,
                    label=f"contact:{name}:{i}",
                    body=name,
                    point_m=point,
                    force_n=_vec3(sign * f_world),
                    torque_nm=None if t_world is None else _vec3(sign * t_world),
                    source=engine,
                )
            )
    return wrenches


def ground_reaction_wrenches(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    engine: str,
    ground_height_m: float = 0.0,
) -> tuple[OverlayWrench, ...]:
    """GRF breakdown overlay of the foot-on-ground contacts (GCV-1 labels).

    The centre of mass is ``data.subtree_com[0]`` (current after ``mj_forward``).
    Empty when no foot is loaded (unavailable, never zero).
    """
    ground = contact_wrenches(model, data, engine=engine, ground_only=True)
    return ground_reaction_overlay(
        ground,
        data.subtree_com[_WORLD_BODY],
        source=f"{engine}:mj_contactForce",
        ground_height_m=ground_height_m,
    )

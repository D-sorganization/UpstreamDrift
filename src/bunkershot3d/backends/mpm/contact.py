"""Contact-wrench extraction from MuJoCo, with an explicit sign convention.

Issue #8612 (findings B5 and B5b).

**Sign (B5).** ``mj_contactForce`` returns the contact wrench in the *contact*
frame; ``contact.frame`` holds the frame's basis vectors as rows in world
coordinates, so ``frame.T @ raw`` maps contact -> world. Verified empirically
against ``qfrc_constraint`` (a sphere at rest on a plane reports exactly
``m g = 41.888 N``): the result is the world force on **geom2's** body. The
force on a given body is therefore ``+`` when it is geom2 and ``-`` when it is
geom1. The previous code added it unsigned, so a body appearing as geom1 in one
pair and geom2 in another had its two contributions cancel instead of add.
MuJoCo orders a contact pair by *collider type*, not geom id, so which side the
clubhead lands on is not something a caller can assume.

**Moment (B5b).** ``raw[3:]`` is only the torsional/rolling friction couple *at
the contact point*. The dominant term for a wedge — the ``(r_contact - r_CoM) x
F`` moment arm, the entire reason a sole digs, twists or resists opening — was
absent. Both terms are summed here about the body's centre of mass.
"""

from __future__ import annotations

import typing

import numpy as np


def contact_wrench_on_body(
    model: typing.Any,
    data: typing.Any,
    body_id: int,
    *,
    mujoco: typing.Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Total contact force and moment acting **on** ``body_id``.

    Args:
        model: ``mujoco.MjModel``.
        data: ``mujoco.MjData`` already advanced to the state of interest.
        body_id: Body the wrench is reported for.
        mujoco: The ``mujoco`` module; imported lazily when omitted so this
            module stays importable without the optional dependency.

    Returns:
        ``(force, moment)``, each ``(3,)`` in world coordinates. The moment is
        taken about the body's centre of mass (``data.xipos``).
    """
    if mujoco is None:  # pragma: no cover - exercised via the driver
        import mujoco as mujoco_module_local

        mujoco = mujoco_module_local

    force_total = np.zeros(3)
    torque_total = np.zeros(3)
    centre_of_mass = np.asarray(data.xipos[body_id], dtype=float)

    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])

        if body1 == body2:
            continue  # self-contact contributes no net wrench
        if body2 == body_id:
            sign = 1.0
        elif body1 == body_id:
            sign = -1.0
        else:
            continue

        raw = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, raw)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)

        force = sign * (frame.T @ raw[:3])
        friction_couple = sign * (frame.T @ raw[3:])
        lever = np.asarray(contact.pos, dtype=float) - centre_of_mass

        force_total += force
        torque_total += np.cross(lever, force) + friction_couple

    return force_total, torque_total

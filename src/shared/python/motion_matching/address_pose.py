"""Address pose of an address-only pipeline report as a MuJoCo ``qpos`` (SHOW-C7, #14563).

``address_report.json`` lists the address coordinates by joint name: hinge
angles in degrees under ``address_coordinates_deg`` and the root slides in
metres under ``address_translations_m``.  Joints the report does not name
(free joints, the club) keep the model default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np


def address_qpos(model: mujoco.MjModel, report: Mapping[str, Any]) -> np.ndarray:
    """``qpos`` with every named slide/hinge joint set from ``report``.

    Raises ``ValueError`` if a slide or hinge joint of ``model`` is named in
    neither report table (the pose would silently be wrong).
    """
    slides = report["address_translations_m"]
    hinges = report["address_coordinates_deg"]
    qpos = np.array(model.qpos0, dtype=float)
    for jid in range(model.njnt):
        kind = model.jnt_type[jid]
        if kind not in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        table = slides if kind == mujoco.mjtJoint.mjJNT_SLIDE else hinges
        if name not in table:
            raise ValueError(f"address report has no value for joint {name!r}")
        value = float(table[name])
        qpos[model.jnt_qposadr[jid]] = (
            value if kind == mujoco.mjtJoint.mjJNT_SLIDE else math.radians(value)
        )
    return qpos

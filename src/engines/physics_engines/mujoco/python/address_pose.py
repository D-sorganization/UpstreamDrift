"""Address pose of an address-only pipeline report as a MuJoCo ``qpos`` (SHOW-C7, #14563).

``address_report.json`` lists the address coordinates by joint name: hinge
angles in degrees under ``address_coordinates_deg`` and the root slides in
metres under ``address_translations_m``.  Free and ball joints are not part of
the report and keep the model default; every slide and hinge joint must be
named, the club's included.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np

from src.shared.python.contracts import require


def address_qpos(model: mujoco.MjModel, report: Mapping[str, Any]) -> np.ndarray:
    """``qpos`` with every named slide/hinge joint set from ``report``.

    Pre-condition (``contracts.require``): every slide or hinge joint of
    ``model`` is named in its report table, else the pose would silently be
    wrong (``ContractViolationError``, a ``ValueError``).
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
        require(name in table, f"address report has no value for joint {name!r}", name)
        value = float(table[name])
        qpos[model.jnt_qposadr[jid]] = (
            value if kind == mujoco.mjtJoint.mjJNT_SLIDE else math.radians(value)
        )
    return qpos

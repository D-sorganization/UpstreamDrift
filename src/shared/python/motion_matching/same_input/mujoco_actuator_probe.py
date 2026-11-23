"""Research seam for memoryless MuJoCo motor effort into the existing weld solve.

Motor XML and bounds are supplied by the caller, not inferred as anatomy. This
is neither a physical feasibility gate nor an actuator model for the v2 spec.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)

Array = NDArray[np.float64]


class ControlledMujocoDrive:
    """Use MuJoCo's saturated motor force once as the KKT generalized input.

    Restricted to scalar-joint memoryless motors. Each call starts from an
    explicit q/v/control, and the temporary model has the same base XML as the
    constrained plant; no implicit actuator activation state is accepted.
    """

    def __init__(self, plant: NativeMujocoFullBodyModel, actuator_xml: str) -> None:
        self.plant = plant
        actuators = ET.fromstring(actuator_xml)
        if actuators.tag != "actuator" or not len(actuators):
            raise ValueError("nonempty actuator set required")
        if any(
            child.tag != "motor" or "joint" not in child.attrib for child in actuators
        ):
            raise ValueError(
                "stateful or unsupported actuator; only joint motors allowed"
            )
        root = ET.fromstring(plant.xml)
        if root.find("actuator") is not None:
            raise ValueError("base model already contains actuators")
        root.append(actuators)
        self.model = plant._mj.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        if self.model.nu == 0 or self.model.na:
            raise ValueError("stateful or missing actuator")
        self.data = plant._mj.MjData(self.model)
        if self.model.nq != plant.model.nq or self.model.nv != plant.model.nv:
            raise ValueError("motor model does not match base coordinate dimensions")

    def acceleration(self, q: Array, v: Array, control: Array) -> tuple[Array, Array]:
        n = len(self.plant.coordinate_order)
        q, v, control = (np.asarray(x, dtype=float) for x in (q, v, control))
        if q.shape != (n,) or v.shape != (n,) or control.shape != (self.model.nu,):
            raise ValueError("state or control dimensions do not match")
        if not all(np.isfinite(x).all() for x in (q, v, control)):
            raise ValueError("state and control must be finite")
        named_q = dict(zip(self.plant.coordinate_order, map(float, q), strict=True))
        named_v = dict(zip(self.plant.coordinate_order, map(float, v), strict=True))
        self.data.qpos[:] = self.plant._vector(named_q)
        self.data.qvel[:] = self.plant._vector(named_v)
        self.data.ctrl[:] = control
        mj = self.plant._mj
        mj.mj_fwdPosition(self.model, self.data)
        mj.mj_fwdVelocity(self.model, self.data)
        mj.mj_fwdActuation(self.model, self.data)
        raw = self.data.qfrc_actuator.copy()
        if not np.isfinite(raw).all() or np.any(raw[self.plant._passive_dofs] != 0):
            raise ValueError(
                "unsupported force on passive shaft or nonfinite actuation"
            )
        applied = raw[self.plant._order_dofs].copy()
        named_tau = dict(
            zip(self.plant.coordinate_order, map(float, applied), strict=True)
        )
        result = self.plant.accelerations(named_q, named_v, named_tau)
        return np.array([result[name] for name in self.plant.coordinate_order]), applied

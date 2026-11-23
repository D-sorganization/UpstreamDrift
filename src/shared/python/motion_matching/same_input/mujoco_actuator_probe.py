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


class MotorRolloutPlant:
    """No-impact research plant: held controls, measured motor input at each RK4 stage."""

    drive_evidence_semantics = "mujoco_post_saturation_generalized_motor"
    supports_impact = False
    supports_projection = False

    def __init__(
        self, drive: ControlledMujocoDrive, control_indices: NDArray[np.int64]
    ) -> None:
        self.drive = drive
        self.coordinate_order = tuple(drive.plant.coordinate_order)
        indices = np.asarray(control_indices)
        if indices.shape != (drive.model.nu,) or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("one coordinate index per motor control is required")
        if (
            np.any(indices < 6)
            or np.any(indices >= len(self.coordinate_order))
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("control indices must be distinct non-root coordinates")
        motor_dofs = drive.model.jnt_dofadr[drive.model.actuator_trnid[:, 0]]
        if not np.array_equal(motor_dofs, drive.plant._order_dofs[indices]):
            raise ValueError("control indices do not match motor joint order")
        self.indices = indices.copy()

    def acceleration_with_drive(
        self, q: Array, v: Array, command: Array
    ) -> tuple[Array, Array, Array]:
        values = np.asarray(command, dtype=float)
        if (
            values.shape != (len(self.coordinate_order),)
            or not np.isfinite(values).all()
        ):
            raise ValueError("command must be a finite coordinate-order vector")
        unsupported = np.ones(len(values), dtype=bool)
        unsupported[self.indices] = False
        if np.any(values[unsupported] != 0):
            raise ValueError("command includes unmapped coordinates")
        acceleration, applied = self.drive.acceleration(q, v, values[self.indices])
        return acceleration, applied, np.zeros_like(applied)

    def acceleration(self, q: Array, v: Array, command: Array) -> Array:
        return self.acceleration_with_drive(q, v, command)[0]

    def closure_pose_residual(self, q: Array) -> Array:
        n = len(self.coordinate_order)
        self.drive.acceleration(q, np.zeros(n), np.zeros(self.drive.model.nu))
        pose, _ = self.drive.plant.closure_errors()
        return np.asarray(pose, dtype=float)

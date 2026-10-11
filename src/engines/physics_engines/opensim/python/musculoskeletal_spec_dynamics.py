"""Inverse dynamics and ground contact of the spec skeleton in OpenSim.

Part of issue #11617 (epic #11605), phase 2.  The reference motion was produced
with a *deterministic* ground-contact law (``contact_law.sphere_ground_contact``
applied to the spec's contact spheres), so the ground reaction at every sample is
a function of ``(q, v)`` alone.  This module evaluates that law on an OpenSim
model of the spec skeleton, so the exact reaction forces replace the previous
friction-pyramid estimate, and checks that OpenSim inverse dynamics with those
forces reproduces the reference joint efforts.

The grip loop (``RHandStandoff`` to ``Clubface``) is *not* modelled: the weld
reaction acts only on the arm/scapula coordinates, so efforts of the legs, pelvis
and trunk coordinates are directly comparable to open-chain inverse dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import musculoskeletal_graft as graft
from src.engines.physics_engines.opensim.python.full_body_osim import (
    clean_osim_body_name,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_grf import (
    ContactPoint,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_spec_model import (
    load_spec_skeleton,
)
from src.shared.python.contracts import require
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
)

#: Coordinates whose efforts carry the grip-weld reaction (arms and scapulae).
LOOP_COORDINATE_PREFIXES: tuple[str, ...] = (
    "LE",
    "LF",
    "LScap",
    "LS",
    "LW",
    "RE",
    "RF",
    "RScap",
    "RS",
    "RW",
)


def is_loop_coordinate(name: str) -> bool:
    """True for the arm/scapula spec coordinates the grip weld acts on."""
    return name.endswith(("Input", "InputX", "InputY", "InputZ")) and name.startswith(
        LOOP_COORDINATE_PREFIXES
    )


@dataclass(frozen=True)
class ContactSphere:
    """A spec contact sphere: body, station (body frame, m) and radius."""

    name: str
    body: str
    station: tuple[float, float, float]
    radius_m: float


def _arr(matrix: Any) -> np.ndarray:
    return np.array(matrix.to_numpy(), dtype=float, copy=True)


def contact_spheres(spec: dict[str, Any]) -> tuple[ContactSphere, ...]:
    """Contact spheres of a full-body spec, in document order."""
    return tuple(
        ContactSphere(
            s["name"],
            clean_osim_body_name(s["body"]),
            tuple(s["position_m"]),
            s["radius_m"],
        )
        for s in spec["contact"]["spheres"]
    )


class SkeletonDynamics:
    """OpenSim spec skeleton with the shared contact law and inverse dynamics.

    Args:
        spec_bytes: ``full-body-v1`` specification the reference was produced with.

    Postconditions: ``coordinate_order`` is the spec order; every method takes and
    returns arrays in that order.
    """

    def __init__(self, spec_bytes: bytes) -> None:
        import opensim

        self._osim = opensim
        self.spec = graft.spec_document(spec_bytes)
        self.coordinate_order: tuple[str, ...] = tuple(self.spec["coordinate_order"])
        self.nv = len(self.coordinate_order)
        self.model = load_spec_skeleton(spec_bytes)
        self.state = self.model.initSystem()
        self.matter = self.model.getMatterSubsystem()
        self._id = opensim.InverseDynamicsSolver(self.model)
        require(
            self.state.getNQ() == self.nv and self.state.getNU() == self.nv,
            "OpenSim coordinate inventory differs from the spec",
        )
        self._slot = self._probe_slots()
        contact = self.spec["contact"]
        self.parameters = ContactParameters(**contact["parameters"])
        gravity = np.asarray(self.spec["gravity_m_s2"], dtype=float)
        up = -gravity / math.sqrt(float(gravity @ gravity))
        self.ground = GroundPlane(
            normal=(float(up[0]), float(up[1]), float(up[2])),
            height_m=float(contact["ground"]["height_m"] or 0.0),
        )
        self.spheres = contact_spheres(self.spec)

    def _single_nonzero(self, vector: Any) -> int:
        values = np.array([vector.get(i) for i in range(vector.size())])
        index = np.flatnonzero(values)
        require(index.size == 1, "coordinate must drive exactly one mobility")
        return int(index[0])

    def _probe_slots(self) -> np.ndarray:
        slots: list[int] = []
        coords = self.model.getCoordinateSet()
        for name in self.coordinate_order:
            self.state.updQ().setToZero()
            coords.get(name).setValue(self.state, 1.0, False)
            slots.append(self._single_nonzero(self.state.getQ()))
        self.state.updQ().setToZero()
        self.state.updU().setToZero()
        require(sorted(slots) == list(range(self.nv)), "unsupported Q/U layout")
        return np.asarray(slots, dtype=int)

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set coordinates and rates (spec order) and realise velocities."""
        require(q.shape == (self.nv,) and v.shape == (self.nv,), "state size mismatch")
        q_raw, u_raw = self.state.updQ(), self.state.updU()
        for k, slot in enumerate(self._slot):
            q_raw.set(int(slot), float(q[k]))
            u_raw.set(int(slot), float(v[k]))
        self.model.realizeVelocity(self.state)

    def body_origins(self, bodies: list[str]) -> np.ndarray:
        """Ground positions of body origins at the current state, ``(n, 3)``."""
        out = []
        for name in bodies:
            p = self.model.getBodySet().get(name).getTransformInGround(self.state).p()
            out.append([p.get(i) for i in range(3)])
        return np.asarray(out, dtype=float)

    def contact_samples(self) -> list[tuple[ContactSphere, ContactSample, np.ndarray]]:
        """Shared-law sample and 3 x nv world Jacobian (spec order) per sphere."""
        osim = self._osim
        u = np.array([self.state.getU().get(int(s)) for s in self._slot])
        result = []
        for sphere in self.spheres:
            body = self.model.getBodySet().get(sphere.body)
            index = body.getMobilizedBodyIndex()
            point = osim.Vec3(*(float(x) for x in sphere.station))
            jac = osim.Matrix(3, self.nv)
            self.matter.calcStationJacobian(self.state, index, point, jac)
            jacobian = np.take(_arr(jac), self._slot, axis=1)
            centre = body.findStationLocationInGround(self.state, point)
            position = np.array([centre.get(i) for i in range(3)])
            sample = sphere_ground_contact(
                position, jacobian @ u, sphere.radius_m, self.ground, self.parameters
            )
            result.append((sphere, sample, jacobian))
        return result

    def contact_generalised_force(self) -> tuple[np.ndarray, np.ndarray]:
        """``(generalised force (nv,), per-sphere world force (n, 3))`` of the contact."""
        total = np.zeros(self.nv)
        forces = []
        for _, sample, jacobian in self.contact_samples():
            force = sample.normal_force_n + sample.friction_force_n
            forces.append(force)
            if sample.penetration_m > 0.0:
                total += jacobian.T @ force
        return total, np.asarray(forces, dtype=float)

    def inverse_dynamics(self, acceleration: np.ndarray) -> np.ndarray:
        """Generalised forces (spec order) for ``acceleration`` at the current state.

        Contact is *not* included: the result is ``M a + C - gravity`` and the
        caller subtracts the contact generalised force.
        """
        udot = self._osim.Vector(self.nv, 0.0)
        for k, slot in enumerate(self._slot):
            udot.set(int(slot), float(acceleration[k]))
        raw = np.array(self._id.solve(self.state, udot).to_numpy(), dtype=float)
        return np.take(raw.reshape(-1), self._slot)

    def required_efforts(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Efforts that open-chain dynamics needs at ``(q, v, a)`` and the contact forces.

        Returns ``(efforts, sphere_forces)`` with
        ``efforts = ID(a) - J_contact^T f_contact``.
        """
        self.set_state(q, v)
        contact, forces = self.contact_generalised_force()
        return self.inverse_dynamics(a) - contact, forces


def foot_contact_points(spheres: tuple[ContactSphere, ...]) -> tuple[ContactPoint, ...]:
    """Contact spheres as ``ContactPoint`` records for ``write_external_loads``."""
    return tuple(
        ContactPoint(s.name, s.body, s.station, "r" if s.body.endswith("_r") else "l")
        for s in spheres
    )

"""OpenSim/Simbody adapter for same-input cross-engine dynamics parity (#11611).

The adapter builds the spec's rigid-body tree with the project's own
``export_full_body_osim`` exporter and then uses OpenSim only for what it is
authoritative at: multibody kinematics, the joint-space mass matrix, the
velocity-dependent plus gravity bias and station Jacobians.  Everything that
defines the *problem* is shared with the Drake and Pinocchio adapters:

* ground contact is the shared law
  :func:`~src.shared.python.motion_matching.contact_law.sphere_ground_contact`
  applied to the spec's contact spheres (OpenSim's ``HuntCrossleyForce`` is a
  different law and is deliberately not used);
* the dual-grip weld (``RHandStandoff`` to ``Clubface Vector``, six
  dimensional) is solved EXACTLY through the same unregularised KKT system,
  with no Baumgarte term and no OpenSim constraint enforcement.

The exporter's force, constraint, marker and contact-geometry sets are removed
after loading, so the Simbody system holds only the bodies, joints and
gravity.  Efforts are joint-conjugate generalized forces in the spec
coordinate order.

Weld residuals use world-frame relative velocity of the two closure frames
(translation of the placement origins, then angular), so the rate residual is
linear in the rates and its bias is the relative station/angular acceleration
at zero generalized acceleration.  On the closure manifold this is the
Pinocchio/Drake constraint up to an invertible row transform; off it, the
shared closest-point projection (``same_input.closure``) removes the
convention dependence.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.drake.python.full_body_model import (
    solve_weld_acceleration,
)
from src.engines.physics_engines.opensim.python.full_body_osim import (
    clean_osim_body_name,
    export_full_body_osim,
)
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
)
from src.shared.python.motion_matching.full_body_spec import FULL_BODY_SCHEMA_VERSION

Array = NDArray[np.float64]

_REMOVED_SETS = (
    "updMarkerSet",
    "updForceSet",
    "updConstraintSet",
    "updContactGeometrySet",
)


def _to_array(matrix: Any) -> Array:
    return np.array(matrix.to_numpy(), dtype=float, copy=True)


class OpenSimFullBodyParityAdapter:
    """Constrained forward dynamics of the full-body spec on OpenSim/Simbody.

    Args:
        spec_bytes: Full-body spec document (``full-body-v1``).

    Postconditions: ``coordinate_order`` equals the spec order; every public
    method takes and returns name-keyed dictionaries in that vocabulary.
    """

    def __init__(self, spec_bytes: bytes) -> None:
        spec = json.loads(spec_bytes)
        if spec.get("schema_version") != FULL_BODY_SCHEMA_VERSION:
            raise ValueError("Unsupported full-body schema: expected full-body-v1")
        self.specification = spec
        self.coordinate_order: tuple[str, ...] = tuple(spec["coordinate_order"])
        self.nv = len(self.coordinate_order)
        self._osim: Any = import_module("opensim")
        self._osim.Logger.setLevelString("Warn")
        self._build_model(spec_bytes)
        self._probe_state_layout()
        self._init_closure(spec["closure"])
        self._init_frames(spec)
        self._init_contact(spec)
        self.mass_kg = float(self._matter.calcSystemMass(self._state))

    # -- construction -------------------------------------------------------
    def _build_model(self, spec_bytes: bytes) -> None:
        osim = self._osim
        xml, _ = export_full_body_osim(spec_bytes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_body.osim"
            path.write_text(xml, encoding="utf-8")
            self._model = osim.Model(str(path))
        for remove in _REMOVED_SETS:
            getattr(self._model, remove)().clearAndDestroy()
        self._state = self._model.initSystem()
        self._matter = self._model.getMatterSubsystem()
        self._inverse_dynamics = osim.InverseDynamicsSolver(self._model)
        if self._state.getNQ() != self.nv or self._state.getNU() != self.nv:
            raise ValueError("OpenSim coordinate inventory differs from the spec")

    def _probe_state_layout(self) -> None:
        """Map each spec coordinate to its slot in Simbody's Q and U vectors."""
        coords = self._model.getCoordinateSet()
        state = self._state
        q_slot: list[int] = []
        u_slot: list[int] = []
        for name in self.coordinate_order:
            coordinate = coords.get(name)
            state.updQ().setToZero()
            state.updU().setToZero()
            coordinate.setValue(state, 1.0, False)
            q_slot.append(self._single_nonzero(state.getQ()))
            state.updQ().setToZero()
            coordinate.setSpeedValue(state, 1.0)
            u_slot.append(self._single_nonzero(state.getU()))
        state.updQ().setToZero()
        state.updU().setToZero()
        if sorted(q_slot) != list(range(self.nv)) or q_slot != u_slot:
            raise ValueError("Unsupported OpenSim Q/U layout for the spec joints")
        self._slot = np.asarray(q_slot, dtype=int)

    def _single_nonzero(self, vector: Any) -> int:
        values = np.array([vector.get(i) for i in range(vector.size())])
        index = np.flatnonzero(values)
        if index.size != 1:
            raise ValueError("Coordinate does not drive exactly one mobility")
        return int(index[0])

    def _body(self, spec_body: str) -> Any:
        return self._model.getBodySet().get(clean_osim_body_name(spec_body))

    def _init_closure(self, closure: Mapping[str, Any]) -> None:
        self._closure = []
        for suffix in ("a", "b"):
            placement = np.asarray(closure[f"placement_{suffix}"], dtype=float)
            self._closure.append((self._body(closure[f"body_{suffix}"]), placement))

    def _init_frames(self, spec: Mapping[str, Any]) -> None:
        self._frames = [
            (frame["name"], self._body(frame["body"]), np.asarray(frame["placement"]))
            for frame in spec["frames"]
        ]

    def _init_contact(self, spec: Mapping[str, Any]) -> None:
        contact = spec["contact"]
        self.contact_parameters = ContactParameters(**contact["parameters"])
        gravity = np.asarray(spec["gravity_m_s2"], dtype=float)
        magnitude = float(math.sqrt(np.dot(gravity, gravity)))
        if magnitude <= 0.0:
            raise ValueError("Gravity must be a nonzero vector")
        up = -gravity / magnitude
        self.ground = GroundPlane(
            normal=(float(up[0]), float(up[1]), float(up[2])),
            height_m=float(contact["ground"]["height_m"] or 0.0),
        )
        self._spheres = [
            (
                sphere["name"],
                self._body(sphere["body"]),
                np.asarray(sphere["position_m"], dtype=float),
                float(sphere["radius_m"]),
            )
            for sphere in contact["spheres"]
        ]

    # -- state --------------------------------------------------------------
    def _vector(self, values: Mapping[str, float]) -> Array:
        if set(values) != set(self.coordinate_order):
            raise ValueError("Provide exactly the model coordinate inventory")
        vector = np.array([values[name] for name in self.coordinate_order], float)
        if not np.isfinite(vector).all():
            raise ValueError("Coordinates, rates and efforts must be finite")
        return vector

    def _set_state(self, q: Array, u: Array | None = None) -> None:
        state = self._state
        q_raw, u_raw = state.updQ(), state.updU()
        for k, slot in enumerate(self._slot):
            q_raw.set(int(slot), float(q[k]))
            u_raw.set(int(slot), 0.0 if u is None else float(u[k]))
        if u is None:
            self._model.realizePosition(state)
        else:
            self._model.realizeVelocity(state)

    def _ordered(self, raw: Array, axis: int = -1) -> Array:
        return np.take(raw, self._slot, axis=axis)

    # -- kinematics ---------------------------------------------------------
    @staticmethod
    def _pose(body: Any, state: Any, placement: Array) -> Array:
        transform = body.getTransformInGround(state)
        pose = np.eye(4)
        rotation = transform.R()
        pose[:3, :3] = [[rotation.get(i, j) for j in range(3)] for i in range(3)]
        pose[:3, 3] = [transform.p().get(i) for i in range(3)]
        return pose @ placement

    def frame_poses(self, coordinates: Mapping[str, float]) -> dict[str, Array]:
        """4x4 world poses of the spec marker-reference frames."""
        self._set_state(self._vector(coordinates))
        return {
            name: self._pose(body, self._state, placement)
            for name, body, placement in self._frames
        }

    def mass_matrix(self, coordinates: Mapping[str, float]) -> Array:
        """Joint-space mass matrix in spec coordinate order."""
        self._set_state(self._vector(coordinates))
        matrix = self._osim.Matrix(self.nv, self.nv)
        self._matter.calcM(self._state, matrix)
        raw = _to_array(matrix)
        return raw[np.ix_(self._slot, self._slot)]

    def _bias_forces(self) -> Array:
        """Generalized force ``C(q, u) - gravity`` (inverse dynamics at udot=0)."""
        zero = self._osim.Vector(self.nv, 0.0)
        raw = np.array(
            self._inverse_dynamics.solve(self._state, zero).to_numpy(), float
        )
        return self._ordered(raw.reshape(-1))

    def _station_rows(
        self, body: Any, station: Array
    ) -> tuple[Array, Array, Array, Array]:
        """World-frame frame Jacobian and zero-udot bias of a body-fixed point.

        Returns ``(J_linear, J_angular, bias_linear, bias_angular)``; the
        Jacobians are 3 x nv in spec order.
        """
        osim = self._osim
        index = body.getMobilizedBodyIndex()
        point = osim.Vec3(*(float(x) for x in station))
        frame = osim.Matrix(6, self.nv)
        self._matter.calcFrameJacobian(self._state, index, point, frame)
        jacobian = self._ordered(_to_array(frame))
        bias = self._matter.calcBiasForFrameJacobian(self._state, index, point)
        bias_angular = np.array([bias.get(0).get(i) for i in range(3)])
        bias_linear = np.array([bias.get(1).get(i) for i in range(3)])
        return jacobian[3:], jacobian[:3], bias_linear, bias_angular

    # -- weld closure -------------------------------------------------------
    def _closure_blocks(self) -> tuple[Array, Array, Array]:
        """Pose residual (6), rate Jacobian (6 x nv) and rate bias (6)."""
        poses, jac_lin, jac_ang, bias_lin, bias_ang = [], [], [], [], []
        for body, placement in self._closure:
            poses.append(self._pose(body, self._state, placement))
            jl, ja, bl, ba = self._station_rows(body, placement[:3, 3])
            jac_lin.append(jl)
            jac_ang.append(ja)
            bias_lin.append(bl)
            bias_ang.append(ba)
        pose_a, pose_b = poses
        pose_residual = np.concatenate(
            (
                pose_b[:3, 3] - pose_a[:3, 3],
                Rotation.from_matrix(pose_b[:3, :3] @ pose_a[:3, :3].T).as_rotvec(),
            )
        )
        jacobian = np.vstack((jac_lin[1] - jac_lin[0], jac_ang[1] - jac_ang[0]))
        bias = np.concatenate((bias_lin[1] - bias_lin[0], bias_ang[1] - bias_ang[0]))
        return pose_residual, jacobian, bias

    def closure_residuals(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> tuple[Array, Array]:
        """Weld pose residual and rate residual (linear in ``rates``)."""
        q, u = self._vector(coordinates), self._vector(rates)
        self._set_state(q, u)
        pose, jacobian, _ = self._closure_blocks()
        return pose, jacobian @ u

    # -- contact ------------------------------------------------------------
    def contact_forces(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> dict[str, ContactSample]:
        """Shared-law contact sample of every spec contact sphere."""
        self._set_state(self._vector(coordinates), self._vector(rates))
        return {name: sample for name, (sample, _) in self._contact().items()}

    def _contact(self) -> dict[str, tuple[ContactSample, Array]]:
        """Per sphere: shared-law sample and its 3 x nv world Jacobian."""
        osim = self._osim
        u = self._ordered(np.array([self._state.getU().get(i) for i in range(self.nv)]))
        result = {}
        for name, body, offset, radius in self._spheres:
            index = body.getMobilizedBodyIndex()
            point = osim.Vec3(*(float(x) for x in offset))
            jac = osim.Matrix(3, self.nv)
            self._matter.calcStationJacobian(self._state, index, point, jac)
            jacobian = self._ordered(_to_array(jac))
            center = body.findStationLocationInGround(self._state, point)
            position = np.array([center.get(i) for i in range(3)])
            sample = sphere_ground_contact(
                position, jacobian @ u, radius, self.ground, self.contact_parameters
            )
            result[name] = (sample, jacobian)
        return result

    # -- dynamics -----------------------------------------------------------
    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Constrained generalized accelerations with the exact weld solve."""
        q, u = self._vector(coordinates), self._vector(rates)
        tau = self._vector(efforts)
        self._set_state(q, u)
        matrix = self._osim.Matrix(self.nv, self.nv)
        self._matter.calcM(self._state, matrix)
        mass = _to_array(matrix)[np.ix_(self._slot, self._slot)]
        force = tau - self._bias_forces()
        for sample, jacobian in self._contact().values():
            if sample.penetration_m > 0.0:
                force += jacobian.T @ (sample.normal_force_n + sample.friction_force_n)
        _, jacobian, bias = self._closure_blocks()
        acceleration = solve_weld_acceleration(mass, force, jacobian, bias)
        return dict(zip(self.coordinate_order, map(float, acceleration), strict=True))


def build_parity_adapter(spec_bytes: bytes) -> OpenSimFullBodyParityAdapter:
    """Entry point used by ``same_input.plant.VectorPlant``."""
    return OpenSimFullBodyParityAdapter(spec_bytes)

"""Phase 5 Closed-Loop Snapshot Engine Adapters (NSX-51 through NSX-54).

Provides canonical ``ClosedLoopSnapshot`` adapters across:
- Pinocchio (``PinocchioClosedLoopSnapshotAdapter``)
- MuJoCo hard-KKT (``MuJoCoClosedLoopSnapshotAdapter``, #12579 / NSX-51)
- MyoSuite (``MyoSuiteClosedLoopSnapshotAdapter``, #12580 / NSX-52)
- Drake native ``MultibodyPlant`` (``DrakeClosedLoopSnapshotAdapter`` / ``DrakeForceAdapter``, #12581 / NSX-53)
- OpenSim ``OpenSimFullBodyParityAdapter`` (``OpenSimClosedLoopSnapshotAdapter`` / ``OpenSimForceAdapter``, #12582 / NSX-54)

All adapters convert their engine-native 6-D weld closure Jacobian and velocity
drift into ``CANONICAL_CLOSURE_CONVENTION`` (``[v_x, v_y, v_z, omega_x, omega_y, omega_z]``
expressed in the grip frame ``G`` about the grip origin ``p_G``, relative direction
``body_a_minus_body_b``) via ``to_canonical_closure``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.engine_core.capabilities import CapabilityLevel
from src.shared.python.engine_core.mujoco_compat import full_mass_matrix

Array = NDArray[np.float64]
DriftMethod = Literal["analytic", "finite_difference_centered"]


class CapabilityUnavailableError(RuntimeError):
    """Raised when an adapter cannot produce a FULL-capability closed-loop snapshot."""


def _skew(vec: Array) -> Array:
    """Return 3x3 skew-symmetric cross-product matrix ``[v]_x``."""
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


try:
    from src.shared.python.loop_nullspace.contracts import (  # type: ignore[import-not-found]
        ClosedLoopSnapshot,
        ConstraintTopology,
        GripClosureMetadata,
        ModelProvenance,
    )
except ImportError:

    @dataclass(frozen=True)
    class GripClosureMetadata:
        """Metadata describing the 6-DOF hand-to-club weld closure convention."""

        hand_a_body: str = "rhand"
        hand_b_body: str = "club"
        frame_convention: str = "grip_local_lin_ang"
        grip_origin_in_club_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @dataclass(frozen=True)
    class ConstraintTopology:
        """Kinematic and constraint topology metadata for a closed-loop snapshot."""

        coordinate_order: tuple[str, ...]
        actuated_indices: tuple[int, ...]
        floating_base_indices: tuple[int, ...] = ()
        contact_names: tuple[str, ...] = ()
        closure: GripClosureMetadata = field(default_factory=GripClosureMetadata)

    @dataclass(frozen=True)
    class ModelProvenance:
        """Cryptographic and inertial provenance of the multibody model."""

        spec_sha256: str
        inertial_source: str = "full_body_spec"
        bare_club_bench_id: str | None = None
        engine_name: str = "analytical"
        drift_method: str = "analytic"
        kkt_regularization: float = 0.0
        matlab_release: str | None = None

    @dataclass(frozen=True)
    class ClosedLoopSnapshot:
        """Immutable closed-loop multibody dynamics snapshot at a single state."""

        q: Array
        v: Array
        a: Array
        M: Array
        h: Array
        J_c: Array
        Jdot_v: Array
        S: Array
        J_g: Array | None
        topology: ConstraintTopology
        model_provenance: ModelProvenance

        def closure_rows(self) -> int:
            """Return the number of scalar closure rows ``m_c``."""
            return int(self.J_c.shape[0])


try:
    from src.shared.python.loop_nullspace.closure_convention import (  # type: ignore[import-not-found]
        CANONICAL_CLOSURE_CONVENTION,
        DRAKE_CLOSURE_CONVENTION,
        MUJOCO_CLOSURE_CONVENTION,
        PINOCCHIO_CLOSURE_CONVENTION,
        SIMSCAPE_CLOSURE_CONVENTION,
        ClosureConvention,
        from_canonical_closure,
        to_canonical_closure,
    )
except ImportError:

    @dataclass(frozen=True)
    class ClosureConvention:
        """Engine-specific 6-D closure row ordering, frame, point, and sign."""

        row_order: Literal["lin_ang", "ang_lin"] = "lin_ang"
        frame: Literal["local", "world"] = "local"
        point: Literal["placement_a", "site_xpos", "body_origin"] = "placement_a"
        relative_sign: Literal[1, -1] = 1

    CANONICAL_CLOSURE_CONVENTION = ClosureConvention(
        "lin_ang", "local", "placement_a", 1
    )
    PINOCCHIO_CLOSURE_CONVENTION = ClosureConvention(
        "lin_ang", "local", "placement_a", 1
    )
    MUJOCO_CLOSURE_CONVENTION = ClosureConvention("lin_ang", "world", "site_xpos", 1)
    DRAKE_CLOSURE_CONVENTION = ClosureConvention("ang_lin", "local", "placement_a", 1)
    SIMSCAPE_CLOSURE_CONVENTION = ClosureConvention(
        "lin_ang", "local", "placement_a", 1
    )

    def _canonical_transform_matrix(
        native: ClosureConvention,
        grip_frame_rotation: Array,
        grip_origin: Array,
        reference_point: Array,
    ) -> Array:
        rot_wg = np.asarray(grip_frame_rotation, dtype=np.float64)
        p_g = np.asarray(grip_origin, dtype=np.float64)
        p_ref = np.asarray(reference_point, dtype=np.float64)
        t_mat = np.eye(6, dtype=np.float64)
        if native.row_order == "ang_lin":
            perm = np.zeros((6, 6), dtype=np.float64)
            perm[:3, 3:] = np.eye(3)
            perm[3:, :3] = np.eye(3)
            t_mat = perm @ t_mat
        if native.frame == "world":
            rot_gw = rot_wg.T
            blk = np.zeros((6, 6), dtype=np.float64)
            blk[:3, :3] = rot_gw
            blk[3:, 3:] = rot_gw
            t_mat = blk @ t_mat
        delta_w = p_g - p_ref
        if float(np.linalg.norm(delta_w)) > 0.0:
            delta_g = rot_wg.T @ delta_w
            shift = np.eye(6, dtype=np.float64)
            shift[:3, 3:] = -_skew(delta_g)
            t_mat = shift @ t_mat
        return float(native.relative_sign) * t_mat

    def to_canonical_closure(
        J_native: Array,
        drift_native: Array,
        *,
        native: ClosureConvention,
        grip_frame_rotation: Array,
        grip_origin: Array,
        reference_point: Array | None = None,
    ) -> tuple[Array, Array]:
        """Convert native 6-D closure Jacobian and drift into canonical grip convention."""
        p_ref = grip_origin if reference_point is None else reference_point
        t_mat = _canonical_transform_matrix(
            native, grip_frame_rotation, grip_origin, p_ref
        )
        j_arr = np.asarray(J_native, dtype=np.float64)
        d_arr = np.asarray(drift_native, dtype=np.float64)
        return t_mat @ j_arr, t_mat @ d_arr

    def from_canonical_closure(
        J_canonical: Array,
        drift_canonical: Array,
        *,
        native: ClosureConvention,
        grip_frame_rotation: Array,
        grip_origin: Array,
        reference_point: Array | None = None,
    ) -> tuple[Array, Array]:
        """Invert canonical closure Jacobian and drift back to native convention."""
        p_ref = grip_origin if reference_point is None else reference_point
        t_mat = _canonical_transform_matrix(
            native, grip_frame_rotation, grip_origin, p_ref
        )
        t_inv = np.linalg.inv(t_mat)
        j_arr = np.asarray(J_canonical, dtype=np.float64)
        d_arr = np.asarray(drift_canonical, dtype=np.float64)
        return t_inv @ j_arr, t_inv @ d_arr


OPENSIM_CLOSURE_CONVENTION = ClosureConvention(
    row_order="lin_ang",
    frame="world",
    point="placement_a",
    relative_sign=-1,
)


def _attach_provenance_attrs(
    prov: ModelProvenance,
    *,
    drift_method: str = "analytic",
    kkt_regularization: float = 0.0,
    matlab_release: str | None = None,
) -> ModelProvenance:
    """Attach Phase 5 provenance metadata to ``ModelProvenance`` cleanly."""
    object.__setattr__(prov, "drift_method", drift_method)
    object.__setattr__(prov, "kkt_regularization", float(kkt_regularization))
    object.__setattr__(prov, "matlab_release", matlab_release)
    return prov


def _build_default_selection_matrix(nv: int, actuated_indices: Sequence[int]) -> Array:
    """Construct actuator selection matrix ``S`` of shape ``(n_u, nv)``."""
    act = tuple(int(i) for i in actuated_indices)
    s_mat = np.zeros((len(act), nv), dtype=np.float64)
    for row_idx, col_idx in enumerate(act):
        s_mat[row_idx, col_idx] = 1.0
    return s_mat


def _build_snapshot_instance(
    q: Array,
    v: Array,
    a: Array,
    quantities: Mapping[str, Any],
) -> ClosedLoopSnapshot:
    """Assemble a validated ``ClosedLoopSnapshot`` from canonical quantities."""
    coord_order = tuple(str(name) for name in quantities["coordinate_order"])
    nv = len(coord_order)
    actuated = tuple(
        int(i)
        for i in quantities.get(
            "actuated_indices",
            range(6, nv) if nv > 6 else range(nv),
        )
    )
    floating = tuple(range(min(6, nv))) if len(actuated) == nv - 6 else ()
    topology = ConstraintTopology(
        coordinate_order=coord_order,
        actuated_indices=actuated,
        floating_base_indices=floating,
        contact_names=tuple(quantities.get("contact_names", ())),
        closure=GripClosureMetadata(),
    )
    prov = ModelProvenance(
        spec_sha256=str(quantities["model_sha256"]),
        inertial_source=str(quantities.get("inertial_source", "full_body_spec")),
        bare_club_bench_id=quantities.get("bare_club_bench_id"),
        engine_name=str(quantities["engine_name"]),
    )
    _attach_provenance_attrs(
        prov,
        drift_method=str(quantities.get("drift_method", "analytic")),
        kkt_regularization=float(quantities.get("kkt_regularization", 0.0)),
        matlab_release=quantities.get("matlab_release"),
    )
    s_mat = quantities.get("S")
    if s_mat is None:
        s_mat = _build_default_selection_matrix(nv, actuated)
    return ClosedLoopSnapshot(
        q=np.asarray(q, dtype=np.float64).copy(),
        v=np.asarray(v, dtype=np.float64).copy(),
        a=np.asarray(a, dtype=np.float64).copy(),
        M=np.asarray(quantities["M"], dtype=np.float64).copy(),
        h=np.asarray(quantities["h"], dtype=np.float64).copy(),
        J_c=np.asarray(quantities["J_c"], dtype=np.float64).copy(),
        Jdot_v=np.asarray(quantities["Jdot_v"], dtype=np.float64).copy(),
        S=np.asarray(s_mat, dtype=np.float64).copy(),
        J_g=quantities.get("J_g"),
        topology=topology,
        model_provenance=prov,
    )


class _BaseSnapshotAdapter:
    """Shared base functionality for closed-loop snapshot adapters."""

    engine_name: str = "analytical"
    native_convention: ClosureConvention = CANONICAL_CLOSURE_CONVENTION

    def __init__(self) -> None:
        self.coordinate_order: tuple[str, ...] = ()
        self.nv: int = 0
        self.model_sha256: str = ""
        self._static_quantities: dict[str, Any] | None = None

    def capability_level(
        self, capability: str = "closed_loop_snapshot"
    ) -> CapabilityLevel:
        """Report capability level for closed-loop snapshot evaluation."""
        if capability == "closed_loop_snapshot":
            return CapabilityLevel.FULL
        return CapabilityLevel.NONE

    def pose_residual(self, q: Array) -> Array:
        """Return 6-D closure pose residual at ``q`` (zero on the manifold)."""
        q_arr = np.asarray(q, dtype=np.float64)
        if self._static_quantities is not None:
            q_man = self._static_quantities["q_manifold"]
            j_c = self._static_quantities["J_c"]
            return np.asarray(j_c @ (q_arr - q_man), dtype=np.float64)
        raise NotImplementedError

    def rate_matrix(self, q: Array) -> Array:
        """Return ``(6, nv)`` closure Jacobian at ``q`` for manifold projection."""
        del q
        if self._static_quantities is not None:
            return np.asarray(self._static_quantities["J_c"], dtype=np.float64).copy()
        raise NotImplementedError

    def closure_pose_residual(self, q: Array) -> Array:
        """ClosurePlant protocol method delegating to ``pose_residual``."""
        return self.pose_residual(q)

    def closure_rate_matrix(self, q: Array) -> Array:
        """ClosurePlant protocol method delegating to ``rate_matrix``."""
        return self.rate_matrix(q)

    def build_snapshot(
        self, q: Array, v: Array, a: Array | None = None
    ) -> ClosedLoopSnapshot:
        """Evaluate and return a canonical ``ClosedLoopSnapshot`` at ``(q, v, a)``."""
        if self.capability_level("closed_loop_snapshot") != CapabilityLevel.FULL:
            raise CapabilityUnavailableError(
                f"{self.engine_name} closed_loop_snapshot capability is not FULL"
            )
        q_arr = np.asarray(q, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        a_arr = np.zeros_like(v_arr) if a is None else np.asarray(a, dtype=np.float64)
        if self._static_quantities is not None:
            return _build_snapshot_instance(
                q_arr, v_arr, a_arr, self._static_quantities
            )
        return self._evaluate_live_snapshot(q_arr, v_arr, a_arr)

    def _evaluate_live_snapshot(
        self, q: Array, v: Array, a: Array
    ) -> ClosedLoopSnapshot:
        raise NotImplementedError


class PinocchioClosedLoopSnapshotAdapter(_BaseSnapshotAdapter):
    """Pinocchio reference closed-loop snapshot adapter."""

    engine_name = "pinocchio"
    native_convention = PINOCCHIO_CLOSURE_CONVENTION

    def __init__(self, spec_bytes: bytes | None = None) -> None:
        super().__init__()
        self._model: Any = None
        if spec_bytes is not None:
            from importlib import import_module

            pin_mod = import_module("pinocchio")
            if type(pin_mod).__module__ == "unittest.mock":
                raise CapabilityUnavailableError(
                    "Real pinocchio C++ bindings required; got unittest.mock"
                )
            from src.engines.physics_engines.pinocchio.python.native_model import (
                FullBodyPinocchioModel,
            )

            spec = json.loads(spec_bytes)
            self._model = FullBodyPinocchioModel(spec)
            self.coordinate_order = tuple(self._model.coordinate_order)
            self.nv = len(self.coordinate_order)
            self.model_sha256 = hashlib.sha256(spec_bytes).hexdigest()

    @classmethod
    def from_reference_backend(cls, backend: Any) -> PinocchioClosedLoopSnapshotAdapter:
        """Construct adapter backed by a pre-evaluated canonical reference fixture."""
        instance = cls()
        instance.coordinate_order = tuple(backend.coordinate_order)
        instance.nv = int(backend.nv)
        instance.model_sha256 = str(backend.model_sha256)
        instance._static_quantities = {
            "coordinate_order": instance.coordinate_order,
            "actuated_indices": tuple(
                getattr(backend, "actuated_indices", range(instance.nv))
            ),
            "model_sha256": instance.model_sha256,
            "engine_name": cls.engine_name,
            "M": np.asarray(backend.M, dtype=np.float64),
            "h": np.asarray(backend.h, dtype=np.float64),
            "J_c": np.asarray(backend.J_c, dtype=np.float64),
            "Jdot_v": np.asarray(backend.Jdot_v, dtype=np.float64),
            "q_manifold": np.asarray(backend.q_ref, dtype=np.float64),
            "drift_method": "analytic",
            "kkt_regularization": 0.0,
        }
        return instance

    def pose_residual(self, q: Array) -> Array:
        if self._static_quantities is not None:
            return super().pose_residual(q)
        coords = dict(zip(self.coordinate_order, q.tolist(), strict=True))
        lin = self._model.closure_position_linearization(coords)
        return np.asarray(lin.position, dtype=np.float64)

    def rate_matrix(self, q: Array) -> Array:
        if self._static_quantities is not None:
            return super().rate_matrix(q)
        coords = dict(zip(self.coordinate_order, q.tolist(), strict=True))
        force_jac = self._model.closure_force_jacobian(coords)
        return np.asarray(force_jac.jacobian, dtype=np.float64)

    def _evaluate_live_snapshot(
        self, q: Array, v: Array, a: Array
    ) -> ClosedLoopSnapshot:
        model = self._model
        pin = model._pin
        coords = dict(zip(self.coordinate_order, q.tolist(), strict=True))
        rates = dict(zip(self.coordinate_order, v.tolist(), strict=True))
        zero_acc = dict.fromkeys(self.coordinate_order, 0.0)
        q_pin = model.configuration(coords)
        v_pin = model.velocity(rates)
        indices = [model._velocity_indices[name] for name in self.coordinate_order]
        m_raw = model.mass_matrix(coords)
        mass = np.asarray(m_raw, dtype=np.float64)[np.ix_(indices, indices)]
        h_raw = pin.nonLinearEffects(model.model, model.data, q_pin, v_pin)
        bias = np.asarray(h_raw, dtype=np.float64)[indices]
        j_raw = np.asarray(
            model.closure_force_jacobian(coords).jacobian, dtype=np.float64
        )
        res = model.closure_trajectory_residuals(coords, rates, zero_acc)
        jdot_raw = np.asarray(res.acceleration, dtype=np.float64)
        rot_wa, pos_a = model.closure_frame_pose(coords)
        j_c, jdot_v = to_canonical_closure(
            -j_raw,
            -jdot_raw,
            native=PINOCCHIO_CLOSURE_CONVENTION,
            grip_frame_rotation=rot_wa,
            grip_origin=pos_a,
            reference_point=pos_a,
        )
        j_g_rows = []
        pin.computeJointJacobians(model.model, model.data, q_pin)
        pin.updateFramePlacements(model.model, model.data)
        for sphere in model.contact_spheres:
            fid = model._contact_frames[sphere.name]
            j_frame = np.asarray(
                pin.getFrameJacobian(
                    model.model,
                    model.data,
                    fid,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                ),
                dtype=np.float64,
            )
            j_g_rows.append(j_frame[:3, indices])
        j_g = (
            np.vstack(j_g_rows)
            if j_g_rows
            else np.zeros((0, self.nv), dtype=np.float64)
        )
        return _build_snapshot_instance(
            q,
            v,
            a,
            {
                "coordinate_order": self.coordinate_order,
                "model_sha256": self.model_sha256,
                "engine_name": self.engine_name,
                "M": mass,
                "h": bias,
                "J_c": j_c,
                "Jdot_v": jdot_v,
                "J_g": j_g,
                "contact_names": tuple(s.name for s in model.contact_spheres),
                "drift_method": "analytic",
                "kkt_regularization": 0.0,
            },
        )


class MuJoCoClosedLoopSnapshotAdapter(_BaseSnapshotAdapter):
    """MuJoCo Closed-Loop Snapshot Adapter via Hard-KKT Evaluation (#12579 / NSX-51).

    Extracts ``M(q)`` via ``mj_fullM``, ``h(q, v)`` via ``data.qfrc_bias``, and
    6-D weld closure Jacobian ``J_c`` and velocity drift ``Jdot_v`` in canonical
    grip coordinates. Records ``drift_method`` (``"analytic"`` or
    ``"finite_difference_centered"``) and ``kkt_regularization`` in ``ModelProvenance``.
    """

    engine_name = "mujoco"
    native_convention = MUJOCO_CLOSURE_CONVENTION

    def __init__(
        self,
        spec_bytes: bytes | None = None,
        *,
        kkt_regularization: float = 0.0,
        drift_method: DriftMethod = "analytic",
        fd_step: float = 1e-7,
    ) -> None:
        super().__init__()
        if kkt_regularization < 0.0 or not np.isfinite(kkt_regularization):
            raise ValueError("kkt_regularization must be finite and >= 0.0")
        if drift_method not in ("analytic", "finite_difference_centered"):
            raise ValueError(f"Unsupported drift_method: {drift_method}")
        self.kkt_regularization = float(kkt_regularization)
        self.drift_method: DriftMethod = drift_method
        self.fd_step = float(fd_step)
        self._model: Any = None
        if spec_bytes is not None:
            self._init_live_model(spec_bytes)

    def _init_live_model(self, spec_bytes: bytes) -> None:
        from src.engines.physics_engines.mujoco.python.full_body_model import (
            NativeMujocoFullBodyModel,
        )

        self._model = NativeMujocoFullBodyModel(spec_bytes)
        self._model.kkt_regularization = self.kkt_regularization
        self.coordinate_order = tuple(self._model.coordinate_order)
        self.nv = len(self.coordinate_order)
        self.model_sha256 = str(self._model.model_sha256)

    def _dof_indices(self) -> list[int]:
        backend = self._model
        return [int(backend._indices[name]) for name in self.coordinate_order]

    @classmethod
    def from_native_quantities(cls, **kwargs: Any) -> MuJoCoClosedLoopSnapshotAdapter:
        """Build an adapter from MuJoCo-native world-frame site quantities."""
        kkt_reg = float(kwargs.get("kkt_regularization", 0.0))
        drift_method: DriftMethod = kwargs.get("drift_method", "analytic")
        instance = cls(None, kkt_regularization=kkt_reg, drift_method=drift_method)
        instance.coordinate_order = tuple(kwargs["coordinate_order"])
        instance.nv = len(instance.coordinate_order)
        instance.model_sha256 = str(kwargs["model_sha256"])
        j_c, jdot_v = to_canonical_closure(
            np.asarray(kwargs["native_jacobian"], dtype=np.float64),
            np.asarray(kwargs["native_drift"], dtype=np.float64),
            native=MUJOCO_CLOSURE_CONVENTION,
            grip_frame_rotation=np.asarray(
                kwargs["closure_rotation_world"], dtype=np.float64
            ),
            grip_origin=np.asarray(kwargs["grip_origin_world"], dtype=np.float64),
            reference_point=np.asarray(kwargs["closure_point_world"], dtype=np.float64),
        )
        instance._static_quantities = {
            "coordinate_order": instance.coordinate_order,
            "actuated_indices": tuple(
                kwargs.get("actuated_indices", range(instance.nv))
            ),
            "model_sha256": instance.model_sha256,
            "engine_name": cls.engine_name,
            "M": np.asarray(kwargs["mass_matrix"], dtype=np.float64),
            "h": np.asarray(kwargs["bias_vector"], dtype=np.float64),
            "J_c": j_c,
            "Jdot_v": jdot_v,
            "q_manifold": np.asarray(kwargs["q_manifold"], dtype=np.float64),
            "drift_method": drift_method,
            "kkt_regularization": kkt_reg,
        }
        return instance

    def capability_level(
        self, capability: str = "closed_loop_snapshot"
    ) -> CapabilityLevel:
        """Report FULL for hard-KKT snapshot (kkt_regularization == 0) and PARTIAL for soft step."""
        if capability == "soft_constraint_stepping":
            return CapabilityLevel.PARTIAL
        if capability == "closed_loop_snapshot":
            if self.kkt_regularization == 0.0:
                return CapabilityLevel.FULL
            return CapabilityLevel.PARTIAL
        return CapabilityLevel.NONE

    def with_drift_method(
        self, drift_method: DriftMethod
    ) -> MuJoCoClosedLoopSnapshotAdapter:
        """Return a clone of this adapter configured with ``drift_method``."""
        clone = self.__class__(
            None,
            kkt_regularization=self.kkt_regularization,
            drift_method=drift_method,
            fd_step=self.fd_step,
        )
        clone.coordinate_order = self.coordinate_order
        clone.nv = self.nv
        clone.model_sha256 = self.model_sha256
        clone._model = self._model
        if self._static_quantities is not None:
            clone._static_quantities = {
                **self._static_quantities,
                "drift_method": drift_method,
            }
        return clone

    def with_kkt_regularization(
        self, kkt_regularization: float
    ) -> MuJoCoClosedLoopSnapshotAdapter:
        """Return a clone of this adapter configured with ``kkt_regularization``."""
        clone = self.__class__(
            None,
            kkt_regularization=kkt_regularization,
            drift_method=self.drift_method,
            fd_step=self.fd_step,
        )
        clone.coordinate_order = self.coordinate_order
        clone.nv = self.nv
        clone.model_sha256 = self.model_sha256
        clone._model = self._model
        if self._static_quantities is not None:
            clone._static_quantities = {
                **self._static_quantities,
                "kkt_regularization": float(kkt_regularization),
            }
        return clone

    def pose_residual(self, q: Array) -> Array:
        if self._static_quantities is not None:
            return super().pose_residual(q)
        from scipy.spatial.transform import Rotation

        backend = self._model
        mj, model, data = backend._mj, backend.model, backend.data
        dof_idx = self._dof_indices()
        data.qpos[dof_idx] = np.asarray(q, dtype=np.float64)
        mj.mj_fwdPosition(model, data)
        site_a, site_b = backend._closure
        rot_a = np.asarray(data.site_xmat[site_a], dtype=np.float64).reshape(3, 3)
        rot_b = np.asarray(data.site_xmat[site_b], dtype=np.float64).reshape(3, 3)
        pos_err = np.asarray(
            data.site_xpos[site_a] - data.site_xpos[site_b], dtype=np.float64
        )
        rot_err = Rotation.from_matrix(rot_a @ rot_b.T).as_rotvec()
        return np.concatenate((pos_err, rot_err))

    def rate_matrix(self, q: Array) -> Array:
        if self._static_quantities is not None:
            return super().rate_matrix(q)
        jac_native, _, _, _ = self._native_site_jacobian(
            np.asarray(q, dtype=np.float64), np.zeros(self.nv, dtype=np.float64)
        )
        return jac_native

    def _native_site_jacobian(
        self, q: Array, v: Array
    ) -> tuple[Array, Array, Array, Array]:
        backend = self._model
        mj, model, data = backend._mj, backend.model, backend.data
        dof_idx = self._dof_indices()
        data.qpos[dof_idx] = q
        data.qvel[dof_idx] = v
        mj.mj_fwdPosition(model, data)
        mj.mj_fwdVelocity(model, data)
        site_a = backend._closure[0]
        rot_wa = (
            np.asarray(data.site_xmat[site_a], dtype=np.float64).reshape(3, 3).copy()
        )
        pos_a = np.asarray(data.site_xpos[site_a], dtype=np.float64).copy()
        jac_raw, drift_analytic = backend.evaluate_weld_closure()
        jac_ordered = np.asarray(jac_raw, dtype=np.float64)[:, dof_idx]
        return jac_ordered, np.asarray(drift_analytic, dtype=np.float64), rot_wa, pos_a

    def _evaluate_canonical_closure(self, q: Array, v: Array) -> tuple[Array, Array]:
        jac_native, drift_native, rot_wa, pos_a = self._native_site_jacobian(q, v)
        if self.drift_method == "finite_difference_centered":
            eps = self.fd_step
            q_plus = q + eps * v
            q_minus = q - eps * v
            jac_plus, _, _, _ = self._native_site_jacobian(q_plus, v)
            jac_minus, _, _, _ = self._native_site_jacobian(q_minus, v)
            drift_native = ((jac_plus - jac_minus) / (2.0 * eps)) @ v
            self._native_site_jacobian(q, v)
        return to_canonical_closure(
            jac_native,
            drift_native,
            native=MUJOCO_CLOSURE_CONVENTION,
            grip_frame_rotation=rot_wa,
            grip_origin=pos_a,
            reference_point=pos_a,
        )

    def _evaluate_live_snapshot(
        self, q: Array, v: Array, a: Array
    ) -> ClosedLoopSnapshot:
        backend = self._model
        mj, model, data = backend._mj, backend.model, backend.data
        dof_idx = self._dof_indices()
        j_c, jdot_v = self._evaluate_canonical_closure(q, v)
        mass_raw = full_mass_matrix(mj, model, data)
        mass = mass_raw[np.ix_(dof_idx, dof_idx)]
        bias = np.asarray(data.qfrc_bias, dtype=np.float64)[dof_idx].copy()
        j_ground = np.zeros((len(backend._spheres) * 3, self.nv), dtype=np.float64)
        for idx, s_info in enumerate(backend._spheres.values()):
            jac_pos = np.zeros((3, self.nv), dtype=np.float64)
            mj.mj_jacSite(model, data, jac_pos, None, s_info["site_id"])
            j_ground[idx * 3 : (idx + 1) * 3, :] = jac_pos[:, dof_idx]
        return _build_snapshot_instance(
            q,
            v,
            a,
            {
                "coordinate_order": self.coordinate_order,
                "model_sha256": self.model_sha256,
                "engine_name": self.engine_name,
                "M": mass,
                "h": bias,
                "J_c": j_c,
                "Jdot_v": jdot_v,
                "J_g": j_ground,
                "contact_names": tuple(backend._spheres.keys()),
                "drift_method": self.drift_method,
                "kkt_regularization": self.kkt_regularization,
            },
        )


class MyoSuiteClosedLoopSnapshotAdapter(MuJoCoClosedLoopSnapshotAdapter):
    """MyoSuite Closed-Loop Snapshot Adapter (#12580 / NSX-52).

    Subclasses ``MuJoCoClosedLoopSnapshotAdapter`` to reuse 100% of the hard-KKT
    snapshot assembly and canonical conversion path without duplicate math.
    Gates ``CapabilityLevel.FULL`` on ``kkt_regularization == 0.0``; positive
    regularization degrades to ``CapabilityLevel.PARTIAL``.
    """

    engine_name = "myosuite"

    def _init_live_model(self, spec_bytes: bytes) -> None:
        from src.engines.physics_engines.myosuite.python.full_body_parity import (
            MyoSuiteFullBodyModel,
        )

        self._model = MyoSuiteFullBodyModel(spec_bytes)
        self._model.kkt_regularization = self.kkt_regularization
        self.coordinate_order = tuple(self._model.coordinate_order)
        self.nv = len(self.coordinate_order)
        self.model_sha256 = str(self._model.model_sha256)

    def build_snapshot(
        self, q: Array, v: Array, a: Array | None = None
    ) -> ClosedLoopSnapshot:
        if self.kkt_regularization > 0.0:
            raise CapabilityUnavailableError(
                f"MyoSuite closed_loop_snapshot requires kkt_regularization == 0.0 "
                f"(got kkt_regularization={self.kkt_regularization}); capability is PARTIAL"
            )
        return super().build_snapshot(q, v, a)


class DrakeClosedLoopSnapshotAdapter(_BaseSnapshotAdapter):
    """Drake Native Force Adapter With Closed-Loop Snapshot (#12581 / NSX-53).

    Evaluates ``CalcMassMatrix``, ``CalcBiasTerm - CalcGravityGeneralizedForces``,
    ``CalcJacobianSpatialVelocity``, and ``CalcBiasSpatialAcceleration`` on Drake's
    native ``MultibodyPlant`` and converts ``DRAKE_CLOSURE_CONVENTION``
    (``[omega; v]`` in local frame ``a``) into ``CANONICAL_CLOSURE_CONVENTION``.
    """

    engine_name = "drake"
    native_convention = DRAKE_CLOSURE_CONVENTION

    def __init__(self, spec_bytes: bytes | None = None) -> None:
        super().__init__()
        self._model: Any = None
        if spec_bytes is not None:
            from importlib import import_module

            drake_mod = import_module("pydrake.all")
            if type(drake_mod).__module__ == "unittest.mock":
                raise CapabilityUnavailableError(
                    "Real pydrake C++ bindings required; got unittest.mock"
                )
            from src.engines.physics_engines.drake.python.full_body_model import (
                FullBodyDrakeModel,
            )

            spec = json.loads(spec_bytes)
            self._model = FullBodyDrakeModel(spec)
            self.coordinate_order = tuple(self._model.coordinate_order)
            self.nv = len(self.coordinate_order)
            self.model_sha256 = hashlib.sha256(spec_bytes).hexdigest()

    def pose_residual(self, q: Array) -> Array:
        if self._static_quantities is not None:
            return super().pose_residual(q)
        coords = dict(zip(self.coordinate_order, q.tolist(), strict=True))
        zeros = dict.fromkeys(self.coordinate_order, 0.0)
        pos_res, _ = self._model.closure_residuals(coords, zeros)
        return -np.asarray(pos_res, dtype=np.float64)

    def rate_matrix(self, q: Array) -> Array:
        if self._static_quantities is not None:
            return super().rate_matrix(q)
        model = self._model
        plant, ctx = model.plant, model.context
        plant.SetPositions(ctx, model.to_plant_positions(q))
        plant.SetVelocities(ctx, np.zeros(self.nv, dtype=np.float64))
        frame_a, _, jac_ba_raw, bias_ba = model._closure_bias_and_jacobian()
        jac_ba = model.named_columns(jac_ba_raw)
        tf_wa = plant.CalcRelativeTransform(ctx, plant.world_frame(), frame_a)
        rot_wa = np.asarray(tf_wa.rotation().matrix(), dtype=np.float64)
        pos_a = np.asarray(tf_wa.translation(), dtype=np.float64)
        j_c, _ = to_canonical_closure(
            -jac_ba,
            -np.asarray(bias_ba, dtype=np.float64),
            native=DRAKE_CLOSURE_CONVENTION,
            grip_frame_rotation=rot_wa,
            grip_origin=pos_a,
            reference_point=pos_a,
        )
        return j_c

    @classmethod
    def from_native_quantities(cls, **kwargs: Any) -> DrakeClosedLoopSnapshotAdapter:
        """Build an adapter from Drake-native ``[omega; v]`` local-frame quantities."""
        instance = cls(None)
        instance.coordinate_order = tuple(kwargs["coordinate_order"])
        instance.nv = len(instance.coordinate_order)
        instance.model_sha256 = str(kwargs["model_sha256"])
        # Drake CalcBiasTerm is C(q, v)v, and CalcGravityGeneralizedForces is tau_g,
        # so generalized bias h(q, v) = C(q, v)v - tau_g.
        bias_h = np.asarray(kwargs["bias_term"], dtype=np.float64) - np.asarray(
            kwargs["gravity_generalized_forces"], dtype=np.float64
        )
        # FullBodyDrakeModel computes spatial velocity of b relative to a (b - a);
        # negate to (a - b) before applying DRAKE_CLOSURE_CONVENTION (ang_lin -> lin_ang).
        j_drake_ab = -np.asarray(kwargs["native_jacobian_ba"], dtype=np.float64)
        d_drake_ab = -np.asarray(kwargs["native_bias_spatial_acc_ba"], dtype=np.float64)
        j_c, jdot_v = to_canonical_closure(
            j_drake_ab,
            d_drake_ab,
            native=DRAKE_CLOSURE_CONVENTION,
            grip_frame_rotation=np.asarray(
                kwargs["closure_rotation_world"], dtype=np.float64
            ),
            grip_origin=np.asarray(kwargs["grip_origin_world"], dtype=np.float64),
            reference_point=np.asarray(kwargs["closure_point_world"], dtype=np.float64),
        )
        instance._static_quantities = {
            "coordinate_order": instance.coordinate_order,
            "actuated_indices": tuple(
                kwargs.get("actuated_indices", range(instance.nv))
            ),
            "model_sha256": instance.model_sha256,
            "engine_name": cls.engine_name,
            "M": np.asarray(kwargs["mass_matrix"], dtype=np.float64),
            "h": bias_h,
            "J_c": j_c,
            "Jdot_v": jdot_v,
            "q_manifold": np.asarray(kwargs["q_manifold"], dtype=np.float64),
            "drift_method": "analytic",
            "kkt_regularization": 0.0,
        }
        return instance

    def _evaluate_live_snapshot(
        self, q: Array, v: Array, a: Array
    ) -> ClosedLoopSnapshot:
        model = self._model
        plant, ctx = model.plant, model.context
        plant.SetPositions(ctx, model.to_plant_positions(q))
        plant.SetVelocities(ctx, model.to_plant_velocities(v))
        v_idx = model.plant_position_indices
        mass_raw = np.asarray(plant.CalcMassMatrix(ctx), dtype=np.float64)
        mass = mass_raw[np.ix_(v_idx, v_idx)]
        bias_raw = np.asarray(plant.CalcBiasTerm(ctx), dtype=np.float64) - np.asarray(
            plant.CalcGravityGeneralizedForces(ctx), dtype=np.float64
        )
        bias = bias_raw[list(v_idx)]
        frame_a, _, jac_ba_raw, bias_ba = model._closure_bias_and_jacobian()
        jac_ba = model.named_columns(jac_ba_raw)
        tf_wa = plant.CalcRelativeTransform(ctx, plant.world_frame(), frame_a)
        rot_wa = np.asarray(tf_wa.rotation().matrix(), dtype=np.float64)
        pos_a = np.asarray(tf_wa.translation(), dtype=np.float64)
        j_c, jdot_v = to_canonical_closure(
            -jac_ba,
            -np.asarray(bias_ba, dtype=np.float64),
            native=DRAKE_CLOSURE_CONVENTION,
            grip_frame_rotation=rot_wa,
            grip_origin=pos_a,
            reference_point=pos_a,
        )
        return _build_snapshot_instance(
            q,
            v,
            a,
            {
                "coordinate_order": self.coordinate_order,
                "model_sha256": self.model_sha256,
                "engine_name": self.engine_name,
                "M": mass,
                "h": bias,
                "J_c": j_c,
                "Jdot_v": jdot_v,
                "J_g": model.sphere_jacobians(),
                "drift_method": "analytic",
                "kkt_regularization": 0.0,
            },
        )


DrakeForceAdapter = DrakeClosedLoopSnapshotAdapter


class OpenSimClosedLoopSnapshotAdapter(_BaseSnapshotAdapter):
    """OpenSim Closed-Loop Snapshot Adapter (#12582 / NSX-54).

    Delegates to ``OpenSimFullBodyParityAdapter`` public methods with LoD depth <= 2
    and converts OpenSim's world-frame ``b - a`` closure quantities into
    ``CANONICAL_CLOSURE_CONVENTION`` via ``OPENSIM_CLOSURE_CONVENTION``.
    """

    engine_name = "opensim"
    native_convention = OPENSIM_CLOSURE_CONVENTION

    def __init__(self, spec_bytes: bytes | None = None) -> None:
        super().__init__()
        self._parity: Any = None
        if spec_bytes is not None:
            from src.engines.physics_engines.opensim.python.full_body_parity import (
                OpenSimFullBodyParityAdapter,
            )

            self._parity = OpenSimFullBodyParityAdapter(spec_bytes)
            self.coordinate_order = tuple(self._parity.coordinate_order)
            self.nv = len(self.coordinate_order)
            self.model_sha256 = hashlib.sha256(spec_bytes).hexdigest()

    @classmethod
    def from_native_quantities(cls, **kwargs: Any) -> OpenSimClosedLoopSnapshotAdapter:
        """Build an adapter from OpenSim-native world-frame ``b - a`` quantities."""
        instance = cls(None)
        instance.coordinate_order = tuple(kwargs["coordinate_order"])
        instance.nv = len(instance.coordinate_order)
        instance.model_sha256 = str(kwargs["model_sha256"])
        j_c, jdot_v = to_canonical_closure(
            np.asarray(kwargs["native_jacobian_ba"], dtype=np.float64),
            np.asarray(kwargs["native_bias_ba"], dtype=np.float64),
            native=OPENSIM_CLOSURE_CONVENTION,
            grip_frame_rotation=np.asarray(
                kwargs["closure_rotation_world"], dtype=np.float64
            ),
            grip_origin=np.asarray(kwargs["grip_origin_world"], dtype=np.float64),
            reference_point=np.asarray(kwargs["closure_point_world"], dtype=np.float64),
        )
        instance._static_quantities = {
            "coordinate_order": instance.coordinate_order,
            "actuated_indices": tuple(
                kwargs.get("actuated_indices", range(instance.nv))
            ),
            "model_sha256": instance.model_sha256,
            "engine_name": cls.engine_name,
            "M": np.asarray(kwargs["mass_matrix"], dtype=np.float64),
            "h": np.asarray(kwargs["bias_forces"], dtype=np.float64),
            "J_c": j_c,
            "Jdot_v": jdot_v,
            "q_manifold": np.asarray(kwargs["q_manifold"], dtype=np.float64),
            "drift_method": "analytic",
            "kkt_regularization": 0.0,
        }
        return instance

    def _evaluate_live_snapshot(
        self, q: Array, v: Array, a: Array
    ) -> ClosedLoopSnapshot:
        parity = self._parity
        coords = dict(zip(self.coordinate_order, q.tolist(), strict=True))
        rates = dict(zip(self.coordinate_order, v.tolist(), strict=True))
        mass = parity.mass_matrix(coords)
        bias = parity.bias_forces(coords, rates)
        _, jac_ba, bias_ba = parity.closure_jacobian_and_bias(coords, rates)
        rot_wa, pos_a = parity.closure_frame_pose(coords)
        j_c, jdot_v = to_canonical_closure(
            jac_ba,
            bias_ba,
            native=OPENSIM_CLOSURE_CONVENTION,
            grip_frame_rotation=rot_wa,
            grip_origin=pos_a,
            reference_point=pos_a,
        )
        return _build_snapshot_instance(
            q,
            v,
            a,
            {
                "coordinate_order": self.coordinate_order,
                "model_sha256": self.model_sha256,
                "engine_name": self.engine_name,
                "M": mass,
                "h": bias,
                "J_c": j_c,
                "Jdot_v": jdot_v,
                "J_g": parity.contact_jacobian(coords),
                "drift_method": "analytic",
                "kkt_regularization": 0.0,
            },
        )


OpenSimForceAdapter = OpenSimClosedLoopSnapshotAdapter


def snapshot_from_adapter(
    adapter: Any,
    q: Array,
    v: Array,
    a: Array | None = None,
) -> ClosedLoopSnapshot:
    """Build a canonical ``ClosedLoopSnapshot`` from any Phase 5 engine adapter."""
    require(adapter is not None, "adapter must not be None")
    result = adapter.build_snapshot(q, v, a)
    ensure(result.closure_rows() == 6, "snapshot must have 6 closure rows")
    return result


__all__ = [
    "CANONICAL_CLOSURE_CONVENTION",
    "DRAKE_CLOSURE_CONVENTION",
    "MUJOCO_CLOSURE_CONVENTION",
    "OPENSIM_CLOSURE_CONVENTION",
    "PINOCCHIO_CLOSURE_CONVENTION",
    "SIMSCAPE_CLOSURE_CONVENTION",
    "CapabilityUnavailableError",
    "ClosedLoopSnapshot",
    "ClosureConvention",
    "ConstraintTopology",
    "DrakeClosedLoopSnapshotAdapter",
    "DrakeForceAdapter",
    "GripClosureMetadata",
    "ModelProvenance",
    "MuJoCoClosedLoopSnapshotAdapter",
    "MyoSuiteClosedLoopSnapshotAdapter",
    "OpenSimClosedLoopSnapshotAdapter",
    "OpenSimForceAdapter",
    "PinocchioClosedLoopSnapshotAdapter",
    "from_canonical_closure",
    "snapshot_from_adapter",
    "to_canonical_closure",
]

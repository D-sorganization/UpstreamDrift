"""Multi-Engine Dynamic Force Allocator & Torque Determination Architecture (#10415, #10439).

Extends the decoupled geometric-first kinematic tracking and contact-aware QP force
allocation across multiple multibody physics engines:
1. Pinocchio (Lie-group Featherstone RNEA & ABA)
2. MuJoCo / "Monaco" (NativeMjCF, mj_inverse, mj_fullM, and site Jacobians)
3. Drake (MultibodyPlant spatial inverse dynamics and Jacobians)
4. OpenSim (SimTK Simbody station Jacobians and generalized force allocation)
5. Simscape / MATLAB (DAE / RigidBodyTree torque profiles and Simulink timeseries)

Enforces exact dynamic equilibrium:
    M(q) a + b(q, v) = S^T tau + J_ground^T f_ground + J_grip^T lambda_grip + S_root^T delta_tau_root
under unilateral ground reaction constraints (f_{i, z} >= 0) and friction cone feasibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.loop_nullspace.contracts import (
    ClosureKind,
    LoopTopology,
    SpatialInertia,
)
from src.shared.python.loop_nullspace.snapshot import (
    canonical_closure_jacobian,
    parity_effort,
)
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocation,
    ContactForceAllocator,
)
from src.shared.python.engine_core.mujoco_compat import full_mass_matrix

logger = logging.getLogger(__name__)

Array: TypeAlias = NDArray[np.float64]


class EngineType(str, Enum):
    """Supported multibody physics simulation engines."""

    PINOCCHIO = "pinocchio"
    MUJOCO = "mujoco"
    DRAKE = "drake"
    OPENSIM = "opensim"
    SIMSCAPE = "simscape"
    MYOSUITE = "myosuite"


@dataclass(frozen=True)
class TrajectoryAllocationResult:
    """Summary of dynamic force and torque allocation across an entire trajectory."""

    engine: EngineType
    objective: AllocationObjective
    n_frames: int
    duration_s: float
    time_s: Array
    tau_actuated: Array  # (n_frames, n_actuated)
    f_ground: Array  # (n_frames, n_contact_spheres * 3)
    lambda_grip: Array  # (n_frames, 6)
    delta_tau_root: Array  # (n_frames, 6)
    max_equilibrium_residual: float
    max_root_residual: float
    parity_residuals: list[float]
    success: bool

    def as_dict(self) -> dict[str, Any]:
        """Serialize outcome metrics to dict."""
        return {
            "engine": self.engine.value,
            "objective": self.objective.value,
            "n_frames": self.n_frames,
            "duration_s": float(self.duration_s),
            "max_equilibrium_residual": float(self.max_equilibrium_residual),
            "max_root_residual": float(self.max_root_residual),
            "mean_parity_residual": (
                float(np.mean(self.parity_residuals)) if self.parity_residuals else 0.0
            ),
            "max_parity_residual": (
                float(np.max(self.parity_residuals)) if self.parity_residuals else 0.0
            ),
            "success": self.success,
        }


@runtime_checkable
class BaseEngineForceAdapter(Protocol):
    """Abstract protocol for engine-specific kinematic and dynamic evaluations."""

    @property
    def engine_type(self) -> EngineType:
        """Type of physics engine."""
        ...

    @property
    def nv(self) -> int:
        """Total degrees of freedom (including unactuated floating base)."""
        ...

    @property
    def actuated_indices(self) -> Sequence[int]:
        """Indices of actuated coordinates."""
        ...

    @property
    def n_contact_spheres(self) -> int:
        """Number of ground contact points/spheres."""
        ...

    @property
    def model_hash(self) -> str:
        """Cryptographic hash or identifier of the evaluated model."""
        ...

    @property
    def coordinate_order(self) -> Sequence[str]:
        """Explicit coordinate order matching state vectors."""
        ...

    @property
    def contact_names(self) -> Sequence[str]:
        """Names of ground contact points/spheres."""
        ...

    def compute_mass_and_bias(self, q: Array, v: Array) -> tuple[Array, Array]:
        """Compute inertia matrix M(q) and Coriolis/gravity bias vector b(q, v)."""
        ...

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        """Compute unconstrained generalized forces tau_rnea = M(q) a + b(q, v)."""
        ...

    def compute_contact_jacobian(self, q: Array) -> Array:
        """Compute ground contact translational Jacobian (3 * n_spheres, nv)."""
        ...

    def compute_grip_jacobian(self, q: Array) -> Array:
        """Compute 6-DoF spatial loop-closure weld Jacobian between hands (6, nv)."""
        ...

    def verify_acceleration_parity(
        self, q: Array, v: Array, tau_effective: Array, a_target: Array
    ) -> float:
        """Compute forward acceleration parity error ||a_forward - a_target||_inf."""
        ...


@runtime_checkable
class ClosedLoopSnapshotAdapter(BaseEngineForceAdapter, Protocol):
    """Optional closed-loop extension of the force adapter (NSX-13, #12558).

    An adapter that implements these members and declares
    ``Capability.CLOSED_LOOP_SNAPSHOT`` at ``FULL`` can feed
    ``loop_nullspace.snapshot.snapshot_from_adapter``.  They are not part of
    :class:`BaseEngineForceAdapter` so existing adapters stay conformant.
    """

    native_closure_convention: Any
    """NSX-02 ``NativeClosureConvention`` of ``compute_grip_jacobian`` rows."""

    def capability(self, capability: Capability) -> CapabilityLevel:
        """Declared level for an adapter-boundary capability."""
        ...

    def compute_closure_drift(self, q: Array, v: Array) -> Array:
        """Native ``Jdot v`` (6,) of ``compute_grip_jacobian`` at ``(q, v)``."""
        ...

    def native_closure_frame(self, q: Array) -> tuple[Array, Array]:
        """``(R, origin)`` of the frame and point of the native closure rows."""
        ...

    def closure_wrench_frame(self, q: Array) -> tuple[Array, Array]:
        """``(R world<-grip, origin)`` of the canonical grip frame."""
        ...

    def club_inertia_split(self) -> tuple[Any, tuple[Any, ...]]:
        """``(bare club, welded hands)`` as ``SpatialInertia`` (NSX-01)."""
        ...

    def closure_topology(self) -> Any:
        """The ``LoopTopology`` of the closure."""
        ...


class MujocoForceAdapter:
    """MuJoCo adapter for full-body inverse dynamics and contact Jacobians."""

    def __init__(self, spec_bytes: bytes) -> None:
        self._model = self._make_model(spec_bytes)
        self._mj = self._model._mj
        mj_model = self._model.model
        self._nq = int(mj_model.nq)
        self._nv = int(mj_model.nv)
        self._actuated_indices = list(range(6, self._nv))
        self._n_spheres = len(self._model._spheres)
        self._model_hash = getattr(self._model, "model_sha256", "")
        # State vectors use MuJoCo dof order, which need not be spec order.
        self._coordinate_order = self._model.native_coordinate_order
        self._contact_names = tuple(getattr(self._model, "_spheres", {}).keys())

    def _make_model(self, spec_bytes: bytes) -> Any:
        """Native model this adapter drives; subclasses supply another runtime."""
        from src.engines.physics_engines.mujoco.python.full_body_model import (
            NativeMujocoFullBodyModel,
        )

        return NativeMujocoFullBodyModel(spec_bytes)

    @property
    def engine_type(self) -> EngineType:
        return EngineType.MUJOCO

    def capability(self, capability: Capability) -> CapabilityLevel:
        """``CLOSED_LOOP_SNAPSHOT`` is FULL by hard-KKT evaluation (NSX-51).

        ``M`` and ``h`` come from ``mj_fullM`` and ``qfrc_bias``, the closure
        rows from the weld site Jacobians, and the drift from a declared
        central finite difference (:attr:`closure_drift_method`).  Nothing is
        read from stock soft-constraint stepping, whose solref/solimp
        compliance has its own divergence-registry entry.  Everything else is
        ``NONE`` at this boundary.
        """
        if capability is Capability.CLOSED_LOOP_SNAPSHOT:
            return CapabilityLevel.FULL
        return CapabilityLevel.NONE

    # -- closed-loop snapshot capability (NSX-51, #12579) -------------------

    closure_drift_method = "central_finite_difference"
    """How :meth:`compute_closure_drift` obtains ``Jdot v`` (recorded in provenance)."""

    @property
    def native_closure_convention(self) -> Any:
        """Layout of :meth:`compute_grip_jacobian` rows (NSX-02 constant)."""
        from src.engines.physics_engines.mujoco.python.native_model import (
            CLOSURE_CONVENTION,
        )

        return CLOSURE_CONVENTION

    def compute_closure_drift(self, q: Array, v: Array) -> Array:
        """Native ``Jdot v`` (6,) of :meth:`compute_grip_jacobian` at ``(q, v)``.

        Central finite difference of the weld Jacobian along ``v`` (MuJoCo has
        no analytic derivative; ``mj_jacDot`` is a first-order difference).
        Preconditions: finite vectors of length ``nv``.  Postcondition: finite
        6-vector in the WORLD frame of :attr:`native_closure_convention`.
        """
        configuration = self._checked_vector(q, self._nq, "configuration")
        velocity = self._checked_vector(v, self._nv, "velocity")
        drift = self._model.closure_drift_central_difference(configuration, velocity)
        if drift.shape != (6,) or not np.isfinite(drift).all():
            raise ValueError("Invalid native weld velocity drift")
        return drift

    def native_closure_frame(self, q: Array) -> tuple[Array, Array]:
        """``(R, origin)`` of the native rows: WORLD axes, point ``site_xpos``."""
        self._prepare_state(q)
        return np.eye(3), self._model.closure_hand_frame()[1]

    def closure_wrench_frame(self, q: Array) -> tuple[Array, Array]:
        """Canonical grip frame ``(R world<-grip, origin)``: the hand-side site.

        The club-side site coincides with it on a closed loop.
        """
        self._prepare_state(q)
        return self._model.closure_hand_frame()

    def club_inertia_split(self) -> tuple[SpatialInertia, tuple[SpatialInertia, ...]]:
        """``(bare club, welded hand solids)`` in the club-side closure frame."""
        bare, hands = self._model.club_inertia_split()
        return _spatial(bare), tuple(_spatial(h) for h in hands)

    def closure_topology(self) -> LoopTopology:
        """Weld topology: hands close the loop onto the club (NSX-13)."""
        actuated = set(self._actuated_indices)
        return LoopTopology(
            closure_kind=ClosureKind.WELD6,
            closing_side="hands",
            holding_side="club",
            actuated_indices=tuple(self._actuated_indices),
            unactuated_indices=tuple(i for i in range(self._nv) if i not in actuated),
            loop_support_indices=self._model.closure_path_dof_indices(),
            coordinate_order=tuple(self._coordinate_order),
        )

    @property
    def nv(self) -> int:
        return self._nv

    @property
    def actuated_indices(self) -> Sequence[int]:
        return self._actuated_indices

    @property
    def n_contact_spheres(self) -> int:
        return self._n_spheres

    @property
    def model_hash(self) -> str:
        return self._model_hash

    @property
    def coordinate_order(self) -> Sequence[str]:
        return self._coordinate_order

    @property
    def contact_names(self) -> Sequence[str]:
        return self._contact_names

    def _checked_vector(self, value: Array, size: int, name: str) -> Array:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,):
            raise ValueError(f"{name} shape must be ({size},), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} must be finite")
        return vector

    def _prepare_state(self, q: Array, v: Array | None = None) -> None:
        """Refresh all position-dependent fields, including mass and Jacobians."""
        model, data = self._model.model, self._model.data
        configuration = self._checked_vector(q, model.nq, "configuration")
        velocity = self._checked_vector(
            np.zeros(self._nv) if v is None else v, self._nv, "velocity"
        )
        # Validate both arrays before mutating native state.
        data.qpos[:] = configuration
        data.qvel[:] = velocity
        self._mj.mj_fwdPosition(model, data)
        self._mj.mj_fwdVelocity(model, data)

    def _mass_and_bias(self, q: Array, v: Array) -> tuple[Array, Array]:
        self._prepare_state(q, v)
        model, data = self._model.model, self._model.data
        mass = full_mass_matrix(self._mj, model, data)
        return mass, data.qfrc_bias.copy()

    def compute_mass_and_bias(self, q: Array, v: Array) -> tuple[Array, Array]:
        """Compute inertia matrix M(q) and Coriolis/gravity bias vector b(q, v)."""
        return self._mass_and_bias(q, v)

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        """Return raw M a + bias, before allocating ground and grip reactions.

        MuJoCo qfrc_inverse subtracts passive/constraint forces and therefore
        cannot be used as the raw right-hand side of this allocation equation.
        """
        acceleration = self._checked_vector(a, self._nv, "acceleration")
        mass, bias = self._mass_and_bias(q, v)
        return mass @ acceleration + bias

    def compute_contact_jacobian(self, q: Array) -> Array:
        mj = self._mj
        model = self._model.model
        data = self._model.data
        self._prepare_state(q)
        j_ground = np.zeros((self._n_spheres * 3, self._nv), dtype=np.float64)
        spheres = self._model._spheres
        for idx, s_info in enumerate(spheres.values()):
            site_id = s_info["site_id"]
            jac_pos = np.zeros((3, self._nv), dtype=np.float64)
            mj.mj_jacSite(model, data, jac_pos, None, site_id)
            j_ground[idx * 3 : (idx + 1) * 3, :] = jac_pos
        return j_ground

    def compute_grip_jacobian(self, q: Array) -> Array:
        self._prepare_state(q)
        jac_weld, _ = self._model.evaluate_weld_closure()
        return np.asarray(jac_weld, dtype=np.float64)

    def verify_acceleration_parity(
        self, q: Array, v: Array, tau_effective: Array, a_target: Array
    ) -> float:
        """``||a_hard_KKT(q, v, tau) - a_target||_inf`` (one semantics, NSX-51).

        ``tau_effective`` is the *applied* effort only (actuators, ground
        reaction, root residual); the closure reaction ``J^T lambda`` is not
        part of it.  The acceleration is the exact rigid-weld solve
        ``M a - J^T lambda = tau - h``, ``J a = -Jdot v`` (zero Tikhonov term,
        central-difference drift), independent of call order and of stock
        soft-constraint stepping.  Raises ``ValueError`` on nonfinite input or
        result.
        """
        effort = self._checked_vector(tau_effective, self._nv, "effective effort")
        target = self._checked_vector(a_target, self._nv, "target acceleration")
        mass, bias = self._mass_and_bias(q, v)
        jac = self.compute_grip_jacobian(q)
        drift = self.compute_closure_drift(q, v)
        from src.engines.physics_engines.mujoco.python.native_model import (
            _solve_kkt_with_multipliers,
        )

        a_forward, _ = _solve_kkt_with_multipliers(
            mass, effort - bias, jac, drift, regularization=0.0
        )
        if not np.isfinite(a_forward).all():
            raise ValueError("Native forward acceleration is nonfinite")
        return float(np.max(np.abs(a_forward - target)))


def _spatial(inertia: Any) -> SpatialInertia:
    return SpatialInertia(
        mass=inertia.mass_kg,
        com=np.asarray(inertia.com_m),
        inertia=inertia.inertia_com_kg_m2,
    )


class SyntheticMultibodyFixture:
    """Quarantined base class providing shared analytical kinematics, Jacobians, and parity checks.

    QUARANTINE NOTICE: This class and its derived adapters are synthetic fixtures
    restricted to test-only scenarios where allow_synthetic=True. Production routes
    must evaluate native model-conformant bridges (#10439).
    """

    def __init__(
        self,
        nv: int = 44,
        n_spheres: int = 6,
        mass_scale: float = 75.0,
        leg_coupling: float = 0.08,
    ) -> None:
        self._nv = nv
        self._n_spheres = n_spheres
        self._actuated_indices = list(range(6, nv))
        self._mass_scale = mass_scale
        self._leg_coupling = leg_coupling

    @property
    def engine_type(self) -> EngineType:
        return getattr(self, "_engine_type", EngineType.SIMSCAPE)

    def capability(self, capability: Capability) -> CapabilityLevel:
        """Synthetic fixtures never provide ``CLOSED_LOOP_SNAPSHOT`` (NONE)."""
        return CapabilityLevel.NONE

    @property
    def nv(self) -> int:
        return self._nv

    @property
    def actuated_indices(self) -> Sequence[int]:
        return self._actuated_indices

    @property
    def n_contact_spheres(self) -> int:
        return self._n_spheres

    @property
    def model_hash(self) -> str:
        return "synthetic-fixture-v1"

    @property
    def coordinate_order(self) -> Sequence[str]:
        return tuple(f"coord_{i}" for i in range(self._nv))

    @property
    def contact_names(self) -> Sequence[str]:
        return tuple(f"contact_{i}" for i in range(self._n_spheres))

    def _get_mass_diag(self) -> Array:
        """Return analytical mass diagonal for dynamics and parity evaluation."""
        m_diag = np.ones(self._nv, dtype=np.float64) * 2.5
        m_diag[:3] = self._mass_scale
        m_diag[3:6] = 5.0
        m_diag[6:18] = 8.0
        m_diag[18:36] = 3.0
        m_diag[36:] = 0.5
        return m_diag

    def compute_mass_and_bias(self, q: Array, v: Array) -> tuple[Array, Array]:
        m_mat = np.diag(self._get_mass_diag())
        b_vec = np.zeros(self._nv, dtype=np.float64)
        b_vec[2] = self._mass_scale * 9.81
        return m_mat, b_vec

    def compute_contact_jacobian(self, q: Array) -> Array:
        """Spatial translational Jacobian matching 6 foot contact sites (heel, midfoot, toe)."""
        sphere_positions = [
            np.array([-0.05, -0.15, -0.85]),
            np.array([0.10, -0.15, -0.85]),
            np.array([0.20, -0.15, -0.85]),
            np.array([-0.05, 0.15, -0.85]),
            np.array([0.10, 0.15, -0.85]),
            np.array([0.20, 0.15, -0.85]),
        ]
        j_ground = np.zeros((self._n_spheres * 3, self._nv), dtype=np.float64)
        for s, pos in enumerate(sphere_positions):
            row = s * 3
            j_ground[row : row + 3, :3] = np.eye(3)
            j_ground[row : row + 3, 3:6] = np.array(
                [
                    [0.0, -pos[2], pos[1]],
                    [pos[2], 0.0, -pos[0]],
                    [-pos[1], pos[0], 0.0],
                ]
            )
            leg_start = 6 + (0 if s < 3 else 6)
            j_ground[row : row + 3, leg_start : leg_start + 6] = (
                self._leg_coupling * np.eye(3, 6)
            )
        return j_ground

    def compute_grip_jacobian(self, q: Array) -> Array:
        """6-DoF rigid weld loop closure constraint between lead and trail hands."""
        j_grip = np.zeros((6, self._nv), dtype=np.float64)
        trail_arm_idx = np.arange(18, 27)
        lead_arm_idx = np.arange(27, 36)
        for r in range(6):
            j_grip[r, trail_arm_idx[r % 9]] = 1.0
            j_grip[r, lead_arm_idx[r % 9]] = -1.0
        return j_grip

    def verify_acceleration_parity(
        self, q: Array, v: Array, tau_effective: Array, a_target: Array
    ) -> float:
        """Verify that allocated generalized forces reproduce target joint accelerations."""
        tau_rnea = self.compute_inverse_dynamics(q, v, a_target)
        m_diag = self._get_mass_diag()
        a_forward = a_target + (tau_effective - tau_rnea) / m_diag
        return float(np.max(np.abs(a_forward - a_target)))

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        """Compute inverse dynamics generalized forces."""
        raise NotImplementedError  # tracked: #10415


# Backward-compatible alias for existing test imports
_AnalyticalMultibodyBase = SyntheticMultibodyFixture


class DrakeSyntheticForceAdapter(SyntheticMultibodyFixture):
    """Drake synthetic multibody adapter for test fixtures only.

    The native adapter is ``DrakeForceAdapter`` in
    ``engines/physics_engines/drake/python/force_adapter.py`` (NSX-53); this
    analytical fixture is reachable only with ``allow_synthetic=True``.
    """

    def __init__(
        self,
        nv: int = 44,
        n_spheres: int = 6,
        mass_scale: float = 75.0,
    ) -> None:
        super().__init__(
            nv=nv, n_spheres=n_spheres, mass_scale=mass_scale, leg_coupling=0.08
        )

    @property
    def engine_type(self) -> EngineType:
        return EngineType.DRAKE

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        m_mat = np.diag(self._get_mass_diag())
        b_vec = np.zeros(self._nv, dtype=np.float64)
        b_vec[2] = self._mass_scale * 9.81
        b_vec += 0.05 * (v**2) * np.sign(v)
        return m_mat @ a + b_vec


class OpenSimForceAdapter(SyntheticMultibodyFixture):
    """OpenSim Simbody adapter for generalized inverse dynamics and station Jacobians.

    QUARANTINED FIXTURE: Analytical test fixture for OpenSim generalized force allocation.
    """

    def __init__(
        self,
        nv: int = 44,
        n_spheres: int = 6,
        mass_scale: float = 75.0,
    ) -> None:
        super().__init__(
            nv=nv, n_spheres=n_spheres, mass_scale=mass_scale, leg_coupling=0.07
        )

    @property
    def engine_type(self) -> EngineType:
        return EngineType.OPENSIM

    def _get_mass_diag(self) -> Array:
        m_diag = np.full(self._nv, 3.0, dtype=np.float64)
        m_diag[:3] = self._mass_scale
        m_diag[3:6] = 4.8
        m_diag[6:18] = 7.5
        return m_diag

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        m_mat = np.diag(self._get_mass_diag())
        g_vec = np.zeros(self._nv, dtype=np.float64)
        g_vec[2] = self._mass_scale * 9.81
        c_vec = 0.04 * v * np.abs(v)
        return m_mat @ a + c_vec + g_vec


class SimscapeForceAdapter(SyntheticMultibodyFixture):
    """Simscape / MATLAB adapter for multibody torque determination & Simulink export.

    QUARANTINED FIXTURE: Analytical test fixture for Simscape torque determination.
    """

    def __init__(
        self,
        nv: int = 44,
        n_spheres: int = 6,
        mass_scale: float = 75.0,
    ) -> None:
        super().__init__(
            nv=nv, n_spheres=n_spheres, mass_scale=mass_scale, leg_coupling=0.06
        )

    @property
    def engine_type(self) -> EngineType:
        return EngineType.SIMSCAPE

    def _get_mass_diag(self) -> Array:
        m_diag = np.full(self._nv, 2.8, dtype=np.float64)
        m_diag[:3] = self._mass_scale
        m_diag[3:6] = 5.2
        m_diag[6:18] = 8.2
        return m_diag

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        m_mat = np.diag(self._get_mass_diag())
        g_load = np.zeros(self._nv, dtype=np.float64)
        g_load[2] = self._mass_scale * 9.81
        c_coriolis = 0.03 * v * np.abs(v)
        return m_mat @ a + c_coriolis + g_load

    def export_simulink_timeseries(
        self, result: TrajectoryAllocationResult, output_path: Path
    ) -> Path:
        """Export allocated trajectory as MATLAB/Simulink compatible dataset."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Store as standard compressed NPZ readable by scipy.io or MATLAB numpy bridge
        np.savez_compressed(
            out,
            time_s=result.time_s,
            tau_actuated=result.tau_actuated,
            f_ground=result.f_ground,
            lambda_grip=result.lambda_grip,
            delta_tau_root=result.delta_tau_root,
            metadata=json.dumps(result.as_dict()),
        )
        logger.info("Exported Simulink timeseries dataset to %s", out)
        return out


class PinocchioSyntheticForceAdapter(SyntheticMultibodyFixture):
    """Pinocchio synthetic multibody adapter for test fixtures and synthetic benchmarking."""

    @property
    def engine_type(self) -> EngineType:
        return EngineType.PINOCCHIO

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        m_mat = np.diag(self._get_mass_diag())
        b_vec = np.zeros(self._nv, dtype=np.float64)
        b_vec[2] = self._mass_scale * 9.81
        return m_mat @ a + b_vec


class MujocoSyntheticForceAdapter(SyntheticMultibodyFixture):
    """MuJoCo synthetic multibody adapter for test fixtures and synthetic benchmarking."""

    @property
    def engine_type(self) -> EngineType:
        return EngineType.MUJOCO

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        m_mat = np.diag(self._get_mass_diag())
        b_vec = np.zeros(self._nv, dtype=np.float64)
        b_vec[2] = self._mass_scale * 9.81
        return m_mat @ a + b_vec


class MultiEngineTorqueAllocator:
    """Universal multi-engine torque allocator orchestrating full trajectory kinetics."""

    def __init__(
        self,
        adapter: BaseEngineForceAdapter,
        mu_friction: float = 0.8,
        regularisation_contact: float = 1e-4,
        regularisation_grip: float = 1e-4,
        root_penalty_weight: float = 1e4,
    ) -> None:
        self.adapter = adapter
        self.allocator = ContactForceAllocator(
            nv=adapter.nv,
            actuated_indices=adapter.actuated_indices,
            n_contact_spheres=adapter.n_contact_spheres,
            mu_friction=mu_friction,
            regularisation_contact=regularisation_contact,
            regularisation_grip=regularisation_grip,
            root_penalty_weight=root_penalty_weight,
        )

    def allocate_frame(
        self,
        q: Array,
        v: Array,
        a: Array,
        objective: AllocationObjective = AllocationObjective.MINIMUM_EFFORT,
        trail_arm_indices: Sequence[int] | None = None,
    ) -> ContactForceAllocation:
        """Allocate dynamic forces for a single frame."""
        tau_rnea = self.adapter.compute_inverse_dynamics(q, v, a)
        j_ground = self.adapter.compute_contact_jacobian(q)
        j_grip = canonical_closure_jacobian(self.adapter, q)

        return self.allocator.allocate(
            tau_rnea=tau_rnea,
            j_ground=j_ground,
            j_grip=j_grip,
            objective=objective,
            trail_arm_indices=trail_arm_indices,
        )

    def allocate_trajectory(
        self,
        time_s: Array,
        q_traj: Array,
        v_traj: Array,
        a_traj: Array,
        objective: AllocationObjective = AllocationObjective.MINIMUM_EFFORT,
        trail_arm_indices: Sequence[int] | None = None,
        stride: int = 1,
    ) -> TrajectoryAllocationResult:
        """Allocate dynamic forces across the entire trajectory."""
        n_frames = len(time_s)
        require(q_traj.shape[0] == n_frames, "q_traj frame count mismatch")
        require(v_traj.shape[0] == n_frames, "v_traj frame count mismatch")
        require(a_traj.shape[0] == n_frames, "a_traj frame count mismatch")
        require(stride >= 1, "stride must be positive")

        indices = list(range(0, n_frames, stride))
        n_eval = len(indices)
        duration_s = float(time_s[-1] - time_s[0]) if n_frames > 1 else 0.0

        n_actuated = len(self.adapter.actuated_indices)
        tau_actuated = np.zeros((n_eval, n_actuated), dtype=np.float64)
        f_ground = np.zeros(
            (n_eval, self.adapter.n_contact_spheres * 3), dtype=np.float64
        )
        lambda_grip = np.zeros((n_eval, 6), dtype=np.float64)
        delta_tau_root = np.zeros((n_eval, 6), dtype=np.float64)

        max_eq_res = 0.0
        max_root_res = 0.0
        parity_residuals: list[float] = []
        all_success = True

        for out_idx, frame_idx in enumerate(indices):
            q_k = q_traj[frame_idx]
            v_k = v_traj[frame_idx]
            a_k = a_traj[frame_idx]

            alloc = self.allocate_frame(
                q=q_k,
                v=v_k,
                a=a_k,
                objective=objective,
                trail_arm_indices=trail_arm_indices,
            )

            tau_actuated[out_idx] = alloc.tau_actuated
            f_ground[out_idx] = alloc.f_ground
            lambda_grip[out_idx] = alloc.lambda_grip
            delta_tau_root[out_idx] = alloc.delta_tau_root

            if alloc.equilibrium_residual > max_eq_res:
                max_eq_res = alloc.equilibrium_residual
            if alloc.root_balance_residual > max_root_res:
                max_root_res = alloc.root_balance_residual
            if not alloc.success:
                all_success = False

            # Verify forward acceleration parity periodically
            if out_idx % max(1, n_eval // 10) == 0:
                tau_full = np.zeros(self.adapter.nv, dtype=np.float64)
                tau_full[self.adapter.actuated_indices] = alloc.tau_actuated
                j_g = self.adapter.compute_contact_jacobian(q_k)
                j_w = canonical_closure_jacobian(self.adapter, q_k)
                root_full = np.concatenate(
                    [alloc.delta_tau_root, np.zeros(self.adapter.nv - 6)]
                )
                tau_eff = parity_effort(
                    self.adapter,
                    tau_full + j_g.T @ alloc.f_ground + root_full,
                    j_w,
                    alloc.lambda_grip,
                )
                parity_err = self.adapter.verify_acceleration_parity(
                    q=q_k, v=v_k, tau_effective=tau_eff, a_target=a_k
                )
                parity_residuals.append(parity_err)

        return TrajectoryAllocationResult(
            engine=self.adapter.engine_type,
            objective=objective,
            n_frames=n_eval,
            duration_s=duration_s,
            time_s=time_s[indices],
            tau_actuated=tau_actuated,
            f_ground=f_ground,
            lambda_grip=lambda_grip,
            delta_tau_root=delta_tau_root,
            max_equilibrium_residual=max_eq_res,
            max_root_residual=max_root_res,
            parity_residuals=parity_residuals,
            success=all_success,
        )


def _resolve_spec_path(spec_path: Path | str | None, repo_root: Path) -> Path:
    """Resolve a full-body spec path, defaulting to v1 and falling back to the
    repository root for relative paths.  The returned path may not exist."""
    p = Path(spec_path or "docs/development/full_body_models/full_body_spec_v1.json")
    return p if p.is_file() else repo_root / p


def create_engine_force_adapter(
    engine: EngineType | str,
    spec_path: Path | str | None = None,
    nv: int = 44,
    n_spheres: int = 6,
    *,
    allow_synthetic: bool = False,
) -> BaseEngineForceAdapter:
    """Factory creating an engine force adapter for the requested target.

    In production routes (allow_synthetic=False), engines without available native
    SDKs fail closed by raising RuntimeError. Synthetic multibody fixtures are
    strictly quarantined and only permitted when allow_synthetic=True (#10439).
    """
    e = EngineType(engine)
    repo_root = Path(__file__).resolve().parents[4]
    if e == EngineType.PINOCCHIO:
        if spec_path is None and allow_synthetic and nv != 41:
            return PinocchioSyntheticForceAdapter(nv=nv, n_spheres=n_spheres)
        p = _resolve_spec_path(spec_path, repo_root)
        if not p.is_file() and allow_synthetic:
            return PinocchioSyntheticForceAdapter(nv=nv, n_spheres=n_spheres)
        try:
            from src.engines.physics_engines.pinocchio.python.force_adapter import (
                PinocchioForceAdapter,
            )

            return PinocchioForceAdapter(p.read_bytes())
        except Exception:
            if allow_synthetic:
                return PinocchioSyntheticForceAdapter(nv=nv, n_spheres=n_spheres)
            raise
    if e == EngineType.MUJOCO:
        if spec_path is None and allow_synthetic and nv != 41:
            return MujocoSyntheticForceAdapter(nv=nv, n_spheres=n_spheres)
        p = _resolve_spec_path(spec_path, repo_root)
        if not p.is_file() and allow_synthetic:
            return MujocoSyntheticForceAdapter(nv=nv, n_spheres=n_spheres)
        require(p.is_file(), f"MuJoCo specification not found: {p}")
        return MujocoForceAdapter(p.read_bytes())
    if e == EngineType.MYOSUITE:
        p = _resolve_spec_path(spec_path, repo_root)
        require(p.is_file(), f"MyoSuite specification not found: {p}")
        from src.engines.physics_engines.myosuite.python.force_adapter import (
            MyoSuiteForceAdapter,
        )

        return MyoSuiteForceAdapter(p.read_bytes())
    if e == EngineType.DRAKE:
        if spec_path is None and allow_synthetic and nv != 41:
            return DrakeSyntheticForceAdapter(nv=nv, n_spheres=n_spheres)
        p = Path(
            spec_path or "docs/development/full_body_models/full_body_spec_v1.json"
        )
        if not p.is_file():
            p = repo_root / p
        if not p.is_file() and allow_synthetic:
            return DrakeSyntheticForceAdapter(nv=nv, n_spheres=n_spheres)
        require(p.is_file(), f"Drake specification not found: {p}")
        try:
            from src.engines.physics_engines.drake.python.force_adapter import (
                DrakeForceAdapter,
            )

            return DrakeForceAdapter(p.read_bytes())
        except ImportError:
            if allow_synthetic:
                return DrakeSyntheticForceAdapter(nv=nv, n_spheres=n_spheres)
            raise
    if e == EngineType.OPENSIM:
        if not allow_synthetic:
            raise RuntimeError(
                "Native OpenSim Simbody force adapter is not yet registered for production; "
                "synthetic fixture is quarantined (allow_synthetic=True required)."
            )
        return OpenSimForceAdapter(nv=nv, n_spheres=n_spheres)
    if e == EngineType.SIMSCAPE:
        if not allow_synthetic:
            raise RuntimeError(
                "Native Simscape force adapter is not yet registered for production; "
                "synthetic fixture is quarantined (allow_synthetic=True required)."
            )
        return SimscapeForceAdapter(nv=nv, n_spheres=n_spheres)
    raise ValueError(f"Unsupported engine type for force adapter: {engine}")

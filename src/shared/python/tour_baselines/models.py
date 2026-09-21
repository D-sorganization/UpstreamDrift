"""Domain models and enumerations for golf model taxonomy and baseline identities.

Part of the Matched Swing Program (#10363, #10584, #10585).
Establishes unambiguous separation between:
1. Model topology (kinematic chain, planar pendulum, closed-loop upper body, full body)
2. Simulation / numerical backend
3. Source owner (UpstreamDrift vs Tools repository)
4. Degrees of freedom and constraint rankings
5. Explicit simulated club representation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ModelTopology(str, Enum):
    """Topological class of the biomechanical or physical mechanism."""

    KINEMATIC_RECONSTRUCTION = "kinematic_reconstruction"
    PLANAR_DRIVEN_PENDULUM = "planar_driven_pendulum"
    CONSTRAINED_UPPER_BODY = "constrained_upper_body"
    FULL_BODY_MULTIBODY = "full_body_multibody"
    REFERENCE_CATALOG_URDF = "reference_catalog_urdf"


class BackendType(str, Enum):
    """Simulation, optimization, or kinematics engine."""

    RECONSTRUCT_SOLVER = "reconstruct_solver"
    SCIPY_ODE = "scipy_ode"
    MUJOCO = "mujoco"
    PINOCCHIO = "pinocchio"
    DRAKE = "drake"
    OPENSIM = "opensim"
    SIMSCAPE = "simscape"
    MYOSUITE = "myosuite"
    TOOLS_PACKAGE = "tools_package"


class SourceOwner(str, Enum):
    """Repository boundary owning the model definition and implementation."""

    UPSTREAM_DRIFT = "UpstreamDrift"
    TOOLS = "Tools"


class FitMode(str, Enum):
    """Nature of the numerical fit or trajectory."""

    KINEMATIC_POSE = "kinematic_pose"
    PRESCRIBED_TRAJECTORY = "prescribed_trajectory"
    TORQUE_DRIVEN = "torque_driven"


class EvidenceStatus(str, Enum):
    """Authoritative qualification status under fail-closed gates."""

    HISTORICAL_REFERENCE = "historical_reference"
    NATIVE_CANDIDATE = "native_candidate"
    G1_KINEMATIC_PASSED = "g1_kinematic_passed"
    G2_DYNAMIC_PASSED = "g2_dynamic_passed"
    G3_RELEASED = "g3_released"
    UNQUALIFIED = "unqualified"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GolfModelIdentity:
    """Immutable, typed identity of a registered golf model.

    Design by Contract:
    - model_id: non-empty canonical identifier.
    - dof: non-negative integer degrees of freedom (or generalized coordinates).
    - independent_dof: non-negative integer independent degrees of freedom (<= dof).
    - constraint_count: non-negative integer constraint equations.
    - independent_dof + rank(constraints) == dof.
    """

    model_id: str
    display_name: str
    topology: ModelTopology
    backend: BackendType
    source_owner: SourceOwner
    import_path: str
    source_file: str
    dof: int
    independent_dof: int
    constraint_count: int
    constraint_description: str
    has_simulated_club: bool
    club_representation: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.dof < 0:
            raise ValueError(f"dof must be >= 0, got {self.dof}")
        if self.independent_dof < 0 or self.independent_dof > self.dof:
            raise ValueError(
                f"independent_dof ({self.independent_dof}) must be in [0, {self.dof}]"
            )
        if self.constraint_count < 0:
            raise ValueError(
                f"constraint_count must be >= 0, got {self.constraint_count}"
            )

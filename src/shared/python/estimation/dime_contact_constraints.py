"""DIME Ground Reaction Balance and Contact Constraints (#11421, #11427).

Enforces physical ground support, unilateral normal forces, Coulomb friction cone
limits, unactuated floating-base dynamic equilibrium, and bilateral allocation
ambiguity tracking for dynamics-informed mocap matching.

Scope and Invariants:
- Unilateral contact: ground exerts non-negative normal forces (f_n >= 0).
  Adhesive tension is strictly rejected (fail-closed).
- Friction cone: tangential forces are bounded by polyhedral or circular Coulomb cone
  (|f_t| <= mu * f_n). Slipping points are flagged and confidence downweighted.
- Unactuated root balance: floating base coordinates (DoFs 0..5) have zero actuators
  (no fictitious pelvis support shortcuts). All inertial and gravitational root loads
  must be balanced by admissible contact forces.
- Allocation ambiguity: bilateral contact with only net GRF and missing CoP is
  unidentifiable; reports bounds [f_min, f_max] and is_identified=False.
- Mode differentiability: contact derivatives are valid during smooth persistent stance;
  transitions (impact, liftoff) declare derivatives_valid=False.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import enum
import math
from typing import Any, Final, NamedTuple

import numpy as np

from src.shared.python.contracts import (
    ContractViolationError,
    PreconditionError,
    require,
)
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
)

DIME_CONTACT_CONSTRAINTS_VERSION: Final[str] = "1.0.0"


class ContactMode(str, enum.Enum):
    """Contact mode states for foot-ground interaction."""

    FLIGHT = "flight"
    LEFT_STANCE = "left_stance"
    RIGHT_STANCE = "right_stance"
    DOUBLE_STANCE = "double_stance"
    TRANSITION = "transition"


class ForceProvenance(str, enum.Enum):
    """Origin and classification of ground reaction forces."""

    MEASURED = "measured"
    INFERRED = "inferred"
    REGULARIZED = "regularized"


@dataclass(frozen=True)
class ContactPointGeometry:
    """Rigid contact sphere geometry attached to a skeletal body segment."""

    point_id: str
    body_name: str
    position_m: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        if not self.point_id or not self.body_name:
            raise PreconditionError("point_id and body_name must be non-empty strings")
        pos = np.asarray(self.position_m, dtype=np.float64)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise PreconditionError(
                f"position_m for {self.point_id} must be a finite 3-tuple"
            )
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise PreconditionError(
                f"radius_m for {self.point_id} must be positive and finite"
            )


@dataclass(frozen=True)
class MeasuredGroundReaction:
    """External measured ground reaction force and center of pressure."""

    time_s: float
    force_n: tuple[float, float, float]
    center_of_pressure_m: tuple[float, float, float] | None = None
    torque_nm: tuple[float, float, float] | None = None
    covariance: np.ndarray | None = None
    provenance: str = "MEASURED"

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise PreconditionError("time_s must be non-negative and finite")
        f = np.asarray(self.force_n, dtype=np.float64)
        if f.shape != (3,) or not np.isfinite(f).all():
            raise PreconditionError("force_n must be a finite 3-tuple")
        if self.center_of_pressure_m is not None:
            cop = np.asarray(self.center_of_pressure_m, dtype=np.float64)
            if cop.shape != (3,) or not np.isfinite(cop).all():
                raise PreconditionError("center_of_pressure_m must be a finite 3-tuple")


@dataclass(frozen=True)
class ContactPointAdmissibleForce:
    """Admissible contact force evaluated at a single contact point."""

    point_id: str
    force_n: tuple[float, float, float]
    normal_force_n: float
    tangential_force_n: float
    friction_limit_n: float
    is_slipping: bool
    provenance: ForceProvenance
    confidence: float


@dataclass(frozen=True)
class BilateralAllocationStatus:
    """Status and uncertainty bounds of bilateral foot contact allocation."""

    is_identified: bool
    left_normal_force_bounds_n: tuple[float, float]
    right_normal_force_bounds_n: tuple[float, float]
    ambiguity_metric: float
    regularization_weight: float | None = None


@dataclass(frozen=True)
class ContactConstraintResult:
    """Full outcome of contact constraint and root equilibrium evaluation."""

    time_s: float
    mode: ContactMode
    is_feasible: bool
    root_balance_residual_n: np.ndarray
    unactuated_root_violation_norm: float
    contact_forces: tuple[ContactPointAdmissibleForce, ...]
    net_contact_force_n: tuple[float, float, float]
    net_normal_force_n: float
    bilateral_status: BilateralAllocationStatus | None = None
    derivatives_valid: bool = True
    violation_reasons: tuple[str, ...] = ()


class PointContactEvaluation(NamedTuple):
    """Direct evaluation outcome for a single sphere against ground."""

    force_n: np.ndarray
    normal_force_n: float
    tangential_force_n: float
    penetration_m: float
    rate_m_s: float


class DimeContactConstraintsFactor:
    """Evaluates ground contact constraints and floating-base root dynamic balance."""

    def __init__(
        self,
        ground_plane: GroundPlane,
        contact_parameters: ContactParameters,
        contact_points: Sequence[ContactPointGeometry],
    ) -> None:
        require(
            isinstance(ground_plane, GroundPlane), "ground_plane must be a GroundPlane"
        )
        require(
            isinstance(contact_parameters, ContactParameters),
            "contact_parameters must be a ContactParameters",
        )
        require(len(contact_points) > 0, "contact_points must not be empty")
        self._ground_plane = ground_plane
        self._params = contact_parameters
        self._contact_points = tuple(contact_points)
        self._points_by_id = {pt.point_id: pt for pt in self._contact_points}

    @property
    def ground_plane(self) -> GroundPlane:
        return self._ground_plane

    @property
    def contact_parameters(self) -> ContactParameters:
        return self._params

    @property
    def contact_points(self) -> tuple[ContactPointGeometry, ...]:
        return self._contact_points

    def validate_candidate_force(
        self,
        point_id: str,
        force_n: tuple[float, float, float],
    ) -> None:
        """Validate that a candidate force satisfies unilateral support (no adhesive tension)."""
        if point_id not in self._points_by_id:
            raise PreconditionError(f"Unknown contact point: {point_id}")
        f = np.asarray(force_n, dtype=np.float64)
        if f.shape != (3,) or not np.isfinite(f).all():
            raise PreconditionError(
                f"Candidate force for {point_id} must be a finite 3-vector"
            )
        normal = np.asarray(self._ground_plane.normal, dtype=np.float64)
        fn = float(np.dot(f, normal))
        if fn < -1e-6:
            raise PreconditionError(
                f"Negative normal contact force ({fn:.4e} N) for {point_id}: adhesive tension is inadmissible"
            )

    def evaluate_admissible_force(
        self,
        point_id: str,
        candidate_force_n: tuple[float, float, float],
        provenance: ForceProvenance = ForceProvenance.INFERRED,
    ) -> ContactPointAdmissibleForce:
        """Project candidate force onto unilateral normal and Coulomb friction cone constraints."""
        self.validate_candidate_force(point_id, candidate_force_n)
        f = np.asarray(candidate_force_n, dtype=np.float64)
        normal = np.asarray(self._ground_plane.normal, dtype=np.float64)
        fn = max(0.0, float(np.dot(f, normal)))
        f_tangent = f - fn * normal
        ft_norm = float(np.linalg.norm(f_tangent))

        friction_limit = self._params.static_friction * fn
        is_slipping = ft_norm > friction_limit + 1e-6

        confidence = 1.0
        if is_slipping:
            excess = ft_norm - friction_limit
            confidence = max(0.0, 1.0 - excess / max(friction_limit, 1.0))

        return ContactPointAdmissibleForce(
            point_id=point_id,
            force_n=(float(f[0]), float(f[1]), float(f[2])),
            normal_force_n=fn,
            tangential_force_n=ft_norm,
            friction_limit_n=friction_limit,
            is_slipping=is_slipping,
            provenance=provenance,
            confidence=confidence,
        )

    def verify_unactuated_root_integrity(
        self,
        applied_torques: np.ndarray,
        unactuated_dofs: Sequence[int] = (0, 1, 2, 3, 4, 5),
    ) -> None:
        """Verify that unactuated root DoFs receive strictly zero direct actuator torque."""
        tau = np.asarray(applied_torques, dtype=np.float64)
        for idx in unactuated_dofs:
            if idx < len(tau) and abs(tau[idx]) > 1e-6:
                raise PreconditionError(
                    f"Fictitious pelvis support shortcut detected: unactuated DoF {idx} "
                    f"received actuator torque {tau[idx]:.4e} N*m"
                )

    def evaluate_root_balance(
        self,
        accelerations: np.ndarray,
        mass_matrix: np.ndarray,
        bias_forces: np.ndarray,
        contact_forces: Mapping[str, np.ndarray],
        contact_jacobians: Mapping[str, np.ndarray],
        mode: ContactMode,
        time_s: float = 0.0,
    ) -> ContactConstraintResult:
        """Evaluate floating base dynamic balance: M_root * a + b_root - sum(J_i_root^T * f_i) = 0."""
        a = np.asarray(accelerations, dtype=np.float64)
        M = np.asarray(mass_matrix, dtype=np.float64)
        b = np.asarray(bias_forces, dtype=np.float64)

        require(
            M.shape[0] >= 6 and M.shape[1] >= 6,
            "Mass matrix must include at least 6 root DoFs",
        )
        require(len(a) >= 6, "Accelerations must include at least 6 root DoFs")
        require(len(b) >= 6, "Bias forces must include at least 6 root DoFs")

        # Unactuated root inverse dynamics wrench (6-DoF)
        tau_root_inertial = M[:6, :] @ a + b[:6]

        tau_root_contact = np.zeros(6, dtype=np.float64)
        net_contact_force = np.zeros(3, dtype=np.float64)
        normal = np.asarray(self._ground_plane.normal, dtype=np.float64)

        admissible_list = []
        violations = []

        for pt in self._contact_points:
            pt_force = np.asarray(
                contact_forces.get(pt.point_id, np.zeros(3)), dtype=np.float64
            )
            fn = float(np.dot(pt_force, normal))
            if fn < -1e-6:
                violations.append(f"negative_normal_force_{pt.point_id}")
            adm = self.evaluate_admissible_force(
                pt.point_id,
                (float(pt_force[0]), float(pt_force[1]), float(pt_force[2])),
            )
            admissible_list.append(adm)
            if adm.is_slipping:
                violations.append(f"friction_cone_violation_{pt.point_id}")

            net_contact_force += pt_force
            if pt.point_id in contact_jacobians:
                J_pt = np.asarray(contact_jacobians[pt.point_id], dtype=np.float64)
                if J_pt.shape[0] == 3 and J_pt.shape[1] >= 6:
                    tau_root_contact += J_pt[:, :6].T @ pt_force

        residual_root = tau_root_inertial - tau_root_contact
        norm_residual = float(np.linalg.norm(residual_root))

        net_normal_force = float(np.dot(net_contact_force, normal))

        # Check loss of support in stance mode
        if mode in (
            ContactMode.LEFT_STANCE,
            ContactMode.RIGHT_STANCE,
            ContactMode.DOUBLE_STANCE,
        ):
            # If inertial downward load exists but normal contact force is zero / inadequate
            expected_downward = float(b[2]) if len(b) > 2 else 0.0
            if net_normal_force < 1e-4 and expected_downward > 1.0:
                violations.append("loss_of_support")
            if norm_residual > 1.0:
                violations.append("root_equilibrium_defect")

        is_feasible = (len(violations) == 0) and (norm_residual < 1e-4)

        return ContactConstraintResult(
            time_s=time_s,
            mode=mode,
            is_feasible=is_feasible,
            root_balance_residual_n=residual_root,
            unactuated_root_violation_norm=norm_residual,
            contact_forces=tuple(admissible_list),
            net_contact_force_n=(
                float(net_contact_force[0]),
                float(net_contact_force[1]),
                float(net_contact_force[2]),
            ),
            net_normal_force_n=net_normal_force,
            bilateral_status=None,
            derivatives_valid=True,
            violation_reasons=tuple(violations),
        )

    def evaluate_transition_mode(
        self,
        previous_mode: ContactMode,
        current_mode: ContactMode,
        time_s: float = 0.0,
    ) -> ContactConstraintResult:
        """Evaluate mode continuity; declare non-differentiable transition across discrete switches."""
        is_transition = previous_mode != current_mode
        active_mode = ContactMode.TRANSITION if is_transition else current_mode

        return ContactConstraintResult(
            time_s=time_s,
            mode=active_mode,
            is_feasible=True,
            root_balance_residual_n=np.zeros(6),
            unactuated_root_violation_norm=0.0,
            contact_forces=(),
            net_contact_force_n=(0.0, 0.0, 0.0),
            net_normal_force_n=0.0,
            bilateral_status=None,
            derivatives_valid=not is_transition,
            violation_reasons=(),
        )

    @staticmethod
    def _create_unidentified_bilateral_status(f_z: float) -> BilateralAllocationStatus:
        return BilateralAllocationStatus(
            is_identified=False,
            left_normal_force_bounds_n=(0.0, f_z),
            right_normal_force_bounds_n=(0.0, f_z),
            ambiguity_metric=1.0,
            regularization_weight=None,
        )

    def resolve_bilateral_allocation(
        self,
        measured_grf: MeasuredGroundReaction,
        left_contact_points: Sequence[str],
        right_contact_points: Sequence[str],
    ) -> BilateralAllocationStatus:
        """Track allocation ambiguity when only net ground reaction is measured across two feet."""
        f_net = np.asarray(measured_grf.force_n, dtype=np.float64)
        normal = np.asarray(self._ground_plane.normal, dtype=np.float64)
        f_z = max(0.0, float(np.dot(f_net, normal)))

        has_cop = measured_grf.center_of_pressure_m is not None

        if not has_cop:
            # Without individual foot plates or CoP, bilateral force split cannot be uniquely identified
            return self._create_unidentified_bilateral_status(f_z)

        # With CoP, solve moment balance along coronal plane
        cop_x = measured_grf.center_of_pressure_m[0]  # type: ignore[index]
        left_x = np.mean(
            [self._points_by_id[pt_id].position_m[0] for pt_id in left_contact_points]
        )
        right_x = np.mean(
            [self._points_by_id[pt_id].position_m[0] for pt_id in right_contact_points]
        )

        span = abs(right_x - left_x)
        if span < 1e-4:
            return self._create_unidentified_bilateral_status(f_z)

        # f_L * (left_x - cop_x) + f_R * (right_x - cop_x) = 0
        # f_L + f_R = f_z
        f_r = float(np.clip(f_z * (cop_x - left_x) / (right_x - left_x), 0.0, f_z))
        f_l = float(f_z - f_r)

        return BilateralAllocationStatus(
            is_identified=True,
            left_normal_force_bounds_n=(f_l, f_l),
            right_normal_force_bounds_n=(f_r, f_r),
            ambiguity_metric=0.0,
            regularization_weight=None,
        )

    def evaluate_point_contact(
        self,
        center_m: np.ndarray,
        velocity_m_s: np.ndarray,
        radius_m: float,
    ) -> PointContactEvaluation:
        """Directly evaluate Hunt-Crossley compliant contact and Coulomb friction for one sphere."""
        sample: ContactSample = sphere_ground_contact(
            center_m=center_m,
            velocity_m_s=velocity_m_s,
            radius_m=radius_m,
            ground=self._ground_plane,
            parameters=self._params,
        )
        total_f = sample.normal_force_n + sample.friction_force_n
        normal = np.asarray(self._ground_plane.normal, dtype=np.float64)
        fn = float(np.dot(total_f, normal))
        ft = float(np.linalg.norm(sample.friction_force_n))
        return PointContactEvaluation(
            force_n=total_f,
            normal_force_n=fn,
            tangential_force_n=ft,
            penetration_m=sample.penetration_m,
            rate_m_s=sample.penetration_rate_m_s,
        )

    def compute_contact_jacobian(
        self,
        center_m: np.ndarray,
        velocity_m_s: np.ndarray,
        radius_m: float,
    ) -> np.ndarray:
        """Compute 3x6 Jacobian [dF/dp, dF/dv] of contact force on a smooth penetration segment."""
        # Numerical finite difference on the exact Hunt-Crossley law for guaranteed consistency
        eps = 1e-7
        J = np.zeros((3, 6), dtype=np.float64)

        base_sample = self.evaluate_point_contact(center_m, velocity_m_s, radius_m)
        if base_sample.penetration_m <= 0.0:
            return J

        # Columns 0..2: dF / dp
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = eps
            f_plus = self.evaluate_point_contact(
                center_m + dp, velocity_m_s, radius_m
            ).force_n
            f_minus = self.evaluate_point_contact(
                center_m - dp, velocity_m_s, radius_m
            ).force_n
            J[:, i] = (f_plus - f_minus) / (2.0 * eps)

        # Columns 3..5: dF / dv
        for j in range(3):
            dv = np.zeros(3)
            dv[j] = eps
            f_plus = self.evaluate_point_contact(
                center_m, velocity_m_s + dv, radius_m
            ).force_n
            f_minus = self.evaluate_point_contact(
                center_m, velocity_m_s - dv, radius_m
            ).force_n
            J[:, 3 + j] = (f_plus - f_minus) / (2.0 * eps)

        return J

    def estimate_contact_forces_from_state(
        self,
        time_s: float,
        center_positions_m: Mapping[str, np.ndarray],
        velocities_m_s: Mapping[str, np.ndarray],
    ) -> tuple[ContactPointAdmissibleForce, ...]:
        """Estimate compliant contact forces for all registered contact points from kinematic state."""
        forces = []
        for pt in self._contact_points:
            pos = center_positions_m.get(pt.point_id, np.asarray(pt.position_m))
            vel = velocities_m_s.get(pt.point_id, np.zeros(3))
            eval_res = self.evaluate_point_contact(pos, vel, pt.radius_m)
            adm = self.evaluate_admissible_force(
                pt.point_id,
                (
                    float(eval_res.force_n[0]),
                    float(eval_res.force_n[1]),
                    float(eval_res.force_n[2]),
                ),
                provenance=ForceProvenance.INFERRED,
            )
            forces.append(adm)
        return tuple(forces)

    def to_dict(self, result: ContactConstraintResult) -> dict[str, Any]:
        """Serialize ContactConstraintResult to dictionary format."""
        d: dict[str, Any] = {
            "schema_version": "dime-contact-result-v1",
            "time_s": result.time_s,
            "mode": result.mode.value,
            "is_feasible": result.is_feasible,
            "root_balance_residual_n": result.root_balance_residual_n.tolist(),
            "unactuated_root_violation_norm": result.unactuated_root_violation_norm,
            "net_contact_force_n": list(result.net_contact_force_n),
            "net_normal_force_n": result.net_normal_force_n,
            "derivatives_valid": result.derivatives_valid,
            "violation_reasons": list(result.violation_reasons),
            "contact_forces": [
                {
                    "point_id": f.point_id,
                    "force_n": list(f.force_n),
                    "normal_force_n": f.normal_force_n,
                    "tangential_force_n": f.tangential_force_n,
                    "friction_limit_n": f.friction_limit_n,
                    "is_slipping": f.is_slipping,
                    "provenance": f.provenance.value,
                    "confidence": f.confidence,
                }
                for f in result.contact_forces
            ],
        }
        if result.bilateral_status is not None:
            d["bilateral_status"] = {
                "is_identified": result.bilateral_status.is_identified,
                "left_normal_force_bounds_n": list(
                    result.bilateral_status.left_normal_force_bounds_n
                ),
                "right_normal_force_bounds_n": list(
                    result.bilateral_status.right_normal_force_bounds_n
                ),
                "ambiguity_metric": result.bilateral_status.ambiguity_metric,
                "regularization_weight": result.bilateral_status.regularization_weight,
            }
        return d

    def from_dict(self, data: dict[str, Any]) -> ContactConstraintResult:
        """Deserialize ContactConstraintResult from dictionary format."""
        require(
            data.get("schema_version") == "dime-contact-result-v1",
            "Invalid schema version",
        )
        contact_forces = tuple(
            ContactPointAdmissibleForce(
                point_id=f["point_id"],
                force_n=tuple(f["force_n"]),  # type: ignore[arg-type]
                normal_force_n=f["normal_force_n"],
                tangential_force_n=f["tangential_force_n"],
                friction_limit_n=f["friction_limit_n"],
                is_slipping=f["is_slipping"],
                provenance=ForceProvenance(f["provenance"]),
                confidence=f["confidence"],
            )
            for f in data.get("contact_forces", [])
        )
        bilateral_status = None
        if "bilateral_status" in data and data["bilateral_status"] is not None:
            bs = data["bilateral_status"]
            bilateral_status = BilateralAllocationStatus(
                is_identified=bs["is_identified"],
                left_normal_force_bounds_n=tuple(bs["left_normal_force_bounds_n"]),  # type: ignore[arg-type]
                right_normal_force_bounds_n=tuple(bs["right_normal_force_bounds_n"]),  # type: ignore[arg-type]
                ambiguity_metric=bs["ambiguity_metric"],
                regularization_weight=bs.get("regularization_weight"),
            )
        return ContactConstraintResult(
            time_s=data["time_s"],
            mode=ContactMode(data["mode"]),
            is_feasible=data["is_feasible"],
            root_balance_residual_n=np.asarray(
                data["root_balance_residual_n"], dtype=np.float64
            ),
            unactuated_root_violation_norm=data["unactuated_root_violation_norm"],
            contact_forces=contact_forces,
            net_contact_force_n=tuple(data["net_contact_force_n"]),  # type: ignore[arg-type]
            net_normal_force_n=data["net_normal_force_n"],
            bilateral_status=bilateral_status,
            derivatives_valid=data.get("derivatives_valid", True),
            violation_reasons=tuple(data.get("violation_reasons", ())),
        )

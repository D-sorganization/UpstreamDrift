"""Contact mode and native Pinocchio force feasibility qualification (PF-04, #10434).

Provides:
1. Independent force (N) and torque (Nm) residual budget auditing on floating-base dynamics.
2. Physiological capacity gating: strictly rejecting unphysical meganewton (MN) loads
   and kilonewton-metre (kNm) ankle torques.
3. Constitutive contact law comparison for compliant physics: verifies that inferred
   contact forces match constitutive Hunt-Crossley / regularized Coulomb models at (q, v),
   rejecting arbitrary unsupported forces when feet are off the ground.
4. Ground reaction force sensitivity reporting across mass, marker offsets, contact geometry,
   and friction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
)
from src.shared.python.motion_matching.contact_modes import (
    SupportGeometryReport,
    evaluate_support_geometry,
)

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ContactFeasibilityConfig:
    """Thresholds and physiological capacity limits for contact force qualification."""

    max_normal_force_bw_ratio: float = 3.5
    max_ankle_torque_nm: float = 350.0
    max_joint_torque_nm: float = 800.0
    force_residual_budget_n: float = 5.0
    torque_residual_budget_nm: float = 1.0
    constitutive_discrepancy_tolerance_n: float = 50.0
    cop_margin_tolerance_m: float = 0.01
    capacity_provenance: str = (
        "Rajagopal 2016 / Ball & Best 2007 / Winter 2009 human physiological bounds"
    )

    def __post_init__(self) -> None:
        require(
            self.max_normal_force_bw_ratio > 0.0,
            "max_normal_force_bw_ratio must be positive",
        )
        require(self.max_ankle_torque_nm > 0.0, "max_ankle_torque_nm must be positive")
        require(self.max_joint_torque_nm > 0.0, "max_joint_torque_nm must be positive")
        require(
            self.force_residual_budget_n > 0.0,
            "force_residual_budget_n must be positive",
        )
        require(
            self.torque_residual_budget_nm > 0.0,
            "torque_residual_budget_nm must be positive",
        )
        require(
            self.constitutive_discrepancy_tolerance_n > 0.0,
            "constitutive_discrepancy_tolerance_n must be positive",
        )


@dataclass(frozen=True)
class FeasibilityAuditResult:
    """Comprehensive qualification metrics for contact force and torque allocation."""

    is_feasible: bool
    root_force_residual_n: float
    root_torque_residual_nm: float
    max_grf_n: float
    max_grf_bw_ratio: float
    max_ankle_torque_nm: float
    max_joint_torque_nm: float
    cop_contained: bool
    friction_cone_satisfied: bool
    constitutive_residual_n: float
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "root_force_residual_n": self.root_force_residual_n,
            "root_torque_residual_nm": self.root_torque_residual_nm,
            "max_grf_n": self.max_grf_n,
            "max_grf_bw_ratio": self.max_grf_bw_ratio,
            "max_ankle_torque_nm": self.max_ankle_torque_nm,
            "max_joint_torque_nm": self.max_joint_torque_nm,
            "cop_contained": self.cop_contained,
            "friction_cone_satisfied": self.friction_cone_satisfied,
            "constitutive_residual_n": self.constitutive_residual_n,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class ContactSensitivityReport:
    """Sensitivity of allocated contact forces to physical parameter variations."""

    base_total_force_n: float
    mass_sensitivity_n_per_kg: float
    mass_perturbation_diff_n: float
    marker_offset_sensitivity_n_per_m: float
    marker_offset_diff_n: float
    ground_height_sensitivity_n_per_m: float
    ground_height_diff_n: float
    friction_sensitivity_n_per_unit: float
    friction_diff_n: float


@dataclass(frozen=True)
class ContactKinematicsContext:
    """Optional geometric and contact kinematics context for force feasibility auditing."""

    ankle_indices: Sequence[int] | None = None
    contact_positions_m: Mapping[str, Array] | None = None
    ground: GroundPlane | None = None
    mu_friction: float = 0.8
    constitutive_forces: Mapping[str, ContactSample] | None = None


class ForceFeasibilityError(ValueError):
    """Raised when contact forces or actuator torques violate physical capacity gates."""


def _check_root_residuals(
    delta_tau_root: Array,
    cfg: ContactFeasibilityConfig,
    failures: list[str],
) -> tuple[float, float]:
    root_force_res = float(np.linalg.norm(delta_tau_root[:3]))
    root_torque_res = float(np.linalg.norm(delta_tau_root[3:]))
    if root_force_res > cfg.force_residual_budget_n:
        failures.append(
            f"Root force residual {root_force_res:.2f} N exceeds budget {cfg.force_residual_budget_n:.2f} N"
        )
    if root_torque_res > cfg.torque_residual_budget_nm:
        failures.append(
            f"Root torque residual {root_torque_res:.2f} Nm exceeds budget {cfg.torque_residual_budget_nm:.2f} Nm"
        )
    return root_force_res, root_torque_res


def _check_grf_limits(
    f_ground: Array,
    body_mass_kg: float,
    cfg: ContactFeasibilityConfig,
    failures: list[str],
) -> tuple[float, float, list[Array]]:
    n_spheres = len(f_ground) // 3
    grf_vectors = [f_ground[s * 3 : (s + 1) * 3] for s in range(n_spheres)]
    grf_norms = (
        [float(np.linalg.norm(v)) for v in grf_vectors] if grf_vectors else [0.0]
    )
    max_grf = float(np.max(grf_norms)) if grf_norms else 0.0
    bw_n = body_mass_kg * 9.81
    grf_bw_ratio = max_grf / bw_n
    if max_grf >= 1e6:
        failures.append(f"Unphysical meganewton load detected: {max_grf:.1e} N")
    elif grf_bw_ratio > cfg.max_normal_force_bw_ratio:
        failures.append(
            f"Ground reaction force {max_grf:.1f} N ({grf_bw_ratio:.2f} BW) exceeds capacity {cfg.max_normal_force_bw_ratio:.1f} BW"
        )
    return max_grf, grf_bw_ratio, grf_vectors


def _check_joint_and_ankle_limits(
    tau_actuated: Array,
    ankle_indices: Sequence[int] | None,
    cfg: ContactFeasibilityConfig,
    failures: list[str],
) -> tuple[float, float]:
    max_joint = float(np.max(np.abs(tau_actuated))) if len(tau_actuated) > 0 else 0.0
    if max_joint > cfg.max_joint_torque_nm:
        failures.append(
            f"Joint torque {max_joint:.1f} Nm exceeds maximum human capacity {cfg.max_joint_torque_nm:.1f} Nm"
        )
    max_ankle = 0.0
    if ankle_indices is not None and len(ankle_indices) > 0:
        ankle_torques = [
            abs(float(tau_actuated[idx]))
            for idx in ankle_indices
            if idx < len(tau_actuated)
        ]
        if ankle_torques:
            max_ankle = float(np.max(ankle_torques))
            if max_ankle >= 1e3:
                failures.append(
                    f"Unphysical kilonewton-metre ankle torque: {max_ankle:.1f} Nm"
                )
            elif max_ankle > cfg.max_ankle_torque_nm:
                failures.append(
                    f"Ankle torque {max_ankle:.1f} Nm exceeds physiological limit {cfg.max_ankle_torque_nm:.1f} Nm"
                )
    return max_joint, max_ankle


def _check_cop_and_friction(
    contact_positions_m: Mapping[str, Array] | None,
    ground: GroundPlane | None,
    grf_vectors: list[Array],
    mu_friction: float,
    cfg: ContactFeasibilityConfig,
    failures: list[str],
) -> tuple[bool, bool]:
    cop_contained = True
    friction_cone_ok = True
    if contact_positions_m is not None and ground is not None:
        sphere_names = list(contact_positions_m.keys())
        contact_forces_dict = {
            name: grf_vectors[idx]
            for idx, name in enumerate(sphere_names[: len(grf_vectors)])
        }
        geo_report = evaluate_support_geometry(
            contact_positions_m,
            contact_forces_dict,
            ground,
            mu_friction=mu_friction,
            tolerance_cop_m=cfg.cop_margin_tolerance_m,
        )
        cop_contained = geo_report.inside_support_polygon
        if not cop_contained and geo_report.total_normal_force_n > 10.0:
            failures.append(
                f"Center of pressure lies outside foot support polygon (margin {geo_report.cop_margin_m:.3f} m)"
            )
        if geo_report.friction_cone_violations:
            friction_cone_ok = False
            viol_str = ", ".join(
                f"{k}: +{v:.1f}N"
                for k, v in geo_report.friction_cone_violations.items()
            )
            failures.append(f"Friction cone violations detected: {viol_str}")
    return cop_contained, friction_cone_ok


def _check_constitutive_compliance(
    constitutive_forces: Mapping[str, ContactSample] | None,
    contact_positions_m: Mapping[str, Array] | None,
    grf_vectors: list[Array],
    cfg: ContactFeasibilityConfig,
    failures: list[str],
) -> float:
    constitutive_res = 0.0
    if constitutive_forces is not None and contact_positions_m is not None:
        sphere_names = list(contact_positions_m.keys())
        discrepancies: list[float] = []
        for s_idx, name in enumerate(sphere_names[: len(grf_vectors)]):
            allocated_f = grf_vectors[s_idx]
            if name in constitutive_forces:
                sample = constitutive_forces[name]
                sample_f = sample.normal_force_n + sample.friction_force_n
                discrepancies.append(float(np.linalg.norm(allocated_f - sample_f)))
                if sample.penetration_m <= 0.0 and np.linalg.norm(allocated_f) > 5.0:
                    failures.append(
                        f"Unsupported contact force ({np.linalg.norm(allocated_f):.1f} N) assumed on {name} while in flight"
                    )
        if discrepancies:
            constitutive_res = float(np.max(discrepancies))
            if constitutive_res > cfg.constitutive_discrepancy_tolerance_n:
                failures.append(
                    f"Allocated forces diverge from constitutive contact law by {constitutive_res:.1f} N (tolerance {cfg.constitutive_discrepancy_tolerance_n:.1f} N)"
                )
    return constitutive_res


def audit_contact_force_feasibility(
    tau_actuated: Array,
    f_ground: Array,
    delta_tau_root: Array,
    *,
    body_mass_kg: float,
    context: ContactKinematicsContext | None = None,
    config: ContactFeasibilityConfig | None = None,
    raise_on_failure: bool = False,
    **kwargs: Any,
) -> FeasibilityAuditResult:
    """Audit contact forces and joint torques against physical and physiological feasibility.

    Args:
        tau_actuated: Generalized actuated coordinate torques (N·m).
        f_ground: Ground reaction forces (flat array of 3-vectors per contact sphere).
        delta_tau_root: Floating base residual wrench (6 elements: [fx, fy, fz, tx, ty, tz]).
        body_mass_kg: Subject total body mass in kilograms.
        context: Optional ContactKinematicsContext bundling geometric and constitutive context.
        config: Feasibility configuration and limits.
        raise_on_failure: If True, raises ForceFeasibilityError when infeasible.
        **kwargs: Optional fallback keyword arguments (ankle_indices, contact_positions_m,
            ground, mu_friction, constitutive_forces).

    Returns:
        FeasibilityAuditResult detailing compliance, residuals, and reasons for failure.
    """
    cfg = config or ContactFeasibilityConfig()
    require(body_mass_kg > 0.0, "body_mass_kg must be positive", body_mass_kg)
    require(len(delta_tau_root) == 6, "delta_tau_root must have length 6")

    ankle_indices: Sequence[int] | None = kwargs.get(
        "ankle_indices", context.ankle_indices if context else None
    )
    contact_positions_m: Mapping[str, Array] | None = kwargs.get(
        "contact_positions_m", context.contact_positions_m if context else None
    )
    ground: GroundPlane | None = kwargs.get(
        "ground", context.ground if context else None
    )
    mu_friction: float = float(
        kwargs.get("mu_friction", context.mu_friction if context else 0.8)
    )
    constitutive_forces: Mapping[str, ContactSample] | None = kwargs.get(
        "constitutive_forces", context.constitutive_forces if context else None
    )

    failures: list[str] = []
    root_force_res, root_torque_res = _check_root_residuals(
        delta_tau_root, cfg, failures
    )
    max_grf, grf_bw_ratio, grf_vectors = _check_grf_limits(
        f_ground, body_mass_kg, cfg, failures
    )
    max_joint, max_ankle = _check_joint_and_ankle_limits(
        tau_actuated, ankle_indices, cfg, failures
    )
    cop_contained, friction_cone_ok = _check_cop_and_friction(
        contact_positions_m, ground, grf_vectors, mu_friction, cfg, failures
    )
    constitutive_res = _check_constitutive_compliance(
        constitutive_forces, contact_positions_m, grf_vectors, cfg, failures
    )

    is_feasible = len(failures) == 0
    if not is_feasible and raise_on_failure:
        raise ForceFeasibilityError("; ".join(failures))

    return FeasibilityAuditResult(
        is_feasible=is_feasible,
        root_force_residual_n=root_force_res,
        root_torque_residual_nm=root_torque_res,
        max_grf_n=max_grf,
        max_grf_bw_ratio=grf_bw_ratio,
        max_ankle_torque_nm=max_ankle,
        max_joint_torque_nm=max_joint,
        cop_contained=cop_contained,
        friction_cone_satisfied=friction_cone_ok,
        constitutive_residual_n=constitutive_res,
        failure_reasons=tuple(failures),
    )


def compute_contact_sensitivity(
    nominal_solve_fn: Callable[..., tuple[Array, Array, Array]],
    *,
    base_mass_kg: float,
    base_ground_height_m: float,
    base_friction: float,
    mass_perturbation_ratio: float = 0.05,
    height_perturbation_m: float = 0.005,
    friction_perturbation: float = 0.1,
    offset_perturbation_m: float = 0.005,
) -> ContactSensitivityReport:
    """Compute sensitivity of allocated ground forces to physical parameter variations.

    Args:
        nominal_solve_fn: Function taking kwargs (mass_kg, ground_height_m, mu_friction, marker_offset_m)
                          and returning (tau_actuated, f_ground, delta_tau_root).
        base_mass_kg: Baseline subject mass.
        base_ground_height_m: Baseline ground height.
        base_friction: Baseline friction coefficient.
        mass_perturbation_ratio: Relative mass perturbation (default 5%).
        height_perturbation_m: Ground height perturbation in meters (default 5 mm).
        friction_perturbation: Friction coefficient delta (default 0.1).
        offset_perturbation_m: Marker/foot position offset perturbation (default 5 mm).

    Returns:
        ContactSensitivityReport with derivative estimates per physical parameter.
    """
    # 1. Base solve
    _, f_base, _ = nominal_solve_fn(
        mass_kg=base_mass_kg,
        ground_height_m=base_ground_height_m,
        mu_friction=base_friction,
        marker_offset_m=0.0,
    )
    total_f_base = float(np.sum(np.linalg.norm(f_base.reshape(-1, 3), axis=1)))

    # 2. Mass sensitivity
    dm = base_mass_kg * mass_perturbation_ratio
    _, f_m_plus, _ = nominal_solve_fn(
        mass_kg=base_mass_kg + dm,
        ground_height_m=base_ground_height_m,
        mu_friction=base_friction,
        marker_offset_m=0.0,
    )
    total_f_m_plus = float(np.sum(np.linalg.norm(f_m_plus.reshape(-1, 3), axis=1)))
    diff_m = abs(total_f_m_plus - total_f_base)
    sens_m = diff_m / dm if dm > 0 else 0.0

    # 3. Ground height sensitivity
    dh = height_perturbation_m
    _, f_h_plus, _ = nominal_solve_fn(
        mass_kg=base_mass_kg,
        ground_height_m=base_ground_height_m + dh,
        mu_friction=base_friction,
        marker_offset_m=0.0,
    )
    total_f_h_plus = float(np.sum(np.linalg.norm(f_h_plus.reshape(-1, 3), axis=1)))
    diff_h = abs(total_f_h_plus - total_f_base)
    sens_h = diff_h / dh if dh > 0 else 0.0

    # 4. Friction sensitivity
    dmu = friction_perturbation
    _, f_mu_plus, _ = nominal_solve_fn(
        mass_kg=base_mass_kg,
        ground_height_m=base_ground_height_m,
        mu_friction=base_friction + dmu,
        marker_offset_m=0.0,
    )
    total_f_mu_plus = float(np.sum(np.linalg.norm(f_mu_plus.reshape(-1, 3), axis=1)))
    diff_mu = abs(total_f_mu_plus - total_f_base)
    sens_mu = diff_mu / dmu if dmu > 0 else 0.0

    # 5. Marker / foot offset sensitivity
    do = offset_perturbation_m
    _, f_off_plus, _ = nominal_solve_fn(
        mass_kg=base_mass_kg,
        ground_height_m=base_ground_height_m,
        mu_friction=base_friction,
        marker_offset_m=do,
    )
    total_f_off_plus = float(np.sum(np.linalg.norm(f_off_plus.reshape(-1, 3), axis=1)))
    diff_off = abs(total_f_off_plus - total_f_base)
    sens_off = diff_off / do if do > 0 else 0.0

    return ContactSensitivityReport(
        base_total_force_n=total_f_base,
        mass_sensitivity_n_per_kg=sens_m,
        mass_perturbation_diff_n=diff_m,
        marker_offset_sensitivity_n_per_m=sens_off,
        marker_offset_diff_n=diff_off,
        ground_height_sensitivity_n_per_m=sens_h,
        ground_height_diff_n=diff_h,
        friction_sensitivity_n_per_unit=sens_mu,
        friction_diff_n=diff_mu,
    )

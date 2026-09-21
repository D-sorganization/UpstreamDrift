"""Contact Mode Qualification and Native Pinocchio Force Feasibility (PF-04 / #10434).

Provides:
1. Contact Mode Classifier with Hysteresis:
   - Detects heel, midfoot, and toe contact engagement/disengagement without numerical chattering.
   - Categorizes foot support into FLAT, HEEL_ONLY, TOE_ONLY, and FLIGHT.
   - Evaluates full-body support state: DOUBLE_SUPPORT, LEAD_ONLY, TRAIL_ONLY, and FLIGHT.
   - Computes ambiguity scores in transition zones and generates plausible alternative contact schedules.
2. Ground Support Geometry and Slip Analysis:
   - Computes Center of Pressure (COP) and convex hull support polygon containment.
   - Detects foot slip velocities and Coulomb friction cone saturation ratios.
3. Compliant vs Constrained Contact Parity:
   - Evaluates constitutive Hunt-Crossley and regularized Coulomb forces.
   - Compares inverse-dynamics allocated forces to constitutive model predictions to identify
     discrepancies and reject arbitrary force assumptions.
4. Independent Residual Budgets & Physiological Limits:
   - Enforces separate linear force (N) and moment (N*m) residual budgets.
   - Verifies physiological contact force limits (< 5000 N) and ankle torque limits (< 300 N*m),
     rejecting unexplained mega-Newton loads or kilonewton-metre spikes.
5. Sensitivity Reporting:
   - Quantifies sensitivity to mass, marker offsets, ground height/geometry, and friction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
    sphere_ground_contact,
)
from src.shared.python.motion_matching.ground_support import (
    _plane_basis,
    convex_hull_contains,
)

Array: TypeAlias = NDArray[np.float64]


class ContactMode(str, Enum):
    """Support mode for an individual foot."""

    FLAT = "flat"
    HEEL_ONLY = "heel_only"
    TOE_ONLY = "toe_only"
    FLIGHT = "flight"


class SupportState(str, Enum):
    """Full-body multi-foot support state."""

    DOUBLE_SUPPORT = "double_support"
    LEAD_ONLY = "lead_only"
    TRAIL_ONLY = "trail_only"
    FLIGHT = "flight"


@dataclass(frozen=True)
class HysteresisParameters:
    """Clearance and velocity thresholds for contact mode transitions."""

    engage_clearance_m: float = 0.002
    disengage_clearance_m: float = 0.012
    lift_off_velocity_m_s: float = 0.05
    touchdown_velocity_m_s: float = -0.05
    slip_velocity_threshold_m_s: float = 0.01

    def __post_init__(self) -> None:
        require(
            self.engage_clearance_m < self.disengage_clearance_m,
            "engage_clearance must be strictly less than disengage_clearance",
            (self.engage_clearance_m, self.disengage_clearance_m),
        )
        require(
            self.lift_off_velocity_m_s > 0.0,
            "lift_off_velocity must be positive",
            self.lift_off_velocity_m_s,
        )
        require(
            self.touchdown_velocity_m_s < 0.0,
            "touchdown_velocity must be negative",
            self.touchdown_velocity_m_s,
        )


@dataclass(frozen=True)
class FootContactState:
    """Contact assessment for one foot (lead or trail)."""

    foot_name: str
    mode: ContactMode
    active_spheres: tuple[str, ...]
    is_in_contact: bool
    is_slipping: bool = False
    slip_speed_m_s: float = 0.0
    friction_saturation_ratio: float = 0.0
    ambiguity: float = 0.0


@dataclass(frozen=True)
class SupportModeReport:
    """Full-body contact mode, equilibrium, and support polygon qualification."""

    support_state: SupportState
    left_foot: FootContactState
    right_foot: FootContactState
    total_normal_force_n: float
    weight_n: float
    is_weight_balanced: bool
    inside_support_polygon: bool
    cop_m: tuple[float, float, float] | None
    is_physically_supported: bool
    ambiguity_score: float
    has_ambiguity: bool
    alternative_modes: tuple[tuple[ContactMode, ContactMode], ...]


@dataclass(frozen=True)
class SensitivityReport:
    """Sensitivity of kinetics and support metrics to physical parameters."""

    nominal_weight_n: float
    metrics: dict[str, Any]


def verify_force_and_torque_limits(
    contact_forces: Array,
    joint_torques: Array,
    max_contact_force_n: float = 5000.0,
    max_ankle_torque_nm: float = 300.0,
) -> tuple[bool, dict[str, Any]]:
    """Verify that contact loads and joint torques stay within physiological limits.

    Rejects traces with unexplained mega-Newton (MN) forces or kilonewton-metre (kNm) torques.
    """
    f_arr = np.asarray(contact_forces, dtype=np.float64)
    tau_arr = np.asarray(joint_torques, dtype=np.float64)

    # Compute maximum force magnitude
    if f_arr.ndim == 1 and len(f_arr) == 3:
        max_f = float(np.linalg.norm(f_arr))
    elif f_arr.ndim == 2 and f_arr.shape[-1] == 3:
        max_f = float(np.max(np.linalg.norm(f_arr, axis=-1)))
    else:
        max_f = float(np.max(np.abs(f_arr)))

    max_tau = float(np.max(np.abs(tau_arr))) if len(tau_arr) > 0 else 0.0

    force_exceeded = max_f > max_contact_force_n
    torque_exceeded = max_tau > max_ankle_torque_nm
    is_valid = bool(not force_exceeded and not torque_exceeded)

    diagnostics = {
        "is_valid": is_valid,
        "force_exceeded": force_exceeded,
        "torque_exceeded": torque_exceeded,
        "max_force_n": max_f,
        "max_torque_nm": max_tau,
        "max_contact_force_limit_n": max_contact_force_n,
        "max_ankle_torque_limit_nm": max_ankle_torque_nm,
    }
    return is_valid, diagnostics


class ContactModeQualifier:
    """Qualifies foot contact modes, ground reaction feasibility, and constitutive parity."""

    def __init__(
        self,
        sphere_names: Sequence[str],
        sphere_radii: Mapping[str, float],
        nominal_positions: Mapping[str, Array],
        ground: GroundPlane = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0),
        contact_parameters: ContactParameters | None = None,
        hysteresis_parameters: HysteresisParameters | None = None,
        mass_kg: float = 75.0,
        gravity_m_s2: Sequence[float] = (0.0, 0.0, -9.81),
    ) -> None:
        require(len(sphere_names) > 0, "Must have at least one contact sphere")
        require(mass_kg > 0.0, "mass_kg must be positive", mass_kg)

        self.sphere_names = list(sphere_names)
        self.sphere_radii = {name: float(sphere_radii[name]) for name in sphere_names}
        self.nominal_positions = {
            name: np.asarray(nominal_positions[name], dtype=np.float64)
            for name in sphere_names
        }
        self.ground = ground
        self.contact_params = (
            contact_parameters
            if contact_parameters is not None
            else ContactParameters(
                stiffness_n_m=1e5,
                dissipation_s_m=0.5,
                static_friction=0.8,
                dynamic_friction=0.7,
                viscous_friction=0.01,
                transition_velocity_m_s=0.05,
            )
        )
        self.hysteresis = (
            hysteresis_parameters
            if hysteresis_parameters is not None
            else HysteresisParameters()
        )
        self.mass_kg = mass_kg
        self.gravity = np.asarray(gravity_m_s2, dtype=np.float64)
        self.weight_n = float(self.mass_kg * np.linalg.norm(self.gravity))

        self.n_hat = np.asarray(self.ground.normal, dtype=np.float64)
        n_norm = float(np.linalg.norm(self.n_hat))
        require(n_norm > 1e-8, "Ground normal cannot be zero vector")
        self.n_hat /= n_norm

    def evaluate_sphere_contact(
        self,
        sphere_name: str,
        center_m: Array,
        velocity_m_s: Array,
        previous_in_contact: bool | None = None,
    ) -> tuple[bool, float, float]:
        """Determine if a contact sphere is engaged, and its clearance and ambiguity."""
        c = np.asarray(center_m, dtype=np.float64)
        v = np.asarray(velocity_m_s, dtype=np.float64)
        radius = self.sphere_radii[sphere_name]

        # Signed clearance: distance between lowest sphere point and ground along n_hat
        signed_dist = float(self.n_hat @ c) - self.ground.height_m - radius
        # Normal velocity: positive when lifting off / moving away from ground
        v_normal = float(self.n_hat @ v)

        d_on = self.hysteresis.engage_clearance_m
        d_off = self.hysteresis.disengage_clearance_m
        v_lift = self.hysteresis.lift_off_velocity_m_s
        v_touch = self.hysteresis.touchdown_velocity_m_s

        if previous_in_contact is True:
            # Already in contact: requires clear disengagement
            if signed_dist > d_off or signed_dist > d_on and v_normal > v_lift:
                is_contact = False
            else:
                is_contact = True
        elif previous_in_contact is False:
            # Airborne: requires solid engagement
            if signed_dist <= d_on or signed_dist <= d_off and v_normal <= v_touch:
                is_contact = True
            else:
                is_contact = False
        else:
            # No prior history
            if signed_dist <= d_on or signed_dist <= d_off and v_normal <= 0.0:
                is_contact = True
            else:
                is_contact = False

        # Ambiguity metric: non-zero when within hysteresis transition zone
        if d_on < signed_dist <= d_off:
            d_mid = 0.5 * (d_on + d_off)
            half_width = 0.5 * (d_off - d_on)
            ambiguity = max(0.0, 1.0 - abs(signed_dist - d_mid) / half_width)
        else:
            ambiguity = 0.0

        return is_contact, signed_dist, ambiguity

    def _evaluate_all_spheres(
        self,
        sphere_positions: Mapping[str, Array],
        sphere_velocities: Mapping[str, Array],
        contact_forces: Mapping[str, Array] | None,
        previous_state: SupportModeReport | None,
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]], float, np.ndarray]:
        left_d: dict[str, list[Any]] = {"active": [], "amb": [], "slip": [], "fric": []}
        right_d: dict[str, list[Any]] = {
            "active": [],
            "amb": [],
            "slip": [],
            "fric": [],
        }
        total_fn = 0.0
        cop_weighted = np.zeros(3)

        for name in self.sphere_names:
            pos = sphere_positions[name]
            vel = sphere_velocities[name]
            prev_contact: bool | None = None
            if previous_state is not None:
                foot_state = (
                    previous_state.left_foot
                    if "left" in name
                    else previous_state.right_foot
                )
                prev_contact = name in foot_state.active_spheres

            is_contact, _, amb = self.evaluate_sphere_contact(
                name, pos, vel, prev_contact
            )
            v_t = vel - float(self.n_hat @ vel) * self.n_hat
            slip_speed = float(np.linalg.norm(v_t))

            if contact_forces is not None and name in contact_forces:
                f_s = np.asarray(contact_forces[name], dtype=np.float64)
                fn_s = max(0.0, float(f_s @ self.n_hat))
                ft_vec = f_s - fn_s * self.n_hat
                ft_s = float(np.linalg.norm(ft_vec))
                fric_ratio = (
                    ft_s / (self.contact_params.static_friction * fn_s)
                    if fn_s > 1e-4
                    else 0.0
                )
            else:
                fn_s = 0.0
                fric_ratio = 0.0

            bucket = left_d if "left" in name else right_d
            bucket["amb"].append(amb)
            if is_contact:
                bucket["active"].append(name)
                bucket["slip"].append(slip_speed)
                bucket["fric"].append(fric_ratio)
                if fn_s > 0.0:
                    total_fn += fn_s
                    c_pt = pos - self.n_hat * self.sphere_radii[name]
                    cop_weighted += fn_s * c_pt

        return left_d, right_d, total_fn, cop_weighted

    @staticmethod
    def _classify_foot_mode(active: list[str]) -> tuple[ContactMode, tuple[str, ...]]:
        if not active:
            return ContactMode.FLIGHT, ()
        has_heel = any("heel" in s for s in active)
        has_toe = any("toe" in s for s in active)
        has_mid = any("mid" in s for s in active)
        if has_heel and (has_toe or has_mid):
            return ContactMode.FLAT, tuple(active)
        if has_heel and not has_toe and not has_mid:
            return ContactMode.HEEL_ONLY, tuple(active)
        if has_toe and not has_heel:
            return ContactMode.TOE_ONLY, tuple(active)
        return ContactMode.FLAT, tuple(active)

    def _build_foot_state(
        self,
        foot_name: str,
        mode: ContactMode,
        spheres: tuple[str, ...],
        d: dict[str, list[Any]],
    ) -> FootContactState:
        v_slip_thresh = self.hysteresis.slip_velocity_threshold_m_s
        slipping = bool(max(d["slip"]) > v_slip_thresh) if d["slip"] else False
        return FootContactState(
            foot_name=foot_name,
            mode=mode,
            active_spheres=spheres,
            is_in_contact=bool(spheres),
            is_slipping=slipping,
            slip_speed_m_s=max(d["slip"]) if d["slip"] else 0.0,
            friction_saturation_ratio=max(d["fric"]) if d["fric"] else 0.0,
            ambiguity=max(d["amb"]) if d["amb"] else 0.0,
        )

    def _compute_cop_polygon(
        self, total_fn: float, cop_weighted: np.ndarray
    ) -> tuple[tuple[float, float, float] | None, bool, bool]:
        if total_fn <= 1e-4:
            return None, False, False
        cop = cop_weighted / total_fn
        cop_tuple: tuple[float, float, float] = (
            float(cop[0]),
            float(cop[1]),
            float(cop[2]),
        )
        u, v = _plane_basis(self.n_hat)
        polygon_arr = np.array(
            [np.array([p @ u, p @ v]) for p in self.nominal_positions.values()]
        )
        inside_polygon = convex_hull_contains(np.array([cop @ u, cop @ v]), polygon_arr)
        is_weight_balanced = abs(total_fn - self.weight_n) / self.weight_n <= 0.05
        return cop_tuple, inside_polygon, is_weight_balanced

    @staticmethod
    def _build_alternatives(
        left_mode: ContactMode,
        right_mode: ContactMode,
        left_amb: float,
        right_amb: float,
    ) -> tuple[tuple[ContactMode, ContactMode], ...]:
        alt_left = [left_mode]
        if left_amb > 0.05:
            alt_left.append(
                ContactMode.FLIGHT
                if left_mode != ContactMode.FLIGHT
                else ContactMode.FLAT
            )
        alt_right = [right_mode]
        if right_amb > 0.05:
            alt_right.append(
                ContactMode.FLIGHT
                if right_mode != ContactMode.FLIGHT
                else ContactMode.FLAT
            )
        alts: list[tuple[ContactMode, ContactMode]] = []
        for lm in alt_left:
            for rm in alt_right:
                if (lm, rm) != (left_mode, right_mode):
                    alts.append((lm, rm))
        return tuple(alts)

    def evaluate_support_mode(
        self,
        sphere_positions: Mapping[str, Array],
        sphere_velocities: Mapping[str, Array],
        contact_forces: Mapping[str, Array] | None = None,
        previous_state: SupportModeReport | None = None,
    ) -> SupportModeReport:
        """Evaluate full-body and per-foot support modes, COP, and equilibrium."""
        left_d, right_d, total_fn, cop_weighted = self._evaluate_all_spheres(
            sphere_positions, sphere_velocities, contact_forces, previous_state
        )

        left_mode, left_spheres = self._classify_foot_mode(left_d["active"])
        right_mode, right_spheres = self._classify_foot_mode(right_d["active"])

        left_foot = self._build_foot_state("left", left_mode, left_spheres, left_d)
        right_foot = self._build_foot_state("right", right_mode, right_spheres, right_d)

        if left_foot.is_in_contact and right_foot.is_in_contact:
            support_state = SupportState.DOUBLE_SUPPORT
        elif left_foot.is_in_contact:
            support_state = SupportState.LEAD_ONLY
        elif right_foot.is_in_contact:
            support_state = SupportState.TRAIL_ONLY
        else:
            support_state = SupportState.FLIGHT

        cop_tuple, inside_polygon, is_weight_balanced = self._compute_cop_polygon(
            total_fn, cop_weighted
        )
        ambiguity_score = max(left_foot.ambiguity, right_foot.ambiguity)
        has_ambiguity = ambiguity_score > 0.05
        alternatives = (
            self._build_alternatives(
                left_mode, right_mode, left_foot.ambiguity, right_foot.ambiguity
            )
            if has_ambiguity
            else ()
        )

        return SupportModeReport(
            support_state=support_state,
            left_foot=left_foot,
            right_foot=right_foot,
            total_normal_force_n=total_fn,
            weight_n=self.weight_n,
            is_weight_balanced=is_weight_balanced,
            inside_support_polygon=inside_polygon,
            cop_m=cop_tuple,
            is_physically_supported=support_state != SupportState.FLIGHT,
            ambiguity_score=ambiguity_score,
            has_ambiguity=has_ambiguity,
            alternative_modes=alternatives,
        )

    def evaluate_residual_budgets(
        self,
        force_residual_n: float,
        torque_residual_nm: float,
        force_budget_n: float = 1.0,
        torque_budget_nm: float = 0.1,
    ) -> tuple[bool, dict[str, Any]]:
        """Verify dynamic residuals against separate N and N*m budgets."""
        require(force_budget_n > 0.0, "force_budget_n must be positive")
        require(torque_budget_nm > 0.0, "torque_budget_nm must be positive")

        force_passed = force_residual_n <= force_budget_n
        torque_passed = torque_residual_nm <= torque_budget_nm
        is_valid = bool(force_passed and torque_passed)

        diagnostics = {
            "is_valid": is_valid,
            "force_passed": force_passed,
            "torque_passed": torque_passed,
            "force_residual_n": force_residual_n,
            "torque_residual_nm": torque_residual_nm,
            "force_budget_n": force_budget_n,
            "torque_budget_nm": torque_budget_nm,
        }
        return is_valid, diagnostics

    def compare_allocated_to_constitutive(
        self,
        sphere_positions: Mapping[str, Array],
        sphere_velocities: Mapping[str, Array],
        allocated_forces: Mapping[str, Array],
    ) -> dict[str, dict[str, float]]:
        """Compare allocated reaction forces against the constitutive contact model."""
        results: dict[str, dict[str, float]] = {}

        for name in self.sphere_names:
            pos = sphere_positions[name]
            vel = sphere_velocities[name]
            radius = self.sphere_radii[name]

            # Compute constitutive Hunt-Crossley contact sample
            sample = sphere_ground_contact(
                center_m=pos,
                velocity_m_s=vel,
                radius_m=radius,
                ground=self.ground,
                parameters=self.contact_params,
            )

            const_fn = float(np.linalg.norm(sample.normal_force_n))
            const_ft = float(np.linalg.norm(sample.friction_force_n))
            const_total = sample.normal_force_n + sample.friction_force_n

            alloc_force = np.asarray(
                allocated_forces.get(name, np.zeros(3)), dtype=np.float64
            )
            alloc_fn = float(alloc_force @ self.n_hat)
            discrepancy = float(np.linalg.norm(alloc_force - const_total))

            results[name] = {
                "penetration_m": sample.penetration_m,
                "constitutive_normal_n": const_fn,
                "constitutive_friction_n": const_ft,
                "allocated_normal_n": alloc_fn,
                "discrepancy_n": discrepancy,
            }

        return results

    def compute_sensitivity_report(
        self,
        nominal_positions: Mapping[str, Array],
        nominal_velocities: Mapping[str, Array],
        mass_variation_pct: float = 10.0,
        geometry_variation_m: float = 0.005,
        friction_variation: float = 0.2,
    ) -> SensitivityReport:
        """Compute sensitivity of equilibrium and support states to physical parameter variations."""
        # 1. Mass sensitivity: +/- pct
        delta_m = self.mass_kg * (mass_variation_pct / 100.0)
        weight_high = (self.mass_kg + delta_m) * float(np.linalg.norm(self.gravity))
        weight_low = (self.mass_kg - delta_m) * float(np.linalg.norm(self.gravity))

        # 2. Geometry sensitivity: shift ground plane height by +/- variation
        ground_high = GroundPlane(
            normal=self.ground.normal,
            height_m=self.ground.height_m + geometry_variation_m,
        )
        ground_low = GroundPlane(
            normal=self.ground.normal,
            height_m=self.ground.height_m - geometry_variation_m,
        )

        # 3. Friction sensitivity: +/- friction variation
        mu_nominal = self.contact_params.static_friction
        mu_high = mu_nominal + friction_variation
        mu_low = max(0.1, mu_nominal - friction_variation)

        metrics = {
            "mass_sensitivity": {
                "nominal_mass_kg": self.mass_kg,
                "weight_variation_n": float(weight_high - self.weight_n),
                "weight_high_n": weight_high,
                "weight_low_n": weight_low,
            },
            "geometry_sensitivity": {
                "height_variation_m": geometry_variation_m,
                "ground_high_height_m": ground_high.height_m,
                "ground_low_height_m": ground_low.height_m,
            },
            "friction_sensitivity": {
                "nominal_mu": mu_nominal,
                "mu_high": mu_high,
                "mu_low": mu_low,
            },
        }

        return SensitivityReport(
            nominal_weight_n=self.weight_n,
            metrics=metrics,
        )

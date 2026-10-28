from __future__ import annotations

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.physics._contact_types import (
    ContactPoint,
    ContactState,
    GripContactState,
    GripParameters,
)
from src.shared.python.physics._friction_laws import (
    _magnitude,
    classify_contact_state,
    decompose_contact_force,
)
from src.shared.python.physics._grip_forces import compute_grip_pressure_centre

logger = get_logger(__name__)


class GripContactModel:
    def __init__(self, params: GripParameters | None = None) -> None:
        self.params = params or GripParameters()
        self.current_state: GripContactState | None = None
        self.contact_history: list[GripContactState] = []

    def update_from_mujoco(
        self,
        contact_positions: np.ndarray,
        contact_normals: np.ndarray,
        contact_forces: np.ndarray,
        contact_velocities: np.ndarray,
        body_names: list[str],
        timestamp: float,
    ) -> GripContactState:
        if contact_positions is None:
            raise ValueError("contact_positions must be provided")
        n_contacts = len(contact_positions)

        contacts: list[ContactPoint] = []

        for i in range(n_contacts):
            normal_force, tangent_force = decompose_contact_force(
                contact_forces[i], contact_normals[i]
            )

            vel = contact_velocities[i]
            vel_normal = np.dot(vel, contact_normals[i]) * contact_normals[i]
            slip_velocity = vel - vel_normal

            state = classify_contact_state(
                normal_force, tangent_force, slip_velocity, self.params
            )

            contacts.append(
                ContactPoint(
                    position=contact_positions[i].copy(),
                    normal=contact_normals[i].copy(),
                    normal_force=normal_force,
                    tangent_force=tangent_force.copy(),
                    slip_velocity=slip_velocity.copy(),
                    state=state,
                    body_name=body_names[i] if i < len(body_names) else "",
                    contact_id=i,
                )
            )

        total_normal = sum(c.normal_force for c in contacts if c.normal_force > 0)
        total_tangent = sum((c.tangent_force for c in contacts), np.zeros(3))
        num_slipping = sum(1 for c in contacts if c.state == ContactState.SLIPPING)
        num_sticking = sum(1 for c in contacts if c.state == ContactState.STICKING)
        # No loaded contact: the pressure centre is unavailable (NaN), not the origin.
        cop = compute_grip_pressure_centre(contacts)
        if cop is None:
            cop = np.full(3, np.nan)

        self.current_state = GripContactState(
            contacts=contacts,
            total_normal_force=total_normal,
            total_tangent_force=total_tangent,
            num_slipping=num_slipping,
            num_sticking=num_sticking,
            center_of_pressure=cop,
            timestamp=timestamp,
        )

        self.contact_history.append(self.current_state)
        return self.current_state

    def check_static_equilibrium(
        self,
        club_weight: float,
        gravity_direction: np.ndarray = np.array([0.0, 0.0, -1.0]),
        tolerance: float = 0.01,
    ) -> dict[str, bool | float]:
        """Check whether contact forces satisfy static equilibrium under gravity.

        Static equilibrium (Newton's 1st law, UD-30) requires:
        1. Vertical support force balances the downward gravitational load
           within tolerance: |support_ratio - 1.0| <= tolerance. Over-supporting
           (support_ratio > 1.0 + tolerance) produces upward acceleration and is
           not in equilibrium.
        2. Transverse (orthogonal to gravity) contact forces are negligible:
           transverse_magnitude <= tolerance * club_weight.

        Args:
            club_weight: Gravitational force magnitude of the club in Newtons (> 0).
            gravity_direction: Direction vector of gravity (default [0, 0, -1]).
            tolerance: Relative balance tolerance (default 0.01 = 1%).

        Returns:
            Dict containing equilibrium verdict, support ratio, and force diagnostics.
        """
        if club_weight is None:
            raise ValueError("club_weight must be provided")
        if not np.isfinite(club_weight) or club_weight <= 0.0:
            raise ValueError(
                f"club_weight must be positive and finite; got {club_weight!r}"
            )
        if tolerance <= 0.0 or not np.isfinite(tolerance):
            raise ValueError(
                f"tolerance must be positive and finite; got {tolerance!r}"
            )

        g_dir = np.asarray(gravity_direction, dtype=np.float64)
        g_norm = float(np.linalg.norm(g_dir))
        if g_norm <= 0.0 or not np.isfinite(g_norm):
            raise ValueError("gravity_direction must be non-zero and finite")
        g_unit = g_dir / g_norm
        support_unit = -g_unit

        if self.current_state is None or not self.current_state.contacts:
            return {
                "equilibrium": False,
                "support_ratio": 0.0,
                "total_normal_force": 0.0,
                "required_force": float(club_weight),
                "transverse_force": 0.0,
                "net_force": float(club_weight),
            }

        total_contact_force = np.zeros(3, dtype=np.float64)
        for c in self.current_state.contacts:
            total_contact_force += c.normal_force * c.normal + c.tangent_force

        support_magnitude = float(np.dot(total_contact_force, support_unit))
        transverse_force_vec = total_contact_force - support_magnitude * support_unit
        transverse_magnitude = float(np.linalg.norm(transverse_force_vec))

        support_ratio = support_magnitude / club_weight
        net_force_vec = total_contact_force + club_weight * g_unit
        net_force_magnitude = float(np.linalg.norm(net_force_vec))

        vertical_balanced = abs(support_ratio - 1.0) <= tolerance
        transverse_negligible = transverse_magnitude <= tolerance * club_weight
        equilibrium = bool(vertical_balanced and transverse_negligible)

        return {
            "equilibrium": equilibrium,
            "support_ratio": float(support_ratio),
            "total_normal_force": float(self.current_state.total_normal_force),
            "required_force": float(club_weight),
            "transverse_force": float(transverse_magnitude),
            "net_force": float(net_force_magnitude),
        }

    def check_slip_margin(self) -> dict[str, float]:
        if self.current_state is None or not self.current_state.contacts:
            return {"min_margin": 0.0, "mean_margin": 0.0, "any_slipping": True}

        margins = [
            (max_tangent - _magnitude(c.tangent_force))
            / max_tangent  # ⚡ Bolt: _magnitude (float-promoted dot product) is faster than np.linalg.norm and safe against integer overflow (#10845)
            for c in self.current_state.contacts
            if c.normal_force > 0
            and (max_tangent := self.params.static_friction * c.normal_force) > 0
        ]

        if not margins:
            return {"min_margin": 0.0, "mean_margin": 0.0, "any_slipping": True}

        return {
            "min_margin": float(np.min(margins)),  # type: ignore[arg-type]
            "mean_margin": float(np.mean(margins)),
            "any_slipping": any(m < 0 for m in margins),
        }

    def get_pressure_distribution(self) -> np.ndarray:
        if self.current_state is None:
            return np.array([])

        n_contacts = len(self.current_state.contacts)
        if n_contacts == 0:
            return np.array([])

        area_per_contact = self.params.hand_contact_area / n_contacts

        pressures = [
            c.normal_force / area_per_contact if area_per_contact > 0 else 0
            for c in self.current_state.contacts
        ]

        return np.array(pressures)

    def reset(self) -> None:
        self.current_state = None
        self.contact_history.clear()

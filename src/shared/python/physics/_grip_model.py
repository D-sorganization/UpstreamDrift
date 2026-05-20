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
    classify_contact_state,
    decompose_contact_force,
)
from src.shared.python.physics._grip_forces import compute_center_of_pressure

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
        cop = compute_center_of_pressure(contacts)

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
    ) -> dict[str, bool | float]:
        if club_weight is None:
            raise ValueError("club_weight must be provided")
        if self.current_state is None:
            return {"equilibrium": False, "support_ratio": 0.0}

        total_contact_force = np.zeros(3)
        for c in self.current_state.contacts:
            total_contact_force += c.normal_force * c.normal + c.tangent_force

        support_magnitude = np.dot(total_contact_force, -gravity_direction)
        support_ratio = support_magnitude / club_weight if club_weight > 0 else 0

        equilibrium = support_ratio >= 0.99

        return {
            "equilibrium": equilibrium,
            "support_ratio": float(support_ratio),
            "total_normal_force": self.current_state.total_normal_force,
            "required_force": float(club_weight),
        }

    def check_slip_margin(self) -> dict[str, float]:
        if self.current_state is None or not self.current_state.contacts:
            return {"min_margin": 0.0, "mean_margin": 0.0, "any_slipping": True}

        margins = [
            (max_tangent - np.linalg.norm(c.tangent_force)) / max_tangent
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

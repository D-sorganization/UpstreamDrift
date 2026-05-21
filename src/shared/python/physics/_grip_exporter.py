from __future__ import annotations

import numpy as np

from src.shared.python.physics._contact_types import GripContactTimestep
from src.shared.python.physics._grip_model import GripContactModel


def create_mujoco_grip_contacts(
    grip_body_name: str = "club_grip",
    hand_body_names: list[str] | None = None,
    friction: tuple[float, float, float] = (0.8, 0.6, 0.001),
) -> dict:
    if grip_body_name is None:
        raise ValueError("grip_body_name must be provided")
    if hand_body_names is None:
        hand_body_names = ["left_hand", "right_hand"]

    contact_pairs = []
    for hand in hand_body_names:
        contact_pairs.append(
            {
                "body1": hand,
                "body2": grip_body_name,
                "friction": list(friction),
                "condim": 4,
                "margin": 0.001,
                "gap": 0.0,
            }
        )

    return {
        "contact_pairs": contact_pairs,
        "default_friction": friction,
        "solver_parameters": {
            "nconmax": 100,
            "njmax": 300,
            "cone": "pyramidal",
        },
    }


class GripContactExporter:
    def __init__(self, model: GripContactModel) -> None:
        if model is None:
            raise ValueError("model must be provided")
        self.model = model
        self.timesteps: list[GripContactTimestep] = []

    def capture_timestep(self) -> GripContactTimestep | None:
        state = self.model.current_state
        if state is None:
            return None

        margins = self.model.check_slip_margin()
        pressures = self.model.get_pressure_distribution()

        contact_forces = np.array([c.normal_force for c in state.contacts])
        contact_positions = (
            np.array([c.position for c in state.contacts])
            if state.contacts
            else np.zeros((0, 3))
        )
        slip_velocities = np.array(
            [np.linalg.norm(c.slip_velocity) for c in state.contacts]
        )

        timestep = GripContactTimestep(
            timestamp=state.timestamp,
            total_normal_force=state.total_normal_force,
            total_tangent_force_mag=float(np.linalg.norm(state.total_tangent_force)),
            num_contacts=len(state.contacts),
            num_slipping=state.num_slipping,
            num_sticking=state.num_sticking,
            slip_ratio=(
                state.num_slipping / len(state.contacts) if state.contacts else 0.0
            ),
            min_slip_margin=margins["min_margin"],
            mean_slip_margin=margins["mean_margin"],
            center_of_pressure=state.center_of_pressure.copy(),
            max_pressure=float(np.max(pressures)) if len(pressures) > 0 else 0.0,
            mean_pressure=float(np.mean(pressures)) if len(pressures) > 0 else 0.0,
            contact_forces=contact_forces,
            contact_positions=contact_positions,
            slip_velocities=slip_velocities,
        )

        self.timesteps.append(timestep)
        return timestep

    def export_to_dict(self) -> dict:
        return {
            "metadata": {
                "num_timesteps": len(self.timesteps),
                "friction_static": self.model.params.static_friction,
                "friction_dynamic": self.model.params.dynamic_friction,
                "grip_diameter": self.model.params.grip_diameter,
            },
            "timesteps": [
                {
                    "timestamp": ts.timestamp,
                    "total_normal_force": ts.total_normal_force,
                    "total_tangent_force_mag": ts.total_tangent_force_mag,
                    "num_contacts": ts.num_contacts,
                    "num_slipping": ts.num_slipping,
                    "num_sticking": ts.num_sticking,
                    "slip_ratio": ts.slip_ratio,
                    "min_slip_margin": ts.min_slip_margin,
                    "mean_slip_margin": ts.mean_slip_margin,
                    "center_of_pressure": ts.center_of_pressure.tolist(),
                    "max_pressure": ts.max_pressure,
                    "mean_pressure": ts.mean_pressure,
                }
                for ts in self.timesteps
            ],
        }

    def export_to_csv_data(self) -> list[dict]:
        return [
            {
                "timestamp": ts.timestamp,
                "total_normal_force": ts.total_normal_force,
                "total_tangent_force_mag": ts.total_tangent_force_mag,
                "num_contacts": ts.num_contacts,
                "num_slipping": ts.num_slipping,
                "num_sticking": ts.num_sticking,
                "slip_ratio": ts.slip_ratio,
                "min_slip_margin": ts.min_slip_margin,
                "mean_slip_margin": ts.mean_slip_margin,
                "cop_x": ts.center_of_pressure[0],
                "cop_y": ts.center_of_pressure[1],
                "cop_z": ts.center_of_pressure[2],
                "max_pressure": ts.max_pressure,
                "mean_pressure": ts.mean_pressure,
            }
            for ts in self.timesteps
        ]

    def get_summary_statistics(self) -> dict:
        if not self.timesteps:
            return {"error": "No timesteps captured"}

        forces = [ts.total_normal_force for ts in self.timesteps]
        slip_ratios = [ts.slip_ratio for ts in self.timesteps]
        margins = [ts.min_slip_margin for ts in self.timesteps]

        return {
            "duration": self.timesteps[-1].timestamp - self.timesteps[0].timestamp,
            "num_timesteps": len(self.timesteps),
            "force_mean": float(np.mean(forces)),
            "force_max": float(np.max(forces)),
            "force_std": float(np.std(forces)),
            "slip_ratio_mean": float(np.mean(slip_ratios)),
            "slip_ratio_max": float(np.max(slip_ratios)),
            "any_slip_detected": any(sr > 0 for sr in slip_ratios),
            "min_margin_ever": float(np.min(margins)),
            "mean_margin": float(np.mean(margins)),
        }

    def reset(self) -> None:
        self.timesteps.clear()

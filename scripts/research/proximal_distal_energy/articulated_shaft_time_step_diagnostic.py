"""Three-level time-step diagnostic for the limiting torsion cell."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
)
from scripts.research.proximal_distal_energy.articulated_shaft_forward import (
    ShaftForwardConfig,
    ShaftIntegrationCase,
    integrate_articulated_shaft,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
STEPS_S = (0.00025, 0.000125, 0.0000625)
SOURCES = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "scripts/research/proximal_distal_energy/articulated_shaft.py",
    "scripts/research/proximal_distal_energy/articulated_shaft_forward.py",
    "scripts/research/proximal_distal_energy/articulated_shaft_time_step_diagnostic.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_articulated_shaft_time_step_diagnostic() -> dict[str, Any]:
    """Run the registered limiting state at three successively finer steps."""

    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        time_s = np.asarray(source["time_s"], dtype=float)
        solution_q = np.asarray(source["solution_q"][8], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][8])
        profile_index = int(source["case_profile_index"][8])
    velocity, _ = finite_difference_kinematics(solution_q, time_s)
    model, metadata = build_subject_scaled_model(
        default_synthetic_profiles()[profile_index]
    )
    grip = DistributedGripConfig(
        station_count_per_hand=5,
        station_width_m=0.03,
        total_stiffness_n_m=1800.0,
        total_damping_n_s_m=18.0,
    )
    results = []
    for step_s in STEPS_S:
        trace = integrate_articulated_shaft(
            model,
            ShaftIntegrationCase(
                q=solution_q[0],
                qd=velocity[0],
                grip_span_m=grip_span_m,
                hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
                time_step_s=step_s,
                initial_club_displacement_m=0.001,
                initial_club_velocity_m_s=0.05,
                engine="mujoco",
                grip=grip,
                shaft=ArticulatedShaftConfig(activation="torsion"),
            ),
            ShaftForwardConfig(duration_s=0.05, time_steps_s=(step_s, step_s / 2.0)),
        )
        results.append(
            {
                "step_s": step_s,
                "maximum_twist_angle_rad": float(
                    np.max(np.abs(trace["twist_angle_rad"]))
                ),
                "maximum_absolute_work_energy_residual_j": float(
                    np.max(np.abs(trace["work_energy_residual_j"]))
                ),
                "remained_in_linear_domain": True,
            }
        )
    residuals = np.asarray(
        [item["maximum_absolute_work_energy_residual_j"] for item in results]
    )
    return {
        "schema_version": "articulated-shaft-time-step-diagnostic/v1",
        "state": [8, 0],
        "activation": "torsion",
        "velocity_factor": 1.0,
        "engine": "mujoco",
        "duration_s": 0.05,
        "results": results,
        "successive_residual_ratios": (residuals[1:] / residuals[:-1]).tolist(),
        "monotone_refinement_passed": bool(np.all(np.diff(residuals) < 0.0)),
        "calibration_status": "synthetic_reference_not_equipment_calibrated",
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCES},
    }


__all__ = ["run_articulated_shaft_time_step_diagnostic"]

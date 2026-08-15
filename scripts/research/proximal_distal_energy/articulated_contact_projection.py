"""Qualify bilateral contact projection on the subject-scaled articulated tree.

This is a same-state initial-acceleration gate. It applies one declared
Kelvin-Voigt perturbation at every closed state, projects equal-and-opposite
point forces through the articulated Jacobians, and compares native MuJoCo and
Pinocchio accelerations. It does not integrate a forward trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    build_pinocchio_articulated_model,
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.spatial_forward_contract import (
    contact_pair,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
    mujoco_mass_matrix_and_bias,
    point_contact_jacobians,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "docs/research/proximal_distal_energy_transfer/data/articulated_inertia_cross_engine.json",
    "scripts/research/proximal_distal_energy/articulated_contact_projection.py",
    "scripts/research/proximal_distal_energy/run_articulated_contact_projection.py",
    "scripts/research/proximal_distal_energy/make_articulated_contact_projection_figure.py",
    "scripts/research/proximal_distal_energy/register_articulated_contact_projection_claims.py",
    "scripts/research/proximal_distal_energy/articulated_inertia_cross_engine.py",
    "scripts/research/proximal_distal_energy/spatial_forward_contract.py",
    "scripts/research/proximal_distal_energy/spatial_full_body.py",
    "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
    "tests/research/test_articulated_contact_projection.py",
)


@dataclass(frozen=True, slots=True)
class ArticulatedContactProjectionConfig:
    """Preoutcome contact, power, and acceleration gates."""

    contact_stiffness: float = 1800.0
    contact_damping: float = 18.0
    club_translation_perturbation_m: float = 1.0e-3
    club_velocity_perturbation_m_s: float = 5.0e-2
    zero_preload_force_tolerance_n: float = 1.0e-6
    action_reaction_tolerance_n: float = 1.0e-12
    virtual_power_tolerance_w: float = 1.0e-10
    geometry_control_tolerance_nm: float = 1.0e-12
    acceleration_relative_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        positive = (
            "contact_stiffness",
            "club_translation_perturbation_m",
            "club_velocity_perturbation_m_s",
            "zero_preload_force_tolerance_n",
            "action_reaction_tolerance_n",
            "virtual_power_tolerance_w",
            "geometry_control_tolerance_nm",
            "acceleration_relative_tolerance",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.contact_damping) or self.contact_damping < 0.0:
            raise ValueError("contact_damping must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ContactProjectionSnapshot:
    """One articulated bilateral contact evaluation."""

    generalized_contact_force: FloatArray
    contact_forces_on_club: FloatArray
    contact_points: FloatArray
    maximum_contact_force_n: float
    action_reaction_residual_n: float
    virtual_power_residual_w: float
    contact_storage_power_w: float
    contact_dissipation_power_w: float
    force_couple_vector_nm: FloatArray
    coincident_force_couple_nm: float
    reversed_couple_sign_residual_nm: float


def _contact_kinematics(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    kin = forward_kinematics(model, q)
    hand_local = np.array([hand_contact_local_x_m, 0.0, 0.0])
    grip_locals = np.array(
        [[0.0, grip_span_m / 2.0, -0.03], [0.0, -grip_span_m / 2.0, -0.03]]
    )
    hand_points, hand_jacobians = [], []
    grip_points, grip_jacobians = [], []
    for hand_joint, grip_local in zip(
        (model.lead_hand_joint, model.trail_hand_joint), grip_locals, strict=True
    ):
        hand_point, hand_jacobian, _ = point_contact_jacobians(
            model, kin, hand_joint, hand_local
        )
        grip_point, grip_jacobian, _ = point_contact_jacobians(
            model, kin, model.club_frame_joint, grip_local
        )
        hand_points.append(hand_point)
        hand_jacobians.append(hand_jacobian)
        grip_points.append(grip_point)
        grip_jacobians.append(grip_jacobian)
    return (
        np.asarray(hand_points),
        np.asarray(hand_jacobians),
        np.asarray(grip_points),
        np.asarray(grip_jacobians),
    )


def evaluate_contact_projection(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    perturb_contact: bool,
    config: ArticulatedContactProjectionConfig = ArticulatedContactProjectionConfig(),
) -> ContactProjectionSnapshot:
    """Apply one bilateral perturbation and project its generalized load."""

    q = np.asarray(q, dtype=float).copy()
    qd = np.asarray(qd, dtype=float).copy()
    if q.shape != (model.nq,) or qd.shape != (model.nq,):
        raise ValueError("q and qd must match the articulated model dimension")
    if not np.isfinite(grip_span_m) or grip_span_m <= 0.0:
        raise ValueError("grip_span_m must be finite and positive")
    if not np.isfinite(hand_contact_local_x_m) or hand_contact_local_x_m <= 0.0:
        raise ValueError("hand_contact_local_x_m must be finite and positive")
    if perturb_contact:
        q[14] += config.club_translation_perturbation_m
        qd[14] += config.club_velocity_perturbation_m_s

    hand_points, hand_jacobians, grip_points, grip_jacobians = _contact_kinematics(
        model,
        q,
        qd,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
    )
    forces = np.zeros((2, 3))
    generalized = np.zeros(model.nq)
    storage_power = 0.0
    dissipation_power = 0.0
    action_residual = 0.0
    physical_power = 0.0
    for index in range(2):
        hand_velocity = hand_jacobians[index] @ qd
        grip_velocity = grip_jacobians[index] @ qd
        force_on_club, force_on_hand, storage, dissipation = contact_pair(
            hand_position=hand_points[index],
            hand_velocity=hand_velocity,
            club_point_position=grip_points[index],
            club_point_velocity=grip_velocity,
            stiffness=config.contact_stiffness,
            damping=config.contact_damping,
        )
        forces[index] = force_on_club
        generalized += grip_jacobians[index].T @ force_on_club
        generalized += hand_jacobians[index].T @ force_on_hand
        action_residual = max(
            action_residual, float(np.linalg.norm(force_on_club + force_on_hand))
        )
        physical_power += float(
            force_on_club @ grip_velocity + force_on_hand @ hand_velocity
        )
        storage_power += storage
        dissipation_power += dissipation

    reference = np.mean(grip_points, axis=0)
    arms = grip_points - reference
    couple = np.sum(np.cross(arms, forces), axis=0)
    coincident = np.sum(np.cross(np.zeros_like(arms), forces), axis=0)
    reversed_couple = np.sum(np.cross(-arms, forces), axis=0)
    return ContactProjectionSnapshot(
        generalized_contact_force=generalized,
        contact_forces_on_club=forces,
        contact_points=grip_points,
        maximum_contact_force_n=float(np.max(np.linalg.norm(forces, axis=1))),
        action_reaction_residual_n=action_residual,
        virtual_power_residual_w=float(abs(generalized @ qd - physical_power)),
        contact_storage_power_w=float(storage_power),
        contact_dissipation_power_w=float(dissipation_power),
        force_couple_vector_nm=couple,
        coincident_force_couple_nm=float(np.linalg.norm(coincident)),
        reversed_couple_sign_residual_nm=float(
            np.linalg.norm(reversed_couple + couple)
        ),
    )


def _relative_error(left: FloatArray, right: FloatArray) -> tuple[float, float]:
    absolute = float(np.max(np.abs(left - right)))
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return absolute, absolute / scale


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_articulated_contact_projection_atlas(
    config: ArticulatedContactProjectionConfig = ArticulatedContactProjectionConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Evaluate contact projection and native acceleration at all closed states."""

    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error

    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        time_s = np.asarray(source["time_s"], dtype=float)
        profile_index = np.asarray(source["case_profile_index"], dtype=int)
        grip_span_m = np.asarray(source["case_grip_span_m"], dtype=float)
        solution_q = np.asarray(source["solution_q"], dtype=float)
        feasible = np.asarray(source["feasible"], dtype=bool)
    if solution_q.shape != (18, 13, 20) or not np.all(feasible):
        raise RuntimeError("the closed-state authority is incomplete or infeasible")

    shape = solution_q.shape[:2]
    acceleration = np.empty((*shape, 2, solution_q.shape[2]))
    acceleration_absolute_error = np.empty(shape)
    acceleration_relative_error = np.empty(shape)
    maximum_force = np.empty(shape)
    couple = np.empty((*shape, 3))
    zero_preload_force = np.empty(shape)
    action_reaction = np.empty(shape)
    virtual_power = np.empty(shape)
    dissipation_power = np.empty(shape)
    coincident_couple = np.empty(shape)
    reversal_residual = np.empty(shape)
    profiles = default_synthetic_profiles()

    for case in range(shape[0]):
        model, metadata = build_subject_scaled_model(profiles[profile_index[case]])
        native = build_pinocchio_articulated_model(pin, model)
        native_data = native.createData()
        velocity, _ = finite_difference_kinematics(solution_q[case], time_s)
        for sample in range(shape[1]):
            q = solution_q[case, sample]
            qd = velocity[sample]
            zero = evaluate_contact_projection(
                model,
                q,
                np.zeros_like(qd),
                grip_span_m=float(grip_span_m[case]),
                hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
                perturb_contact=False,
                config=config,
            )
            perturbed = evaluate_contact_projection(
                model,
                q,
                qd,
                grip_span_m=float(grip_span_m[case]),
                hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
                perturb_contact=True,
                config=config,
            )
            q_perturbed = q.copy()
            q_perturbed[14] += config.club_translation_perturbation_m
            qd_perturbed = qd.copy()
            qd_perturbed[14] += config.club_velocity_perturbation_m_s
            matrix_m, bias_m = mujoco_mass_matrix_and_bias(
                model, q_perturbed, qd_perturbed
            )
            matrix_p = np.asarray(pin.crba(native, native_data, q_perturbed)).copy()
            bias_p = np.asarray(
                pin.nonLinearEffects(native, native_data, q_perturbed, qd_perturbed)
            ).copy()
            acceleration[case, sample, 0] = np.linalg.solve(
                matrix_m, perturbed.generalized_contact_force - bias_m
            )
            acceleration[case, sample, 1] = np.linalg.solve(
                matrix_p, perturbed.generalized_contact_force - bias_p
            )
            (
                acceleration_absolute_error[case, sample],
                acceleration_relative_error[case, sample],
            ) = _relative_error(
                acceleration[case, sample, 0], acceleration[case, sample, 1]
            )
            maximum_force[case, sample] = perturbed.maximum_contact_force_n
            couple[case, sample] = perturbed.force_couple_vector_nm
            zero_preload_force[case, sample] = zero.maximum_contact_force_n
            action_reaction[case, sample] = perturbed.action_reaction_residual_n
            virtual_power[case, sample] = perturbed.virtual_power_residual_w
            dissipation_power[case, sample] = perturbed.contact_dissipation_power_w
            coincident_couple[case, sample] = perturbed.coincident_force_couple_nm
            reversal_residual[case, sample] = perturbed.reversed_couple_sign_residual_nm

    gates = (
        (zero_preload_force <= config.zero_preload_force_tolerance_n)
        & (action_reaction <= config.action_reaction_tolerance_n)
        & (virtual_power <= config.virtual_power_tolerance_w)
        & (dissipation_power <= 0.0)
        & (coincident_couple <= config.geometry_control_tolerance_nm)
        & (reversal_residual <= config.geometry_control_tolerance_nm)
        & (acceleration_relative_error <= config.acceleration_relative_tolerance)
    )
    arrays: dict[str, NDArray[Any]] = {
        "time_s": time_s,
        "case_profile_index": profile_index,
        "case_grip_span_m": grip_span_m,
        "initial_acceleration": acceleration,
        "acceleration_absolute_error": acceleration_absolute_error,
        "acceleration_relative_error": acceleration_relative_error,
        "maximum_contact_force_n": maximum_force,
        "force_couple_vector_nm": couple,
        "zero_preload_force_n": zero_preload_force,
        "action_reaction_residual_n": action_reaction,
        "virtual_power_residual_w": virtual_power,
        "contact_dissipation_power_w": dissipation_power,
        "coincident_force_couple_nm": coincident_couple,
        "reversed_couple_sign_residual_nm": reversal_residual,
        "all_gates_passed": gates,
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
    }
    record: dict[str, Any] = {
        "schema_version": "articulated-contact-projection/v1",
        "study_id": "subject-scaled-articulated-contact-initial-acceleration",
        "model_tier": "same_state_articulated_bilateral_contact_projection",
        "design": {
            "profile_count": len(profiles),
            "grip_span_count": int(np.unique(grip_span_m).size),
            "case_count": int(shape[0]),
            "samples_per_case": int(shape[1]),
            "state_count": int(np.prod(shape)),
            "coordinate_count": int(solution_q.shape[2]),
            "forward_steps": 0,
        },
        "engines": {
            "mujoco": str(mujoco.__version__),
            "pinocchio": str(pin.__version__),
        },
        "contact_contract": {
            "kind": "paired Kelvin-Voigt point interfaces",
            "force_origin": "achieved hand-grip displacement and relative velocity",
            "projection": "J_grip.T force_on_club + J_hand.T force_on_hand",
            "direct_club_actuation": "none",
        },
        "tolerances": asdict(config),
        "results": {
            "maximum_contact_force_n": float(np.max(maximum_force)),
            "maximum_zero_preload_force_n": float(np.max(zero_preload_force)),
            "maximum_action_reaction_residual_n": float(np.max(action_reaction)),
            "maximum_virtual_power_residual_w": float(np.max(virtual_power)),
            "maximum_contact_dissipation_power_w": float(np.max(dissipation_power)),
            "minimum_contact_dissipation_power_w": float(np.min(dissipation_power)),
            "maximum_coincident_force_couple_nm": float(np.max(coincident_couple)),
            "maximum_reversal_sign_residual_nm": float(np.max(reversal_residual)),
            "maximum_generalized_acceleration_absolute_error": float(
                np.max(acceleration_absolute_error)
            ),
            "maximum_acceleration_relative_error": float(
                np.max(acceleration_relative_error)
            ),
            "failed_state_count": int(np.count_nonzero(~gates)),
            "all_registered_gates_passed": bool(np.all(gates)),
        },
        "claim_boundary": {
            "supported": (
                "bilateral compliant point forces project consistently into the "
                "qualified articulated tree and yield matching native initial acceleration"
            ),
            "forward_trajectory": "not_executed",
            "contact_loss_or_recovery": "not_tested",
            "anatomy_and_equipment": "not_calibrated",
            "human_transfer_or_strategy": "untested",
        },
        "next_gate": (
            "integrate the articulated bilateral contact state through a bounded "
            "horizon with contact-loss, convergence, energy, and adverse-load controls"
        ),
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in SOURCE_PATHS},
    }
    return record, arrays


__all__ = [
    "ArticulatedContactProjectionConfig",
    "ContactProjectionSnapshot",
    "evaluate_contact_projection",
    "run_articulated_contact_projection_atlas",
]

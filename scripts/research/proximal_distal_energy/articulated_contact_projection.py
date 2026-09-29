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
    pinocchio_crba_mass_matrix,
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
    attachment_strain_energy_j: float
    maximum_attachment_separation_m: float
    force_couple_vector_nm: FloatArray
    coincident_force_couple_nm: float
    reversed_couple_sign_residual_nm: float


def contact_kinematics(
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

    hand_points, hand_jacobians, grip_points, grip_jacobians = contact_kinematics(
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
    strain_energy = 0.0
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
        displacement = hand_points[index] - grip_points[index]
        strain_energy += (
            0.5 * config.contact_stiffness * float(displacement @ displacement)
        )
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
        attachment_strain_energy_j=float(strain_energy),
        maximum_attachment_separation_m=float(
            np.max(np.linalg.norm(hand_points - grip_points, axis=1))
        ),
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


@dataclass(frozen=True, slots=True)
class _ProjectionAuthority:
    time_s: FloatArray
    profile_index: NDArray[np.int_]
    grip_span_m: FloatArray
    solution_q: FloatArray


@dataclass(slots=True)
class _ProjectionBuffers:
    acceleration: FloatArray
    acceleration_absolute_error: FloatArray
    acceleration_relative_error: FloatArray
    maximum_force: FloatArray
    couple: FloatArray
    zero_preload_force: FloatArray
    action_reaction: FloatArray
    virtual_power: FloatArray
    dissipation_power: FloatArray
    coincident_couple: FloatArray
    reversal_residual: FloatArray


def _load_projection_authority() -> _ProjectionAuthority:
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        authority = _ProjectionAuthority(
            time_s=np.asarray(source["time_s"], dtype=float),
            profile_index=np.asarray(source["case_profile_index"], dtype=int),
            grip_span_m=np.asarray(source["case_grip_span_m"], dtype=float),
            solution_q=np.asarray(source["solution_q"], dtype=float),
        )
        feasible = np.asarray(source["feasible"], dtype=bool)
    if authority.solution_q.shape != (18, 13, 20) or not np.all(feasible):
        raise RuntimeError("the closed-state authority is incomplete or infeasible")
    return authority


def _projection_buffers(shape: tuple[int, int], nq: int) -> _ProjectionBuffers:
    return _ProjectionBuffers(
        acceleration=np.empty((*shape, 2, nq)),
        acceleration_absolute_error=np.empty(shape),
        acceleration_relative_error=np.empty(shape),
        maximum_force=np.empty(shape),
        couple=np.empty((*shape, 3)),
        zero_preload_force=np.empty(shape),
        action_reaction=np.empty(shape),
        virtual_power=np.empty(shape),
        dissipation_power=np.empty(shape),
        coincident_couple=np.empty(shape),
        reversal_residual=np.empty(shape),
    )


def _evaluate_projection_case(
    authority: _ProjectionAuthority,
    buffers: _ProjectionBuffers,
    config: ArticulatedContactProjectionConfig,
    case: int,
    profiles: tuple[Any, ...],
    pin: Any,
) -> None:
    model, metadata = build_subject_scaled_model(
        profiles[authority.profile_index[case]]
    )
    native = build_pinocchio_articulated_model(pin, model)
    native_data = native.createData()
    velocity, _ = finite_difference_kinematics(
        authority.solution_q[case], authority.time_s
    )
    for sample, (q, qd) in enumerate(
        zip(authority.solution_q[case], velocity, strict=True)
    ):
        grip_span_m = float(authority.grip_span_m[case])
        hand_contact_local_x_m = float(metadata["hand_contact_local_x_m"])
        zero = evaluate_contact_projection(
            model,
            q,
            np.zeros_like(qd),
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_contact_local_x_m,
            perturb_contact=False,
            config=config,
        )
        perturbed = evaluate_contact_projection(
            model,
            q,
            qd,
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_contact_local_x_m,
            perturb_contact=True,
            config=config,
        )
        q_perturbed, qd_perturbed = q.copy(), qd.copy()
        q_perturbed[14] += config.club_translation_perturbation_m
        qd_perturbed[14] += config.club_velocity_perturbation_m_s
        matrix_m, bias_m = mujoco_mass_matrix_and_bias(model, q_perturbed, qd_perturbed)
        matrix_p = pinocchio_crba_mass_matrix(pin, native, native_data, q_perturbed)
        bias_p = np.asarray(
            pin.nonLinearEffects(  # type: ignore[attr-defined]
                native, native_data, q_perturbed, qd_perturbed
            )
        ).copy()
        buffers.acceleration[case, sample, 0] = np.linalg.solve(
            matrix_m, perturbed.generalized_contact_force - bias_m
        )
        buffers.acceleration[case, sample, 1] = np.linalg.solve(
            matrix_p, perturbed.generalized_contact_force - bias_p
        )
        errors = _relative_error(
            buffers.acceleration[case, sample, 0],
            buffers.acceleration[case, sample, 1],
        )
        buffers.acceleration_absolute_error[case, sample] = errors[0]
        buffers.acceleration_relative_error[case, sample] = errors[1]
        buffers.maximum_force[case, sample] = perturbed.maximum_contact_force_n
        buffers.couple[case, sample] = perturbed.force_couple_vector_nm
        buffers.zero_preload_force[case, sample] = zero.maximum_contact_force_n
        buffers.action_reaction[case, sample] = perturbed.action_reaction_residual_n
        buffers.virtual_power[case, sample] = perturbed.virtual_power_residual_w
        buffers.dissipation_power[case, sample] = perturbed.contact_dissipation_power_w
        buffers.coincident_couple[case, sample] = perturbed.coincident_force_couple_nm
        buffers.reversal_residual[case, sample] = (
            perturbed.reversed_couple_sign_residual_nm
        )


def _projection_gates(
    buffers: _ProjectionBuffers, config: ArticulatedContactProjectionConfig
) -> NDArray[np.bool_]:
    return (
        (buffers.zero_preload_force <= config.zero_preload_force_tolerance_n)
        & (buffers.action_reaction <= config.action_reaction_tolerance_n)
        & (buffers.virtual_power <= config.virtual_power_tolerance_w)
        & (buffers.dissipation_power <= 0.0)
        & (buffers.coincident_couple <= config.geometry_control_tolerance_nm)
        & (buffers.reversal_residual <= config.geometry_control_tolerance_nm)
        & (
            buffers.acceleration_relative_error
            <= config.acceleration_relative_tolerance
        )
    )


def _projection_arrays(
    authority: _ProjectionAuthority,
    buffers: _ProjectionBuffers,
    gates: NDArray[np.bool_],
) -> dict[str, NDArray[Any]]:
    return {
        "time_s": authority.time_s,
        "case_profile_index": authority.profile_index,
        "case_grip_span_m": authority.grip_span_m,
        "initial_acceleration": buffers.acceleration,
        "acceleration_absolute_error": buffers.acceleration_absolute_error,
        "acceleration_relative_error": buffers.acceleration_relative_error,
        "maximum_contact_force_n": buffers.maximum_force,
        "force_couple_vector_nm": buffers.couple,
        "zero_preload_force_n": buffers.zero_preload_force,
        "action_reaction_residual_n": buffers.action_reaction,
        "virtual_power_residual_w": buffers.virtual_power,
        "contact_dissipation_power_w": buffers.dissipation_power,
        "coincident_force_couple_nm": buffers.coincident_couple,
        "reversed_couple_sign_residual_nm": buffers.reversal_residual,
        "all_gates_passed": gates,
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
    }


def _projection_record(
    authority: _ProjectionAuthority,
    buffers: _ProjectionBuffers,
    gates: NDArray[np.bool_],
    config: ArticulatedContactProjectionConfig,
    engine_versions: dict[str, str],
) -> dict[str, Any]:
    shape = authority.solution_q.shape[:2]
    return {
        "schema_version": "articulated-contact-projection/v1",
        "study_id": "subject-scaled-articulated-contact-initial-acceleration",
        "model_tier": "same_state_articulated_bilateral_contact_projection",
        "design": {
            "profile_count": len(default_synthetic_profiles()),
            "grip_span_count": int(np.unique(authority.grip_span_m).size),
            "case_count": int(shape[0]),
            "samples_per_case": int(shape[1]),
            "state_count": int(np.prod(shape)),
            "coordinate_count": int(authority.solution_q.shape[2]),
            "forward_steps": 0,
        },
        "engines": engine_versions,
        "contact_contract": {
            "kind": "paired Kelvin-Voigt point interfaces",
            "force_origin": "achieved hand-grip displacement and relative velocity",
            "projection": "J_grip.T force_on_club + J_hand.T force_on_hand",
            "direct_club_actuation": "none",
        },
        "tolerances": asdict(config),
        "results": {
            "maximum_contact_force_n": float(np.max(buffers.maximum_force)),
            "maximum_zero_preload_force_n": float(np.max(buffers.zero_preload_force)),
            "maximum_action_reaction_residual_n": float(
                np.max(buffers.action_reaction)
            ),
            "maximum_virtual_power_residual_w": float(np.max(buffers.virtual_power)),
            "maximum_contact_dissipation_power_w": float(
                np.max(buffers.dissipation_power)
            ),
            "minimum_contact_dissipation_power_w": float(
                np.min(buffers.dissipation_power)
            ),
            "maximum_coincident_force_couple_nm": float(
                np.max(buffers.coincident_couple)
            ),
            "maximum_reversal_sign_residual_nm": float(
                np.max(buffers.reversal_residual)
            ),
            "maximum_generalized_acceleration_absolute_error": float(
                np.max(buffers.acceleration_absolute_error)
            ),
            "maximum_acceleration_relative_error": float(
                np.max(buffers.acceleration_relative_error)
            ),
            "failed_state_count": int(np.count_nonzero(~gates)),
            "all_registered_gates_passed": bool(np.all(gates)),
        },
        "claim_boundary": {
            "supported": "bilateral compliant point forces project consistently into the qualified articulated tree and yield matching native initial acceleration",
            "forward_trajectory": "not_executed",
            "contact_loss_or_recovery": "not_tested",
            "anatomy_and_equipment": "not_calibrated",
            "human_transfer_or_strategy": "untested",
        },
        "next_gate": "integrate the articulated bilateral contact state through a bounded horizon with contact-loss, convergence, energy, and adverse-load controls",
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in SOURCE_PATHS},
    }


def run_articulated_contact_projection_atlas(
    config: ArticulatedContactProjectionConfig = ArticulatedContactProjectionConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Evaluate contact projection and native acceleration at all closed states."""

    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error

    authority = _load_projection_authority()
    shape = (authority.solution_q.shape[0], authority.solution_q.shape[1])
    buffers = _projection_buffers(shape, authority.solution_q.shape[2])
    profiles = default_synthetic_profiles()
    for case in range(authority.solution_q.shape[0]):
        _evaluate_projection_case(authority, buffers, config, case, profiles, pin)
    gates = _projection_gates(buffers, config)
    arrays = _projection_arrays(authority, buffers, gates)
    versions = {
        "mujoco": str(mujoco.__version__),
        "pinocchio": str(pin.__version__),  # type: ignore[attr-defined]
    }
    return _projection_record(authority, buffers, gates, config, versions), arrays


__all__ = [
    "ArticulatedContactProjectionConfig",
    "ContactProjectionSnapshot",
    "contact_kinematics",
    "evaluate_contact_projection",
    "run_articulated_contact_projection_atlas",
]

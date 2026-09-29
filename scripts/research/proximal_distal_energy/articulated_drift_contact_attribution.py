"""Same-state drift, contact, and applied-input attribution for #9151.

The decomposition is an identity at one generalized state. It does not state
that any contribution persists under forward integration, identify biological
torque, or establish a human strategy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_contact_projection import (
    ArticulatedContactProjectionConfig,
    evaluate_contact_projection,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    build_pinocchio_articulated_model,
    finite_difference_kinematics,
    pinocchio_crba_mass_matrix,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    mujoco_mass_matrix_and_bias,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
CONTRIBUTION_NAMES = ("configuration", "velocity", "contact", "active")
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "docs/research/proximal_distal_energy_transfer/data/articulated_contact_projection.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_contact_projection.npz",
    "scripts/research/proximal_distal_energy/articulated_drift_contact_attribution.py",
    "scripts/research/proximal_distal_energy/run_articulated_drift_contact_attribution.py",
    "tests/research/test_articulated_drift_contact_attribution.py",
)


class AttributionAdequacy(str, Enum):
    """Whether a signed share has a sufficiently large denominator."""

    ADEQUATE = "adequate"
    SUPPRESSED = "suppressed_below_denominator_floor"


@dataclass(frozen=True, slots=True)
class GeneralizedDynamicsAttribution:
    """Exact same-state generalized-force and acceleration decomposition."""

    generalized_forces: FloatArray
    acceleration_contributions: FloatArray
    total_generalized_force: FloatArray
    total_acceleration: FloatArray
    generalized_powers_w: FloatArray
    total_generalized_power_w: float
    mass_metric_acceleration_shares: FloatArray
    generalized_power_shares: FloatArray
    acceleration_share_adequacy: AttributionAdequacy
    power_share_adequacy: AttributionAdequacy
    acceleration_cancellation_index: float
    power_cancellation_index: float
    acceleration_closure_residual: float
    power_closure_residual_w: float


def _finite_vector(name: str, value: FloatArray, size: int) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _validated_mass_matrix(value: FloatArray) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mass_matrix must be a square matrix")
    if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("mass_matrix must be nonempty and finite")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("mass_matrix must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass_matrix must be positive definite") from error
    return matrix


def _shares(
    numerators: FloatArray, denominator: float, floor: float
) -> tuple[FloatArray, AttributionAdequacy, float]:
    if abs(denominator) < floor:
        return (
            np.full(numerators.shape, np.nan, dtype=np.float64),
            AttributionAdequacy.SUPPRESSED,
            float("nan"),
        )
    shares = numerators / denominator
    cancellation = float(np.sum(np.abs(numerators)) / abs(denominator))
    return shares, AttributionAdequacy.ADEQUATE, cancellation


def decompose_generalized_dynamics(
    *,
    mass_matrix: FloatArray,
    bias_force: FloatArray,
    zero_velocity_bias_force: FloatArray,
    contact_force: FloatArray,
    active_force: FloatArray,
    velocity: FloatArray,
    share_denominator_floor: float = 1.0e-12,
) -> GeneralizedDynamicsAttribution:
    """Decompose ``M qdd = tau_active + Q_contact - h`` at one state.

    ``h(q, 0)`` is called the configuration contribution because its precise
    content depends on the native operator. The velocity contribution is the
    exact residual ``h(q, qdot) - h(q, 0)``. Postcondition: the four returned
    force and acceleration contributions sum to their returned totals.
    """

    matrix = _validated_mass_matrix(mass_matrix)
    size = matrix.shape[0]
    bias = _finite_vector("bias_force", bias_force, size)
    static_bias = _finite_vector(
        "zero_velocity_bias_force", zero_velocity_bias_force, size
    )
    contact = _finite_vector("contact_force", contact_force, size)
    active = _finite_vector("active_force", active_force, size)
    qd = _finite_vector("velocity", velocity, size)
    if not np.isfinite(share_denominator_floor) or share_denominator_floor <= 0.0:
        raise ValueError("share_denominator_floor must be finite and positive")

    forces = np.stack(
        (
            -static_bias,
            -(bias - static_bias),
            contact,
            active,
        )
    )
    accelerations = np.asarray(
        [np.linalg.solve(matrix, force) for force in forces], dtype=np.float64
    )
    total_force = np.sum(forces, axis=0)
    total_acceleration = np.linalg.solve(matrix, total_force)
    acceleration_closure = float(
        np.linalg.norm(np.sum(accelerations, axis=0) - total_acceleration)
    )

    powers = forces @ qd
    total_power = float(total_force @ qd)
    power_closure = float(abs(np.sum(powers) - total_power))

    mass_projection_numerators = np.asarray(
        [part @ matrix @ total_acceleration for part in accelerations],
        dtype=np.float64,
    )
    mass_projection_denominator = float(
        total_acceleration @ matrix @ total_acceleration
    )
    acceleration_shares, acceleration_adequacy, acceleration_cancellation = _shares(
        mass_projection_numerators,
        mass_projection_denominator,
        share_denominator_floor,
    )
    power_shares, power_adequacy, power_cancellation = _shares(
        powers, total_power, share_denominator_floor
    )
    return GeneralizedDynamicsAttribution(
        generalized_forces=forces,
        acceleration_contributions=accelerations,
        total_generalized_force=total_force,
        total_acceleration=total_acceleration,
        generalized_powers_w=powers,
        total_generalized_power_w=total_power,
        mass_metric_acceleration_shares=acceleration_shares,
        generalized_power_shares=power_shares,
        acceleration_share_adequacy=acceleration_adequacy,
        power_share_adequacy=power_adequacy,
        acceleration_cancellation_index=acceleration_cancellation,
        power_cancellation_index=power_cancellation,
        acceleration_closure_residual=acceleration_closure,
        power_closure_residual_w=power_closure,
    )


def scale_generalized_coordinates(
    *,
    mass_matrix: FloatArray,
    bias_force: FloatArray,
    zero_velocity_bias_force: FloatArray,
    contact_force: FloatArray,
    active_force: FloatArray,
    velocity: FloatArray,
    coordinate_scale: FloatArray,
) -> dict[str, Any]:
    """Return the same physical state under ``q_scaled = S q``.

    Generalized forces transform contragrediently. The returned mapping is
    accepted directly by :func:`decompose_generalized_dynamics`.
    """

    matrix = _validated_mass_matrix(mass_matrix)
    size = matrix.shape[0]
    scale = _finite_vector("coordinate_scale", coordinate_scale, size)
    if np.any(scale <= 0.0):
        raise ValueError("coordinate_scale must be positive")
    inverse = np.diag(1.0 / scale)

    def transform_force(name: str, value: FloatArray) -> FloatArray:
        return inverse @ _finite_vector(name, value, size)

    return {
        "mass_matrix": inverse @ matrix @ inverse,
        "bias_force": transform_force("bias_force", bias_force),
        "zero_velocity_bias_force": transform_force(
            "zero_velocity_bias_force", zero_velocity_bias_force
        ),
        "contact_force": transform_force("contact_force", contact_force),
        "active_force": transform_force("active_force", active_force),
        "velocity": scale * _finite_vector("velocity", velocity, size),
    }


@dataclass(frozen=True, slots=True)
class ArticulatedAttributionConfig:
    """Preoutcome numerical gates for the 234-state native-engine atlas."""

    acceleration_closure_tolerance: float = 1.0e-10
    power_closure_tolerance_w: float = 1.0e-10
    engine_relative_tolerance: float = 1.0e-8
    scaling_invariance_tolerance: float = 1.0e-10
    pathway_killswitch_tolerance: float = 1.0e-12
    share_denominator_floor: float = 1.0e-10
    corruption_force_nm: float = 1.0e-3
    corruption_detection_floor: float = 1.0e-6
    contact: ArticulatedContactProjectionConfig = ArticulatedContactProjectionConfig()

    def __post_init__(self) -> None:
        for name in (
            "acceleration_closure_tolerance",
            "power_closure_tolerance_w",
            "engine_relative_tolerance",
            "scaling_invariance_tolerance",
            "pathway_killswitch_tolerance",
            "share_denominator_floor",
            "corruption_force_nm",
            "corruption_detection_floor",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class _AttributionAuthority:
    time_s: FloatArray
    profile_index: NDArray[np.int64]
    grip_span_m: FloatArray
    solution_q: FloatArray


@dataclass(slots=True)
class _AttributionBuffers:
    force: FloatArray
    acceleration: FloatArray
    total_acceleration: FloatArray
    power: FloatArray
    total_power: FloatArray
    acceleration_share: FloatArray
    power_share: FloatArray
    acceleration_share_adequate: NDArray[np.bool_]
    power_share_adequate: NDArray[np.bool_]
    acceleration_cancellation: FloatArray
    power_cancellation: FloatArray
    acceleration_closure: FloatArray
    power_closure: FloatArray
    engine_contribution_error: FloatArray
    engine_total_error: FloatArray
    scaling_power_error: FloatArray
    scaling_share_error: FloatArray
    zero_contact_norm: FloatArray
    zero_velocity_norm: FloatArray
    gravity_off_norm: FloatArray
    corruption_detection_residual: FloatArray
    coincident_couple: FloatArray
    reversal_residual: FloatArray


def _load_authority() -> _AttributionAuthority:
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        authority = _AttributionAuthority(
            time_s=np.asarray(source["time_s"], dtype=np.float64),
            profile_index=np.asarray(source["case_profile_index"], dtype=np.int64),
            grip_span_m=np.asarray(source["case_grip_span_m"], dtype=np.float64),
            solution_q=np.asarray(source["solution_q"], dtype=np.float64),
        )
        feasible = np.asarray(source["feasible"], dtype=bool)
    if authority.solution_q.shape != (18, 13, 20) or not np.all(feasible):
        raise RuntimeError("the closed-state authority is incomplete or infeasible")
    return authority


def _buffers(shape: tuple[int, int], nq: int) -> _AttributionBuffers:
    engine_shape = (*shape, 2)
    return _AttributionBuffers(
        force=np.empty((*engine_shape, 4, nq)),
        acceleration=np.empty((*engine_shape, 4, nq)),
        total_acceleration=np.empty((*engine_shape, nq)),
        power=np.empty((*engine_shape, 4)),
        total_power=np.empty(engine_shape),
        acceleration_share=np.empty((*engine_shape, 4)),
        power_share=np.empty((*engine_shape, 4)),
        acceleration_share_adequate=np.empty(engine_shape, dtype=bool),
        power_share_adequate=np.empty(engine_shape, dtype=bool),
        acceleration_cancellation=np.empty(engine_shape),
        power_cancellation=np.empty(engine_shape),
        acceleration_closure=np.empty(engine_shape),
        power_closure=np.empty(engine_shape),
        engine_contribution_error=np.empty((*shape, 4)),
        engine_total_error=np.empty(shape),
        scaling_power_error=np.empty((*engine_shape, 4)),
        scaling_share_error=np.empty((*engine_shape, 4)),
        zero_contact_norm=np.empty(engine_shape),
        zero_velocity_norm=np.empty(engine_shape),
        gravity_off_norm=np.empty(engine_shape),
        corruption_detection_residual=np.empty(engine_shape),
        coincident_couple=np.empty(shape),
        reversal_residual=np.empty(shape),
    )


def _relative_error(left: FloatArray, right: FloatArray) -> float:
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return float(np.max(np.abs(left - right)) / scale)


def _coordinate_scale(model: Any) -> FloatArray:
    return np.asarray(
        [
            1000.0 if joint.kind == "prismatic" else 180.0 / np.pi
            for joint in model.joints
        ],
        dtype=np.float64,
    )


def _store_attribution(
    buffers: _AttributionBuffers,
    index: tuple[int, int, int],
    result: GeneralizedDynamicsAttribution,
) -> None:
    buffers.force[index] = result.generalized_forces
    buffers.acceleration[index] = result.acceleration_contributions
    buffers.total_acceleration[index] = result.total_acceleration
    buffers.power[index] = result.generalized_powers_w
    buffers.total_power[index] = result.total_generalized_power_w
    buffers.acceleration_share[index] = result.mass_metric_acceleration_shares
    buffers.power_share[index] = result.generalized_power_shares
    buffers.acceleration_share_adequate[index] = (
        result.acceleration_share_adequacy is AttributionAdequacy.ADEQUATE
    )
    buffers.power_share_adequate[index] = (
        result.power_share_adequacy is AttributionAdequacy.ADEQUATE
    )
    buffers.acceleration_cancellation[index] = result.acceleration_cancellation_index
    buffers.power_cancellation[index] = result.power_cancellation_index
    buffers.acceleration_closure[index] = result.acceleration_closure_residual
    buffers.power_closure[index] = result.power_closure_residual_w


def _evaluate_native_result(
    *,
    matrix: FloatArray,
    bias: FloatArray,
    static_bias: FloatArray,
    contact_force: FloatArray,
    velocity: FloatArray,
    config: ArticulatedAttributionConfig,
) -> GeneralizedDynamicsAttribution:
    return decompose_generalized_dynamics(
        mass_matrix=matrix,
        bias_force=bias,
        zero_velocity_bias_force=static_bias,
        contact_force=contact_force,
        active_force=np.zeros_like(contact_force),
        velocity=velocity,
        share_denominator_floor=config.share_denominator_floor,
    )


def _evaluate_controls(
    *,
    buffers: _AttributionBuffers,
    index: tuple[int, int, int],
    result: GeneralizedDynamicsAttribution,
    matrix: FloatArray,
    bias: FloatArray,
    static_bias: FloatArray,
    contact_force: FloatArray,
    velocity: FloatArray,
    coordinate_scale: FloatArray,
    config: ArticulatedAttributionConfig,
) -> None:
    zero = np.zeros_like(contact_force)
    zero_contact = _evaluate_native_result(
        matrix=matrix,
        bias=bias,
        static_bias=static_bias,
        contact_force=zero,
        velocity=velocity,
        config=config,
    )
    buffers.zero_contact_norm[index] = np.linalg.norm(
        zero_contact.acceleration_contributions[2]
    )
    zero_velocity = _evaluate_native_result(
        matrix=matrix,
        bias=static_bias,
        static_bias=static_bias,
        contact_force=contact_force,
        velocity=zero,
        config=config,
    )
    buffers.zero_velocity_norm[index] = np.linalg.norm(
        zero_velocity.acceleration_contributions[1]
    )
    gravity_off = _evaluate_native_result(
        matrix=matrix,
        bias=bias - static_bias,
        static_bias=zero,
        contact_force=contact_force,
        velocity=velocity,
        config=config,
    )
    buffers.gravity_off_norm[index] = np.linalg.norm(
        gravity_off.acceleration_contributions[0]
    )

    scaled_arguments = scale_generalized_coordinates(
        mass_matrix=matrix,
        bias_force=bias,
        zero_velocity_bias_force=static_bias,
        contact_force=contact_force,
        active_force=zero,
        velocity=velocity,
        coordinate_scale=coordinate_scale,
    )
    scaled = decompose_generalized_dynamics(
        **scaled_arguments, share_denominator_floor=config.share_denominator_floor
    )
    buffers.scaling_power_error[index] = np.abs(
        scaled.generalized_powers_w - result.generalized_powers_w
    )
    if result.acceleration_share_adequacy is AttributionAdequacy.ADEQUATE:
        buffers.scaling_share_error[index] = np.abs(
            scaled.mass_metric_acceleration_shares
            - result.mass_metric_acceleration_shares
        )
    else:
        buffers.scaling_share_error[index] = 0.0

    corruption = np.zeros_like(contact_force)
    corruption[0] = config.corruption_force_nm
    corrupted_contact_acceleration = np.linalg.solve(matrix, contact_force + corruption)
    corrupted_sum = np.sum(result.acceleration_contributions, axis=0)
    corrupted_sum += (
        corrupted_contact_acceleration - result.acceleration_contributions[2]
    )
    buffers.corruption_detection_residual[index] = np.linalg.norm(
        corrupted_sum - result.total_acceleration
    )


def _evaluate_case(
    authority: _AttributionAuthority,
    buffers: _AttributionBuffers,
    case: int,
    pin: Any,
    config: ArticulatedAttributionConfig,
) -> None:
    profiles = default_synthetic_profiles()
    model, metadata = build_subject_scaled_model(
        profiles[authority.profile_index[case]]
    )
    native = build_pinocchio_articulated_model(pin, model)
    native_data = native.createData()
    velocity, _ = finite_difference_kinematics(
        authority.solution_q[case], authority.time_s
    )
    scale = _coordinate_scale(model)
    for sample, (q, qd) in enumerate(
        zip(authority.solution_q[case], velocity, strict=True)
    ):
        contact = evaluate_contact_projection(
            model,
            q,
            qd,
            grip_span_m=float(authority.grip_span_m[case]),
            hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
            perturb_contact=True,
            config=config.contact,
        )
        q_eval = q.copy()
        qd_eval = qd.copy()
        q_eval[14] += config.contact.club_translation_perturbation_m
        qd_eval[14] += config.contact.club_velocity_perturbation_m_s
        matrix_m, bias_m = mujoco_mass_matrix_and_bias(model, q_eval, qd_eval)
        _, static_m = mujoco_mass_matrix_and_bias(model, q_eval, np.zeros_like(qd_eval))
        matrix_p = pinocchio_crba_mass_matrix(pin, native, native_data, q_eval)
        bias_p = np.asarray(
            pin.nonLinearEffects(native, native_data, q_eval, qd_eval), dtype=np.float64
        )
        static_p = np.asarray(
            pin.nonLinearEffects(native, native_data, q_eval, np.zeros_like(qd_eval)),
            dtype=np.float64,
        )
        results = (
            _evaluate_native_result(
                matrix=matrix_m,
                bias=bias_m,
                static_bias=static_m,
                contact_force=contact.generalized_contact_force,
                velocity=qd_eval,
                config=config,
            ),
            _evaluate_native_result(
                matrix=matrix_p,
                bias=bias_p,
                static_bias=static_p,
                contact_force=contact.generalized_contact_force,
                velocity=qd_eval,
                config=config,
            ),
        )
        for engine, (result, matrix, bias, static_bias) in enumerate(
            zip(
                results,
                (matrix_m, matrix_p),
                (bias_m, bias_p),
                (static_m, static_p),
                strict=True,
            )
        ):
            index = (case, sample, engine)
            _store_attribution(buffers, index, result)
            _evaluate_controls(
                buffers=buffers,
                index=index,
                result=result,
                matrix=matrix,
                bias=bias,
                static_bias=static_bias,
                contact_force=contact.generalized_contact_force,
                velocity=qd_eval,
                coordinate_scale=scale,
                config=config,
            )
        for contribution in range(len(CONTRIBUTION_NAMES)):
            buffers.engine_contribution_error[case, sample, contribution] = (
                _relative_error(
                    results[0].acceleration_contributions[contribution],
                    results[1].acceleration_contributions[contribution],
                )
            )
        buffers.engine_total_error[case, sample] = _relative_error(
            results[0].total_acceleration, results[1].total_acceleration
        )
        buffers.coincident_couple[case, sample] = contact.coincident_force_couple_nm
        buffers.reversal_residual[case, sample] = (
            contact.reversed_couple_sign_residual_nm
        )


def _finite_max(values: FloatArray) -> float:
    finite = values[np.isfinite(values)]
    return 0.0 if finite.size == 0 else float(np.max(finite))


def _gates(
    buffers: _AttributionBuffers, config: ArticulatedAttributionConfig
) -> NDArray[np.bool_]:
    return (
        (buffers.acceleration_closure <= config.acceleration_closure_tolerance)
        & (buffers.power_closure <= config.power_closure_tolerance_w)
        & (buffers.engine_total_error[..., None] <= config.engine_relative_tolerance)
        & (
            np.max(buffers.engine_contribution_error, axis=-1)[..., None]
            <= config.engine_relative_tolerance
        )
        & (
            np.max(buffers.scaling_power_error, axis=-1)
            <= config.scaling_invariance_tolerance
        )
        & (
            np.max(buffers.scaling_share_error, axis=-1)
            <= config.scaling_invariance_tolerance
        )
        & (buffers.zero_contact_norm <= config.pathway_killswitch_tolerance)
        & (buffers.zero_velocity_norm <= config.pathway_killswitch_tolerance)
        & (buffers.gravity_off_norm <= config.pathway_killswitch_tolerance)
        & (buffers.corruption_detection_residual >= config.corruption_detection_floor)
        & (
            buffers.coincident_couple[..., None]
            <= config.contact.geometry_control_tolerance_nm
        )
        & (
            buffers.reversal_residual[..., None]
            <= config.contact.geometry_control_tolerance_nm
        )
    )


def _arrays(
    authority: _AttributionAuthority,
    buffers: _AttributionBuffers,
    gates: NDArray[np.bool_],
) -> dict[str, NDArray[Any]]:
    return {
        "time_s": authority.time_s,
        "case_profile_index": authority.profile_index,
        "case_grip_span_m": authority.grip_span_m,
        "generalized_force_contribution": buffers.force,
        "acceleration_contribution": buffers.acceleration,
        "total_acceleration": buffers.total_acceleration,
        "generalized_power_contribution_w": buffers.power,
        "total_generalized_power_w": buffers.total_power,
        "mass_metric_acceleration_share": buffers.acceleration_share,
        "generalized_power_share": buffers.power_share,
        "acceleration_share_adequate": buffers.acceleration_share_adequate,
        "power_share_adequate": buffers.power_share_adequate,
        "acceleration_cancellation_index": buffers.acceleration_cancellation,
        "power_cancellation_index": buffers.power_cancellation,
        "acceleration_closure_residual": buffers.acceleration_closure,
        "power_closure_residual_w": buffers.power_closure,
        "engine_contribution_relative_error": buffers.engine_contribution_error,
        "engine_total_relative_error": buffers.engine_total_error,
        "coordinate_scaling_power_error_w": buffers.scaling_power_error,
        "coordinate_scaling_share_error": buffers.scaling_share_error,
        "zero_contact_acceleration_norm": buffers.zero_contact_norm,
        "zero_velocity_velocity_contribution_norm": buffers.zero_velocity_norm,
        "gravity_off_configuration_contribution_norm": buffers.gravity_off_norm,
        "corruption_detection_residual": buffers.corruption_detection_residual,
        "all_gates_passed": gates,
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
        "contribution_names": np.asarray(CONTRIBUTION_NAMES),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    authority: _AttributionAuthority,
    buffers: _AttributionBuffers,
    gates: NDArray[np.bool_],
    config: ArticulatedAttributionConfig,
    versions: dict[str, str],
) -> dict[str, Any]:
    state_count = int(np.prod(authority.solution_q.shape[:2]))
    return {
        "schema_version": "articulated-drift-contact-attribution/v1",
        "study_id": "subject-scaled-articulated-same-state-attribution",
        "classification": "synthetic_pointwise_mechanics_attribution",
        "design": {
            "state_count": state_count,
            "profile_count": len(default_synthetic_profiles()),
            "grip_span_count": int(np.unique(authority.grip_span_m).size),
            "coordinate_count": int(authority.solution_q.shape[-1]),
            "engine_names": ["mujoco", "pinocchio"],
            "contribution_names": list(CONTRIBUTION_NAMES),
            "applied_input": "exactly_zero_in_baseline",
            "forward_steps": 0,
        },
        "equations": {
            "configuration": "M(q)^-1 [-h(q, 0)]",
            "velocity": "M(q)^-1 {-[h(q, qdot) - h(q, 0)]}",
            "contact": "M(q)^-1 Q_contact(q, qdot)",
            "active": "M(q)^-1 tau_applied",
            "total": "sum(configuration, velocity, contact, active)",
            "power": "Q_i dot qdot",
            "acceleration_share": "(qdd_i^T M qdd_total)/(qdd_total^T M qdd_total)",
        },
        "engines": versions,
        "tolerances": {**asdict(config), "contact": asdict(config.contact)},
        "results": {
            "failed_engine_state_count": int(np.count_nonzero(~gates)),
            "all_registered_gates_passed": bool(np.all(gates)),
            "maximum_acceleration_closure_residual": float(
                np.max(buffers.acceleration_closure)
            ),
            "maximum_power_closure_residual_w": float(np.max(buffers.power_closure)),
            "maximum_engine_contribution_relative_error": float(
                np.max(buffers.engine_contribution_error)
            ),
            "maximum_engine_total_relative_error": float(
                np.max(buffers.engine_total_error)
            ),
            "maximum_coordinate_scaling_power_error_w": float(
                np.max(buffers.scaling_power_error)
            ),
            "maximum_coordinate_scaling_share_error": float(
                np.max(buffers.scaling_share_error)
            ),
            "maximum_zero_contact_acceleration_norm": float(
                np.max(buffers.zero_contact_norm)
            ),
            "maximum_zero_velocity_velocity_contribution_norm": float(
                np.max(buffers.zero_velocity_norm)
            ),
            "maximum_gravity_off_configuration_contribution_norm": float(
                np.max(buffers.gravity_off_norm)
            ),
            "minimum_corruption_detection_residual": float(
                np.min(buffers.corruption_detection_residual)
            ),
            "acceleration_share_suppressed_count": int(
                np.count_nonzero(~buffers.acceleration_share_adequate)
            ),
            "power_share_suppressed_count": int(
                np.count_nonzero(~buffers.power_share_adequate)
            ),
            "maximum_finite_acceleration_cancellation_index": _finite_max(
                buffers.acceleration_cancellation
            ),
            "maximum_finite_power_cancellation_index": _finite_max(
                buffers.power_cancellation
            ),
        },
        "controls": {
            "zero_contact": "contact contribution must vanish exactly",
            "zero_velocity": "velocity-dependent bias contribution must vanish exactly",
            "gravity_off": "configuration contribution must vanish exactly",
            "geometry": "coincident moment arms remove the couple and reversed arms reverse its sign",
            "coordinate_scaling": "millimetre and degree coordinates preserve physical power and mass-metric shares",
            "corrupted_force": "a 1e-3 N m first-coordinate sentinel must violate the uncorrupted acceleration closure",
        },
        "claim_boundary": {
            "supported": "same-state configuration, velocity, contact, and zero applied-input contributions close and agree across the declared native operators",
            "forward_persistence_impulse_or_work": "not_executed",
            "zvcf": "not_evaluated",
            "biological_torque_or_effort": "not_identified",
            "human_transfer_or_strategy": "untested",
            "coaching_or_safety": "unavailable",
        },
        "next_gate": "integrate matched forward trajectories and attribute impulse and work through contact transitions, shaft/base coupling, uncertainty, and adverse loads",
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in SOURCE_PATHS},
    }


def run_articulated_drift_contact_attribution(
    config: ArticulatedAttributionConfig = ArticulatedAttributionConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Execute the registered 234-state, two-native-engine attribution atlas."""

    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - optional native stack
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error
    authority = _load_authority()
    buffers = _buffers(authority.solution_q.shape[:2], authority.solution_q.shape[-1])
    for case in range(authority.solution_q.shape[0]):
        _evaluate_case(authority, buffers, case, pin, config)
    gates = _gates(buffers, config)
    versions = {
        "mujoco": str(mujoco.__version__),
        "pinocchio": str(pin.__version__),  # type: ignore[attr-defined]
    }
    return _record(authority, buffers, gates, config, versions), _arrays(
        authority, buffers, gates
    )


__all__ = [
    "ArticulatedAttributionConfig",
    "AttributionAdequacy",
    "CONTRIBUTION_NAMES",
    "GeneralizedDynamicsAttribution",
    "decompose_generalized_dynamics",
    "run_articulated_drift_contact_attribution",
    "scale_generalized_coordinates",
]

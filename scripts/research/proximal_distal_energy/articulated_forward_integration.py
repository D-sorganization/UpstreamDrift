"""Single-trajectory integration for articulated bilateral attachment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_contact_projection import (
    ArticulatedContactProjectionConfig,
    ContactProjectionSnapshot,
    evaluate_contact_projection,
)
from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    ArticulatedForwardContactConfig,
    mechanical_energy,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    build_pinocchio_articulated_model,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ForwardIntegrationCase:
    """All varying inputs for one deterministic trajectory."""

    q: FloatArray
    qd: FloatArray
    grip_span_m: float
    hand_contact_local_x_m: float
    time_step_s: float
    contact_stiffness: float
    contact_damping: float
    initial_club_displacement_m: float
    initial_club_velocity_m_s: float
    engine: str


@dataclass(slots=True)
class _TraceBuffers:
    q: FloatArray
    qd: FloatArray
    force: FloatArray
    separation: FloatArray
    virtual_power: FloatArray
    dissipation: FloatArray
    strain: FloatArray
    mechanical: FloatArray


def native_dynamics_operator(
    engine: str, model: SpatialModel
) -> Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]:
    if engine == "mujoco":
        import mujoco
        from scripts.research.proximal_distal_energy.spatial_full_body import (
            _compiled_mujoco_model,
            _mujoco_xml,
        )

        mj_model = _compiled_mujoco_model(model.canonical_hash, _mujoco_xml(model))
        data = mujoco.MjData(mj_model)
        matrix = np.empty((model.nq, model.nq), dtype=np.float64)

        def evaluate_mujoco(
            q: FloatArray, qd: FloatArray
        ) -> tuple[FloatArray, FloatArray]:
            data.qpos[:] = q
            data.qvel[:] = qd
            mujoco.mj_forward(mj_model, data)
            mujoco.mj_fullM(mj_model, matrix, data.qM)
            return matrix.copy(), np.asarray(data.qfrc_bias, dtype=np.float64).copy()

        return evaluate_mujoco

    if engine != "pinocchio":
        raise ValueError("engine must be 'mujoco' or 'pinocchio'")
    try:
        import pinocchio as pin

        native = build_pinocchio_articulated_model(pin, model)
        data_pin = native.createData()

        def evaluate(q: FloatArray, qd: FloatArray) -> tuple[FloatArray, FloatArray]:
            matrix = np.asarray(pin.crba(native, data_pin, q)).copy()
            bias = np.asarray(
                pin.nonLinearEffects(native, data_pin, q, qd)  # type: ignore[attr-defined]
            ).copy()
            return matrix, bias

        return evaluate
    except (ImportError, RuntimeError):
        import mujoco
        from scripts.research.proximal_distal_energy.spatial_full_body import (
            _compiled_mujoco_model,
            _mujoco_xml,
        )

        mj_model = _compiled_mujoco_model(model.canonical_hash, _mujoco_xml(model))
        data = mujoco.MjData(mj_model)
        matrix = np.empty((model.nq, model.nq), dtype=np.float64)

        def evaluate_fallback(
            q: FloatArray, qd: FloatArray
        ) -> tuple[FloatArray, FloatArray]:
            data.qpos[:] = q
            data.qvel[:] = qd
            mujoco.mj_forward(mj_model, data)
            mujoco.mj_fullM(mj_model, matrix, data.qM)
            return matrix.copy(), np.asarray(data.qfrc_bias, dtype=np.float64).copy()

        return evaluate_fallback


def advance_semi_implicit(
    position: FloatArray,
    velocity: FloatArray,
    generalized_force: FloatArray,
    time_step_s: float,
    dynamics_operator: Callable[
        [FloatArray, FloatArray], tuple[FloatArray, FloatArray]
    ],
) -> tuple[FloatArray, FloatArray]:
    """Advance the production articulated stepping kernel by one time step."""

    matrix, bias = dynamics_operator(position, velocity)
    acceleration = np.linalg.solve(matrix, generalized_force - bias)
    next_velocity = velocity + time_step_s * acceleration
    next_position = position + time_step_s * next_velocity
    return next_position, next_velocity


def _validate_case(
    model: SpatialModel,
    case: ForwardIntegrationCase,
    config: ArticulatedForwardContactConfig,
) -> int:
    if not isinstance(case, ForwardIntegrationCase):
        raise TypeError("case must be a ForwardIntegrationCase")
    if not isinstance(config, ArticulatedForwardContactConfig):
        raise TypeError("config must be an ArticulatedForwardContactConfig")
    scalars = (
        case.time_step_s,
        case.contact_stiffness,
        case.contact_damping,
        case.initial_club_displacement_m,
        case.initial_club_velocity_m_s,
        case.grip_span_m,
        case.hand_contact_local_x_m,
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("integration scalars must be finite")
    if case.time_step_s <= 0.0 or case.contact_stiffness <= 0.0:
        raise ValueError("step and stiffness must be positive")
    if case.contact_damping < 0.0:
        raise ValueError("damping must be nonnegative")
    if np.asarray(case.q).shape != (model.nq,) or np.asarray(case.qd).shape != (
        model.nq,
    ):
        raise ValueError("q and qd must match the articulated model dimension")
    step_count = int(round(config.duration_s / case.time_step_s))
    if not np.isclose(step_count * case.time_step_s, config.duration_s):
        raise ValueError("time_step_s must divide the configured duration")
    return step_count


def _trace_buffers(sample_count: int, nq: int) -> _TraceBuffers:
    states = np.empty((sample_count, nq))
    return _TraceBuffers(
        q=states,
        qd=np.empty_like(states),
        force=np.empty(sample_count),
        separation=np.empty(sample_count),
        virtual_power=np.empty(sample_count),
        dissipation=np.empty(sample_count),
        strain=np.empty(sample_count),
        mechanical=np.empty(sample_count),
    )


def _record_sample(
    buffers: _TraceBuffers,
    index: int,
    position: FloatArray,
    velocity: FloatArray,
    snapshot: ContactProjectionSnapshot,
    model: SpatialModel,
) -> None:
    buffers.q[index] = position
    buffers.qd[index] = velocity
    buffers.force[index] = snapshot.maximum_contact_force_n
    buffers.separation[index] = snapshot.maximum_attachment_separation_m
    buffers.virtual_power[index] = snapshot.virtual_power_residual_w
    buffers.dissipation[index] = snapshot.contact_dissipation_power_w
    buffers.strain[index] = snapshot.attachment_strain_energy_j
    buffers.mechanical[index] = mechanical_energy(model, position, velocity)


def _trace_result(
    buffers: _TraceBuffers, case: ForwardIntegrationCase
) -> dict[str, FloatArray | float]:
    cumulative = np.zeros(buffers.dissipation.size)
    cumulative[1:] = np.cumsum(
        0.5 * (buffers.dissipation[1:] + buffers.dissipation[:-1]) * case.time_step_s
    )
    total = buffers.mechanical + buffers.strain
    return {
        "time_s": np.arange(buffers.dissipation.size) * case.time_step_s,
        "q": buffers.q,
        "qd": buffers.qd,
        "maximum_contact_force_n": buffers.force,
        "maximum_attachment_separation_m": buffers.separation,
        "virtual_power_residual_w": buffers.virtual_power,
        "contact_dissipation_power_w": buffers.dissipation,
        "attachment_strain_energy_j": buffers.strain,
        "mechanical_energy_j": buffers.mechanical,
        "total_energy_j": total,
        "cumulative_dissipation_j": cumulative,
        "work_energy_residual_j": total - total[0] - cumulative,
    }


def integrate_articulated_contact(
    model: SpatialModel,
    case: ForwardIntegrationCase,
    config: ArticulatedForwardContactConfig = ArticulatedForwardContactConfig(),
) -> dict[str, FloatArray | float]:
    """Advance one engine with semi-implicit Euler and a named energy ledger."""

    step_count = _validate_case(model, case, config)
    position, velocity = np.asarray(case.q).copy(), np.asarray(case.qd).copy()
    position[14] += case.initial_club_displacement_m
    velocity[14] += case.initial_club_velocity_m_s
    contact_config = ArticulatedContactProjectionConfig(
        contact_stiffness=case.contact_stiffness,
        contact_damping=case.contact_damping,
    )
    native_operator = native_dynamics_operator(case.engine, model)
    buffers = _trace_buffers(step_count + 1, model.nq)
    for index in range(step_count + 1):
        snapshot = evaluate_contact_projection(
            model,
            position,
            velocity,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            perturb_contact=False,
            config=contact_config,
        )
        _record_sample(buffers, index, position, velocity, snapshot, model)
        if index < step_count:
            position, velocity = advance_semi_implicit(
                position,
                velocity,
                snapshot.generalized_contact_force,
                case.time_step_s,
                native_operator,
            )
    return _trace_result(buffers, case)


__all__ = [
    "ForwardIntegrationCase",
    "advance_semi_implicit",
    "integrate_articulated_contact",
    "native_dynamics_operator",
]

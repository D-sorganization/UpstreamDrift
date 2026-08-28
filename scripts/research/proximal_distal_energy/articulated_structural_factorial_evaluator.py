"""Native one-case evaluator for the prospective structural factorial."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from importlib import import_module
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    distributed_reference_lengths,
)
from scripts.research.proximal_distal_energy.articulated_forward_atlas import (
    load_forward_authority,
)
from scripts.research.proximal_distal_energy.articulated_ground import (
    ArticulatedGroundConfig,
    build_articulated_ground,
    evaluate_ground_coupled_grip,
)
from scripts.research.proximal_distal_energy.articulated_ground_forward import (
    GroundForwardConfig,
    GroundIntegrationCase,
    integrate_articulated_ground,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    finite_difference_kinematics,
    require_robotics_pinocchio,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
    build_articulated_shaft,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    NativeEngineUnavailable,
    StructuralCase,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[3]
AUTHORITY_PATHS = {
    "closed_state_npz": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "shaft_atlas_json": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_atlas.json",
    "shaft_atlas_npz": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_atlas.npz",
    "ground_atlas_json": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_ground_atlas.json",
    "ground_atlas_npz": ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_ground_atlas.npz",
}


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_authorities(plan: Mapping[str, object]) -> None:
    """Fail closed if any preregistered authority bytes have drifted."""

    identity = _mapping(plan.get("identity"), name="identity")
    expected = _mapping(identity.get("authority_sha256"), name="authority_sha256")
    if set(expected) != set(AUTHORITY_PATHS):
        raise ValueError(
            "authority SHA-256 keys do not match evaluator authority paths"
        )
    for name, path in AUTHORITY_PATHS.items():
        if expected.get(name) != _sha256(path):
            raise ValueError(f"authority SHA-256 mismatch: {name}")


def require_native_engine(engine: str) -> dict[str, str]:
    """Return the native identity or one typed unavailability outcome."""

    if engine not in {"mujoco", "pinocchio"}:
        raise ValueError("engine must be mujoco or pinocchio")
    try:
        module = import_module(engine)
    except (ImportError, OSError) as exc:
        raise NativeEngineUnavailable(
            engine=engine, detail=f"registered native module is unavailable: {exc}"
        ) from exc
    if engine == "mujoco":
        missing = [
            name for name in ("MjData", "mj_forward") if not hasattr(module, name)
        ]
        if missing:
            raise NativeEngineUnavailable(
                engine=engine,
                detail="MuJoCo is missing required native API: " + ", ".join(missing),
            )
        version = str(getattr(module, "__version__", "unknown"))
    else:
        try:
            version = require_robotics_pinocchio(module)
        except RuntimeError as exc:
            raise NativeEngineUnavailable(engine=engine, detail=str(exc)) from exc
    return {"name": engine, "version": version, "operator": "native"}


def _club_momentum_and_head_velocity(
    model: SpatialModel, q: FloatArray, qd: FloatArray
) -> tuple[FloatArray, FloatArray]:
    kinematics = forward_kinematics(model, q)
    momentum = np.zeros(3)
    head_velocity: FloatArray | None = None
    for index, body in enumerate(model.bodies):
        velocity = kinematics.body_linear_jacobian[index] @ qd
        if body.region == "club":
            momentum += body.mass_kg * velocity
        if body.name == "clubhead_mass":
            head_velocity = np.asarray(velocity, dtype=float)
    if head_velocity is None:
        raise ValueError("model has no clubhead_mass body")
    return momentum, head_velocity


def _trapz(values: FloatArray, time_s: FloatArray, end: int) -> FloatArray:
    return np.asarray(np.trapezoid(values[: end + 1], time_s[: end + 1], axis=0))


def _contact_histories(
    *,
    model: SpatialModel,
    trace: Mapping[str, NDArray[Any]],
    case: GroundIntegrationCase,
) -> tuple[FloatArray, FloatArray]:
    shaft = build_articulated_shaft(model, case.shaft)
    ground = build_articulated_ground(case.ground)
    reference = distributed_reference_lengths(
        model,
        case.q,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        config=case.grip,
    )
    q = np.asarray(trace["q"], dtype=float)
    qd = np.asarray(trace["qd"], dtype=float)
    eta_dot = np.asarray(trace["elastic_velocities"], dtype=float)
    base = np.asarray(trace["base_coordinates"], dtype=float)
    base_velocity = np.asarray(trace["base_velocities"], dtype=float)
    net_force = np.empty((q.shape[0], 3))
    power = np.empty(q.shape[0])
    for index in range(q.shape[0]):
        contact = evaluate_ground_coupled_grip(
            model,
            q[index],
            qd[index],
            eta_dot[index],
            base[index],
            base_velocity[index],
            shaft,
            ground,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            reference_lengths_m=reference,
            config=case.grip,
        )
        generalized = np.asarray(contact.generalized_contact_force, dtype=float)
        club_power = float(
            generalized[model.club_dof_indices] @ qd[index, model.club_dof_indices]
        )
        if shaft.coordinate_count:
            elastic_slice = slice(model.nq, model.nq + shaft.coordinate_count)
            club_power += float(generalized[elastic_slice] @ eta_dot[index])
        net_force[index] = contact.net_club_force_n
        power[index] = club_power
    return net_force, power


def _horizon_rows(
    *,
    model: SpatialModel,
    trace: Mapping[str, NDArray[Any]],
    contact_force: FloatArray,
    contact_power: FloatArray,
    horizons_s: tuple[float, ...],
) -> list[dict[str, object]]:
    time_s = np.asarray(trace["time_s"], dtype=float)
    q = np.asarray(trace["q"], dtype=float)
    qd = np.asarray(trace["qd"], dtype=float)
    initial_momentum, _ = _club_momentum_and_head_velocity(model, q[0], qd[0])
    rows: list[dict[str, object]] = []
    for horizon_s in horizons_s:
        end = int(np.searchsorted(time_s, horizon_s))
        if end >= time_s.size or not np.isclose(time_s[end], horizon_s):
            raise RuntimeError("registered horizon is absent from the trajectory")
        momentum, head_velocity = _club_momentum_and_head_velocity(
            model, q[end], qd[end]
        )
        speed = float(np.linalg.norm(head_velocity))
        if not math.isfinite(speed) or speed <= 1e-12:
            raise RuntimeError(
                "terminal clubhead velocity is inadequate for projection"
            )
        axis = head_velocity / speed
        impulse = _trapz(contact_force, time_s, end)
        rows.append(
            {
                "horizon_s": horizon_s,
                "final_club_translation_speed_m_s": speed,
                "club_linear_momentum_change_kg_m_s": float(
                    (momentum - initial_momentum) @ axis
                ),
                "signed_contact_impulse_n_s": float(impulse @ axis),
                "signed_contact_work_j": float(_trapz(contact_power, time_s, end)),
                "terminal_contact_dissipation_j": float(
                    -_trapz(
                        np.asarray(trace["grip_dissipation_power_w"], dtype=float),
                        time_s,
                        end,
                    )
                ),
                "terminal_shaft_dissipation_j": float(
                    -_trapz(
                        np.asarray(trace["shaft_damping_power_w"], dtype=float),
                        time_s,
                        end,
                    )
                ),
                "terminal_ground_dissipation_j": float(
                    -_trapz(
                        np.asarray(trace["ground_damping_power_w"], dtype=float),
                        time_s,
                        end,
                    )
                ),
                "peak_grip_force_n": float(
                    np.max(np.asarray(trace["maximum_station_force_n"])[: end + 1])
                ),
            }
        )
    return rows


def evaluate_structural_case(
    case: StructuralCase, plan: Mapping[str, object]
) -> dict[str, object]:
    """Execute one registered intervention and return all nested horizons."""

    if not isinstance(case, StructuralCase):
        raise TypeError("case must be a StructuralCase")
    validate_authorities(plan)
    engine = require_native_engine(case.engine)
    design = _mapping(plan.get("design"), name="design")
    steps = tuple(float(value) for value in design["time_steps_s"])
    horizons = tuple(float(value) for value in design["horizons_s"])
    authority = load_forward_authority()
    if case.source_case_index >= authority.solution_q.shape[0]:
        raise ValueError("source case index is outside the authority")
    if case.source_sample_index >= authority.solution_q.shape[1]:
        raise ValueError("source sample index is outside the authority")
    if not np.isclose(
        authority.time_s[case.source_sample_index], case.source_time_s, atol=1e-15
    ):
        raise ValueError("source time does not match the authority")
    velocity, _ = finite_difference_kinematics(
        authority.solution_q[case.source_case_index], authority.time_s
    )
    profiles = default_synthetic_profiles()
    model, metadata = build_subject_scaled_model(
        profiles[int(authority.profile_index[case.source_case_index])]
    )
    integration_case = GroundIntegrationCase(
        q=authority.solution_q[case.source_case_index, case.source_sample_index],
        qd=velocity[case.source_sample_index],
        grip_span_m=float(authority.grip_span_m[case.source_case_index]),
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        time_step_s=case.time_step_s,
        initial_club_displacement_m=0.001,
        initial_club_velocity_m_s=0.05 * case.velocity_factor,
        initial_base_displacement=(0.0, 0.0, 0.0),
        initial_base_velocity=(0.0, 0.0, 0.0),
        engine=case.engine,
        grip=DistributedGripConfig(
            station_count_per_hand=5,
            station_width_m=0.03,
            total_stiffness_n_m=1800.0,
            total_damping_n_s_m=18.0,
        ),
        shaft=ArticulatedShaftConfig(activation=case.shaft_activation),  # type: ignore[arg-type]
        ground=ArticulatedGroundConfig(activation=case.ground_activation),  # type: ignore[arg-type]
    )
    forward = GroundForwardConfig(duration_s=0.05, time_steps_s=steps)
    trace = integrate_articulated_ground(model, integration_case, forward)
    contact_force, contact_power = _contact_histories(
        model=model, trace=trace, case=integration_case
    )
    residual = np.asarray(trace["work_energy_residual_j"], dtype=float)
    energy = np.asarray(trace["total_energy_j"], dtype=float)
    normalized_residual = float(np.max(np.abs(residual)) / max(1.0, np.ptp(energy)))
    return {
        "engine": engine,
        "estimand": "within_synthetic_model_structural_intervention",
        "horizons": _horizon_rows(
            model=model,
            trace=trace,
            contact_force=contact_force,
            contact_power=contact_power,
            horizons_s=horizons,
        ),
        "numerical": {
            "normalized_work_energy_residual": normalized_residual,
            "maximum_virtual_power_residual_w": float(
                np.max(np.abs(trace["virtual_power_residual_w"]))
            ),
            "maximum_shaft_power_residual_w": float(
                np.max(np.abs(trace["shaft_power_residual_w"]))
            ),
            "maximum_ground_power_residual_w": float(
                np.max(np.abs(trace["ground_power_residual_w"]))
            ),
            "maximum_small_deflection_ratio": float(
                np.max(trace["small_deflection_ratio"])
            ),
            "maximum_twist_angle_rad": float(np.max(np.abs(trace["twist_angle_rad"]))),
            "maximum_base_translation_m": float(
                np.max(np.linalg.norm(trace["base_translation_m"], axis=1))
            ),
            "maximum_base_pitch_rad": float(np.max(np.abs(trace["base_pitch_rad"]))),
        },
        "claim_boundary": {
            "causal_scope": "declared synthetic pathway intervention only",
            "equipment_optimization": False,
            "human_or_coaching_inference": False,
        },
    }


__all__ = [
    "evaluate_structural_case",
    "require_native_engine",
    "validate_authorities",
]

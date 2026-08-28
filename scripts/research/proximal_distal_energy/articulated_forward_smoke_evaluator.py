"""Native rigid-case adapter for the preregistered #9153 serial smoke."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
from importlib import import_module
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_contact_projection import (
    ArticulatedContactProjectionConfig,
)
from scripts.research.proximal_distal_energy.articulated_forward_atlas import (
    build_forward_integration_case,
    load_forward_authority,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    CaseCheckpoint,
    NativeEngineUnavailable,
    StudyCase,
    run_serial_cases,
)
from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    ArticulatedForwardContactConfig,
    ForwardVariant,
)
from scripts.research.proximal_distal_energy.articulated_distributed_event_attribution import (
    attribute_distributed_contact_trajectory,
)
from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
    DistributedIntegrationCase,
)
from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    require_robotics_pinocchio,
)
from scripts.research.proximal_distal_energy.articulated_rigid_forward_attribution import (
    attribute_rigid_contact_trajectory,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _number(mapping: Mapping[str, Any], name: str, *, positive: bool = True) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path_and_hash(manifest: Mapping[str, object]) -> Path:
    identity = _mapping(manifest.get("identity"), name="identity")
    expected_hash = identity.get("source_data_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("identity.source_data_sha256 must be a SHA-256")
    design = _mapping(manifest.get("design"), name="design")
    authority = _mapping(design.get("model_authority"), name="design.model_authority")
    relative_path = authority.get("path")
    expected_path = (
        "docs/research/proximal_distal_energy_transfer/data/"
        "subject_scaled_closed_contact.npz"
    )
    if relative_path != expected_path:
        raise ValueError("model authority path does not match the registered source")
    path = REPO_ROOT / expected_path
    if _sha256(path) != expected_hash:
        raise ValueError("source-data SHA-256 does not match the registered authority")
    return path


def require_registered_native_engine(engine: str) -> dict[str, str]:
    """Return a qualified engine identity or one typed unavailable condition."""

    if engine not in {"mujoco", "pinocchio"}:
        raise ValueError("engine must be 'mujoco' or 'pinocchio'")
    try:
        module = import_module(engine)
    except (ImportError, OSError) as exc:
        raise NativeEngineUnavailable(
            engine=engine,
            detail=f"registered native module is unavailable: {exc}",
        ) from exc
    if engine == "mujoco":
        required = ("MjData", "mj_forward")
        missing = tuple(name for name in required if not hasattr(module, name))
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


def _variant_parameter_row(design: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    rows = design.get("variant_parameters")
    if not isinstance(rows, list):
        raise ValueError("design.variant_parameters must be a list")
    matches = [
        row for row in rows if isinstance(row, Mapping) and row.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError("case variant must have exactly one registered parameter row")
    return matches[0]


def _variant(design: Mapping[str, Any], name: str) -> ForwardVariant:
    row = _variant_parameter_row(design, name)
    return ForwardVariant(
        name=name,
        stiffness_factor=_number(row, "stiffness_factor"),
        damping_factor=_number(row, "damping_factor"),
        displacement_factor=_number(row, "displacement_factor", positive=False),
        velocity_factor=_number(row, "velocity_factor", positive=False),
    )


def _configuration(manifest: Mapping[str, object]) -> ArticulatedForwardContactConfig:
    design = _mapping(manifest.get("design"), name="design")
    contact = _mapping(design.get("contact_law"), name="design.contact_law")
    if contact.get("name") != "bilateral_kelvin_voigt_point_attachment_always_active":
        raise ValueError("contact law does not match the registered rigid smoke")
    if contact.get("unilateral_contact") is not False:
        raise ValueError("rigid smoke must retain bilateral always-active attachment")
    return ArticulatedForwardContactConfig(
        duration_s=_number(design, "duration_s"),
        time_steps_s=tuple(float(value) for value in design["time_steps_s"]),
        contact_stiffness=_number(contact, "contact_stiffness_n_m"),
        contact_damping=_number(contact, "contact_damping_n_s_m"),
        initial_club_displacement_m=_number(contact, "initial_club_displacement_m"),
        initial_club_velocity_m_s=_number(contact, "initial_club_velocity_m_s"),
    )


def _nullable(values: np.ndarray) -> list[Any]:
    array = np.asarray(values)
    if array.ndim == 1:
        return [float(value) if np.isfinite(value) else None for value in array]
    return [_nullable(row) for row in array]


def _club_outcomes(
    model: SpatialModel, q: np.ndarray, qd: np.ndarray
) -> dict[str, object]:
    names = tuple(body.name for body in model.bodies)
    if names.count("clubhead_mass") != 1:
        raise ValueError("model must contain exactly one clubhead_mass body")
    body_index = names.index("clubhead_mass")
    kinematics = forward_kinematics(model, q[-1])
    velocity = kinematics.body_linear_jacobian[body_index] @ qd[-1]
    speed = float(np.linalg.norm(velocity))
    if not math.isfinite(speed) or speed <= 1.0e-12:
        raise ValueError("terminal clubhead velocity is inadequate for direction")
    direction = velocity / speed
    face_axis = kinematics.body_rotation[body_index] @ np.array([1.0, 0.0, 0.0])
    cosine = float(np.clip(face_axis @ direction, -1.0, 1.0))
    return {
        "clubhead_speed_m_s": speed,
        "clubhead_direction_world": direction.tolist(),
        "club_face_proxy_axis_world": face_axis.tolist(),
        "face_path_proxy_deg": float(np.degrees(np.arccos(cosine))),
        "face_path_proxy_definition": (
            "angle between club local +x axis and clubhead velocity in the world frame"
        ),
    }


def evaluate_rigid_smoke_case(
    case: StudyCase, manifest: Mapping[str, object]
) -> dict[str, object]:
    """Evaluate one frozen rigid smoke case without changing its trajectory."""

    if not isinstance(case, StudyCase):
        raise TypeError("case must be a StudyCase")
    _source_path_and_hash(manifest)
    engine = require_registered_native_engine(case.engine)
    authority = load_forward_authority()
    if case.source_case_index >= authority.solution_q.shape[0]:
        raise ValueError("source case index is outside the registered authority")
    if case.source_sample_index >= authority.solution_q.shape[1]:
        raise ValueError("source sample index is outside the registered authority")
    actual_time = float(authority.time_s[case.source_sample_index])
    if not np.isclose(actual_time, case.source_time_s, rtol=0.0, atol=1.0e-15):
        raise ValueError("source state time does not match the registered authority")
    profiles = default_synthetic_profiles()
    profile_index = int(authority.profile_index[case.source_case_index])
    model, metadata = build_subject_scaled_model(profiles[profile_index])
    config = _configuration(manifest)
    design = _mapping(manifest.get("design"), name="design")
    variant = _variant(design, case.variant)
    integration_case = build_forward_integration_case(
        authority=authority,
        config=config,
        variant=variant,
        case=case.source_case_index,
        sample=case.source_sample_index,
        time_step_s=case.time_step_s,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        engine=case.engine,
    )
    contact = ArticulatedContactProjectionConfig(
        contact_stiffness=integration_case.contact_stiffness,
        contact_damping=integration_case.contact_damping,
    )
    evidence = attribute_rigid_contact_trajectory(
        model, integration_case, config, contact
    )
    attribution = evidence.attribution
    trace = evidence.trace
    residual = np.asarray(trace["work_energy_residual_j"], dtype=np.float64)
    total_energy = np.asarray(trace["total_energy_j"], dtype=np.float64)
    normalized_energy = float(np.max(np.abs(residual)) / max(1.0, np.ptp(total_energy)))
    pointwise = float(np.max(np.abs(evidence.pointwise_force_closure_residual)))
    tolerances = _mapping(manifest.get("tolerances"), name="tolerances")
    momentum_tolerance = _number(tolerances, "momentum_relative")
    work_tolerance = _number(tolerances, "work_relative")
    pointwise_tolerance = _number(tolerances, "pointwise_force_closure")
    energy_tolerance = _number(tolerances, "trajectory_energy_relative")
    failures: list[str] = []
    if attribution.momentum_closure_relative_residual > momentum_tolerance:
        failures.append("momentum_closure")
    if attribution.work_closure_relative_residual > work_tolerance:
        failures.append("work_closure")
    if pointwise > pointwise_tolerance:
        failures.append("pointwise_force_closure")
    if normalized_energy > energy_tolerance:
        failures.append("trajectory_energy_closure")
    return {
        "source_state": {
            "source_case_index": case.source_case_index,
            "source_sample_index": case.source_sample_index,
            "source_time_s": case.source_time_s,
        },
        "engine": engine,
        "variant": case.variant,
        "time_step_s": case.time_step_s,
        "estimand": "same_trajectory_descriptive_attribution",
        "contributions": {
            "names": list(attribution.contribution_names),
            "continuous_impulses": attribution.continuous_impulses.tolist(),
            "generalized_work_j": attribution.generalized_work_j.tolist(),
            "kinetic_transport_work_j": attribution.kinetic_transport_work_j,
            "mass_transport_impulse": attribution.transport_impulse.tolist(),
            "event_impulse": attribution.total_event_impulse.tolist(),
            "event_work_j": attribution.total_event_work_j,
            "impulse_shares": _nullable(attribution.impulse_shares),
            "impulse_share_adequacy": attribution.impulse_share_adequacy.tolist(),
            "work_shares": _nullable(attribution.work_shares),
            "work_share_adequate": attribution.work_share_adequate,
        },
        "closure": {
            "momentum_relative_residual": (
                attribution.momentum_closure_relative_residual
            ),
            "work_relative_residual": attribution.work_closure_relative_residual,
            "pointwise_force_residual": pointwise,
            "trajectory_energy_relative_residual": normalized_energy,
            "failure_codes": failures,
            "passes_registered_tolerances": not failures,
        },
        "outcomes": {
            **_club_outcomes(
                model,
                np.asarray(trace["q"], dtype=np.float64),
                np.asarray(trace["qd"], dtype=np.float64),
            ),
            "maximum_contact_load_n": float(
                np.max(np.asarray(trace["maximum_contact_force_n"], dtype=np.float64))
            ),
        },
        "events": {
            "model": "bilateral_always_active_rigid_attachment",
            "count": 0,
            "times_s": [],
        },
        "claim_boundary": {
            "human_or_coaching_inference": False,
            "smoke_state_representative_of_humans": False,
            "causal_counterfactual": False,
        },
    }


def _distributed_grip(
    design: Mapping[str, Any], integration_case: Any, variant_row: Mapping[str, Any]
) -> DistributedGripConfig:
    law = _mapping(
        design.get("distributed_contact_law"),
        name="design.distributed_contact_law",
    )
    if law.get("name") != "distributed_tension_with_regularized_coulomb_limit":
        raise ValueError("distributed contact law is not registered")
    if law.get("static_stick_modeled") is not False:
        raise ValueError("regularized distributed smoke cannot claim static stick")
    station_count = law.get("station_count_per_hand")
    if not isinstance(station_count, int):
        raise ValueError("station_count_per_hand must be an integer")
    return DistributedGripConfig(
        station_count_per_hand=station_count,
        station_width_m=(
            _number(law, "station_width_m", positive=False)
            * float(variant_row.get("station_width_factor", 1.0))
        ),
        total_stiffness_n_m=float(integration_case.contact_stiffness),
        total_damping_n_s_m=float(integration_case.contact_damping),
        tangential_damping_n_s_m=_number(law, "tangential_damping_n_s_m"),
        friction_coefficient=(
            _number(law, "friction_coefficient", positive=False)
            * float(variant_row.get("friction_coefficient_factor", 1.0))
        ),
        slack_distance_m=(
            _number(law, "slack_distance_m", positive=False)
            * float(variant_row.get("slack_distance_factor", 1.0))
        ),
    )


def _serialized_events(events: tuple[Any, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        row: dict[str, object] = {
            "kind": event.kind.value,
            "time_s": event.time_s,
            "left_index": event.left_index,
            "right_index": event.right_index,
            "hand_index": event.hand_index,
            "station_index": event.station_index,
            "position": event.position.tolist(),
            "velocity": event.velocity.tolist(),
            "final_bracket_width_s": event.final_bracket_width_s,
            "path_model": event.path_model,
        }
        if hasattr(event, "gap_residual_m"):
            row["gap_residual_m"] = event.gap_residual_m
        else:
            row["friction_margin_residual_n"] = event.friction_margin_residual_n
            row["static_stick_modeled"] = event.static_stick_modeled
        rows.append(row)
    return rows


def evaluate_distributed_smoke_case(
    case: StudyCase, manifest: Mapping[str, object]
) -> dict[str, object]:
    """Evaluate one distributed, event-aligned, same-trajectory smoke case."""

    if not isinstance(case, StudyCase):
        raise TypeError("case must be a StudyCase")
    _source_path_and_hash(manifest)
    engine = require_registered_native_engine(case.engine)
    authority = load_forward_authority()
    if case.source_case_index >= authority.solution_q.shape[0]:
        raise ValueError("source case index is outside the registered authority")
    if case.source_sample_index >= authority.solution_q.shape[1]:
        raise ValueError("source sample index is outside the registered authority")
    actual_time = float(authority.time_s[case.source_sample_index])
    if not np.isclose(actual_time, case.source_time_s, rtol=0.0, atol=1.0e-15):
        raise ValueError("source state time does not match the registered authority")
    profiles = default_synthetic_profiles()
    profile_index = int(authority.profile_index[case.source_case_index])
    model, metadata = build_subject_scaled_model(profiles[profile_index])
    rigid_config = _configuration(manifest)
    design = _mapping(manifest.get("design"), name="design")
    variant_row = _variant_parameter_row(design, case.variant)
    variant = _variant(design, case.variant)
    rigid_case = build_forward_integration_case(
        authority=authority,
        config=rigid_config,
        variant=variant,
        case=case.source_case_index,
        sample=case.source_sample_index,
        time_step_s=case.time_step_s,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        engine=case.engine,
    )
    grip = _distributed_grip(design, rigid_case, variant_row)
    full_state_velocity_factor = float(
        variant_row.get("full_state_velocity_factor", 1.0)
    )
    if not math.isfinite(full_state_velocity_factor):
        raise ValueError("full_state_velocity_factor must be finite")
    distributed_case = DistributedIntegrationCase(
        q=rigid_case.q,
        qd=rigid_case.qd,
        grip_span_m=rigid_case.grip_span_m,
        hand_contact_local_x_m=rigid_case.hand_contact_local_x_m,
        time_step_s=rigid_case.time_step_s,
        initial_club_displacement_m=rigid_case.initial_club_displacement_m,
        initial_club_velocity_m_s=rigid_case.initial_club_velocity_m_s,
        engine=rigid_case.engine,
        grip=grip,
        initial_state_velocity_factor=full_state_velocity_factor,
    )
    distributed_config = DistributedForwardConfig(
        duration_s=rigid_config.duration_s,
        time_steps_s=rigid_config.time_steps_s,
    )
    evidence = attribute_distributed_contact_trajectory(
        model=model,
        case=distributed_case,
        config=distributed_config,
    )
    attribution = evidence.attribution
    trace = evidence.trace
    energy_residual = np.asarray(trace["work_energy_residual_j"], dtype=np.float64)
    total_energy = np.asarray(trace["total_energy_j"], dtype=np.float64)
    normalized_energy = float(
        np.max(np.abs(energy_residual)) / max(1.0, np.ptp(total_energy))
    )
    pointwise = float(np.max(np.abs(evidence.pointwise_force_closure_residual)))
    tolerances = _mapping(manifest.get("tolerances"), name="tolerances")
    failures: list[str] = []
    if attribution.momentum_closure_relative_residual > _number(
        tolerances, "momentum_relative"
    ):
        failures.append("momentum_closure")
    if attribution.work_closure_relative_residual > _number(
        tolerances, "work_relative"
    ):
        failures.append("work_closure")
    if pointwise > _number(tolerances, "pointwise_force_closure"):
        failures.append("pointwise_force_closure")
    if normalized_energy > _number(tolerances, "trajectory_energy_relative"):
        failures.append("trajectory_energy_closure")
    event_rows = _serialized_events(evidence.events)
    event_counts = Counter(row["kind"] for row in event_rows)
    positions = np.asarray(trace["q"], dtype=np.float64)
    velocities = np.asarray(trace["qd"], dtype=np.float64)
    return {
        "source_state": {
            "source_case_index": case.source_case_index,
            "source_sample_index": case.source_sample_index,
            "source_time_s": case.source_time_s,
        },
        "engine": engine,
        "variant": case.variant,
        "time_step_s": case.time_step_s,
        "estimand": "same_trajectory_descriptive_attribution",
        "contact_model": {
            "name": "distributed_tension_with_regularized_coulomb_limit",
            "station_count_per_hand": grip.station_count_per_hand,
            "station_width_m": grip.station_width_m,
            "friction_coefficient": grip.friction_coefficient,
            "slack_distance_m": grip.slack_distance_m,
            "static_stick_modeled": False,
        },
        "contributions": {
            "names": list(attribution.contribution_names),
            "continuous_impulses": attribution.continuous_impulses.tolist(),
            "generalized_work_j": attribution.generalized_work_j.tolist(),
            "kinetic_transport_work_j": attribution.kinetic_transport_work_j,
            "mass_transport_impulse": attribution.transport_impulse.tolist(),
            "event_impulse": attribution.total_event_impulse.tolist(),
            "event_work_j": attribution.total_event_work_j,
            "impulse_shares": _nullable(attribution.impulse_shares),
            "work_shares": _nullable(attribution.work_shares),
        },
        "closure": {
            "momentum_relative_residual": attribution.momentum_closure_relative_residual,
            "work_relative_residual": attribution.work_closure_relative_residual,
            "pointwise_force_residual": pointwise,
            "trajectory_energy_relative_residual": normalized_energy,
            "failure_codes": failures,
            "passes_registered_tolerances": not failures,
        },
        "outcomes": {
            **_club_outcomes(model, positions, velocities),
            "maximum_contact_load_n": float(
                np.max(np.asarray(trace["maximum_station_force_n"], dtype=np.float64))
            ),
        },
        "events": {
            "path_model": "linear_state_interpolant",
            "discrete_impulse_modeled": False,
            "count": len(event_rows),
            "counts_by_kind": dict(event_counts),
            "records": event_rows,
        },
        "claim_boundary": {
            "human_or_coaching_inference": False,
            "static_stick_inference": False,
            "causal_counterfactual": False,
            "smoke_state_representative_of_humans": False,
        },
    }


def run_registered_rigid_smoke(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    checkpoint_dir: Path,
) -> tuple[CaseCheckpoint, ...]:
    """Run or resume the exact serial rigid smoke through atomic checkpoints."""

    return run_serial_cases(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
        evaluator=lambda case: evaluate_rigid_smoke_case(case, manifest),
    )


__all__ = [
    "evaluate_distributed_smoke_case",
    "evaluate_rigid_smoke_case",
    "require_registered_native_engine",
    "run_registered_rigid_smoke",
]

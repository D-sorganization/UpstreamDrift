"""Native rigid-case adapter for the preregistered #9153 serial smoke."""

from __future__ import annotations

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


def _variant(design: Mapping[str, Any], name: str) -> ForwardVariant:
    rows = design.get("variant_parameters")
    if not isinstance(rows, list):
        raise ValueError("design.variant_parameters must be a list")
    matches = [
        row for row in rows if isinstance(row, Mapping) and row.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError("case variant must have exactly one registered parameter row")
    row = matches[0]
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
    "evaluate_rigid_smoke_case",
    "require_registered_native_engine",
    "run_registered_rigid_smoke",
]

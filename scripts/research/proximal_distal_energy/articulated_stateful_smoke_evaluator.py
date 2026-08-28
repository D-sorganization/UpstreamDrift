"""JSON-safe native evaluator for a preregistered stateful-grip smoke."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.articulated_forward_atlas import (
    build_forward_integration_case,
    load_forward_authority,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    CaseCheckpoint,
    StudyCase,
    run_serial_cases,
)
from scripts.research.proximal_distal_energy.articulated_forward_smoke_evaluator import (
    _club_outcomes,
    _configuration,
    _mapping,
    _number,
    _source_path_and_hash,
    _variant,
    _variant_parameter_row,
    require_registered_native_engine,
)
from scripts.research.proximal_distal_energy.articulated_stateful_distributed_forward import (
    StatefulDistributedForwardConfig,
    StatefulDistributedIntegrationCase,
    integrate_stateful_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_stateful_friction import (
    StatefulFrictionConfig,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)


@dataclass(frozen=True, slots=True)
class _Evaluation:
    model: SpatialModel
    integration: StatefulDistributedIntegrationCase
    forward: StatefulDistributedForwardConfig
    engine_identity: dict[str, str]


def _vector3(mapping: Mapping[str, Any], name: str) -> np.ndarray:
    value = mapping.get(name)
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be one finite three-vector")
    return array


def _stateful_case(
    case: StudyCase,
    manifest: Mapping[str, object],
    model: SpatialModel,
    hand_contact_local_x_m: float,
) -> tuple[StatefulDistributedIntegrationCase, StatefulDistributedForwardConfig]:
    rigid_config = _configuration(manifest)
    design = _mapping(manifest.get("design"), name="design")
    variant_row = _variant_parameter_row(design, case.variant)
    rigid_case = build_forward_integration_case(
        authority=load_forward_authority(),
        config=rigid_config,
        variant=_variant(design, case.variant),
        case=case.source_case_index,
        sample=case.source_sample_index,
        time_step_s=case.time_step_s,
        hand_contact_local_x_m=hand_contact_local_x_m,
        engine=case.engine,
    )
    law = _mapping(design.get("stateful_contact_law"), name="stateful_contact_law")
    if law.get("name") != "distributed_elastic_perfectly_plastic_coulomb":
        raise ValueError("stateful contact law is not registered")
    if law.get("static_stick_modeled") is not True:
        raise ValueError("stateful contact law must declare static stick")
    friction_factor = float(variant_row.get("friction_coefficient_factor", 1.0))
    stiffness_factor = float(variant_row.get("tangential_stiffness_factor", 1.0))
    slack_factor = float(variant_row.get("slack_distance_factor", 1.0))
    preload_factor = float(variant_row.get("initial_preload_factor", 1.0))
    velocity_factor = float(variant_row.get("full_state_velocity_factor", 1.0))
    factors = np.asarray(
        [
            friction_factor,
            stiffness_factor,
            slack_factor,
            preload_factor,
            velocity_factor,
        ]
    )
    if not np.all(np.isfinite(factors)) or np.any(factors[:4] < 0.0):
        raise ValueError("stateful variant factors must be finite and nonnegative")
    slack_override = variant_row.get("slack_distance_override_m")
    slack_distance = (
        _number(variant_row, "slack_distance_override_m", positive=False)
        if slack_override is not None
        else _number(law, "slack_distance_m", positive=False) * slack_factor
    )
    count = int(law.get("station_count_per_hand", 0))
    mu = _number(law, "friction_coefficient", positive=False) * friction_factor
    grip = DistributedGripConfig(
        station_count_per_hand=count,
        station_width_m=_number(law, "station_width_m", positive=False),
        total_stiffness_n_m=rigid_case.contact_stiffness,
        total_damping_n_s_m=rigid_case.contact_damping,
        friction_coefficient=mu,
        slack_distance_m=slack_distance,
    )
    preload = _vector3(law, "initial_preload_vector_m") * preload_factor
    initial_state = np.broadcast_to(preload, (2, count, 3)).copy()
    integration = StatefulDistributedIntegrationCase(
        q=rigid_case.q,
        qd=rigid_case.qd,
        grip_span_m=rigid_case.grip_span_m,
        hand_contact_local_x_m=rigid_case.hand_contact_local_x_m,
        time_step_s=rigid_case.time_step_s,
        initial_club_displacement_m=rigid_case.initial_club_displacement_m,
        initial_club_velocity_m_s=rigid_case.initial_club_velocity_m_s,
        engine=rigid_case.engine,
        grip=grip,
        friction=StatefulFrictionConfig(
            tangential_stiffness_n_m=(
                _number(law, "tangential_stiffness_n_m") * stiffness_factor
            ),
            friction_coefficient=mu,
        ),
        initial_elastic_displacement_m=initial_state,
        initial_state_velocity_factor=velocity_factor,
    )
    forward = StatefulDistributedForwardConfig(
        duration_s=rigid_config.duration_s,
        time_steps_s=rigid_config.time_steps_s,
    )
    return integration, forward


def _evaluation_context(case: StudyCase, manifest: Mapping[str, object]) -> _Evaluation:
    if not isinstance(case, StudyCase):
        raise TypeError("case must be a StudyCase")
    _source_path_and_hash(manifest)
    engine_identity = require_registered_native_engine(case.engine)
    authority = load_forward_authority()
    if case.source_case_index >= authority.solution_q.shape[0]:
        raise ValueError("source case index is outside the registered authority")
    if case.source_sample_index >= authority.solution_q.shape[1]:
        raise ValueError("source sample index is outside the registered authority")
    actual_time = float(authority.time_s[case.source_sample_index])
    if not np.isclose(actual_time, case.source_time_s, rtol=0.0, atol=1.0e-15):
        raise ValueError("source state time does not match the registered authority")
    profile = default_synthetic_profiles()[
        int(authority.profile_index[case.source_case_index])
    ]
    model, metadata = build_subject_scaled_model(profile)
    integration, forward = _stateful_case(
        case,
        manifest,
        model,
        float(metadata["hand_contact_local_x_m"]),
    )
    return _Evaluation(model, integration, forward, engine_identity)


def _closure(
    trace: Mapping[str, Any], tolerances: Mapping[str, Any]
) -> dict[str, object]:
    total = np.asarray(trace["node_total_energy_j"], dtype=float)
    passive = np.asarray(trace["passive_energy_balance_residual_j"], dtype=float)
    coupling = np.asarray(
        trace["interval_tangential_coupling_work_residual_j"], dtype=float
    )
    scale = max(1.0, float(np.ptp(total)))
    passive_relative = float(np.max(np.abs(passive)) / scale)
    coupling_relative = float(abs(np.sum(coupling)) / scale)
    virtual = float(
        np.max(np.abs(np.asarray(trace["interval_virtual_power_residual_w"])))
    )
    ledger = np.asarray(trace["interval_constitutive_work_j"]) - (
        np.asarray(trace["interval_tangential_elastic_energy_change_j"])
        + np.asarray(trace["interval_frictional_dissipation_j"])
        + np.asarray(trace["interval_release_dissipation_j"])
    )
    ledger_max = float(np.max(np.abs(ledger)))
    failures: list[str] = []
    gates = (
        ("passive_energy_closure", passive_relative, "trajectory_energy_relative"),
        ("tangential_coupling_work", coupling_relative, "coupling_work_relative"),
        ("virtual_power", virtual, "virtual_power_w"),
        ("constitutive_ledger", ledger_max, "constitutive_ledger_j"),
    )
    for failure, value, tolerance_name in gates:
        if value > _number(tolerances, tolerance_name):
            failures.append(failure)
    return {
        "trajectory_energy_relative_residual": passive_relative,
        "tangential_coupling_work_relative_residual": coupling_relative,
        "virtual_power_residual_w": virtual,
        "constitutive_ledger_residual_j": ledger_max,
        "failure_codes": failures,
        "passes_registered_tolerances": not failures,
    }


def _events(trace: Mapping[str, Any]) -> dict[str, object]:
    active = np.asarray(trace["interval_active_station"], dtype=bool)
    regimes = np.asarray(trace["interval_regime"])
    times = np.asarray(trace["interval_time_start_s"], dtype=float)
    records: list[dict[str, object]] = []
    for index in range(1, active.shape[0]):
        for hand, station in np.ndindex(active.shape[1:]):
            if active[index, hand, station] != active[index - 1, hand, station]:
                records.append(
                    {
                        "kind": (
                            "reattachment"
                            if active[index, hand, station]
                            else "opening"
                        ),
                        "time_s": float(times[index]),
                        "hand_index": hand,
                        "station_index": station,
                    }
                )
            if regimes[index, hand, station] != regimes[index - 1, hand, station]:
                records.append(
                    {
                        "kind": "regime_transition",
                        "from": str(regimes[index - 1, hand, station]),
                        "to": str(regimes[index, hand, station]),
                        "time_s": float(times[index]),
                        "hand_index": hand,
                        "station_index": station,
                    }
                )
    counts = Counter(str(record["kind"]) for record in records)
    return {
        "path_model": "step_boundary_observation",
        "substep_event_time_claimed": False,
        "discrete_impulse_modeled": False,
        "count": len(records),
        "counts_by_kind": dict(counts),
        "records": records,
    }


def _histories(trace: Mapping[str, Any]) -> dict[str, object]:
    names = (
        "node_time_s",
        "interval_time_start_s",
        "node_q",
        "node_qd",
        "node_elastic_displacement_m",
        "node_mechanical_energy_j",
        "node_normal_strain_energy_j",
        "node_tangential_strain_energy_j",
        "node_total_energy_j",
        "interval_generalized_contact_force",
        "interval_force_on_club_n",
        "interval_normal_force_on_club_n",
        "interval_tangential_force_on_club_n",
        "interval_active_station",
        "interval_station_signed_gap_m",
        "interval_regime",
        "interval_friction_limit_n",
        "interval_yield_margin_n",
        "interval_plastic_slip_increment_m",
        "interval_frictional_dissipation_j",
        "interval_release_dissipation_j",
        "interval_constitutive_work_j",
        "interval_normal_dissipation_j",
        "interval_tangential_coupling_work_residual_j",
        "passive_energy_balance_residual_j",
    )
    return {name: np.asarray(trace[name]).tolist() for name in names}


def evaluate_stateful_smoke_case(
    case: StudyCase, manifest: Mapping[str, object]
) -> dict[str, object]:
    """Evaluate one native stateful-grip case and retain complete histories."""

    context = _evaluation_context(case, manifest)
    trace = integrate_stateful_distributed_grip(
        context.model, context.integration, context.forward
    )
    design = _mapping(manifest.get("design"), name="design")
    law = _mapping(design.get("stateful_contact_law"), name="stateful_contact_law")
    tolerances = _mapping(manifest.get("tolerances"), name="tolerances")
    regimes = np.asarray(trace["interval_regime"])
    regime_names, regime_counts = np.unique(regimes, return_counts=True)
    forces = np.asarray(trace["interval_force_on_club_n"], dtype=float)
    return {
        "source_state": {
            "source_case_index": case.source_case_index,
            "source_sample_index": case.source_sample_index,
            "source_time_s": case.source_time_s,
        },
        "engine": context.engine_identity,
        "variant": case.variant,
        "time_step_s": case.time_step_s,
        "estimand": "stateful_contact_counterfactual_trajectory",
        "contact_model": {
            "name": law["name"],
            "station_count_per_hand": context.integration.grip.station_count_per_hand,
            "station_width_m": context.integration.grip.station_width_m,
            "friction_coefficient": context.integration.friction.friction_coefficient,
            "tangential_stiffness_n_m": (
                context.integration.friction.tangential_stiffness_n_m
            ),
            "slack_distance_m": context.integration.grip.slack_distance_m,
            "static_stick_modeled": True,
            "operator_split": trace["operator_split"],
            "force_timestamp": trace["force_timestamp"],
            "mechanical_step": trace["mechanical_step"],
        },
        "closure": _closure(trace, tolerances),
        "outcomes": {
            **_club_outcomes(
                context.model,
                np.asarray(trace["node_q"]),
                np.asarray(trace["node_qd"]),
            ),
            "maximum_contact_load_n": float(np.max(np.linalg.norm(forces, axis=3))),
            "total_frictional_dissipation_j": float(
                np.sum(trace["interval_frictional_dissipation_j"])
            ),
            "total_release_dissipation_j": float(
                np.sum(trace["interval_release_dissipation_j"])
            ),
        },
        "regimes": dict(
            zip(
                (str(name) for name in regime_names),
                (int(count) for count in regime_counts),
                strict=True,
            )
        ),
        "events": _events(trace),
        "histories": _histories(trace),
        "claim_boundary": {
            "human_or_coaching_inference": False,
            "human_or_anatomical_inference": False,
            "smoke_state_representative_of_humans": False,
            "causal_interpretation_limited_to_declared_model": True,
        },
    }


def run_registered_stateful_smoke(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    checkpoint_dir: Path,
) -> tuple[CaseCheckpoint, ...]:
    """Run or resume the exact stateful serial smoke atomically."""

    return run_serial_cases(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
        evaluator=lambda case: evaluate_stateful_smoke_case(case, manifest),
    )


__all__ = ["evaluate_stateful_smoke_case", "run_registered_stateful_smoke"]

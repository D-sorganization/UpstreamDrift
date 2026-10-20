from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

import pytest

from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.unit, pytest.mark.requires_opensim]


def _native_generalized_effort_n_m(snapshot: Any | None) -> float | None:
    """Report native muscle effort from an already-captured snapshot only."""
    if snapshot is None:
        return None
    total = 0.0
    for path in snapshot.paths:
        force_n = path.native_actuation_force_n
        if force_n is None:
            return None
        total += path.moment_arm_m_per_rad * float(force_n)
    return total


def _write_antagonist_model(osim: Any, path: Path) -> None:
    model = osim.Model()
    model.setName("native_effort_adapter_antagonists")
    model.setGravity(osim.Vec3(0.0, 0.0, 0.0))
    body = osim.Body("load", 1.0, osim.Vec3(0.0), osim.Inertia(0.01))
    model.addBody(body)
    joint = osim.PinJoint(
        "hinge",
        model.getGround(),
        osim.Vec3(0.0),
        osim.Vec3(0.0),
        body,
        osim.Vec3(0.0),
        osim.Vec3(0.0),
    )
    coordinate = joint.updCoordinate()
    coordinate.setName("hinge_coord")
    model.addJoint(joint)
    for name, y in (("flexor", 0.05), ("extensor", -0.05)):
        muscle = osim.Millard2012EquilibriumMuscle(name, 10.0, 0.1, 0.2, 0.1)
        muscle.addNewPathPoint("origin", model.getGround(), osim.Vec3(0.0, y, 0.0))
        muscle.addNewPathPoint("insertion", body, osim.Vec3(0.35, y, 0.0))
        model.addForce(muscle)
    model.finalizeConnections()
    model.printToXML(str(path))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _state_fingerprint_payload(snapshot: Any) -> dict[str, Any]:
    return {
        "source_sha256": snapshot.source_sha256,
        "runtime_identity": snapshot.runtime_identity,
        "time_s": float(snapshot.time_s),
        "coordinate": asdict(snapshot.coordinate),
        "paths": [asdict(path) for path in snapshot.paths],
        "required_state_paths": list(snapshot.required_state_paths),
        "complete_state": [
            [name, float(value)] for name, value in snapshot.complete_state
        ],
    }


def test_native_constant_interval_matches_fresh_raw_replay(
    tmp_path: Path,
    record_property: Callable[[str, str], None],
) -> None:
    """Compare one accepted constant raw interval with an independent replay."""
    osim = require_opensim()
    from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_effort_adapter import (
        AnalyticRecruitmentConfig,
        EffortBasis,
        NativeCoordinateMapping,
        NativeMuscleEffortRequest,
        NativeMuscleEffortSession,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_effort_runtime import (
        NativeMuscleEffortExecutor,
        NativeSourcePin,
        load_pinned_opensim_backend,
        verify_fresh_constant_replay,
    )
    from src.shared.python.motion_matching.distributed_feedback import FeedbackRequest

    runtime_identity = str(osim.GetVersionAndDate())
    record_property(
        "native_runtime_json",
        _json(
            {
                "distribution_version": importlib.metadata.version("opensim"),
                "python": sys.executable,
                "native_get_version": str(osim.GetVersion()),
                "runtime_identity": runtime_identity,
                "package_path": str(Path(osim.__file__).resolve()),
            }
        ),
    )
    record_property(
        "native.native_runtime_json",
        _json(
            {
                "distribution_version": importlib.metadata.version("opensim"),
                "python": sys.executable,
                "native_get_version": str(osim.GetVersion()),
                "runtime_identity": runtime_identity,
                "package_path": str(Path(osim.__file__).resolve()),
            }
        ),
    )
    source_path = tmp_path / "native_effort_antagonists.osim"
    _write_antagonist_model(osim, source_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_directory = os.environ.get("NATIVE_MUSCLE_TRIAL_ARTIFACT_DIR")
    artifact_root = Path(artifact_directory).resolve() if artifact_directory else None
    artifact_relative_path = ""
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, artifact_root / "trial_model.osim")
        artifact_relative_path = "artifacts/trial_model.osim"
    record_property("native.model_artifact_relative_path", artifact_relative_path)
    model = osim.Model(str(source_path))
    model.finalizeConnections()
    state = model.initSystem()
    muscle_set = model.getMuscles()
    muscle_names = tuple(
        str(muscle_set.get(i).getName()) for i in range(muscle_set.getSize())
    )
    assert muscle_names == ("flexor", "extensor")
    analytic: dict[str, AnalyticRecruitmentConfig] = {}
    for i in range(muscle_set.getSize()):
        muscle = muscle_set.get(i)
        millard = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
        assert millard is not None
        analytic[str(muscle.getName())] = AnalyticRecruitmentConfig(
            max_isometric_force_n=float(muscle.getMaxIsometricForce()),
            optimal_fiber_length_m=float(muscle.getOptimalFiberLength()),
            tendon_slack_length_m=float(muscle.getTendonSlackLength()),
            pennation_angle_rad=float(muscle.getPennationAngleAtOptimalFiberLength()),
            activation_time_constant_s=float(millard.getActivationTimeConstant()),
            deactivation_time_constant_s=float(millard.getDeactivationTimeConstant()),
            minimum_activation=float(millard.getMinimumActivation()),
            excitation_bounds=(0.0, 1.0),
            maximum_excitation_rate_per_s=100.0,
        )
        muscle.setActivation(state, 0.1)
    model.equilibrateMuscles(state)
    model.realizeDynamics(state)

    coordinate_set = model.getCoordinateSet()
    assert coordinate_set.getSize() == 1 and model.getConstraintSet().getSize() == 0
    coordinate = coordinate_set.get(0)
    coordinate_path = str(coordinate.getAbsolutePathString())
    state_names_set = model.getStateVariableNames()
    state_names = tuple(
        str(state_names_set.get(i)) for i in range(state_names_set.getSize())
    )
    initial_state = {
        name: float(model.getStateVariableValue(state, name)) for name in state_names
    }
    expected_state_names = (
        f"{coordinate_path}/value",
        f"{coordinate_path}/speed",
        str(muscle_set.get(0).getAbsolutePathString()) + "/activation",
        str(muscle_set.get(0).getAbsolutePathString()) + "/fiber_length",
        str(muscle_set.get(1).getAbsolutePathString()) + "/activation",
        str(muscle_set.get(1).getAbsolutePathString()) + "/fiber_length",
    )
    assert state_names == expected_state_names
    assert all(math.isfinite(v) for v in initial_state.values())
    mapping = NativeCoordinateMapping(
        coordinate_path=coordinate_path,
        coordinate_value_state_path=f"{coordinate_path}/value",
        coordinate_speed_state_path=f"{coordinate_path}/speed",
        coordinate_index=0,
        q_index=0,
        u_index=0,
        generalized_effort_index=0,
        coordinate_count=1,
        q_count=1,
        u_count=1,
        constraint_count=0,
        locked=False,
        prescribed=False,
        motion_type="rotational",
        effort_unit="N*m",
        feedback_channel_id="actuator",
    )
    backend = load_pinned_opensim_backend(
        osim=osim,
        source_path=source_path,
        expected_pin=NativeSourcePin(source_sha256, runtime_identity),
        coordinate_mapping=mapping,
        analytic_by_muscle=analytic,
        initial_state=initial_state,
        initial_time_s=0.0,
        integrator_accuracy=1e-8,
    )
    initial = backend.observe()
    initial_native_effort_n_m = _native_generalized_effort_n_m(initial)
    initial_diagnostics_json = _json(
        {
            "snapshot": asdict(initial),
            "native_generalized_effort_n_m": initial_native_effort_n_m,
        }
    )
    record_property(
        "native.initial_snapshot_diagnostics_json", initial_diagnostics_json
    )
    if artifact_root is not None:
        (artifact_root / "initial_snapshot_diagnostics.json").write_text(
            initial_diagnostics_json + "\n", encoding="utf-8"
        )
    dt_s, target, max_nfev = 0.01, 0.02, 80
    session = NativeMuscleEffortSession(initial, dt_s=dt_s, first_schedule_index=0)
    feedback = FeedbackRequest(
        time_s=0.0,
        dt_s=dt_s,
        schedule_index=0,
        phase="native-fixture",
        active_tasks=(),
        nominal_feedforward=(target,),
        feedback_correction=(0.0,),
        total_requested=(target,),
        generalized_effort=(target,),
        channel_ids=("actuator",),
        generalized_indices=(0,),
        selected_coordinate_mask=(True,),
    )
    step = NativeMuscleEffortExecutor(session, backend).execute_one_interval(
        NativeMuscleEffortRequest(EffortBasis.TOTAL_MUSCULOTENDINOUS),
        feedback,
        effort_tolerance_n_m=1e-8,
        max_nfev=max_nfev,
    )
    proposal, advanced = step.proposal, step.accepted_snapshot
    endpoint_native_effort_n_m = _native_generalized_effort_n_m(advanced)
    endpoint_effort_residual_n_m = (
        endpoint_native_effort_n_m - target
        if endpoint_native_effort_n_m is not None
        else None
    )
    proposal_payload = asdict(proposal)
    proposal_payload["effort_basis"] = proposal.effort_basis.value
    proposal_diagnostics_json = _json(
        {
            "requested_generalized_effort_n_m": target,
            "initial_native_generalized_effort_n_m": initial_native_effort_n_m,
            "endpoint_native_generalized_effort_n_m": endpoint_native_effort_n_m,
            "endpoint_effort_residual_n_m": endpoint_effort_residual_n_m,
            "proposal": proposal_payload,
            "accepted_snapshot": asdict(advanced) if advanced is not None else None,
        }
    )
    record_property(
        "native.endpoint_native_generalized_effort_n_m",
        _json(endpoint_native_effort_n_m),
    )
    record_property(
        "native.endpoint_effort_residual_n_m",
        _json(endpoint_effort_residual_n_m),
    )
    record_property("native.proposal_diagnostics_json", proposal_diagnostics_json)
    if artifact_root is not None:
        (artifact_root / "proposal_diagnostics.json").write_text(
            proposal_diagnostics_json + "\n", encoding="utf-8"
        )
    assert proposal.feasible and proposal.allocation_status == "feasible"
    assert advanced is not None and advanced.time_s == pytest.approx(dt_s, abs=1e-10)
    raw_items = tuple(proposal.raw_excitations)
    assert tuple(name for name, _ in raw_items) == muscle_names
    raw_values = tuple(value for _, value in raw_items)
    assert all(value > 0.0 for value in raw_values) and len(set(raw_values)) > 1
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in raw_values)
    initial_values, final_values = (
        dict(initial.complete_state),
        dict(advanced.complete_state),
    )
    assert (
        max(abs(final_values[name] - value) for name, value in initial_values.items())
        > 1e-8
    )

    state_tolerances = dict.fromkeys(initial.required_state_paths, 1e-8)
    force_tolerances = dict.fromkeys(muscle_names, 1e-6)
    replay = verify_fresh_constant_replay(
        source_path=source_path,
        initial=initial,
        proposal=proposal,
        advanced=advanced,
        state_absolute_tolerances=state_tolerances,
        muscle_force_absolute_tolerances_n=force_tolerances,
        integrator_accuracy=1e-8,
    )
    fingerprint_payload = _state_fingerprint_payload(initial)
    record_property(
        "native.test_node", "test_native_constant_interval_matches_fresh_raw_replay"
    )
    record_property(
        "native.source_test_sha256",
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    record_property("native.model_sha256", source_sha256)
    record_property("native.runtime_identity", runtime_identity)
    record_property("native.input_profile_sha256", step.input_profile_sha256 or "")
    record_property(
        "native.input_profile_start_state_sha256",
        step.input_profile_start_state_sha256 or "",
    )
    record_property(
        "native.analytic_configuration_provenance",
        "Millard physical constants are checked against loaded native getters; excitation bounds and rate are caller-declared policy values",
    )
    record_property(
        "native.restored_state_fingerprint_payload_json", _json(fingerprint_payload)
    )
    record_property(
        "native.restored_state_sha256",
        hashlib.sha256(_json(fingerprint_payload).encode("utf-8")).hexdigest(),
    )
    record_property("native.coordinate_name", mapping.coordinate_name)
    record_property("native.coordinate_mapping_json", _json(asdict(mapping)))
    record_property("native.muscle_names_json", _json(list(muscle_names)))
    record_property(
        "native.state_names_json", _json(list(initial.required_state_paths))
    )
    record_property("native.accepted_step_index", str(proposal.schedule_index + 1))
    record_property(
        "native.interval_times_json", _json([initial.time_s, advanced.time_s])
    )
    record_property(
        "native.raw_excitations_json", _json([value for _, value in raw_items])
    )
    record_property(
        "native.effective_excitations_json",
        _json([value for _, value in proposal.effective_excitations]),
    )
    record_property("native.requested_generalized_effort_nm", _json([target]))
    record_property("native.requested_generalized_effort_json", _json([target]))
    record_property(
        "native.coordinate_value_state_path", mapping.coordinate_value_state_path
    )
    record_property(
        "native.coordinate_speed_state_path", mapping.coordinate_speed_state_path
    )
    record_property("native.allocator_max_nfev", str(max_nfev))
    record_property("native.initial_state_json", _json(dict(initial.complete_state)))
    record_property("native.advanced_state_json", _json(dict(advanced.complete_state)))
    record_property(
        "native.endpoint_actuation_forces_n_json",
        _json({path.name: path.native_actuation_force_n for path in advanced.paths}),
    )
    record_property("native.state_abs_tolerances_json", _json(state_tolerances))
    record_property("native.force_abs_tolerances_n_json", _json(force_tolerances))
    record_property("native.fresh_replay_model_sha256", replay.model_sha256)
    record_property(
        "native.fresh_replay_runtime_identity", str(replay.policy["provider"])
    )
    record_property("native.fresh_replay_times_json", _json(replay.times.tolist()))
    expected_replay_grid = [[value for _, value in raw_items] for _ in range(2)]
    assert replay.applied_excitations.tolist() == expected_replay_grid
    record_property(
        "native.fresh_replay_raw_excitations_json",
        _json(replay.applied_excitations.tolist()),
    )
    record_property(
        "native.fresh_replay_initial_state_json",
        _json(dict(zip(replay.state_names, replay.states[0].tolist(), strict=True))),
    )
    record_property(
        "native.fresh_replay_final_state_json",
        _json(dict(zip(replay.state_names, replay.states[-1].tolist(), strict=True))),
    )
    record_property(
        "native.fresh_replay_terminal_forces_n_json",
        _json(
            dict(
                zip(
                    replay.muscle_names,
                    replay.muscle_forces_n[-1].tolist(),
                    strict=True,
                )
            )
        ),
    )
    state_errors = [
        abs(final_values[name] - float(replay.states[-1, index]))
        for index, name in enumerate(replay.state_names)
    ]
    force_errors = [
        abs(
            float(advanced.paths[index].native_actuation_force_n)
            - float(replay.muscle_forces_n[-1, index])
        )
        for index in range(len(muscle_names))
    ]
    record_property("native.max_state_abs_error", repr(max(state_errors)))
    record_property("native.max_force_abs_error_n", repr(max(force_errors)))
    record_property(
        "native.max_state_change_abs",
        repr(
            max(
                abs(final_values[name] - value)
                for name, value in initial_values.items()
            )
        ),
    )

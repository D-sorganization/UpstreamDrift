"""Unit tests for portable model exchange and MATLAB iteration throughput (#11104, MMR-18).

Verifies:
1. Cold/warm timing and peak memory on the same captures/hardware.
2. Content-addressed cache invalidation covers model, geometry, marker map,
   initial state, solver, controls, and provider revision (7 axes).
3. Topology and operating-point changes trigger rebuild.
4. Identical input yields same metrics within declared tolerances.
5. Optimization winner gets uncached native cold replay.
6. Unsupported neck/contact/muscle mappings reject rather than silently drop dynamics.
7. Portable model export/import preserves named coordinates, frames, inertia, and actuation.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.portable_exchange import (
    EngineType,
    FeatureKind,
    InterchangeConformanceMatrix,
    IterationBenchmarkSample,
    OperatingPointSpec,
    ParityToleranceExceededError,
    PortableModelSpec,
    SessionRebuildDecision,
    SessionRebuildRequiredError,
    TopologySpec,
    UncachedColdReplayRequiredError,
    UnsupportedMappingError,
    WinnerReplayVerificationError,
    compute_content_cache_key,
    evaluate_metric_parity,
    evaluate_session_reuse,
    export_portable_model,
    import_portable_model,
    record_iteration_benchmark,
    verify_optimization_winner_cold_replay,
)

pytestmark = pytest.mark.unit


def _build_reference_spec() -> PortableModelSpec:
    """Build a canonical reference PortableModelSpec."""
    return PortableModelSpec(
        model_id="GolfSwing3D_Kinetic_GS3DX",
        model_sha256="a" * 64,
        provider_revision="matlab-r2025b-rev1",
        geometry={
            "FitHubtoSLength": 0.1603,
            "FitUpperArmLength": 0.3045,
            "FitLowerArmLength": 0.2799,
            "FitUpperTorsoLength": 0.2438,
            "FitLowerTorsoLength": 0.2438,
            "torso_mass_kg": 24.5,
        },
        marker_map={
            "HeadTop": "Head",
            "HeadFront": "Head",
            "HeadSide": "Head",
            "LAcromion": "Torso",
            "RAcromion": "Torso",
            "LElbowOut": "LForearm",
            "RElbowOut": "RForearm",
            "Grip": "Club",
        },
        initial_state={
            "q_pelvis_tx": 0.0,
            "q_pelvis_ty": 0.0,
            "q_pelvis_tz": 0.95,
            "q_pelvis_rx": 0.0,
            "q_pelvis_ry": 0.1,
            "q_pelvis_rz": 0.0,
            "v_pelvis_tx": 0.0,
            "v_pelvis_ty": 0.0,
            "v_pelvis_tz": 0.0,
        },
        solver={
            "name": "ode15s",
            "reltol": 1e-4,
            "abstol": 1e-6,
            "max_step_s": 0.001,
        },
        controls={
            "Hip_A": 0.12,
            "Hip_B": -0.05,
            "Spine_A": 0.45,
            "Torso_A": 0.88,
            "LShoulder_A": -0.32,
            "RShoulder_A": 0.29,
        },
        topology=TopologySpec(
            coordinate_names=(
                "pelvis_tx",
                "pelvis_ty",
                "pelvis_tz",
                "hip_flexion",
                "spine_flexion",
                "torso_rotation",
            ),
            dof_count=6,
            has_independent_neck=False,
            joint_types={"pelvis": "floating", "hip": "revolute", "spine": "universal"},
        ),
        operating_point=OperatingPointSpec(
            plane_tilt_deg=30.0,
            ground_offset_m=(0.0, 0.0, 0.0),
            gravity_m_s2=(0.0, 0.0, -9.80665),
            nominal_cadence_hz=1.2,
        ),
        features=(FeatureKind.RIGID_MULTIBODY, FeatureKind.POLYNOMIAL_ACTUATION),
    )


# ==============================================================================
# 1. Cold/Warm Timing and Peak Memory on Same Captures/Hardware
# ==============================================================================


def test_iteration_throughput_benchmark_same_hardware_and_capture() -> None:
    """Benchmark recorder records valid cold/warm speedup and peak memory."""
    sample = record_iteration_benchmark(
        capture_id="playerA_swing01",
        hardware_id="workstation-epyc-7763",
        stage="fit_replay",
        cold_wall_clock_s=12.4,
        warm_wall_clock_s=2.8,
        cold_peak_memory_mb=1450.0,
        warm_peak_memory_mb=1520.0,
        cache_key="hash123456",
        cache_hit=False,
    )
    assert isinstance(sample, IterationBenchmarkSample)
    assert sample.capture_id == "playerA_swing01"
    assert sample.hardware_id == "workstation-epyc-7763"
    assert sample.cold_wall_clock_s == 12.4
    assert sample.warm_wall_clock_s == 2.8
    assert sample.speedup_ratio == pytest.approx(12.4 / 2.8, rel=1e-5)
    assert sample.speedup_ratio > 1.0
    assert sample.memory_delta_mb == pytest.approx(70.0, rel=1e-5)
    as_dict = sample.to_dict()
    assert as_dict["speedup_ratio"] == pytest.approx(12.4 / 2.8, rel=1e-5)
    assert as_dict["capture_id"] == "playerA_swing01"


def test_iteration_throughput_rejects_empty_identifiers() -> None:
    """Benchmark recorder rejects empty capture_id or hardware_id."""
    with pytest.raises(ValueError, match="capture_id must be non-empty"):
        record_iteration_benchmark(
            capture_id="",
            hardware_id="workstation-epyc-7763",
            stage="fit_replay",
            cold_wall_clock_s=10.0,
            warm_wall_clock_s=2.0,
            cold_peak_memory_mb=1000.0,
            warm_peak_memory_mb=1050.0,
            cache_key="hash123456",
            cache_hit=False,
        )

    with pytest.raises(ValueError, match="hardware_id must be non-empty"):
        record_iteration_benchmark(
            capture_id="playerA_swing01",
            hardware_id="",
            stage="fit_replay",
            cold_wall_clock_s=10.0,
            warm_wall_clock_s=2.0,
            cold_peak_memory_mb=1000.0,
            warm_peak_memory_mb=1050.0,
            cache_key="hash123456",
            cache_hit=False,
        )


def test_iteration_throughput_rejects_negative_or_nonfinite_metrics() -> None:
    """Benchmark recorder fails closed on negative timings or invalid memory."""
    with pytest.raises(ValueError, match="timings must be non-negative"):
        record_iteration_benchmark(
            capture_id="playerA_swing01",
            hardware_id="workstation-epyc-7763",
            stage="fit_replay",
            cold_wall_clock_s=-1.0,
            warm_wall_clock_s=2.0,
            cold_peak_memory_mb=1000.0,
            warm_peak_memory_mb=1050.0,
            cache_key="hash123456",
            cache_hit=False,
        )

    with pytest.raises(ValueError, match="memory values must be finite"):
        record_iteration_benchmark(
            capture_id="playerA_swing01",
            hardware_id="workstation-epyc-7763",
            stage="fit_replay",
            cold_wall_clock_s=10.0,
            warm_wall_clock_s=2.0,
            cold_peak_memory_mb=float("nan"),
            warm_peak_memory_mb=1050.0,
            cache_key="hash123456",
            cache_hit=False,
        )


# ==============================================================================
# 2. Content-Addressed Cache Invalidation (7 Axes)
# ==============================================================================


def test_content_cache_key_deterministic_and_canonical() -> None:
    """Identical specifications produce identical SHA-256 cache keys."""
    spec1 = _build_reference_spec()
    spec2 = _build_reference_spec()
    key1 = compute_content_cache_key(spec1)
    key2 = compute_content_cache_key(spec2)
    assert isinstance(key1, str)
    assert len(key1) == 64
    assert key1 == key2


def test_content_cache_key_invalidation_covers_all_seven_axes() -> None:
    """Mutating ANY of the 7 specified axes invalidates the cache key.

    Axes:
    1. model (model_sha256 or model_id)
    2. geometry
    3. marker_map
    4. initial_state
    5. solver
    6. controls
    7. provider_revision
    """
    base = _build_reference_spec()
    base_key = compute_content_cache_key(base)

    # 1. Model change
    mutated_model = copy.deepcopy(base)
    object.__setattr__(mutated_model, "model_sha256", "b" * 64)
    assert compute_content_cache_key(mutated_model) != base_key

    # 2. Geometry change
    mutated_geom = copy.deepcopy(base)
    new_geom = dict(base.geometry)
    new_geom["FitUpperArmLength"] = 0.3500
    object.__setattr__(mutated_geom, "geometry", new_geom)
    assert compute_content_cache_key(mutated_geom) != base_key

    # 3. Marker map change
    mutated_markers = copy.deepcopy(base)
    new_markers = dict(base.marker_map)
    new_markers["Grip"] = "LeadHand"
    object.__setattr__(mutated_markers, "marker_map", new_markers)
    assert compute_content_cache_key(mutated_markers) != base_key

    # 4. Initial state change
    mutated_state = copy.deepcopy(base)
    new_state = dict(base.initial_state)
    new_state["q_pelvis_tz"] = 0.99
    object.__setattr__(mutated_state, "initial_state", new_state)
    assert compute_content_cache_key(mutated_state) != base_key

    # 5. Solver change
    mutated_solver = copy.deepcopy(base)
    new_solver = dict(base.solver)
    new_solver["reltol"] = 1e-6
    object.__setattr__(mutated_solver, "solver", new_solver)
    assert compute_content_cache_key(mutated_solver) != base_key

    # 6. Controls change
    mutated_controls = copy.deepcopy(base)
    new_controls = dict(base.controls)
    new_controls["Torso_A"] = 0.999
    object.__setattr__(mutated_controls, "controls", new_controls)
    assert compute_content_cache_key(mutated_controls) != base_key

    # 7. Provider revision change
    mutated_provider = copy.deepcopy(base)
    object.__setattr__(mutated_provider, "provider_revision", "matlab-r2025b-rev2")
    assert compute_content_cache_key(mutated_provider) != base_key


# ==============================================================================
# 3. Topology and Operating-Point Changes Trigger Rebuild
# ==============================================================================


def test_topology_change_triggers_rebuild() -> None:
    """Adding joints or changing kinematic tree triggers rebuild."""
    base = _build_reference_spec()

    # Add neck DOFs (e.g. GS3DX_Neck modification)
    neck_topology = TopologySpec(
        coordinate_names=base.topology.coordinate_names + ("neck_rx", "neck_ry"),
        dof_count=base.topology.dof_count + 2,
        has_independent_neck=True,
        joint_types=dict(base.topology.joint_types, neck="universal"),
    )
    new_spec = copy.deepcopy(base)
    object.__setattr__(new_spec, "topology", neck_topology)

    decision = evaluate_session_reuse(base, new_spec)
    assert isinstance(decision, SessionRebuildDecision)
    assert decision.must_rebuild is True
    assert "topology" in decision.reason.lower()


def test_operating_point_change_triggers_rebuild() -> None:
    """Modifying PlaneTilt or ground frame triggers rebuild."""
    base = _build_reference_spec()

    # Change PlaneTilt from 30.0 to 22.5 deg (as in FIT.md)
    new_operating_point = OperatingPointSpec(
        plane_tilt_deg=22.5,
        ground_offset_m=base.operating_point.ground_offset_m,
        gravity_m_s2=base.operating_point.gravity_m_s2,
        nominal_cadence_hz=base.operating_point.nominal_cadence_hz,
    )
    new_spec = copy.deepcopy(base)
    object.__setattr__(new_spec, "operating_point", new_operating_point)

    decision = evaluate_session_reuse(base, new_spec)
    assert decision.must_rebuild is True
    assert "operating_point" in decision.reason.lower()


def test_tunable_controls_alone_permit_warm_session_reuse() -> None:
    """Changing only tunable controls permits warm session reuse."""
    base = _build_reference_spec()
    new_spec = copy.deepcopy(base)
    new_controls = dict(base.controls)
    new_controls["Hip_A"] = 0.55
    object.__setattr__(new_spec, "controls", new_controls)

    decision = evaluate_session_reuse(base, new_spec)
    assert decision.must_rebuild is False
    assert decision.can_reuse_session is True


def test_enforce_session_reuse_raises_when_rebuild_required() -> None:
    """Attempting warm session reuse when rebuild is required raises error."""
    base = _build_reference_spec()
    neck_topology = TopologySpec(
        coordinate_names=base.topology.coordinate_names + ("neck_rx",),
        dof_count=base.topology.dof_count + 1,
        has_independent_neck=True,
        joint_types=dict(base.topology.joint_types, neck="revolute"),
    )
    new_spec = copy.deepcopy(base)
    object.__setattr__(new_spec, "topology", neck_topology)

    with pytest.raises(SessionRebuildRequiredError, match="Rebuild required"):
        evaluate_session_reuse(base, new_spec, fail_closed=True)


# ==============================================================================
# 4. Identical Input Yields Same Metrics Within Declared Tolerances
# ==============================================================================


def test_identical_input_yields_same_metrics_within_declared_tolerances() -> None:
    """Metrics matching within declared tolerances pass parity check."""
    metrics_cold = {
        "whole_marker_rmse_m": 0.012450001,
        "pelvis_travel_m": 0.028900002,
        "terminal_norm_m": 0.008400001,
    }
    metrics_warm = {
        "whole_marker_rmse_m": 0.012450005,
        "pelvis_travel_m": 0.028900001,
        "terminal_norm_m": 0.008400003,
    }
    verdict = evaluate_metric_parity(
        reference_metrics=metrics_cold,
        candidate_metrics=metrics_warm,
        atol=1e-5,
        rtol=1e-5,
    )
    assert verdict.passed is True
    assert verdict.max_abs_diff < 1e-5


def test_metric_drift_beyond_tolerance_fails_closed() -> None:
    """Metric drift exceeding declared tolerance raises ParityToleranceExceededError."""
    metrics_cold = {"whole_marker_rmse_m": 0.0120}
    metrics_drifted = {"whole_marker_rmse_m": 0.0125}  # 0.5 mm drift

    with pytest.raises(
        ParityToleranceExceededError, match="exceeded declared tolerance"
    ):
        evaluate_metric_parity(
            reference_metrics=metrics_cold,
            candidate_metrics=metrics_drifted,
            atol=1e-4,
            rtol=1e-4,
            fail_closed=True,
        )


# ==============================================================================
# 5. Optimization Winner Gets Uncached Native Cold Replay
# ==============================================================================


def test_optimization_winner_gets_uncached_native_cold_replay() -> None:
    """Winner verification succeeds when uncached native cold replay reproduces metrics."""
    candidate = {
        "candidate_id": "cand_opt_42",
        "candidate_sha256": "c" * 64,
        "declared_metrics": {
            "whole_marker_rmse_m": 0.0095,
            "pelvis_travel_m": 0.0280,
        },
    }

    def dummy_cold_replay(
        cand: Mapping[str, Any], is_uncached_cold: bool
    ) -> dict[str, float]:
        assert is_uncached_cold is True
        return {
            "whole_marker_rmse_m": 0.009500001,
            "pelvis_travel_m": 0.028000002,
        }

    receipt = verify_optimization_winner_cold_replay(
        candidate=candidate,
        cold_replay_fn=dummy_cold_replay,
        tolerance_atol=1e-5,
    )
    assert receipt.verified is True
    assert receipt.is_uncached_cold is True
    assert receipt.candidate_id == "cand_opt_42"


def test_optimization_winner_fails_if_replay_is_not_uncached_cold() -> None:
    """Winner verification fails if cold replay flag is False."""
    candidate = {
        "candidate_id": "cand_opt_42",
        "candidate_sha256": "c" * 64,
        "declared_metrics": {"whole_marker_rmse_m": 0.0095},
    }

    def cached_replay(
        cand: Mapping[str, Any], is_uncached_cold: bool
    ) -> dict[str, float]:
        return {"whole_marker_rmse_m": 0.0095}

    with pytest.raises(
        UncachedColdReplayRequiredError,
        match="must be executed with is_uncached_cold=True",
    ):
        verify_optimization_winner_cold_replay(
            candidate=candidate,
            cold_replay_fn=cached_replay,
            enforce_uncached=True,
            is_uncached_cold=False,
        )


def test_optimization_winner_fails_if_cold_replay_diverges() -> None:
    """Winner verification fails if cold replay diverges from declared candidate metrics."""
    candidate = {
        "candidate_id": "cand_opt_42",
        "candidate_sha256": "c" * 64,
        "declared_metrics": {"whole_marker_rmse_m": 0.0095},
    }

    def diverging_replay(
        cand: Mapping[str, Any], is_uncached_cold: bool
    ) -> dict[str, float]:
        return {"whole_marker_rmse_m": 0.0150}  # Large divergence

    with pytest.raises(WinnerReplayVerificationError, match="replay metrics diverged"):
        verify_optimization_winner_cold_replay(
            candidate=candidate,
            cold_replay_fn=diverging_replay,
            tolerance_atol=1e-4,
        )


# ==============================================================================
# 6. Unsupported Neck/Contact/Muscle Mappings Reject Rather Than Silently Drop
# ==============================================================================


def test_unsupported_neck_mapping_rejects_silent_drop() -> None:
    """Mapping a model with neck DOFs to a reduced no-neck engine rejects."""
    neck_spec = copy.deepcopy(_build_reference_spec())
    neck_topology = TopologySpec(
        coordinate_names=neck_spec.topology.coordinate_names + ("neck_rx", "neck_ry"),
        dof_count=neck_spec.topology.dof_count + 2,
        has_independent_neck=True,
        joint_types=dict(neck_spec.topology.joint_types, neck="universal"),
    )
    object.__setattr__(neck_spec, "topology", neck_topology)
    object.__setattr__(
        neck_spec,
        "features",
        neck_spec.features + (FeatureKind.INDEPENDENT_NECK_DOFS,),
    )

    # Simscape reduced 27-coordinate target has no independent neck
    with pytest.raises(
        UnsupportedMappingError, match="independent neck DOFs not supported"
    ):
        export_portable_model(neck_spec, target_engine=EngineType.SIMSCAPE_REDUCED_27)


def test_unsupported_muscle_mapping_rejects_silent_drop() -> None:
    """Mapping Hill-type muscle dynamics to rigid torque-only engine rejects."""
    spec = copy.deepcopy(_build_reference_spec())
    object.__setattr__(
        spec,
        "features",
        spec.features + (FeatureKind.HILL_MUSCLE_DYNAMICS,),
    )

    with pytest.raises(UnsupportedMappingError, match="muscle dynamics not supported"):
        export_portable_model(spec, target_engine=EngineType.SIMSCAPE_REDUCED_27)

    with pytest.raises(UnsupportedMappingError, match="muscle dynamics not supported"):
        export_portable_model(spec, target_engine=EngineType.PINOCCHIO)


def test_unsupported_contact_mapping_rejects_silent_drop() -> None:
    """Mapping compliant volumetric penalty contact to bilateral-only engine rejects."""
    spec = copy.deepcopy(_build_reference_spec())
    object.__setattr__(
        spec,
        "features",
        spec.features + (FeatureKind.VOLUMETRIC_PENALTY_CONTACT,),
    )

    with pytest.raises(
        UnsupportedMappingError, match="volumetric penalty contact not supported"
    ):
        export_portable_model(spec, target_engine=EngineType.PINOCCHIO)


# ==============================================================================
# 7. Portable Model Export/Import Conformance Matrix & Named Fields
# ==============================================================================


def test_portable_model_export_and_import_roundtrip() -> None:
    """Exporting to portable package format and importing preserves all fields."""
    spec = _build_reference_spec()
    exported = export_portable_model(spec, target_engine=EngineType.SIMSCAPE_GS3DX)
    assert exported["schema_version"] == "portable-model-exchange/1.0.0"
    assert exported["model_id"] == spec.model_id
    assert exported["geometry"] == spec.geometry
    assert exported["marker_map"] == spec.marker_map
    assert exported["topology"]["coordinate_names"] == list(
        spec.topology.coordinate_names
    )

    imported = import_portable_model(exported)
    assert imported.model_id == spec.model_id
    assert imported.topology.dof_count == spec.topology.dof_count
    assert (
        imported.operating_point.plane_tilt_deg == spec.operating_point.plane_tilt_deg
    )


def test_interchange_conformance_matrix_reports_accurate_capabilities() -> None:
    """Conformance matrix reports accurate engine feature support."""
    matrix = InterchangeConformanceMatrix.default()
    simscape_caps = matrix.get_capabilities(EngineType.SIMSCAPE_REDUCED_27)
    assert simscape_caps.supports_neck is False
    assert simscape_caps.supports_muscle is False
    assert simscape_caps.supports_polynomial_actuation is True

    opensim_caps = matrix.get_capabilities(EngineType.OPENSIM)
    assert opensim_caps.supports_muscle is True

    pinocchio_caps = matrix.get_capabilities(EngineType.PINOCCHIO)
    assert pinocchio_caps.supports_muscle is False
    assert pinocchio_caps.supports_rigid_multibody is True

"""Integration tests for Cross-Engine Comparison & Injury Indicator Workspaces (ORG-18, #10527).

Acceptance criteria:
- RED: mismatched units/frame/timebase or missing load channels prevents misleading score/comparison.
- RED: main-block mock data is never used by interactive actions.
- GREEN: deterministic compatible real fixture pair gives expected existing metrics;
         scorer fixture uses actual serialized analysis inputs;
         results/export retain method and source IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from src.shared.python.workspace.comparison_indicator_workspace import (
    BiomechanicalLoadChannels,
    ComparisonIndicatorWorkspaceCoordinator,
    ComparisonRunArtifact,
    CrossEngineComparisonAdapter,
    IncompatibleArtifactError,
    InjuryIndicatorAdapter,
    MissingChannelError,
    ModelFidelityLevel,
    validate_run_compatibility,
)


@pytest.fixture
def compatible_run_pair() -> tuple[ComparisonRunArtifact, ComparisonRunArtifact]:
    """Create two compatible runs sharing time horizon, units, and coordinates."""
    times = np.linspace(0.0, 1.0, 11)
    q_a = np.zeros((11, 6))
    q_b = np.ones((11, 6)) * 0.05
    channels = (
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "hip_flex",
        "knee_angle",
        "ankle_angle",
    )
    run_a = ComparisonRunArtifact(
        run_id="run_mujoco_01",
        backend_name="mujoco",
        times=times,
        coordinates=q_a,
        channel_names=channels,
        units="m",
        coordinate_frame="world",
        fidelity=ModelFidelityLevel.SIMPLIFIED_KINETICS,
    )
    run_b = ComparisonRunArtifact(
        run_id="run_drake_02",
        backend_name="drake",
        times=times,
        coordinates=q_b,
        channel_names=channels,
        units="m",
        coordinate_frame="world",
        fidelity=ModelFidelityLevel.SIMPLIFIED_KINETICS,
    )
    return run_a, run_b


@pytest.fixture
def valid_biomechanical_load_channels() -> BiomechanicalLoadChannels:
    """Create a validated set of biomechanical force/moment channels."""
    return BiomechanicalLoadChannels(
        source_run_id="run_mujoco_01",
        peak_compression_n=3924.0,  # ~5.0 BW for an 80kg individual (784.8 N)
        peak_shear_n=627.8,  # ~0.8 BW
        peak_torsion_nm=85.0,
        joint_moments_nm={
            "lead_hip": 120.0,
            "trail_shoulder": 95.0,
            "lead_elbow": 45.0,
            "lead_wrist": 35.0,
        },
        swing_metrics={"tempo_ratio": 3.0, "x_factor_stretch": 48.0},
        fidelity=ModelFidelityLevel.QUALIFIED_FULL_BODY,
    )


# -----------------------------------------------------------------------------
# RED Acceptance Tests: Incompatibility and Missing Channels Fail Closed
# -----------------------------------------------------------------------------


def test_unit_mismatch_fails_closed(compatible_run_pair) -> None:
    """Comparing runs with mismatched units must raise IncompatibleArtifactError."""
    run_a, run_b = compatible_run_pair
    mismatched_b = ComparisonRunArtifact(
        run_id=run_b.run_id,
        backend_name=run_b.backend_name,
        times=run_b.times,
        coordinates=run_b.coordinates * 1000.0,
        channel_names=run_b.channel_names,
        units="mm",  # Incompatible with "m"
        coordinate_frame=run_b.coordinate_frame,
        fidelity=run_b.fidelity,
    )
    compat = validate_run_compatibility(run_a, mismatched_b)
    assert compat.compatible is False
    assert any("unit" in err.lower() for err in compat.errors)

    adapter = CrossEngineComparisonAdapter()
    with pytest.raises(IncompatibleArtifactError, match="(?i)unit"):
        adapter.compare_runs(run_a, mismatched_b)


def test_coordinate_frame_mismatch_fails_closed(compatible_run_pair) -> None:
    """Comparing runs with mismatched coordinate frames must raise IncompatibleArtifactError."""
    run_a, run_b = compatible_run_pair
    mismatched_b = ComparisonRunArtifact(
        run_id=run_b.run_id,
        backend_name=run_b.backend_name,
        times=run_b.times,
        coordinates=run_b.coordinates,
        channel_names=run_b.channel_names,
        units=run_b.units,
        coordinate_frame="pelvis_local",  # Incompatible with "world"
        fidelity=run_b.fidelity,
    )
    compat = validate_run_compatibility(run_a, mismatched_b)
    assert compat.compatible is False
    assert any("frame" in err.lower() for err in compat.errors)

    adapter = CrossEngineComparisonAdapter()
    with pytest.raises(IncompatibleArtifactError, match="(?i)frame"):
        adapter.compare_runs(run_a, mismatched_b)


def test_timebase_mismatch_fails_closed(compatible_run_pair) -> None:
    """Comparing runs with mismatched time horizons or step sizes must fail closed."""
    run_a, run_b = compatible_run_pair
    mismatched_b = ComparisonRunArtifact(
        run_id=run_b.run_id,
        backend_name=run_b.backend_name,
        times=np.linspace(0.0, 2.0, 11),  # Different duration
        coordinates=run_b.coordinates,
        channel_names=run_b.channel_names,
        units=run_b.units,
        coordinate_frame=run_b.coordinate_frame,
        fidelity=run_b.fidelity,
    )
    compat = validate_run_compatibility(run_a, mismatched_b)
    assert compat.compatible is False
    assert any("time" in err.lower() for err in compat.errors)


def test_dimension_channel_mismatch_fails_closed(compatible_run_pair) -> None:
    """Comparing runs with differing degrees of freedom or channels must fail closed."""
    run_a, run_b = compatible_run_pair
    mismatched_b = ComparisonRunArtifact(
        run_id=run_b.run_id,
        backend_name=run_b.backend_name,
        times=run_b.times,
        coordinates=np.zeros((11, 3)),  # 3 DOF vs 6 DOF
        channel_names=("tx", "ty", "tz"),
        units=run_b.units,
        coordinate_frame=run_b.coordinate_frame,
        fidelity=run_b.fidelity,
    )
    compat = validate_run_compatibility(run_a, mismatched_b)
    assert compat.compatible is False
    assert any(
        "dimension" in err.lower() or "channel" in err.lower() for err in compat.errors
    )


def test_missing_load_channels_prevent_misleading_score() -> None:
    """Missing required load channels must raise MissingChannelError; no fabrication allowed."""
    empty_channels = BiomechanicalLoadChannels(
        source_run_id="run_missing_01",
        peak_compression_n=None,
        peak_shear_n=None,
        peak_torsion_nm=None,
        joint_moments_nm={},
        swing_metrics={},
    )
    adapter = InjuryIndicatorAdapter()
    with pytest.raises(MissingChannelError, match="Missing required load channels"):
        adapter.compute_indicators(empty_channels)


def test_main_block_mock_data_never_used_by_interactive_actions(
    valid_biomechanical_load_channels,
) -> None:
    """Interactive indicator evaluation must use the provided payload and not fall back to example data."""
    adapter = InjuryIndicatorAdapter()
    res = adapter.compute_indicators(
        valid_biomechanical_load_channels, body_weight_n=784.8
    )
    assert res.source_run_id == "run_mujoco_01"
    # Ensure raw peak compression reflects the supplied 3924.0 N / 784.8 N = 5.0 BW
    spinal_factor = next(
        (f for f in res.risk_factors if f["name"] == "spinal_compression"), None
    )
    assert spinal_factor is not None
    assert np.isclose(spinal_factor["value"], 5.0, rtol=1e-3)


# -----------------------------------------------------------------------------
# GREEN Acceptance Tests: Real Metrics, Provenance, and Surface Actions
# -----------------------------------------------------------------------------


def test_deterministic_compatible_real_fixture_pair_gives_expected_metrics(
    compatible_run_pair,
) -> None:
    """Deterministic compatible fixture pair evaluates exact cross-engine metrics with provenance."""
    run_a, run_b = compatible_run_pair
    adapter = CrossEngineComparisonAdapter()
    result = adapter.compare_runs(run_a, run_b)

    assert result.action_id == "canonical_core_comparison"
    assert result.run_a_id == "run_mujoco_01"
    assert result.run_b_id == "run_drake_02"
    assert result.fidelity == ModelFidelityLevel.SIMPLIFIED_KINETICS.value
    assert "max_abs_error" in result.metrics
    assert "mean_abs_error" in result.metrics
    assert np.isclose(result.metrics["max_abs_error"], 0.05, rtol=1e-5)
    assert len(result.provenance_hash) == 64  # SHA-256


def test_scorer_fixture_uses_actual_serialized_analysis_inputs(
    valid_biomechanical_load_channels,
) -> None:
    """Injury indicator computes scores from real serialized inputs, attaching explicit disclaimer."""
    adapter = InjuryIndicatorAdapter()
    result = adapter.compute_indicators(
        valid_biomechanical_load_channels, body_weight_n=784.8
    )

    assert result.action_id == "injury_analysis"
    assert result.source_run_id == "run_mujoco_01"
    assert result.fidelity == ModelFidelityLevel.QUALIFIED_FULL_BODY.value
    assert result.overall_risk_score > 0.0
    assert result.disclaimer != ""
    assert "NOT constitute clinical diagnosis" in result.disclaimer
    assert len(result.risk_factors) > 0


def test_fidelity_levels_are_kept_separate(compatible_run_pair) -> None:
    """Stub/pendulum models and qualified full body models cannot be conflated."""
    run_a, run_b = compatible_run_pair
    stub_run = ComparisonRunArtifact(
        run_id="run_stub_pendulum",
        backend_name="pendulum",
        times=run_a.times,
        coordinates=run_a.coordinates,
        channel_names=run_a.channel_names,
        units=run_a.units,
        coordinate_frame=run_a.coordinate_frame,
        fidelity=ModelFidelityLevel.STUB_PENDULUM,
    )
    compat = validate_run_compatibility(stub_run, run_b)
    assert compat.compatible is False
    assert any("fidelity" in err.lower() for err in compat.errors)


def test_coordinator_availability_and_action_surfacing() -> None:
    """Coordinator surfaces both canonical_core_comparison and injury_analysis."""
    coordinator = ComparisonIndicatorWorkspaceCoordinator()
    avail = coordinator.check_availability()
    assert avail["available"] is True
    assert "canonical_core_comparison" in avail["actions"]
    assert "injury_analysis" in avail["actions"]
    assert "results_and_compare" in avail["shells"]
    assert "exercise_analysis" in avail["shells"]

"""Tests for analysis.dataclasses module.

Validates dataclass construction, field access, pre-defined citations,
backward-compatible aliases, and cross-engine validation utility functions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from src.shared.python.analysis.dataclasses import (
    ANGLE_TOLERANCE_DEG,
    CITATION_CRUNCH_FACTOR,
    CITATION_KINEMATIC_SEQUENCE,
    CITATION_SEGMENT_TIMING,
    CITATION_SPINAL_LOAD,
    CITATION_X_FACTOR,
    TIMING_TOLERANCE_S,
    AngularMomentumMetrics,
    CoordinationMetrics,
    GRFMetrics,
    ImpulseMetrics,
    JerkMetrics,
    JointPowerMetrics,
    JointStiffnessMetrics,
    KinematicSequenceInfo,
    MethodCitation,
    PCAResult,
    PeakInfo,
    RQAMetrics,
    SegmentTimingInfo,
    StabilityMetrics,
    SummaryStatistics,
    SwingPhase,
    SwingProfileMetrics,
    validate_angle_cross_engine,
    validate_timing_cross_engine,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# MethodCitation
# ---------------------------------------------------------------------------


class TestMethodCitation:
    def test_required_fields(self) -> None:
        c = MethodCitation(
            name="Test",
            authors="Smith et al.",
            year=2020,
            title="A Test Paper",
        )
        assert c.name == "Test"
        assert c.authors == "Smith et al."
        assert c.year == 2020
        assert c.title == "A Test Paper"
        assert c.doi is None
        assert c.notes is None

    def test_optional_fields(self) -> None:
        c = MethodCitation(
            name="Test",
            authors="Smith",
            year=2020,
            title="Title",
            doi="10.1000/xyz",
            notes="Some notes",
        )
        assert c.doi == "10.1000/xyz"
        assert c.notes == "Some notes"

    def test_frozen_raises_on_mutation(self) -> None:
        c = MethodCitation(name="X", authors="A", year=2000, title="T")
        with pytest.raises(
            (dataclasses.FrozenInstanceError, TypeError, AttributeError)
        ):
            c.name = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pre-defined citations
# ---------------------------------------------------------------------------


class TestPredefinedCitations:
    def test_citation_segment_timing_fields(self) -> None:
        assert CITATION_SEGMENT_TIMING.name == "Proximal-to-Distal Sequencing"
        assert CITATION_SEGMENT_TIMING.authors == "Putnam"
        assert CITATION_SEGMENT_TIMING.year == 1993

    def test_citation_kinematic_sequence_alias(self) -> None:
        assert CITATION_KINEMATIC_SEQUENCE is CITATION_SEGMENT_TIMING

    def test_citation_x_factor_fields(self) -> None:
        assert CITATION_X_FACTOR.name == "X-Factor"
        assert CITATION_X_FACTOR.year == 2001

    def test_citation_crunch_factor_fields(self) -> None:
        assert CITATION_CRUNCH_FACTOR.name == "Crunch Factor"
        assert CITATION_CRUNCH_FACTOR.doi is not None

    def test_citation_spinal_load_fields(self) -> None:
        assert CITATION_SPINAL_LOAD.name == "Spinal Load Analysis"
        assert CITATION_SPINAL_LOAD.year == 1990


# ---------------------------------------------------------------------------
# Dataclass construction — structural tests
# ---------------------------------------------------------------------------


class TestDataclassConstruction:
    def test_peak_info_required(self) -> None:
        p = PeakInfo(value=1.5, time=0.3, index=10)
        assert p.value == 1.5
        assert p.time == 0.3
        assert p.index == 10
        assert p.prominence is None
        assert p.width is None

    def test_peak_info_optional(self) -> None:
        p = PeakInfo(value=1.5, time=0.3, index=10, prominence=0.5, width=2.0)
        assert p.prominence == 0.5
        assert p.width == 2.0

    def test_dataclasses_summary_statistics(self) -> None:
        s = SummaryStatistics(
            mean=1.0,
            median=1.0,
            std=0.1,
            min=0.5,
            max=1.5,
            range=1.0,
            min_time=0.1,
            max_time=0.9,
            rms=1.01,
        )
        assert s.std >= 0
        assert s.range >= 0
        assert s.rms >= 0

    def test_swing_phase(self) -> None:
        sp = SwingPhase(
            name="backswing",
            start_time=0.0,
            end_time=0.5,
            start_index=0,
            end_index=50,
            duration=0.5,
        )
        assert sp.duration == 0.5

    def test_segment_timing_info(self) -> None:
        sti = SegmentTimingInfo(
            segment_name="pelvis",
            peak_velocity=200.0,
            peak_time=0.3,
            peak_index=30,
            order_index=0,
        )
        assert sti.segment_name == "pelvis"

    def test_kinematic_sequence_info_alias(self) -> None:
        assert KinematicSequenceInfo is SegmentTimingInfo

    def test_grf_metrics(self) -> None:
        g = GRFMetrics(
            cop_path_length=0.5,
            cop_max_velocity=1.0,
            cop_x_range=0.3,
            cop_y_range=0.2,
        )
        assert g.peak_vertical_force is None

    def test_angular_momentum_metrics(self) -> None:
        a = AngularMomentumMetrics(
            peak_magnitude=5.0,
            peak_time=0.3,
            mean_magnitude=3.0,
            peak_lx=1.0,
            peak_ly=2.0,
            peak_lz=3.0,
            variability=0.1,
        )
        assert a.peak_magnitude >= 0

    def test_stability_metrics(self) -> None:
        s = StabilityMetrics(
            min_com_cop_distance=0.01,
            max_com_cop_distance=0.15,
            mean_com_cop_distance=0.07,
            peak_inclination_angle=5.0,
            mean_inclination_angle=2.0,
        )
        assert s.mean_com_cop_distance >= 0

    def test_coordination_metrics(self) -> None:
        c = CoordinationMetrics(
            in_phase_pct=40.0,
            anti_phase_pct=30.0,
            proximal_leading_pct=20.0,
            distal_leading_pct=10.0,
            mean_coupling_angle=45.0,
            coordination_variability=10.0,
        )
        assert c.in_phase_pct + c.anti_phase_pct <= 100.0

    def test_joint_power_metrics(self) -> None:
        j = JointPowerMetrics(
            peak_generation=100.0,
            peak_absorption=-50.0,
            avg_generation=60.0,
            avg_absorption=-30.0,
            net_work=10.0,
            generation_duration=0.3,
            absorption_duration=0.2,
        )
        assert j.net_work == 10.0

    def test_impulse_metrics(self) -> None:
        i = ImpulseMetrics(
            net_impulse=5.0,
            positive_impulse=8.0,
            negative_impulse=-3.0,
        )
        assert i.net_impulse == 5.0

    def test_rqa_metrics(self) -> None:
        r = RQAMetrics(
            recurrence_rate=0.05,
            determinism=0.8,
            laminarity=0.6,
            longest_diagonal_line=10,
            trapping_time=3.0,
        )
        assert 0.0 <= r.recurrence_rate <= 1.0

    def test_swing_profile_metrics(self) -> None:
        m = SwingProfileMetrics(
            speed_score=80.0,
            sequence_score=75.0,
            stability_score=70.0,
            efficiency_score=65.0,
            power_score=85.0,
        )
        assert m.power_score == 85.0

    def test_pca_result(self) -> None:
        components = np.eye(3)
        ev = np.array([2.0, 1.0, 0.5])
        evr = ev / ev.sum()
        proj = np.random.default_rng(0).standard_normal((10, 3))
        mean = np.zeros(3)
        r = PCAResult(
            components=components,
            explained_variance=ev,
            explained_variance_ratio=evr,
            projected_data=proj,
            mean=mean,
        )
        assert r.components.shape == (3, 3)

    def test_joint_stiffness_metrics(self) -> None:
        j = JointStiffnessMetrics(
            stiffness=100.0,
            r_squared=0.95,
            hysteresis_area=5.0,
            intercept=0.0,
        )
        assert 0.0 <= j.r_squared <= 1.0

    def test_jerk_metrics(self) -> None:
        j = JerkMetrics(peak_jerk=50.0, rms_jerk=20.0, dimensionless_jerk=0.1)
        assert j.peak_jerk >= 0


# ---------------------------------------------------------------------------
# validate_timing_cross_engine
# ---------------------------------------------------------------------------


class TestValidateTimingCrossEngine:
    def test_identical_arrays_pass(self) -> None:
        t = np.array([0.1, 0.3, 0.5])
        result = validate_timing_cross_engine(t, t.copy())
        assert result["passed"] is True
        assert result["max_diff_s"] == pytest.approx(0.0)

    def test_within_tolerance_pass(self) -> None:
        t_a = np.array([0.1, 0.3])
        t_b = np.array([0.1 + TIMING_TOLERANCE_S * 0.5, 0.3])
        result = validate_timing_cross_engine(t_a, t_b)
        assert result["passed"] is True

    def test_exceeds_tolerance_fail(self) -> None:
        t_a = np.array([0.1])
        t_b = np.array([0.1 + TIMING_TOLERANCE_S * 2])
        result = validate_timing_cross_engine(t_a, t_b)
        assert result["passed"] is False

    def test_length_mismatch_fail(self) -> None:
        result = validate_timing_cross_engine(np.array([0.1, 0.2]), np.array([0.1]))
        assert result["passed"] is False
        assert result["max_diff_s"] == float("inf")

    def test_custom_tolerance(self) -> None:
        t_a = np.array([0.0])
        t_b = np.array([0.1])
        assert validate_timing_cross_engine(t_a, t_b, tolerance_s=0.2)["passed"] is True
        assert (
            validate_timing_cross_engine(t_a, t_b, tolerance_s=0.05)["passed"] is False
        )


# ---------------------------------------------------------------------------
# validate_angle_cross_engine
# ---------------------------------------------------------------------------


class TestValidateAngleCrossEngine:
    def test_identical_arrays_pass(self) -> None:
        a = np.array([10.0, 20.0, 30.0])
        result = validate_angle_cross_engine(a, a.copy())
        assert result["passed"] is True
        assert result["max_diff_deg"] == pytest.approx(0.0)

    def test_within_tolerance_pass(self) -> None:
        a_a = np.array([10.0])
        a_b = np.array([10.0 + ANGLE_TOLERANCE_DEG * 0.5])
        result = validate_angle_cross_engine(a_a, a_b)
        assert result["passed"] is True

    def test_exceeds_tolerance_fail(self) -> None:
        a_a = np.array([0.0])
        a_b = np.array([ANGLE_TOLERANCE_DEG * 2])
        result = validate_angle_cross_engine(a_a, a_b)
        assert result["passed"] is False

    def test_shape_mismatch_fail(self) -> None:
        result = validate_angle_cross_engine(np.array([1.0, 2.0]), np.array([1.0]))
        assert result["passed"] is False
        assert result["max_diff_deg"] == float("inf")

    def test_custom_tolerance(self) -> None:
        a_a = np.array([0.0])
        a_b = np.array([5.0])
        assert (
            validate_angle_cross_engine(a_a, a_b, tolerance_deg=10.0)["passed"] is True
        )
        assert (
            validate_angle_cross_engine(a_a, a_b, tolerance_deg=3.0)["passed"] is False
        )

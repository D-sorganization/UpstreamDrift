"""Tests for fail-closed cross-engine canonical trial comparison."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python import perturbation
from src.shared.python.perturbation.cross_engine_trial_parity import (
    CrossEngineCompatibilityError,
    CrossEngineTolerances,
    compare_cross_engine_trials,
)
from src.shared.python.perturbation.trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    SampledInput,
    TrialTrace,
)

pytestmark = pytest.mark.unit


def test_cross_engine_comparison_is_publicly_exposed() -> None:
    assert perturbation.compare_cross_engine_trials is compare_cross_engine_trials
    assert perturbation.CrossEngineTolerances is CrossEngineTolerances


def _trace(*, marker_offset_m: float = 0.0) -> TrialTrace:
    markers = np.array(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [0.6 + marker_offset_m, 0.1, 0.0]],
        ]
    )
    return TrialTrace(
        times_s=np.array([0.0, 0.01]),
        q=np.array([[0.0, 0.0], [0.1, 0.2]]),
        v=np.array([[0.0, 0.0], [1.0, 2.0]]),
        coordinate_ids=("shoulder_angle", "club_angle"),
        coordinate_units=("rad", "rad"),
        velocity_units=("rad/s", "rad/s"),
        markers_m=markers,
        marker_ids=("lead_hand", "clubhead"),
        frame_id="world-z-up",
        alignment_id="downswing-start/v1",
        complete=True,
    )


def _trial(engine: str, *, marker_offset_m: float = 0.0) -> CanonicalTrialEvidence:
    return CanonicalTrialEvidence(
        trial_index=0,
        seed=71,
        plan_sha256="a" * 64,
        tools_revision="b" * 40,
        engine_id=engine,
        engine_revision=("c" if engine == "mujoco" else "d") * 40,
        model_id="two-dof-double-pendulum/v1",
        sampled_inputs=(
            SampledInput("proximal_torque_peak", 42.0, "N*m"),
            SampledInput("distal_torque_peak", -3.0, "N*m"),
        ),
        outcome="no_impact",
        trace=_trace(marker_offset_m=marker_offset_m),
        closest_approach=ClosestApproach(0.01, 0.02, "clubhead", "ball-center", False),
    )


def _tolerances() -> CrossEngineTolerances:
    return CrossEngineTolerances(
        time_atol_s=1e-12,
        coordinate_atol=(1e-10, 1e-10),
        velocity_atol=(1e-9, 1e-9),
        marker_atol_m=1e-5,
    )


def test_identical_compatible_trials_pass_with_zero_discrepancy() -> None:
    metrics = compare_cross_engine_trials(
        _trial("mujoco"), _trial("pinocchio"), _tolerances()
    )

    assert metrics.tolerance_equivalent is True
    assert metrics.outcome_match is True
    assert metrics.max_time_error_s == 0.0
    assert metrics.max_coordinate_normalized_error == 0.0
    assert metrics.max_velocity_normalized_error == 0.0
    assert metrics.max_marker_error_m == 0.0


def test_declared_marker_tolerance_exposes_non_equivalent_geometry() -> None:
    metrics = compare_cross_engine_trials(
        _trial("mujoco"),
        _trial("pinocchio", marker_offset_m=2e-5),
        _tolerances(),
    )

    assert metrics.tolerance_equivalent is False
    assert metrics.max_marker_error_m == pytest.approx(2e-5)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda trial: replace(trial, plan_sha256="f" * 64),
            "plan digest",
        ),
        (
            lambda trial: replace(
                trial,
                sampled_inputs=(SampledInput("proximal_torque_peak", 43.0, "N*m"),),
            ),
            "sampled inputs",
        ),
        (
            lambda trial: replace(
                trial,
                trace=replace(trial.trace, frame_id="club-local"),
            ),
            "frame",
        ),
        (
            lambda trial: replace(
                trial,
                trace=replace(
                    trial.trace,
                    marker_ids=("clubhead", "lead_hand"),
                ),
            ),
            "marker IDs",
        ),
        (
            lambda trial: replace(
                trial,
                trace=replace(
                    trial.trace,
                    coordinate_units=("deg", "rad"),
                ),
            ),
            "coordinate units",
        ),
    ],
)
def test_incompatible_identity_topology_frame_or_units_fail_before_ranking(
    mutator, message: str
) -> None:
    reference = _trial("mujoco")
    candidate = mutator(_trial("pinocchio"))

    with pytest.raises(CrossEngineCompatibilityError, match=message):
        compare_cross_engine_trials(reference, candidate, _tolerances())


def test_tolerances_require_one_positive_value_per_coordinate() -> None:
    with pytest.raises(ValueError, match="positive"):
        CrossEngineTolerances(
            time_atol_s=1e-12,
            coordinate_atol=(1e-10, 0.0),
            velocity_atol=(1e-9, 1e-9),
            marker_atol_m=1e-5,
        )

    with pytest.raises(CrossEngineCompatibilityError, match="tolerance dimensions"):
        compare_cross_engine_trials(
            _trial("mujoco"),
            _trial("pinocchio"),
            CrossEngineTolerances(1e-12, (1e-10,), (1e-9,), 1e-5),
        )

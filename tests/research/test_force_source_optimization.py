"""TDD contracts for the Coriolis-impulse optimization study."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.force_source_optimization import (
    ForceSourceCandidate,
    evaluate_candidate,
    summarize_optimization,
)
from src.shared.python.simulation_backends import GolfModelParams
from scripts.research.proximal_distal_energy.run_force_source_optimization import (
    registered_candidates,
)


@pytest.mark.scientific
def test_candidate_reports_component_impulse_power_work_and_mapping_status() -> None:
    outcome = evaluate_candidate(
        GolfModelParams.default(),
        ForceSourceCandidate(
            shoulder_torque_nm=100.0,
            wrist_drive_nm=15.0,
            wrist_restrain_nm=5.0,
            onset_s=0.2,
        ),
    )

    assert outcome.status == "qualified_impact"
    assert outcome.clubhead_speed_m_s is not None
    assert outcome.coriolis_tangent_impulse_n_s is not None
    assert outcome.coriolis_absolute_tangent_impulse_n_s is not None
    assert outcome.coriolis_work_j is not None
    assert outcome.tangent_valid_fraction is not None
    assert 0.0 < outcome.tangent_valid_fraction <= 1.0
    assert outcome.mapping_status == "rank_deficient_force_only"
    assert outcome.maximum_mapping_residual_nm is not None


@pytest.mark.unit
def test_summary_keeps_coriolis_and_speed_optima_separate() -> None:
    from dataclasses import replace

    baseline = evaluate_candidate(
        GolfModelParams.default(),
        ForceSourceCandidate(100.0, 15.0, 5.0, 0.2),
    )
    higher_impulse = replace(
        baseline,
        candidate=ForceSourceCandidate(90.0, 15.0, 0.0, 0.1),
        coriolis_absolute_tangent_impulse_n_s=(
            baseline.coriolis_absolute_tangent_impulse_n_s + 1.0
        ),
        clubhead_speed_m_s=baseline.clubhead_speed_m_s - 1.0,
    )

    summary = summarize_optimization((baseline, higher_impulse))

    assert summary["maximum_coriolis_impulse"]["onset_s"] == pytest.approx(0.1)
    assert summary["maximum_signed_coriolis_impulse"]["onset_s"] == pytest.approx(0.2)
    assert summary["maximum_clubhead_speed"]["onset_s"] == pytest.approx(0.2)
    assert summary["same_candidate"] is False


@pytest.mark.unit
def test_registered_grid_is_complete_and_unique() -> None:
    candidates = registered_candidates()

    assert len(candidates) == 135
    assert len(set(candidates)) == len(candidates)

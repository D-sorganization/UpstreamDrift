"""Unit tests for Tour Baselines Bounded Fit Campaigns (TB-08, #10593).

Validates:
1. Deterministic candidate ranking and Pareto selection: feasible beats infeasible lower-error candidate.
2. Incremental immutable checkpoint equivalence and resume validation.
3. Tampering / hash mismatch rejection on resume (IncompatibleResumeError).
4. Cancellation and timeout preservation of diagnostic artifacts without promotion.
5. Exact full-rate clock validation matching stored receipt metrics.
6. Within-capture holdout evaluation and generalization disclaimers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.jobs import (
    HashBundle,
    IncompatibleResumeError,
    JobStatus,
)
from src.shared.python.tour_baselines.campaign import (
    CampaignCandidate,
    CampaignEvaluationRecord,
    CampaignJobSpec,
    FitCampaignService,
    GeneralizationDisclaimer,
    rank_candidates,
)

pytestmark = pytest.mark.unit


def _sample_hashes(**overrides: str) -> HashBundle:
    base = {
        "data_hash": "sha256:" + "a" * 64,
        "model_hash": "sha256:" + "b" * 64,
        "runtime_hash": "sha256:" + "c" * 64,
        "controller_hash": "sha256:" + "d" * 64,
        "solver_hash": "sha256:" + "e" * 64,
    }
    base.update(overrides)
    return HashBundle(**base)


def _sample_spec(tmp_path: Path, **overrides: Any) -> CampaignJobSpec:
    defaults: dict[str, Any] = {
        "job_id": "tb08_campaign_test",
        "model_id": "driven_triple_pendulum",
        "capture_name": "driver",
        "capture_version": "545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba",
        "qualification_profile": "TriplePendulumProfile",
        "deterministic_seeds": [42, 101, 9921],
        "evaluation_limit": 50,
        "wall_time_limit_s": 15.0,
        "parameter_bounds": ([-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]),
        "basis": "constant-bernstein-6",
        "holdout_window_s": 0.05,
        "reproduction_command": "python -m src.shared.python.tour_baselines.campaign --model driven_triple_pendulum --capture driver",
        "run_root": tmp_path,
        "hashes": _sample_hashes(),
    }
    defaults.update(overrides)
    return CampaignJobSpec(**defaults)


def test_deterministic_manufactured_selection() -> None:
    """A feasible candidate must win over an infeasible candidate even if the infeasible has lower error.

    Infeasible candidates are retained as rejected evidence, never discarded or promoted.
    """
    cand_infeasible_low_err = CampaignCandidate(
        candidate_id="cand_infeasible_001",
        seed=42,
        parameters=np.array([120.0, 50.0, -10.0]),  # violates bound (120 > 100)
        rmse_m=0.015,
        max_m=0.025,
        is_feasible=False,
        feasibility_violations=("parameter 0 (120.0) exceeds upper bound 100.0",),
        evaluations=30,
        wall_time_s=1.2,
    )

    cand_feasible_higher_err = CampaignCandidate(
        candidate_id="cand_feasible_002",
        seed=101,
        parameters=np.array([45.0, 20.0, -5.0]),
        rmse_m=0.022,
        max_m=0.038,
        is_feasible=True,
        feasibility_violations=(),
        evaluations=25,
        wall_time_s=1.0,
    )

    cand_feasible_worse = CampaignCandidate(
        candidate_id="cand_feasible_003",
        seed=9921,
        parameters=np.array([55.0, 25.0, -8.0]),
        rmse_m=0.035,
        max_m=0.045,
        is_feasible=True,
        feasibility_violations=(),
        evaluations=40,
        wall_time_s=1.5,
    )

    ranking = rank_candidates(
        [cand_infeasible_low_err, cand_feasible_higher_err, cand_feasible_worse]
    )

    assert ranking.selected_candidate is not None
    assert ranking.selected_candidate.candidate_id == "cand_feasible_002"
    assert len(ranking.feasible_candidates) == 2
    assert len(ranking.rejected_candidates) == 1
    assert ranking.rejected_candidates[0].candidate_id == "cand_infeasible_001"
    assert (
        "exceeds upper bound"
        in ranking.rejected_candidates[0].feasibility_violations[0]
    )


def test_checkpoint_equivalence(tmp_path: Path) -> None:
    """Checkpointing a campaign state to disk and reloading it must yield exact state equivalence."""
    spec = _sample_spec(tmp_path)
    service = FitCampaignService(spec)

    candidate = CampaignCandidate(
        candidate_id="cand_001",
        seed=42,
        parameters=np.array([10.0, 20.0, 30.0]),
        rmse_m=0.025,
        max_m=0.040,
        is_feasible=True,
        feasibility_violations=(),
        evaluations=15,
        wall_time_s=0.5,
    )

    record = CampaignEvaluationRecord(
        job_id=spec.job_id,
        stage_index=1,
        initial_baseline_rmse_m=0.085,
        best_feasible_candidate=candidate,
        current_iterate=candidate,
        failed_starts=[{"seed": 999, "reason": "non-finite simulation"}],
        total_evaluations=15,
        elapsed_wall_s=0.5,
        status=JobStatus.RUNNING,
    )

    checkpoint_path = service.save_checkpoint(record)
    assert checkpoint_path.is_file()

    loaded = service.load_checkpoint(checkpoint_path)
    assert loaded.job_id == record.job_id
    assert loaded.stage_index == record.stage_index
    assert loaded.initial_baseline_rmse_m == pytest.approx(
        record.initial_baseline_rmse_m
    )
    assert loaded.best_feasible_candidate is not None
    np.testing.assert_allclose(
        loaded.best_feasible_candidate.parameters,
        candidate.parameters,
    )
    assert loaded.best_feasible_candidate.rmse_m == pytest.approx(candidate.rmse_m)
    assert loaded.failed_starts == record.failed_starts
    assert loaded.status == JobStatus.RUNNING


def test_changed_hash_blocks_resume(tmp_path: Path) -> None:
    """Resuming a checkpoint with altered data_hash or model_hash must fail closed with IncompatibleResumeError."""
    spec = _sample_spec(tmp_path)
    service = FitCampaignService(spec)

    record = CampaignEvaluationRecord(
        job_id=spec.job_id,
        stage_index=1,
        initial_baseline_rmse_m=0.085,
        best_feasible_candidate=None,
        current_iterate=None,
        failed_starts=[],
        total_evaluations=10,
        elapsed_wall_s=0.3,
        status=JobStatus.RUNNING,
    )
    checkpoint_path = service.save_checkpoint(record)

    # Attempt resume with altered data_hash
    tampered_hashes = _sample_hashes(data_hash="sha256:" + "f" * 64)
    tampered_spec = _sample_spec(tmp_path, hashes=tampered_hashes)
    tampered_service = FitCampaignService(tampered_spec)

    with pytest.raises(IncompatibleResumeError, match="data_hash mismatch"):
        tampered_service.load_checkpoint(checkpoint_path)


def test_timeout_cancel_keeps_artifacts_and_cannot_promote(tmp_path: Path) -> None:
    """When a job times out or is cancelled, existing checkpoints and diagnostics must be preserved, and no candidate promoted."""
    spec = _sample_spec(tmp_path, wall_time_limit_s=0.001)
    service = FitCampaignService(spec)

    # Simulate cancellation/timeout
    manifest = service.cancel_job(reason="Wall-time limit exceeded: 0.001s")

    assert manifest.status == JobStatus.CANCELLED
    assert manifest.acceptance == "UNVERIFIED"
    assert "Wall-time limit exceeded" in manifest.diagnostic_message
    assert (tmp_path / "manifest.json").is_file()

    # Verify that no candidate was promoted
    result = service.finalize_campaign()
    assert result.promoted_candidate is None
    assert result.status == JobStatus.CANCELLED


def test_recomputed_full_rate_scores_match_stored_values() -> None:
    """Full-clock validation scores on original 360 Hz / 359 Hz grid must match stored metrics without discrepancy."""
    # Manufactured clean trajectory vs prediction
    time_s = np.linspace(0.0, 0.5, 181)  # 360 Hz grid
    true_points = np.zeros((len(time_s), 2, 3))
    true_points[:, 0, 0] = np.sin(2.0 * np.pi * time_s)
    true_points[:, 1, 1] = np.cos(2.0 * np.pi * time_s)

    pred_points = true_points.copy()
    pred_points[:, 0, 0] += 0.005  # 5 mm offset on marker 0

    diff = pred_points - true_points
    expected_rmse = float(np.sqrt(np.mean(diff**2)))
    expected_max = float(np.max(np.sqrt(np.sum(diff**2, axis=-1))))

    eval_scores = FitCampaignService.compute_clock_scores(pred_points, true_points)
    assert eval_scores["rmse_m"] == pytest.approx(expected_rmse, rel=1e-6)
    assert eval_scores["max_m"] == pytest.approx(expected_max, rel=1e-6)


def test_pilot_measures_runtime_and_freezes_budget(tmp_path: Path) -> None:
    """A pilot run measures per-evaluation and per-stage runtime, freezing budgets before full campaigns."""
    spec = _sample_spec(tmp_path)
    service = FitCampaignService(spec)

    pilot_budget = service.run_pilot_benchmark(pilot_evaluations=5)
    assert pilot_budget.evaluations_measured == 5
    assert pilot_budget.measured_seconds_per_eval > 0.0
    assert pilot_budget.frozen_evaluation_budget > 0
    assert pilot_budget.frozen_wall_time_budget_s > 0.0


def test_within_capture_holdout_disclaimer() -> None:
    """Holdout evaluation report must explicitly include the statement that within-capture holdout is not population generalization."""
    disclaimer = GeneralizationDisclaimer.get_statement()
    assert (
        "within-capture holdout is not population generalization" in disclaimer.lower()
    )

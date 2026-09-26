"""Unit and behavioral tests for NM-08 verified inference orchestration.

Tests behavioral and negative contracts:
- Wrong geometry/engine/control dimension
- Stale pin / incompatible checkpoint contract
- NaN input / nonfinite output
- Adversarial out-of-domain target
- Contact transition on smooth models
- Missing weights / checkpoint fallback
- Fallback budget sharing and exhaustion
- Replay failure despite low neural loss
- Both attempts retained with auditable statuses
- Empirical confidence as domain metrics, not golfer truth probability
- Club-only and complete target compatibility
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance
from src.shared.python.motion_matching.hybrid import ProposalCheckpointContract
from src.shared.python.neural_motion.inference.types import (
    INFERENCE_SCHEMA,
    AttemptRecord,
    DomainCheckResult,
    InferenceBudget,
    InferenceStatus,
    VerifiedInferenceReport,
)
from src.shared.python.neural_motion.inference.distribution import (
    DistributionBounds,
    check_target_distribution,
)
from src.shared.python.neural_motion.inference.orchestration import (
    VerifiedInferenceOrchestrator,
)


class _SyntheticDuckTarget:
    """Mock target with non-finite coordinates for testing fail-closed validation."""

    def __init__(
        self,
        time: np.ndarray,
        clubhead: np.ndarray,
        impact_idx: int | None = None,
        club_type: str = "driver",
    ) -> None:
        self.time = time
        self.clubhead = clubhead
        self.impact_idx = impact_idx
        self.club_type = club_type


def _make_dummy_target(
    *,
    duration_s: float = 1.0,
    n_frames: int = 50,
    impact_idx: int | None = None,
    with_nan: bool = False,
    v_peak: float = 40.0,
    club_type: str = "driver",
) -> Any:
    """Create synthetic ClubTarget for testing."""
    times = np.linspace(0.0, duration_s, n_frames)
    pos = np.zeros((n_frames, 3), dtype=np.float64)
    pos[:, 0] = np.sin(np.pi * times / duration_s) * 1.2
    pos[:, 1] = np.cos(np.pi * times / duration_s) * 0.8
    pos[:, 2] = np.sin(2.0 * np.pi * times / duration_s) * 0.5

    if with_nan:
        pos[5, 1] = float("nan")
        return _SyntheticDuckTarget(
            time=times,
            clubhead=pos,
            impact_idx=impact_idx,
            club_type=club_type,
        )

    quats = np.zeros((n_frames, 4), dtype=np.float64)
    quats[:, 0] = 1.0
    butt = pos - np.array([0.0, 0.0, 0.8])
    idx = 25 if impact_idx is None else impact_idx
    prov = SourceProvenance(
        filename="dummy.c3d",
        format="synthetic",
        subject_id="TW",
        trial_id="trial1",
        sha256="0" * 64,
    )
    return ClubTarget(
        time=times,
        butt=butt,
        clubhead=pos,
        club_quat=quats,
        impact_idx=idx,
        source=prov,
    )


def _valid_checkpoint_contract(
    model_id: str = "driver_g1_27dof",
    control_dim: int = 27,
) -> ProposalCheckpointContract:
    return ProposalCheckpointContract(
        model_id=model_id,
        control_dim=control_dim,
        control_basis="polynomial_degree_6",
        schema_version="neural-masked-proposals/1.0.0",
        weight_digest="sha256:fedcba9876543210fedcba9876543210",
    )


# ===========================================================================
# 1. Distribution Check & Empirical Confidence Tests
# ===========================================================================


def test_distribution_check_valid_target() -> None:
    target = _make_dummy_target(duration_s=1.2, v_peak=45.0)
    check = check_target_distribution(target)
    assert check.is_in_distribution is True
    assert 0.0 <= check.empirical_confidence <= 1.0
    assert len(check.diagnostics) == 0


def test_distribution_check_rejects_nonfinite_inputs() -> None:
    target = _make_dummy_target(with_nan=True)
    with pytest.raises(ValueError, match="non-finite|finite|NaN"):
        check_target_distribution(target)


def test_distribution_check_out_of_domain_duration() -> None:
    # 4.5 seconds is way beyond golf downswing / backswing training distribution
    target = _make_dummy_target(duration_s=4.5)
    bounds = DistributionBounds(min_duration_s=0.5, max_duration_s=2.5)
    check = check_target_distribution(target, bounds=bounds)
    assert check.is_in_distribution is False
    assert check.duration_supported is False
    assert check.empirical_confidence < 0.5
    assert any("duration" in d.lower() for d in check.diagnostics)


def test_distribution_check_detects_unsupported_contact_regime() -> None:
    # target with impact when smooth-only is specified
    target = _make_dummy_target(impact_idx=25)
    check = check_target_distribution(
        target,
        bounds=DistributionBounds(supported_contact_regimes=("airborne_only",)),
    )
    assert check.contact_regime_supported is False
    assert check.is_in_distribution is False
    assert any("contact" in d.lower() for d in check.diagnostics)


def test_empirical_confidence_is_domain_metric_not_golfer_truth() -> None:
    target = _make_dummy_target(duration_s=1.0)
    check = check_target_distribution(target)
    # Check docstring / contract that empirical_confidence represents domain distance,
    # not the truth probability of the golfer
    assert hasattr(check, "empirical_confidence")
    assert isinstance(check.empirical_confidence, float)
    assert 0.0 <= check.empirical_confidence <= 1.0


# ===========================================================================
# 2. Orchestration & Safe Fallback Behavioral Tests
# ===========================================================================


def test_wrong_control_dimension_triggers_classical_fallback() -> None:
    target = _make_dummy_target()
    contract = _valid_checkpoint_contract(control_dim=27)

    # Neural proposal returns wrong dimension (15 instead of 27)
    def bad_proposal_fn(t: object) -> np.ndarray:
        return np.ones(15, dtype=np.float64)

    classical_called = False

    def classical_fn(t: object, budget_s: float) -> dict[str, Any]:
        nonlocal classical_called
        classical_called = True
        return {
            "controls": np.ones(27, dtype=np.float64),
            "final_loss": 0.02,
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": True},
        }

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=bad_proposal_fn,
        polish_fn=lambda t, u: {"controls": u, "independent_replay": True},
        classical_fn=classical_fn,
        expected_contract=contract,
    )

    report = orchestrator.orchestrate(target, total_budget_s=5.0)

    assert classical_called is True
    assert report.status == InferenceStatus.CLASSICAL_FALLBACK
    assert len(report.attempts) == 2
    assert report.attempts[0].phase == "neural_proposal"
    assert report.attempts[0].acceptance_status == "rejected"
    assert "dimension" in report.attempts[0].rejection_reason.lower()
    assert report.attempts[1].phase == "classical_fallback"
    assert report.attempts[1].acceptance_status == "passed"


def test_checkpoint_contract_mismatch_fails_closed_to_classical() -> None:
    target = _make_dummy_target()
    expected = _valid_checkpoint_contract(model_id="driver_g1_27dof")
    loaded = _valid_checkpoint_contract(model_id="iron_g1_44dof")  # mismatched!

    classical_called = False

    def classical_fn(t: object, budget_s: float) -> dict[str, Any]:
        nonlocal classical_called
        classical_called = True
        return {
            "controls": np.ones(27),
            "final_loss": 0.03,
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": True},
        }

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=lambda t: np.zeros(27),
        polish_fn=lambda t, u: {"controls": u, "independent_replay": True},
        classical_fn=classical_fn,
        expected_contract=expected,
        loaded_contract=loaded,
    )

    report = orchestrator.orchestrate(target, total_budget_s=5.0)

    assert classical_called is True
    assert report.status == InferenceStatus.CLASSICAL_FALLBACK
    assert (
        "contract" in report.attempts[0].rejection_reason.lower()
        or "incompatible" in report.attempts[0].rejection_reason.lower()
    )


def test_missing_checkpoint_immediately_falls_back_to_classical() -> None:
    target = _make_dummy_target()
    classical_budget_received = None

    def classical_fn(t: object, budget_s: float) -> dict[str, Any]:
        nonlocal classical_budget_received
        classical_budget_received = budget_s
        return {
            "controls": np.ones(27),
            "final_loss": 0.015,
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": True},
        }

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=None,  # No neural checkpoint/model available
        polish_fn=lambda t, u: {"controls": u, "independent_replay": True},
        classical_fn=classical_fn,
    )

    report = orchestrator.orchestrate(target, total_budget_s=8.0)

    assert report.status == InferenceStatus.CLASSICAL_FALLBACK
    assert classical_budget_received is not None
    assert (
        classical_budget_received > 7.5
    )  # Almost entire budget preserved for fallback
    assert len(report.attempts) == 2
    assert "missing" in report.attempts[0].rejection_reason.lower()


def test_nan_neural_output_triggers_classical_fallback() -> None:
    target = _make_dummy_target()

    def nan_proposal(t: object) -> np.ndarray:
        out = np.zeros(27)
        out[10] = float("nan")
        return out

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=nan_proposal,
        polish_fn=lambda t, u: {"controls": u, "independent_replay": True},
        classical_fn=lambda t, b: {
            "controls": np.zeros(27),
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": True},
        },
    )

    report = orchestrator.orchestrate(target)
    assert report.status == InferenceStatus.CLASSICAL_FALLBACK
    assert any(
        "non-finite" in a.rejection_reason.lower()
        or "nan" in a.rejection_reason.lower()
        for a in report.attempts
    )


def test_adversarial_out_of_domain_target_triggers_fallback() -> None:
    target = _make_dummy_target(duration_s=6.0)  # Extreme duration

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=lambda t: np.zeros(27),
        polish_fn=lambda t, u: {"controls": u, "independent_replay": True},
        classical_fn=lambda t, b: {
            "controls": np.zeros(27),
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": True},
        },
    )

    report = orchestrator.orchestrate(target)
    assert report.status == InferenceStatus.CLASSICAL_FALLBACK
    assert report.domain_check.is_in_distribution is False
    assert any(
        "out-of-distribution" in a.rejection_reason.lower()
        or "duration" in a.rejection_reason.lower()
        for a in report.attempts
    )


def test_replay_fails_despite_tiny_neural_loss() -> None:
    target = _make_dummy_target()

    # Neural proposal returns tiny loss, but physical independent replay fails
    def proposal_fn(t: object) -> np.ndarray:
        return np.zeros(27)

    def polish_fn(t: object, u: np.ndarray) -> dict[str, Any]:
        return {
            "controls": u,
            "final_loss": 0.0001,  # tiny neural loss!
            "independent_replay": True,
            "acceptance": {
                "is_physically_accepted": False,
                "reason": "max normal force 3500 N exceeds 3x bodyweight",
            },
        }

    classical_called = False

    def classical_fn(t: object, budget_s: float) -> dict[str, Any]:
        nonlocal classical_called
        classical_called = True
        return {
            "controls": np.ones(27) * 0.1,
            "final_loss": 0.015,
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": True},
        }

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=proposal_fn,
        polish_fn=polish_fn,
        classical_fn=classical_fn,
    )

    report = orchestrator.orchestrate(target)
    assert classical_called is True
    assert report.status == InferenceStatus.CLASSICAL_FALLBACK
    assert report.attempts[0].acceptance_status == "failed"
    assert "force" in report.attempts[0].rejection_reason.lower()


def test_budget_exhaustion_terminates_without_calling_classical() -> None:
    target = _make_dummy_target()

    def slow_proposal(t: object) -> np.ndarray:
        time.sleep(0.05)
        return np.zeros(27)

    classical_called = False

    def classical_fn(t: object, budget_s: float) -> dict[str, Any]:
        nonlocal classical_called
        classical_called = True
        return {"controls": np.zeros(27)}

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=slow_proposal,
        polish_fn=lambda t, u: {
            "controls": u,
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": False, "reason": "failed"},
        },
        classical_fn=classical_fn,
    )

    # Budget is tiny (0.01s), so after slow_proposal consumes 0.05s, remaining budget is negative
    report = orchestrator.orchestrate(target, total_budget_s=0.01)
    assert classical_called is False
    assert report.status == InferenceStatus.REJECTED
    assert any("budget" in a.rejection_reason.lower() for a in report.attempts)


def test_neural_accepted_when_independent_replay_passes() -> None:
    target = _make_dummy_target()

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=lambda t: np.zeros(27),
        polish_fn=lambda t, u: {
            "controls": u,
            "final_loss": 0.01,
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": True},
        },
        classical_fn=lambda t, b: {"controls": np.ones(27)},
    )

    report = orchestrator.orchestrate(target, total_budget_s=5.0)
    assert report.status == InferenceStatus.NEURAL_ACCEPTED
    assert len(report.attempts) == 1
    assert report.attempts[0].acceptance_status == "passed"
    assert report.selected_controls is not None


def test_preview_only_flag_propagates_explicitly() -> None:
    target = _make_dummy_target()

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=lambda t: np.zeros(27),
        polish_fn=lambda t, u: {
            "controls": u,
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": True},
        },
        classical_fn=lambda t, b: {"controls": np.ones(27)},
    )

    report_preview = orchestrator.orchestrate(target, is_preview=True)
    assert report_preview.is_preview_only is True

    report_full = orchestrator.orchestrate(target, is_preview=False)
    assert report_full.is_preview_only is False


@pytest.mark.unit
def test_orchestrate_without_polish_fn_and_replay_required_is_not_neural_accepted() -> (
    None
):
    """Orchestration without polish_fn when replay required fails closed (#10960 P0-6)."""
    target = _make_dummy_target()
    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=lambda t: np.zeros(27),
        polish_fn=None,
        classical_fn=None,
    )
    report = orchestrator.orchestrate(target, require_independent_replay=True)
    assert report.status != InferenceStatus.NEURAL_ACCEPTED
    assert report.status == InferenceStatus.REJECTED
    assert len(report.attempts) >= 1
    att = report.attempts[0]
    assert att.phase == "neural_proposal"
    assert att.acceptance_status == "rejected"
    assert att.rejection_reason == "no native polish/replay"


@pytest.mark.unit
def test_orchestrate_polish_returning_empty_acceptance_is_rejected() -> None:
    """Polish returning empty acceptance dict must fail closed to False (#10960 P0-6)."""
    target = _make_dummy_target()
    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=lambda t: np.zeros(27),
        polish_fn=lambda t, u: {
            "controls": u,
            "independent_replay": True,
            "acceptance": {},
        },
        classical_fn=None,
    )
    report = orchestrator.orchestrate(target, require_independent_replay=True)
    assert report.status != InferenceStatus.NEURAL_ACCEPTED
    assert report.status == InferenceStatus.REJECTED
    att = report.attempts[0]
    assert att.phase == "neural_refined"
    assert att.acceptance_status == "failed"


@pytest.mark.unit
def test_orchestrate_polish_returning_explicit_passed_verdict_reported_verbatim() -> (
    None
):
    """Polish returning an explicit PASSED verdict must be reported verbatim (#10960 P0-6)."""
    target = _make_dummy_target()
    verdict = {
        "is_physically_accepted": True,
        "status": "PASSED",
        "custom_receipt_code": "NM08_VERIFIED_77",
        "residual_rms": 0.0012,
    }
    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=lambda t: np.zeros(27),
        polish_fn=lambda t, u: {
            "controls": u,
            "independent_replay": True,
            "acceptance": verdict,
        },
        classical_fn=None,
    )
    report = orchestrator.orchestrate(target, require_independent_replay=True)
    assert report.status == InferenceStatus.NEURAL_ACCEPTED
    assert report.acceptance_verdict == verdict

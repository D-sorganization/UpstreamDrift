"""Tests for NM-08 verified inference via motion_matching.hybrid facade."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance
from src.shared.python.motion_matching.hybrid import (
    INFERENCE_SCHEMA,
    DomainCheckResult,
    InferenceStatus,
    VerifiedInferenceOptions,
    VerifiedInferenceOrchestrator,
    VerifiedInferenceReport,
    check_target_distribution,
    fit_swing_verified_inference,
)
from src.shared.python.motion_matching.provider import MultiSourceTarget


def _make_dummy_target() -> ClubTarget:
    times = np.linspace(0.0, 1.0, 50)
    pos = np.zeros((50, 3), dtype=np.float64)
    pos[:, 0] = np.sin(np.pi * times) * 1.2
    pos[:, 1] = np.cos(np.pi * times) * 0.8
    pos[:, 2] = np.sin(2.0 * np.pi * times) * 0.5
    quats = np.zeros((50, 4), dtype=np.float64)
    quats[:, 0] = 1.0
    butt = pos - np.array([0.0, 0.0, 0.8])
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
        impact_idx=25,
        source=prov,
    )


def test_fit_swing_verified_inference_facade() -> None:
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

    report = fit_swing_verified_inference(
        target,
        orchestrator=orchestrator,
        options=VerifiedInferenceOptions(total_budget_s=5.0, is_preview=True),
    )

    assert isinstance(report, VerifiedInferenceReport)
    assert report.schema == INFERENCE_SCHEMA
    assert report.status == InferenceStatus.NEURAL_ACCEPTED
    assert report.is_preview_only is True
    assert isinstance(report.domain_check, DomainCheckResult)
    assert report.domain_check.is_in_distribution is True


def test_verified_inference_supports_multi_source_target() -> None:
    club = _make_dummy_target()
    multi_target = MultiSourceTarget(club=club, body=None)

    orchestrator = VerifiedInferenceOrchestrator(
        proposal_fn=lambda t: np.zeros(27),
        polish_fn=lambda t, u: {
            "controls": u,
            "independent_replay": True,
            "acceptance": {"is_physically_accepted": True},
        },
        classical_fn=lambda t, b: {"controls": np.ones(27)},
    )

    report = fit_swing_verified_inference(multi_target, orchestrator=orchestrator)
    assert report.status == InferenceStatus.NEURAL_ACCEPTED
    assert report.selected_controls is not None

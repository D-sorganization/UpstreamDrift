"""Tests for UI verified_fit seed honesty and qualification (Issue #10960 P1-4).

Prevents fabricated software-contract verified fit badges and ledger row
publication on synthetic seeds. Fail closed on missing observations and missing
seeds, and reject ledger rows for ui_synthetic seeds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.fast_matching import MatchPreset
from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.motion_matching.club_only.seeds import (
    CandidateSeed,
    geometry_content_hash,
    profile_content_hash,
)
from src.shared.python.motion_matching.club_only.ui_integration import (
    VerificationDisplayStatus,
    _synthetic_seed,
    build_club_only_result_view,
    create_club_only_session,
    publish_club_only_ledger_row,
    run_club_only_ui_match,
    write_club_only_result_package,
)

pytestmark = pytest.mark.unit


def _valid_retrieval_seed(obs: ClubObservation, model_id: str) -> CandidateSeed:
    """Build a valid CO-03 retrieval seed for unit tests."""
    profile = get_club_only_profile(model_id)
    times = np.asarray(obs.native_time_s, dtype=np.float64)
    nq = 4 if "double" in model_id else 6 if "triple" in model_id else 8
    q = np.zeros(nq, dtype=np.float64)
    geom = geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=float(obs.catalog_length_m),
        tool_to_model_residual_m=0.0,
    )
    return CandidateSeed(
        seed_id=f"retrieval-test-{obs.trial_id}-{model_id}",
        trial_id=obs.trial_id,
        model_id=model_id,
        source="retrieval",
        q=q,
        body_configuration_hash=hashlib.sha256(q.tobytes()).hexdigest(),
        observed_residual_m=0.01,
        prior_score=0.8,
        feasibility_reasons=("single_rigid_placement",),
        timestamps_s=times,
        geometry_hash=geom,
        profile_hash=profile_content_hash(profile),
        body_is_prior=True,
        is_kinematic_preview=True,
    )


def test_observation_none_raises_value_error() -> None:
    """(a) run_club_only_ui_match with observation=None raises ValueError (no silent fallback)."""
    session = create_club_only_session(
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        preset=MatchPreset.FAST_PREVIEW,
    )
    obs = build_calibrated_observation_fixture("TW_wiffle")
    seed = _valid_retrieval_seed(obs, session.model_id)

    with pytest.raises(ValueError, match="observation"):
        run_club_only_ui_match(session, observation=None, seed=seed)


def test_missing_seed_raises_naming_co03_sources() -> None:
    """(b) Missing seed raises ValueError naming the CO-03 retrieval/IK sources."""
    session = create_club_only_session(
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        preset=MatchPreset.FAST_PREVIEW,
    )
    obs = build_calibrated_observation_fixture("TW_wiffle")

    with pytest.raises(ValueError) as excinfo:
        run_club_only_ui_match(session, observation=obs, seed=None)

    msg = str(excinfo.value)
    assert "CO-03" in msg
    assert "retrieval" in msg
    assert "constrained_ik" in msg or "ik" in msg.lower()


def test_ui_synthetic_seed_never_yields_software_verified_fit() -> None:
    """(c) A ui_synthetic seed never yields SOFTWARE_VERIFIED_FIT."""
    session = create_club_only_session(
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        preset=MatchPreset.VERIFIED_FIT,
    )
    obs = build_calibrated_observation_fixture("TW_wiffle")
    synthetic = _synthetic_seed(obs, session.model_id)
    assert synthetic.source == "ui_synthetic"

    result = run_club_only_ui_match(session, observation=obs, seed=synthetic)
    view = build_club_only_result_view(result)

    assert view.display_status != VerificationDisplayStatus.SOFTWARE_VERIFIED_FIT
    assert view.display_status == VerificationDisplayStatus.UNQUALIFIED
    assert any(
        "synthetic" in blocker.lower() for blocker in view.qualification_blockers
    )


def test_ui_synthetic_seed_never_appends_ledger_row(tmp_path: Path) -> None:
    """(c) A ui_synthetic seed result cannot publish or append a ledger row."""
    session = create_club_only_session(
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        preset=MatchPreset.VERIFIED_FIT,
    )
    obs = build_calibrated_observation_fixture("TW_wiffle")
    synthetic = _synthetic_seed(obs, session.model_id)

    result = run_club_only_ui_match(session, observation=obs, seed=synthetic)
    view = build_club_only_result_view(result)

    receipt = tmp_path / "synthetic_receipt.json"
    write_club_only_result_package(view, receipt)

    ledger_file = (
        tmp_path / "artifacts" / "matched_swings" / "matched_swing_ledger.json"
    )
    ledger_file.parent.mkdir(parents=True, exist_ok=True)
    initial_ledger_content = json.dumps(
        {
            "schema_version": "1.0.0",
            "generated_at": "2026-09-26T00:00:00Z",
            "total_receipts": 0,
            "rows": [],
        }
    )
    ledger_file.write_text(initial_ledger_content, encoding="utf-8")

    with pytest.raises(ValueError, match="synthetic"):
        publish_club_only_ledger_row(view, receipt_path=receipt, repo_root=tmp_path)

    # Ensure no row was appended to ledger
    current_ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    assert len(current_ledger["rows"]) == 0


def test_gui_shows_no_verified_seed_status_when_seed_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUI shows 'no verified seed' and never runs match or appends ledger when seed missing."""
    try:
        from PyQt6.QtWidgets import QApplication
        from src.tools.motion_matching.gui import MotionMatchingWidget
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable in this environment: {exc}")

    _ = QApplication.instance() or QApplication([])

    widget = MotionMatchingWidget()
    try:
        widget._club_seed = None
        ran_worker = False

        def fake_run_in_worker(*args: Any, **kwargs: Any) -> Any:
            nonlocal ran_worker
            ran_worker = True
            raise AssertionError(
                "worker should not be called when no seed is available"
            )

        monkeypatch.setattr("src.tools.async_action.run_in_worker", fake_run_in_worker)

        # Trigger verified fit run
        widget._run_club_only_match(preset="verified_fit")

        assert not ran_worker
        log_text = widget.club_log.toPlainText().lower()
        assert "no verified seed" in log_text

        # Also test with a synthetic seed set on widget
        obs = build_calibrated_observation_fixture("TW_wiffle")
        widget._club_seed = _synthetic_seed(obs, "driven_double_pendulum")
        widget.club_log.clear()

        widget._run_club_only_match(preset="verified_fit")
        assert not ran_worker
        log_text2 = widget.club_log.toPlainText().lower()
        assert "no verified seed" in log_text2
    finally:
        widget.cleanup()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("retrieval", True),
        ("constrained_ik", True),
        ("ui_synthetic", False),
        ("synthetic", False),
    ],
)
def test_is_verified_seed_accepts_only_verified_sources(
    source: str, expected: bool
) -> None:
    from types import SimpleNamespace

    from src.shared.python.motion_matching.club_only.seeds import is_verified_seed

    assert is_verified_seed(SimpleNamespace(source=source)) is expected


def test_is_verified_seed_rejects_missing_seed() -> None:
    from src.shared.python.motion_matching.club_only.seeds import is_verified_seed

    assert is_verified_seed(None) is False

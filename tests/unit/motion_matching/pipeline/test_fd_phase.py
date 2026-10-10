"""#12042: forward-dynamics marker RMS split at the detected impact."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import fd_phase as fp

pytestmark = pytest.mark.unit
RATE = 360.0


def _clubhead(frames: int = 720, impact_s: float = 1.3) -> tuple[np.ndarray, ...]:
    """Head swings up to a top at 0.9 s and back through the address point
    (the ball) at ``impact_s``, then on to a finish."""
    t = np.arange(frames) / RATE
    angle = np.where(
        t <= 0.9,
        np.pi * t / 0.9,
        np.pi - np.pi * (t - 0.9) / (impact_s - 0.9),
    )
    head = np.stack(
        [1.2 * np.sin(angle), np.zeros_like(t), 1.2 - 1.2 * np.cos(angle)], 1
    )
    return t, head


def test_phase_rms_splits_at_impact_and_respects_validity() -> None:
    t = np.arange(10) / 10.0
    errors = np.zeros((10, 2))
    errors[:6] = 0.01
    errors[6:] = 0.03
    valid = np.ones((10, 2), dtype=bool)
    valid[8, 0] = False
    errors[8, 0] = 9.0  # ignored: invalid
    out = fp.phase_marker_rms(t, errors, valid, impact_time_s=0.5)
    assert out["fd_rms_address_to_impact_m"] == pytest.approx(0.01)
    assert out["fd_rms_after_impact_m"] == pytest.approx(0.03)
    assert out["address_to_impact_frames"] == 6
    assert out["after_impact_frames"] == 4


def test_phase_rms_whole_swing_is_consistent() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(50) / 50.0
    errors = rng.uniform(0, 0.05, (50, 3))
    valid = rng.uniform(size=(50, 3)) > 0.2
    out = fp.phase_marker_rms(t, errors, valid, impact_time_s=0.61)
    pre, post = t <= 0.61, t > 0.61
    whole2 = (
        out["fd_rms_address_to_impact_m"] ** 2 * valid[pre].sum()
        + out["fd_rms_after_impact_m"] ** 2 * valid[post].sum()
    ) / valid.sum()
    assert math.sqrt(whole2) == pytest.approx(np.sqrt(np.mean(errors[valid] ** 2)))


def test_phase_rms_validates_inputs() -> None:
    t = np.arange(4) / 4.0
    with pytest.raises(ValueError, match="shape"):
        fp.phase_marker_rms(t, np.zeros((3, 2)), np.ones((3, 2), bool), 0.5)
    with pytest.raises(ValueError, match="inside"):
        fp.phase_marker_rms(t, np.zeros((4, 2)), np.ones((4, 2), bool), 5.0)
    empty = fp.phase_marker_rms(
        t, np.zeros((4, 1)), np.array([[True], [False], [False], [False]]), 0.3
    )
    assert empty["fd_rms_after_impact_m"] is None


def test_report_uses_the_shared_ball_passage_rule() -> None:
    t, head = _clubhead()
    errors = np.where(t[:, None] <= 1.3, 0.02, 0.05) * np.ones((len(t), 2))
    report = fp.fd_phase_report(t, errors, np.ones_like(errors, bool), head)
    assert report["status"] == "ok"
    assert report["impact_time_s"] == pytest.approx(1.3, abs=1.0 / RATE)
    assert report["impact_detector"] == fp.IMPACT_DETECTOR
    assert report["fd_rms_address_to_impact_m"] == pytest.approx(0.02)
    assert report["fd_rms_after_impact_m"] == pytest.approx(0.05)


def test_report_is_unavailable_not_zero_without_an_impact() -> None:
    t, head = _clubhead()
    head[:, 2] += np.where(t > 0.9, 1.0, 0.0)  # never returns to the ball
    errors = np.full((len(t), 2), 0.02)
    report = fp.fd_phase_report(t, errors, np.ones_like(errors, bool), head)
    assert report["status"] == "unavailable"
    assert report["reason"]
    assert report["fd_rms_address_to_impact_m"] is None
    assert report["fd_rms_after_impact_m"] is None


def test_reference_clubhead_tolerates_stand_in_kinematics() -> None:
    assert fp.reference_clubhead(MagicMock(), np.zeros((3, 6))) is None


def test_receipt_schema_accepts_the_block() -> None:
    from src.shared.python.motion_matching.pipeline.receipt_dynamics import (
        FdPhaseReceipt,
    )

    t, head = _clubhead()
    errors = np.full((len(t), 2), 0.02)
    block = FdPhaseReceipt.model_validate(
        fp.fd_phase_report(t, errors, np.ones_like(errors, bool), head)
    )
    assert block.fd_rms_address_to_impact_m == pytest.approx(0.02)
    with pytest.raises(ValueError):
        FdPhaseReceipt.model_validate({"status": "ok", "fd_rms_after_impact_m": -1})


def test_ledger_extracts_the_phase_columns() -> None:
    from src.shared.python.motion_matching.ledger import extract_metrics

    receipt = {
        "dynamics": {
            "marker_rms_m": 0.06,
            "fd_phase": {
                "fd_rms_address_to_impact_m": 0.02,
                "fd_rms_after_impact_m": 0.11,
            },
        }
    }
    metrics = extract_metrics(receipt)
    assert metrics.whole_marker_rmse_m == pytest.approx(0.06)
    assert metrics.fd_address_to_impact_rmse_m == pytest.approx(0.02)
    assert metrics.fd_after_impact_rmse_m == pytest.approx(0.11)
    assert (
        extract_metrics({"dynamics": {"marker_rms_m": 0.06}}).fd_after_impact_rmse_m
        is None
    )


def test_an_impact_without_a_preceding_backswing_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A follow-through IK branch jump can make the shared detector take a
    near-address sample as the ball passage; that must not be reported."""
    from src.shared.python.model_appearance import club_face

    t, head = _clubhead()
    monkeypatch.setattr(club_face, "ball_passage", lambda *a, **k: (t[20], 20, 0.0))
    errors = np.full((len(t), 2), 0.02)
    report = fp.fd_phase_report(t, errors, np.ones_like(errors, bool), head)
    assert report["status"] == "unavailable"
    assert "backswing" in report["reason"]
    assert report["fd_rms_address_to_impact_m"] is None

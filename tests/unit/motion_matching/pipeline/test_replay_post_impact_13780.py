"""ML-WS-3 (#13780): post-impact replay error growth on capture-A."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import replay_post_impact as rpi

pytestmark = [pytest.mark.unit, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[4]
SEEDS = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)
RECEIPT_PATH = SEEDS / "replay_post_impact_capture_a.json"


def test_resolve_phase_boundaries_rejects_mismatched_shapes() -> None:
    t = np.linspace(0.0, 2.0, 5)
    with pytest.raises(ValueError, match="face_centres"):
        rpi.resolve_phase_boundaries(t, np.zeros((4, 3)))


def test_three_phase_rms_partitions_frames_without_overlap() -> None:
    t = np.linspace(0.0, 2.0, 21)
    bounds = rpi.PhaseBoundaries(
        top_time_s=0.6,
        impact_time_s=1.2,
        top_frame=6,
        impact_frame=12,
    )
    errors = np.linspace(0.01, 0.05, len(t))[:, None] * np.ones((len(t), 2))
    valid = np.ones_like(errors, dtype=bool)
    out = rpi.three_phase_marker_rms(t, errors, valid, bounds)
    total = (
        out["address_to_top_frames"]
        + out["top_to_impact_frames"]
        + out["after_impact_frames"]
    )
    assert total == len(t)
    assert out["fd_rms_after_impact_m"] > out["fd_rms_address_to_top_m"]


def test_assert_post_impact_acceptance_fails_when_post_dominates() -> None:
    report = {
        "phases": {
            "fd_rms_address_to_impact_m": 0.03,
            "fd_rms_after_impact_m": 0.12,
        }
    }
    with pytest.raises(AssertionError, match="post-impact RMS"):
        rpi.assert_post_impact_acceptance(report)


@pytest.fixture(scope="module")
def capture_a_report() -> dict:
    pytest.importorskip("mujoco")
    try:
        from src.motion_capture.capture_registry import resolve_capture

        resolve_capture("capture-A")
    except (OSError, ValueError, ModuleNotFoundError) as exc:
        pytest.skip(f"capture-A unavailable: {exc}")
    return rpi.capture_a_replay_report(seeds_dir=SEEDS)


def test_capture_a_report_splits_at_detected_impact(capture_a_report: dict) -> None:
    phases = capture_a_report["phases"]
    assert phases["fd_rms_address_to_impact_m"] is not None
    assert phases["fd_rms_after_impact_m"] is not None
    assert phases["address_to_impact_frames"] > 0
    assert phases["after_impact_frames"] > 0
    three = phases["three_phase"]
    assert three["top_time_s"] < three["impact_time_s"]
    assert three["top_to_impact_frames"] > 0
    assert capture_a_report["impact_time_s"] == pytest.approx(three["impact_time_s"])


def test_capture_a_address_to_impact_within_documented_band(
    capture_a_report: dict,
) -> None:
    pre = capture_a_report["phases"]["fd_rms_address_to_impact_m"]
    assert pre <= rpi.ADDRESS_TO_IMPACT_DOCUMENTED_MAX_M


def test_capture_a_perturbation_sweep_shows_post_impact_amplification(
    capture_a_report: dict,
) -> None:
    sweep = capture_a_report["perturbation_sweep"]
    sens = sweep["replay_sensitivity"]
    assert sens["growth_rate_after_impact_per_s"] > 0
    assert sens["amplification_after_impact"] > sens["amplification_address_to_impact"]
    assert sweep["timestep"][0]["dt_scale"] == min(rpi.DT_SCALE_FACTORS)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Post-impact RMS exceeds 2x address-to-impact: follow-through reference "
        "branching (#12042), not impact timing or integrator step alone"
    ),
)
def test_capture_a_post_impact_acceptance(capture_a_report: dict) -> None:
    rpi.assert_post_impact_acceptance(capture_a_report)


def test_committed_capture_a_receipt_matches_structure() -> None:
    assert RECEIPT_PATH.is_file(), f"missing receipt: {RECEIPT_PATH}"
    stored = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert stored["capture"] == rpi.CAPTURE_A_ID
    assert "perturbation_sweep" in stored
    assert stored["phases"]["fd_rms_address_to_impact_m"] <= (
        rpi.ADDRESS_TO_IMPACT_DOCUMENTED_MAX_M
    )

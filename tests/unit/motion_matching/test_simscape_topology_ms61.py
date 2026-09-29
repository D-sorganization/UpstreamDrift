"""MS-61 (#10348): Simscape topology + full-marker terminal accuracy contracts.

TDD: these tests define fail-closed software contracts for the 27-DOF reduced
Simscape model (no independent neck) and dual terminal-metric disclosure.
Native R2025b fit/qualification remains a licensed-host gate and must not be
faked when MATLAB is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import (
    GateStatus,
    Horizon,
    evaluate,
)
from src.shared.python.motion_matching.full_marker_terminal import (
    HEAD_MARKER_NAMES,
    TerminalMarkerBreakdown,
    compute_terminal_marker_breakdown,
    evaluate_full_marker_terminal_disclosure,
    require_full_marker_acceptance_terminal,
)
from src.shared.python.motion_matching.simscape_topology import (
    REDUCED_27_COORDINATE_COUNT,
    SimscapeTopologyProfile,
    SimscapeTopologyReport,
    classify_simscape_topology,
    validate_simscape_topology,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN102_DIR = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_102"
)
RUN103_DIR = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_103"
)
RETURNED_CANDIDATE = RUN102_DIR / "returned-candidate.json"
RETURNED_REPLAY = RUN102_DIR / "returned-replay.npz"
RIGIDITY_FLOOR = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "fixed_attachment_rigidity_floor.json"
)


def _run102_identity() -> tuple[list[str], list[str], list[str]]:
    doc = json.loads(RETURNED_CANDIDATE.read_text(encoding="utf-8"))
    return (
        list(doc["coordinate_names"]),
        list(doc["marker_labels"]),
        list(doc["marker_bodies"]),
    )


def test_classify_run102_topology_as_reduced_no_neck() -> None:
    """Run-102 27-DOF model has no independent neck — reduced topology profile."""
    coords, labels, bodies = _run102_identity()
    report = classify_simscape_topology(
        coordinate_names=coords,
        marker_labels=labels,
        marker_bodies=bodies,
    )
    assert isinstance(report, SimscapeTopologyReport)
    assert report.coordinate_count == REDUCED_27_COORDINATE_COUNT
    assert report.has_independent_neck is False
    assert report.profile == SimscapeTopologyProfile.REDUCED_27_NO_NECK
    assert report.head_markers == HEAD_MARKER_NAMES
    assert set(report.head_marker_bodies) == {"Hub"}
    assert (
        "independent neck" in report.limitation.lower()
        or "neck" in report.limitation.lower()
    )


def test_validate_topology_fail_closed_on_length_mismatch() -> None:
    """DbC: marker label/body length mismatch must raise."""
    coords, labels, bodies = _run102_identity()
    with pytest.raises(ValueError, match="marker_bodies"):
        validate_simscape_topology(
            coordinate_names=coords,
            marker_labels=labels,
            marker_bodies=bodies[:-1],
        )


def test_neck_coordinates_flip_topology_profile() -> None:
    """Presence of neck DOFs classifies as full-body-with-neck capability."""
    coords, labels, bodies = _run102_identity()
    coords_with_neck = [*coords, "NeckInputX", "NeckInputY", "NeckInputZ"]
    report = classify_simscape_topology(
        coordinate_names=coords_with_neck,
        marker_labels=labels,
        marker_bodies=bodies,
    )
    assert report.has_independent_neck is True
    assert report.profile == SimscapeTopologyProfile.FULL_BODY_WITH_NECK
    assert report.coordinate_count == REDUCED_27_COORDINATE_COUNT + 3


def test_terminal_breakdown_reports_full_and_head_cluster() -> None:
    """Every terminal receipt must expose full-marker and head-cluster RMS."""
    coords, labels, bodies = _run102_identity()
    with np.load(RETURNED_REPLAY) as raw:
        breakdown = compute_terminal_marker_breakdown(
            pred_markers_m=raw["markers_m"],
            target_markers_m=raw["target_m"],
            valid=raw["valid"],
            marker_labels=labels,
            marker_bodies=bodies,
        )
    assert isinstance(breakdown, TerminalMarkerBreakdown)
    # Pinocchio/Simscape run-102 plateau (~40.3 mm full; head worse).
    assert breakdown.full_marker_terminal_rms_m == pytest.approx(0.0403, abs=5e-4)
    assert breakdown.head_cluster_terminal_rms_m is not None
    assert breakdown.head_cluster_terminal_rms_m > breakdown.full_marker_terminal_rms_m
    assert breakdown.body_excluding_head_terminal_rms_m is not None
    # Body-only is below the 35 mm G1 terminal ceiling — must not be misused.
    assert breakdown.body_excluding_head_terminal_rms_m < 0.035
    assert breakdown.full_marker_terminal_rms_m > 0.035
    payload = breakdown.as_dict()
    assert "full_marker_terminal_rms_m" in payload
    assert "head_cluster_terminal_rms_m" in payload
    assert "body_excluding_head_terminal_rms_m" in payload


def test_body_only_terminal_cannot_satisfy_full_body_acceptance() -> None:
    """Reducing the marker set by dropping head markers must not pass full-body G1."""
    # Craft a receipt whose body-only terminal would pass 35 mm but full fails.
    receipt = {
        "whole_rms_m": 0.020,
        "early_rms_m": 0.010,
        "terminal_rms_m": 0.0403,  # full-marker (honest)
        "club_cluster_rms_m": 0.008,
        "pelvis_yaw_error_pct": 0.6,
        "terminal_breakdown": {
            "full_marker_terminal_rms_m": 0.0403,
            "head_cluster_terminal_rms_m": 0.0723,
            "body_excluding_head_terminal_rms_m": 0.0337,
        },
        "model_profile": "reduced_27_no_neck",
        "acceptance_terminal_source": "body_excluding_head",
    }
    # Disclosure gate must fail closed when acceptance tries to use body-only.
    disclosure = evaluate_full_marker_terminal_disclosure(receipt)
    assert any(g.status == GateStatus.FAILED for g in disclosure)
    assert any(
        "body" in g.reason.lower() or "head" in g.reason.lower() for g in disclosure
    )

    # Even if a caller substitutes body-only into terminal_rms_m, require_helper refuses.
    with pytest.raises(ValueError, match="full.marker|full_marker|head"):
        require_full_marker_acceptance_terminal(
            {
                **receipt,
                "terminal_rms_m": 0.0337,
                "acceptance_terminal_source": "body_excluding_head",
            }
        )


def test_plain_g1_metrics_do_not_require_head_cluster_disclosure() -> None:
    """Ordinary flat G1 metric dicts must not trip MS-61 head-cluster disclosure."""
    metrics = {
        "whole_marker_rmse_m": 0.01,
        "early_marker_rmse_m": 0.005,
        "terminal_marker_rmse_m": 0.01,
        "club_marker_rmse_m": 0.01,
        "pelvis_yaw_rmse_rad": 0.01,
        "max_normal_force_n": 800.0,
        "max_penetration_m": 0.001,
        "max_closure_residual_m": 0.001,
        "max_closure_residual_rad": 0.001,
    }
    disclosure = evaluate_full_marker_terminal_disclosure(metrics)
    assert disclosure == []
    verdict = evaluate(metrics, horizon=Horizon.G1)
    assert "head_cluster_terminal_rms_m" not in {g.name for g in verdict.gates}


def test_evaluate_rejects_missing_head_cluster_disclosure() -> None:
    """Full-body G1 receipts that omit head-cluster terminal fail closed (MS-61)."""
    receipt = {
        "whole_rms_m": 0.020,
        "early_rms_m": 0.010,
        "terminal_rms_m": 0.030,
        "club_cluster_rms_m": 0.008,
        "pelvis_yaw_error_pct": 0.5,
        "model_profile": "full_body",
        # Intentionally omit terminal_breakdown / head_cluster_terminal_rms_m
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    assert verdict.is_physically_accepted is False
    disclosure_names = {g.name for g in verdict.gates}
    assert "head_cluster_terminal_rms_m" in disclosure_names
    head_gate = next(
        g for g in verdict.gates if g.name == "head_cluster_terminal_rms_m"
    )
    assert head_gate.status == GateStatus.MISSING


def test_evaluate_passes_disclosure_when_both_terminals_present() -> None:
    """Disclosure passes when full + head cluster are reported (gate values still apply)."""
    receipt = {
        "whole_rms_m": 0.020,
        "early_rms_m": 0.010,
        "terminal_rms_m": 0.030,
        "club_cluster_rms_m": 0.008,
        "pelvis_yaw_error_pct": 0.5,
        "model_profile": "full_body",
        "terminal_breakdown": {
            "full_marker_terminal_rms_m": 0.030,
            "head_cluster_terminal_rms_m": 0.028,
            "body_excluding_head_terminal_rms_m": 0.031,
        },
        "acceptance_terminal_source": "full_marker",
    }
    disclosure = evaluate_full_marker_terminal_disclosure(receipt)
    assert all(g.status == GateStatus.PASSED for g in disclosure)


def test_run103_native_gate_is_blocked_without_fake_success() -> None:
    """Run-103 scaffolding must document blocked native qualification, not invent a pass."""
    assert RUN103_DIR.is_dir(), "MS-61 must scaffold two_window_fit_9967_103/"
    native_gate = RUN103_DIR / "native_gate.json"
    topology = RUN103_DIR / "topology_report.json"
    terminal = RUN103_DIR / "terminal_breakdown.json"
    runtime = RUN103_DIR / "runtime_license_receipt.json"
    parity = RUN103_DIR / "parity_receipt.json"
    assert native_gate.is_file()
    assert topology.is_file()
    assert terminal.is_file()
    assert runtime.is_file()
    assert parity.is_file()

    gate = json.loads(native_gate.read_text(encoding="utf-8"))
    assert gate["schema_version"] == "simscape-native-gate/1"
    assert gate["issue"] == "#10348"
    assert gate["run_id"] == "two_window_fit_9967_103"
    assert gate["status"] in {"blocked", "failed", "pending_native"}
    assert gate.get("is_physically_accepted") is not True
    assert gate.get("g1_full_marker_terminal_passed") is not True
    assert "ms104" in json.dumps(gate).lower() or "10378" in json.dumps(gate)

    topo = json.loads(topology.read_text(encoding="utf-8"))
    assert topo["profile"] == SimscapeTopologyProfile.REDUCED_27_NO_NECK.value
    assert topo["has_independent_neck"] is False

    breakdown = json.loads(terminal.read_text(encoding="utf-8"))
    assert breakdown["full_marker_terminal_rms_m"] > 0.035
    assert "head_cluster_terminal_rms_m" in breakdown
    # Must not claim G1 success from body-only diagnostic.
    assert breakdown["body_excluding_head_terminal_rms_m"] < 0.035

    license_receipt = json.loads(runtime.read_text(encoding="utf-8"))
    assert license_receipt["matlab_release"] == "2025b"
    assert license_receipt.get("license_cap_assumed") is False

    parity_receipt = json.loads(parity.read_text(encoding="utf-8"))
    assert parity_receipt["max_marker_euclidean_discrepancy_m"] is not None
    assert float(parity_receipt["max_marker_euclidean_discrepancy_m"]) < 1e-3


def test_rigidity_floor_hub_cluster_documented() -> None:
    """Committed rigidity floor remains the independent Hub diagnostic (not dynamics)."""
    assert RIGIDITY_FLOOR.is_file()
    floor = json.loads(RIGIDITY_FLOOR.read_text(encoding="utf-8"))
    assert "Hub" in floor["body_rms_mm"]
    assert floor["body_rms_mm"]["Hub"] > 30.0  # mm; thorax-head coupling floor
    assert "not forward dynamics" in floor["qualification"]

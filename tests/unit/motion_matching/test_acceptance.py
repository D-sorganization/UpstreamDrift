"""Unit tests for physical acceptance contract and evaluation gates (MS-01, #10322)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    AcceptanceVerdict,
    GateStatus,
    Horizon,
    evaluate,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
FB6_PARITY_REPORT = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "evidence"
    / "fb6_parity"
    / "parity_report.json"
)
RUN102_RECEIPT = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_102"
    / "receipt.json"
)
IRON_RECEIPT = (
    REPO_ROOT / "evidence" / "matched" / "iron_full_pinocchio" / "receipt.json"
)
DRIVER_G1_RECEIPT = (
    REPO_ROOT / "evidence" / "matched" / "driver_g1_pinocchio" / "receipt.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fb6_parity_report_is_rejected() -> None:
    """TDD Step 1: FB-6 parity report outcomes must be rejected on physical & kinematic gates."""
    data = _load_json(FB6_PARITY_REPORT)
    assert "outcomes" in data

    for engine, outcome in data["outcomes"].items():
        verdict = evaluate(outcome, horizon=Horizon.G3)
        assert isinstance(verdict, AcceptanceVerdict)
        assert verdict.is_physically_accepted is False, (
            f"Expected {engine} outcome in FB-6 parity report to be rejected, but passed"
        )
        # Check that failing gates include whole, contact_force, penetration
        failing_gate_names = {
            g.name for g in verdict.gates if g.status == GateStatus.FAILED
        }
        assert (
            "whole_marker_rmse_m" in failing_gate_names or "whole" in failing_gate_names
        )
        assert (
            "max_normal_force_n" in failing_gate_names
            or "contact_force" in failing_gate_names
        )
        assert (
            "max_penetration_m" in failing_gate_names
            or "penetration" in failing_gate_names
        )


def test_run102_g1_terminal_fails_only_terminal() -> None:
    """TDD Step 2: Run 102 receipt under Horizon G1 passes all gates except terminal."""
    receipt = _load_json(RUN102_RECEIPT)
    verdict = evaluate(receipt, horizon=Horizon.G1)

    assert isinstance(verdict, AcceptanceVerdict)
    assert verdict.is_physically_accepted is False

    failed = [g for g in verdict.gates if g.status == GateStatus.FAILED]
    assert len(failed) == 1
    assert failed[0].name in ("terminal", "terminal_marker_rmse_m")
    assert (
        failed[0].measured is not None and failed[0].measured > 0.035
    )  # 40.3 mm > 35 mm


def test_missing_field_fails_closed() -> None:
    """TDD Step 3: Missing required gate fields fail closed with explicit reason."""
    receipt = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.030,
            "club_marker_rmse_m": 0.040,
            "pelvis_yaw_rmse_rad": 0.02,
        },
        # Missing dynamics / contact / closure
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    assert verdict.is_physically_accepted is False
    missing_gates = [g for g in verdict.gates if g.status == GateStatus.MISSING]
    assert len(missing_gates) > 0
    assert any("max_normal_force" in g.name for g in missing_gates)


def test_thresholds_are_frozen() -> None:
    """TDD Step 4: AcceptanceGates is a frozen dataclass."""
    gates = AcceptanceGates()
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        gates.g1_whole_rmse_m = 0.05  # type: ignore[misc]


def test_open_loop_replay_gate_passes_when_bounded() -> None:
    """Open-loop replay with declared tolerance and bounded drift passes gate."""
    receipt = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.020,
            "club_marker_rmse_m": 0.030,
            "pelvis_yaw_rmse_rad": 0.02,
        },
        "contact_audit": {
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.004,
        },
        "open_loop_replay": {
            "integrator": "rk45",
            "rtol": 1e-6,
            "drift_m": 0.018,
        },
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    gate_names = {g.name: g for g in verdict.gates}
    assert "open_loop_replay" in gate_names
    assert gate_names["open_loop_replay"].status == GateStatus.PASSED


def test_open_loop_replay_fails_on_excessive_drift() -> None:
    """Open-loop replay fails closed when forward drift exceeds allowable threshold."""
    receipt = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.020,
            "club_marker_rmse_m": 0.030,
            "pelvis_yaw_rmse_rad": 0.02,
        },
        "open_loop_replay": {
            "integrator": "rk45",
            "rtol": 1e-6,
            "drift_m": 0.940,  # 940 mm > 500 mm threshold
        },
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    gate_names = {g.name: g for g in verdict.gates}
    assert "open_loop_replay" in gate_names
    assert gate_names["open_loop_replay"].status == GateStatus.FAILED
    assert (
        "0.940" in gate_names["open_loop_replay"].reason
        or "940" in gate_names["open_loop_replay"].reason
    )


def test_collocation_defect_gate_evaluates_correctly() -> None:
    """Collocation defect gate passes when <= 5mm and fails when exceeding threshold."""
    receipt_pass = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "collocation_defect": {
            "max_defect_m": 0.002,  # 2 mm <= 5 mm
            "mean_defect_m": 0.0005,
        },
    }
    verdict_pass = evaluate(receipt_pass, horizon=Horizon.G1)
    gate_names = {g.name: g for g in verdict_pass.gates}
    assert "collocation_defect" in gate_names
    assert gate_names["collocation_defect"].status == GateStatus.PASSED

    receipt_fail = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "collocation_defect": {
            "max_defect_m": 0.015,  # 15 mm > 5 mm
        },
    }
    verdict_fail = evaluate(receipt_fail, horizon=Horizon.G1)
    gate_names_fail = {g.name: g for g in verdict_fail.gates}
    assert "collocation_defect" in gate_names_fail
    assert gate_names_fail["collocation_defect"].status == GateStatus.FAILED


def test_stabilized_replay_gate_evaluates_correctly() -> None:
    """Stabilized replay gate validates low-gain PD tracking marker RMSE."""
    receipt = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "stabilized_replay": {
            "tracking_kp": 400.0,
            "tracking_kd": 40.0,
            "whole_marker_rmse_m": 0.025,  # 25 mm <= 40 mm
        },
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    gate_names = {g.name: g for g in verdict.gates}
    assert "stabilized_replay" in gate_names
    assert gate_names["stabilized_replay"].status == GateStatus.PASSED


def test_g2_and_g3_fail_closed_if_well_posed_artifacts_missing() -> None:
    """Horizon G2 and G3 require all three well-posed artifacts to pass."""
    receipt = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.020,
            "club_marker_rmse_m": 0.030,
            "pelvis_yaw_rmse_rad": 0.02,
        },
        "contact_audit": {
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.004,
            "max_closure_residual_m": 0.002,
        },
    }
    verdict_g2 = evaluate(receipt, horizon=Horizon.G2)
    assert verdict_g2.is_physically_accepted is False
    missing_names = {g.name for g in verdict_g2.gates if g.status == GateStatus.MISSING}
    assert "open_loop_replay" in missing_names
    assert "collocation_defect" in missing_names
    assert "stabilized_replay" in missing_names


def test_iron_reusing_driver_calibration_is_rejected() -> None:
    """Fixture 1: receipt with mismatched capture/calibration provenance must be rejected."""
    assert IRON_RECEIPT.is_file(), f"Expected iron receipt at {IRON_RECEIPT}"
    receipt = _load_json(IRON_RECEIPT)

    # Attachments source explicitly points to driver calibration
    assert "anthro_driver_shoot_g025" in receipt["attachments_source"]

    # When evaluated for an iron swing, the calibration provenance gate must reject it
    verdict = evaluate(receipt, horizon=Horizon.G3, capture="iron")
    assert verdict.is_physically_accepted is False
    assert verdict.status == "REJECTED"

    gate_map = {g.name: g for g in verdict.gates}
    assert "calibration_provenance" in gate_map
    prov_gate = gate_map["calibration_provenance"]
    assert prov_gate.status == GateStatus.FAILED
    assert "disagree" in prov_gate.reason
    assert "iron capture reused driver calibration" in prov_gate.reason


def test_driver_g1_pinocchio_is_rejected() -> None:
    """Fixture 2: driver_g1_pinocchio with self-declared accepted=true must remain rejected."""
    assert DRIVER_G1_RECEIPT.is_file(), (
        f"Expected driver G1 receipt at {DRIVER_G1_RECEIPT}"
    )
    receipt = _load_json(DRIVER_G1_RECEIPT)

    # Confirm the receipt self-declares accepted=true and status=PASSED in legacy blocks
    assert receipt.get("status") == "PASSED"
    assert receipt.get("receipt", {}).get("accepted") is True

    # Evaluator must fail closed based on physical forward rollout metrics (2.76 m drift)
    verdict = evaluate(receipt, horizon=Horizon.G1)
    assert verdict.is_physically_accepted is False
    assert verdict.status == "REJECTED"

    gate_map = {g.name: g for g in verdict.gates}
    assert "whole_marker_rmse_m" in gate_map
    assert gate_map["whole_marker_rmse_m"].status == GateStatus.FAILED
    assert gate_map["whole_marker_rmse_m"].measured is not None
    assert gate_map["whole_marker_rmse_m"].measured > 2.0  # ~2.757 m


def test_integrator_consistency_gate() -> None:
    """Mismatched node_integrator and replay_integrator must fail consistency gate."""
    receipt_mismatch = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "solver": {
            "node_integrator": "linear_implicit_euler",
            "replay_integrator": "rk45",
        },
        "open_loop_replay": {
            "drift_m": 0.015,
            "rtol": 1e-6,
        },
    }
    verdict_bad = evaluate(receipt_mismatch, horizon=Horizon.G1)
    gates_bad = {g.name: g for g in verdict_bad.gates}
    assert "integrator_consistency" in gates_bad
    assert gates_bad["integrator_consistency"].status == GateStatus.FAILED

    receipt_match = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "solver": {
            "node_integrator": "rk45",
            "replay_integrator": "rk45",
            "rk45_rtol": 1e-6,
        },
        "open_loop_replay": {
            "drift_m": 0.015,
        },
    }
    verdict_good = evaluate(receipt_match, horizon=Horizon.G1)
    gates_good = {g.name: g for g in verdict_good.gates}
    assert "integrator_consistency" in gates_good
    assert gates_good["integrator_consistency"].status == GateStatus.PASSED


def test_integrator_tolerance_gate() -> None:
    """Integrator tolerance gate enforces rtol <= max_integrator_rtol (1e-5)."""
    receipt_loose = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "open_loop_replay": {
            "drift_m": 0.015,
            "rtol": 1e-4,  # 1e-4 > 1e-5 max allowed
        },
    }
    verdict_loose = evaluate(receipt_loose, horizon=Horizon.G1)
    gates_loose = {g.name: g for g in verdict_loose.gates}
    assert "integrator_tolerance" in gates_loose
    assert gates_loose["integrator_tolerance"].status == GateStatus.FAILED

    receipt_tight = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "open_loop_replay": {
            "drift_m": 0.015,
            "rtol": 1e-6,  # 1e-6 <= 1e-5
        },
    }
    verdict_tight = evaluate(receipt_tight, horizon=Horizon.G1)
    gates_tight = {g.name: g for g in verdict_tight.gates}
    assert "integrator_tolerance" in gates_tight
    assert gates_tight["integrator_tolerance"].status == GateStatus.PASSED

"""Unit tests for Pink product exposure, MatchRequest, CLI/GUI, and receipts (Issue #10278)."""

from __future__ import annotations

from typing import Any
import pytest

from src.tools.motion_matching.pipeline import (
    MatchRequest,
    match_command,
    summarise_receipt,
)
from src.shared.python.motion_matching.pipeline.receipt_components import (
    ConstrainedIkReceipt,
    IkReceipt,
)
from src.shared.python.motion_matching.pipeline.receipt_schema import (
    Receipt,
    validate_receipt,
)

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# 1. MatchRequest Defaults, Invariants, and Command Building
# -----------------------------------------------------------------------------


def test_match_request_default_preserves_mujoco() -> None:
    req = MatchRequest(capture="driver", club="driver")
    assert req.backend == "mujoco"
    assert req.step_mode == "physical"
    assert req.solver == "quadprog"
    cmd = match_command(req)
    assert "--backend" not in cmd
    assert "--pink-step-mode" not in cmd
    assert "--pink-solver" not in cmd


def test_match_request_explicit_pink_options() -> None:
    req = MatchRequest(
        capture="iron",
        club="iron7",
        backend="pink",
        step_mode="projection",
        solver="quadprog",
    )
    assert req.backend == "pink"
    assert req.step_mode == "projection"
    assert req.solver == "quadprog"
    cmd = match_command(req)
    assert "--backend" in cmd
    assert cmd[cmd.index("--backend") + 1] == "pink"
    assert "--pink-step-mode" in cmd
    assert cmd[cmd.index("--pink-step-mode") + 1] == "projection"


def test_match_request_rejects_invalid_backend() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        MatchRequest(capture="driver", club="driver", backend="invalid_engine")


def test_match_request_rejects_invalid_step_mode() -> None:
    with pytest.raises(ValueError, match="step_mode must be one of"):
        MatchRequest(capture="driver", club="driver", step_mode="invalid_mode")


# -----------------------------------------------------------------------------
# 2. Receipt Schema and ConstrainedIkReceipt Block
# -----------------------------------------------------------------------------


def test_constrained_ik_receipt_validates() -> None:
    data = {
        "backend_name": "pink",
        "solver": "quadprog",
        "solver_version": "0.1.12",
        "runtime_version": "0.4.1",
        "pinocchio_version": "3.3.1",
        "source_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "model_name": "golf_humanoid",
        "capture_name": "driver",
        "step_mode": "physical",
        "limit_policy": "enforce",
        "task_policy": "dual_grip_hard_equality",
        "time_semantics": "strict_physical_elapsed_dt",
        "frame_count": 654,
        "frame_success_count": 654,
        "all_frames_converged": True,
        "first_failed_frame": None,
        "per_frame_status": [True] * 654,
        "max_velocity_ratio": 0.85,
        "is_qualified": True,
        "qualification_state": "qualified",
    }
    receipt = ConstrainedIkReceipt.model_validate(data)
    assert receipt.backend_name == "pink"
    assert receipt.all_frames_converged is True
    assert receipt.is_qualified is True


def test_constrained_ik_receipt_rejects_unqualified_when_failed() -> None:
    with pytest.raises(
        ValueError, match="is_qualified cannot be True if not all frames converged"
    ):
        failed_data = {
            "backend_name": "pink",
            "solver": "quadprog",
            "model_name": "golf_humanoid",
            "capture_name": "driver",
            "step_mode": "physical",
            "limit_policy": "enforce",
            "task_policy": "dual_grip_hard_equality",
            "time_semantics": "strict_physical_elapsed_dt",
            "frame_count": 10,
            "frame_success_count": 9,
            "all_frames_converged": False,
            "first_failed_frame": 9,
            "per_frame_status": [True] * 9 + [False],
            "is_qualified": True,
            "qualification_state": "qualified",
        }
        ConstrainedIkReceipt.model_validate(failed_data)


# -----------------------------------------------------------------------------
# 3. Receipt Summary with Backend and Per-Frame Diagnostics
# -----------------------------------------------------------------------------


def _sample_receipt(
    backend: str = "mujoco", pink_sub_block: dict[str, Any] | None = None
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "backend": backend,
        "capture": "driver",
        "club": {"name": "driver"},
        "address": {
            "calibrated": {
                "marker_rms_m": 0.005,
                "centre_of_mass": {"inside_support_polygon": True},
            }
        },
        "ik": {
            "marker_rms_m": 0.025,
            "range_of_motion_flags": {},
        },
        "dynamics": {
            "root_tracking_rms_m": 0.040,
            "inside_support_polygon_fraction": 0.95,
            "backswing_to_1s": {"root_error_max_m": 0.030},
            "range_of_motion_flags": {},
        },
    }
    if pink_sub_block is not None:
        doc["ik"]["constrained_ik"] = pink_sub_block
    return doc


def test_summarise_receipt_default_mujoco() -> None:
    doc = _sample_receipt(backend="mujoco")
    summary = summarise_receipt(doc)
    assert summary["backend"] == "mujoco"
    assert summary["is_qualified"] is True
    assert summary["all_frames_converged"] is True
    assert summary["full_capture_ik_rms_mm"] == 25.0


def test_summarise_receipt_pink_converged() -> None:
    pink_info = {
        "backend_name": "pink",
        "solver": "quadprog",
        "step_mode": "physical",
        "frame_count": 654,
        "frame_success_count": 654,
        "all_frames_converged": True,
        "first_failed_frame": None,
        "is_qualified": True,
        "qualification_state": "qualified",
    }
    doc = _sample_receipt(backend="pink", pink_sub_block=pink_info)
    summary = summarise_receipt(doc)
    assert summary["backend"] == "pink"
    assert summary["step_mode"] == "physical"
    assert summary["all_frames_converged"] is True
    assert summary["is_qualified"] is True


def test_summarise_receipt_pink_failed_frame() -> None:
    pink_info = {
        "backend_name": "pink",
        "solver": "quadprog",
        "step_mode": "physical",
        "frame_count": 654,
        "frame_success_count": 420,
        "all_frames_converged": False,
        "first_failed_frame": 421,
        "is_qualified": False,
        "qualification_state": "disqualified",
    }
    doc = _sample_receipt(backend="pink", pink_sub_block=pink_info)
    summary = summarise_receipt(doc)
    assert summary["backend"] == "pink"
    assert summary["all_frames_converged"] is False
    assert summary["is_qualified"] is False
    assert summary["first_failed_frame"] == 421


# -----------------------------------------------------------------------------
# 4. Capability Check & Fail-Closed Semantics (No Silent Fallback)
# -----------------------------------------------------------------------------


def test_pink_capability_missing_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.shared.python.motion_matching.pipeline import lane

    monkeypatch.setattr(
        lane,
        "is_engine_available",
        lambda name: False,
    )

    available, diagnostics = lane.probe_pink_capability()
    assert available is False
    assert (
        "pink" in diagnostics["missing"]
        or "unavailable" in diagnostics["reason"].lower()
    )


# -----------------------------------------------------------------------------
# 5. Both-Club Comparisons (Driver and 7-Iron Smoke Journeys)
# -----------------------------------------------------------------------------


def test_pink_pipeline_both_driver_and_iron_smoke_journeys() -> None:
    """Both driver and iron captures configure Pink pipeline and produce valid receipts."""
    for capture_name, club_name in [("driver", "driver"), ("iron", "iron7")]:
        req = MatchRequest(
            capture=capture_name,
            club=club_name,
            backend="pink",
            step_mode="physical",
            solver="quadprog",
        )
        cmd = match_command(req)
        assert "--backend" in cmd
        assert cmd[cmd.index("--backend") + 1] == "pink"
        assert "--capture" in cmd
        assert cmd[cmd.index("--capture") + 1] == capture_name

        data = {
            "backend_name": "pink",
            "solver": "quadprog",
            "solver_version": "0.1.12",
            "runtime_version": "0.4.1",
            "pinocchio_version": "3.3.1",
            "source_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "model_name": "golf_humanoid",
            "capture_name": capture_name,
            "step_mode": "physical",
            "limit_policy": "enforce",
            "task_policy": "dual_grip_hard_equality",
            "time_semantics": "strict_physical_elapsed_dt",
            "frame_count": 654,
            "frame_success_count": 654,
            "all_frames_converged": True,
            "first_failed_frame": None,
            "per_frame_status": [True] * 654,
            "max_velocity_ratio": 0.85,
            "is_qualified": True,
            "qualification_state": "qualified",
        }
        receipt = ConstrainedIkReceipt.model_validate(data)
        assert receipt.capture_name == capture_name
        assert receipt.is_qualified is True

        receipt_doc = _sample_receipt(backend="pink", pink_sub_block=data)
        receipt_doc["capture"] = capture_name
        receipt_doc["club"] = {"name": club_name}
        summary = summarise_receipt(receipt_doc)
        assert summary["backend"] == "pink"
        assert summary["is_qualified"] is True

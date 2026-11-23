"""Qualification ladder must follow AcceptanceGates, not hardcoded Passed rows (#15244)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_matched_swing_status import (
    _format_ladder_engine_cell,
    _render_qualification_ladder,
    render_matched_swing_status,
)
from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    AcceptanceVerdict,
    GateResult,
    GateStatus,
    Horizon,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "reports" / "matched_swing_ledger.json"

SUPERSEDED_PASSED_SNIPPETS = (
    "Passed (27.3 mm)",
    "Passed (74.6 mm)",
)


def test_committed_ladder_rejects_superseded_hardcoded_passed_rows() -> None:
    if not LEDGER_PATH.is_file():
        pytest.skip("matched_swing_ledger.json not present")
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    rendered = render_matched_swing_status(ledger)
    for snippet in SUPERSEDED_PASSED_SNIPPETS:
        assert snippet not in rendered, (
            "superseded widened-bound receipts must not render as active Passed"
        )
    assert "superseded" in rendered.lower()
    lines = _render_qualification_ladder(ledger.get("rows", []), repo_root=REPO_ROOT)
    ladder_block = "\n".join(lines)
    assert "g1_whole_rmse_m" in ladder_block or "AcceptanceGates" in ladder_block


def test_format_ladder_cell_passed_only_when_gates_accept() -> None:
    gates = AcceptanceGates()
    threshold_m = gates.g1_whole_rmse_m
    failing = AcceptanceVerdict(
        horizon=Horizon.G1,
        is_physically_accepted=False,
        status="REJECTED",
        gates=(
            GateResult(
                name="whole_marker_rmse_m",
                status=GateStatus.FAILED,
                threshold=threshold_m,
                measured=0.031,
                unit="m",
                reason="whole RMSE 31.00 mm > 25.00 mm",
            ),
        ),
    )
    cell = _format_ladder_engine_cell(failing, threshold_m, saw_engine_rows=True)
    assert "Passed" not in cell
    assert "31" in cell
    assert "Failed" in cell

    passing = AcceptanceVerdict(
        horizon=Horizon.G1,
        is_physically_accepted=True,
        status="PASSED",
        gates=(
            GateResult(
                name="whole_marker_rmse_m",
                status=GateStatus.PASSED,
                threshold=threshold_m,
                measured=0.024,
                unit="m",
                reason="",
            ),
        ),
    )
    passed_cell = _format_ladder_engine_cell(passing, threshold_m, saw_engine_rows=True)
    assert passed_cell.startswith("✅ Passed")
    assert "24" in passed_cell


def _rejected(*gates: GateResult) -> AcceptanceVerdict:
    return AcceptanceVerdict(
        horizon=Horizon.G1,
        is_physically_accepted=False,
        status="REJECTED",
        gates=gates,
    )


def _whole(status: GateStatus, measured: float | None, threshold: float) -> GateResult:
    return GateResult(
        name="whole_marker_rmse_m",
        status=status,
        threshold=threshold,
        measured=measured,
        unit="m",
    )


def test_cell_passing_whole_rmse_with_missing_gate_is_unverified_not_failed() -> None:
    """Passing whole RMSE + MISSING club gate must not claim '20.0 mm > 25 mm' (#15244)."""
    threshold_m = AcceptanceGates().g1_whole_rmse_m
    verdict = _rejected(
        _whole(GateStatus.PASSED, 0.020, threshold_m),
        GateResult(
            name="club_marker_rmse_m",
            status=GateStatus.MISSING,
            threshold=0.04,
            measured=None,
        ),
    )
    cell = _format_ladder_engine_cell(verdict, threshold_m, saw_engine_rows=True)
    assert "Unverified" in cell
    assert "club_marker_rmse_m" in cell
    assert ">" not in cell
    assert "Failed" not in cell


def test_cell_passing_whole_rmse_with_other_failed_gate_names_that_gate() -> None:
    """A different failed gate is reported by name, never as a whole-RMSE excess."""
    threshold_m = AcceptanceGates().g1_whole_rmse_m
    verdict = _rejected(
        _whole(GateStatus.PASSED, 0.020, threshold_m),
        GateResult(
            name="max_penetration_m",
            status=GateStatus.FAILED,
            threshold=0.005,
            measured=0.009,
        ),
    )
    cell = _format_ladder_engine_cell(verdict, threshold_m, saw_engine_rows=True)
    assert cell.startswith("❌ Failed")
    assert "max_penetration_m" in cell
    assert "20.0 mm" not in cell
    assert ">" not in cell


def test_cell_failed_whole_rmse_uses_the_gates_own_threshold() -> None:
    """The numeric '>' comparison appears only for a measured whole-RMSE failure."""
    verdict = _rejected(_whole(GateStatus.FAILED, 0.0268, 0.040))
    cell = _format_ladder_engine_cell(verdict, 0.025, saw_engine_rows=True)
    assert cell == "❌ Failed (26.8 mm > 40 mm)"


def test_cell_failed_other_gate_beats_missing_gate() -> None:
    """A measured failure is definitive, so it outranks missing evidence."""
    threshold_m = AcceptanceGates().g1_whole_rmse_m
    verdict = _rejected(
        _whole(GateStatus.PASSED, 0.020, threshold_m),
        GateResult(
            name="club_marker_rmse_m",
            status=GateStatus.MISSING,
            threshold=0.04,
            measured=None,
        ),
        GateResult(
            name="max_penetration_m",
            status=GateStatus.FAILED,
            threshold=0.005,
            measured=0.009,
        ),
    )
    cell = _format_ladder_engine_cell(verdict, threshold_m, saw_engine_rows=True)
    assert cell.startswith("❌ Failed")
    assert "max_penetration_m" in cell

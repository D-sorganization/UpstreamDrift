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

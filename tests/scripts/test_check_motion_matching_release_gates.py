"""Unit tests for check_motion_matching_release_gates CLI script (MMR-17 #11103)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.ci import check_motion_matching_release_gates as cli
from src.shared.python.motion_matching.release_gates import (
    EngineTestStatus,
    ReleaseVerdict,
)

pytestmark = pytest.mark.unit


def test_discover_nightly_receipts_from_fixtures(tmp_path: Path) -> None:
    """Receipt discovery accurately indexes engine files."""
    nightly_dir = tmp_path / "nightly"
    nightly_dir.mkdir()

    receipt_file = nightly_dir / "drake_receipt.json"
    receipt_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "engine": "drake",
                "status": "fail",
                "engine_inventory": {"available": False},
            }
        ),
        encoding="utf-8",
    )

    discovered = cli.discover_nightly_receipts(nightly_dir)
    assert "drake" in discovered
    assert discovered["drake"]["engine"] == "drake"


def test_discover_club_receipts_from_fixtures(tmp_path: Path) -> None:
    """Dual club discovery indexes driver and 7-iron files."""
    ev_dir = tmp_path / "evidence"
    sub_dir = ev_dir / "drake"
    sub_dir.mkdir(parents=True)

    driver_file = sub_dir / "driver_receipt.json"
    driver_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "engine": "drake",
                "club": "driver",
                "status": "qualified",
            }
        ),
        encoding="utf-8",
    )

    iron_file = sub_dir / "iron_receipt.json"
    iron_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "engine": "drake",
                "club": "7-iron",
                "status": "qualified",
            }
        ),
        encoding="utf-8",
    )

    discovered = cli.discover_club_receipts(ev_dir)
    assert "drake" in discovered
    assert "driver" in discovered["drake"]
    assert "7-iron" in discovered["drake"]


def test_cli_audit_only_flag_exits_zero(tmp_path: Path, capsys) -> None:
    """The --audit-only flag outputs the matrix and returns code 0."""
    matrix_out = tmp_path / "matrix.json"
    code = cli.main(["--audit-only", "--matrix-out", str(matrix_out)])
    assert code == 0
    assert matrix_out.is_file()

    captured = capsys.readouterr().out
    assert "Motion Matching Release Gate Matrix" in captured
    assert "Verdict:" in captured

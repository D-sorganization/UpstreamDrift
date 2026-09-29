"""MatchedSwingBrowserModel parity-artifact resolution (#10960 P0-9).

- The parity artifact resolves to reevaluation.json when it sits beside the receipt.
- parity_vs_mujoco.json is used only when no reevaluation exists.
- A parity file never overrides the ledger row's own acceptance verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.ledger_schema import (
    ArtefactPaths,
    LedgerRow,
    SharedMetrics,
)
from src.tools.matched_swing_browser.model import MatchedSwingBrowserModel

pytestmark = pytest.mark.unit


def _create_mock_row(
    tmp_path: Path,
    include_parity: bool = False,
    include_reevaluation: bool = False,
    parity_status: str = "PASSED",
    reevaluation_status: str = "REJECTED",
    acceptance: dict | None = None,
) -> tuple[LedgerRow, Path]:
    run_dir = tmp_path / "evidence" / "matched" / "mock_engine"
    run_dir.mkdir(parents=True, exist_ok=True)

    receipt_file = run_dir / "receipt.json"
    receipt_file.write_text(json.dumps({"status": "PASSED"}), encoding="utf-8")

    if include_parity:
        parity_file = run_dir / "parity_vs_mujoco.json"
        parity_file.write_text(
            json.dumps(
                {"status": parity_status, "parity_passed": (parity_status == "PASSED")}
            ),
            encoding="utf-8",
        )

    if include_reevaluation:
        reeval_file = run_dir / "reevaluation.json"
        reeval_file.write_text(
            json.dumps(
                {
                    "verdict": {
                        "status": reevaluation_status,
                        "is_physically_accepted": (reevaluation_status == "PASSED"),
                    }
                }
            ),
            encoding="utf-8",
        )

    rel_receipt_path = receipt_file.relative_to(tmp_path).as_posix()
    row = LedgerRow(
        receipt_path=rel_receipt_path,
        sha256="0" * 64,
        engine="mock_engine",
        lane="matched",
        capture="driver",
        metrics=SharedMetrics(),
        artefacts=ArtefactPaths(),
        acceptance=acceptance,
    )
    return row, run_dir


class TestMatchedSwingBrowserModelVerdicts:
    def test_prefers_reevaluation_over_parity_when_both_exist(
        self, tmp_path: Path
    ) -> None:
        row, run_dir = _create_mock_row(
            tmp_path,
            include_parity=True,
            include_reevaluation=True,
            parity_status="PASSED",
            reevaluation_status="REJECTED",
        )
        model = MatchedSwingBrowserModel(repo_root=tmp_path)

        parity_art = model.resolve_artifact_path(row, "parity")
        assert parity_art is not None
        assert parity_art.name == "reevaluation.json"

    def test_uses_parity_when_reevaluation_absent(self, tmp_path: Path) -> None:
        row, run_dir = _create_mock_row(
            tmp_path,
            include_parity=True,
            include_reevaluation=False,
            parity_status="PASSED",
        )
        model = MatchedSwingBrowserModel(repo_root=tmp_path)

        parity_art = model.resolve_artifact_path(row, "parity")
        assert parity_art is not None
        assert parity_art.name == "parity_vs_mujoco.json"

    def test_uses_reevaluation_when_parity_absent(self, tmp_path: Path) -> None:
        row, run_dir = _create_mock_row(
            tmp_path,
            include_parity=False,
            include_reevaluation=True,
            reevaluation_status="REJECTED",
        )
        model = MatchedSwingBrowserModel(repo_root=tmp_path)

        parity_art = model.resolve_artifact_path(row, "parity")
        assert parity_art is not None
        assert parity_art.name == "reevaluation.json"

    def test_parity_file_never_overrides_row_acceptance(self, tmp_path: Path) -> None:
        row, _ = _create_mock_row(
            tmp_path,
            include_parity=True,
            parity_status="PASSED",
            acceptance={"status": "REJECTED", "is_physically_accepted": False},
        )
        model = MatchedSwingBrowserModel(repo_root=tmp_path)

        assert model.extract_verdict_string(row) != "PASSED"

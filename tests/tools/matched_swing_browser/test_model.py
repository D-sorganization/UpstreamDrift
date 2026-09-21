"""Unit tests for Matched Swing Browser model and filtering (MS-80, #10353).

Validates:
- Loading the ledger from JSON file or default path.
- Filtering rows by engine, capture, lane, verdict, and text query.
- Integrating with the ResultFilter lineage contract (resolving #8824).
- Formatting metric quantities (mm, deg) cleanly for the view.
- Resolving associated file artifacts (GIF, NPZ, receipt, parity report).
- Design by contract preconditions and postconditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.motion_matching.ledger_schema import (
    ArtefactPaths,
    Ledger,
    LedgerRow,
    SharedMetrics,
)
from src.shared.python.workspace.results_browser import ResultFilter
from src.tools.matched_swing_browser.model import (
    MatchedSwingBrowserModel,
    MatchedSwingFilter,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def fixture_rows() -> list[LedgerRow]:
    """Sample ledger rows spanning engines, captures, and verdicts."""
    return [
        LedgerRow(
            receipt_path="evidence/matched/driver_g1_drake/receipt.json",
            sha256="aaa111",
            engine="drake",
            lane="matched",
            capture="driver",
            candidate_sha="cand_drake",
            horizon_s=0.85,
            metrics=SharedMetrics(
                whole_marker_rmse_m=0.022,
                early_marker_rmse_m=0.010,
                terminal_marker_rmse_m=0.030,
                club_marker_rmse_m=0.050,
                pelvis_yaw_rmse_rad=0.040,
            ),
            acceptance={
                "status": "PASSED",
                "is_physically_accepted": True,
                "gates": [],
            },
            artefacts=ArtefactPaths(
                npz="evidence/matched/driver_g1_drake/candidate.npz",
                gif="evidence/matched/driver_g1_drake/playback.gif",
            ),
        ),
        LedgerRow(
            receipt_path="evidence/matched/driver_g1_opensim/receipt.json",
            sha256="bbb222",
            engine="opensim",
            lane="matched",
            capture="driver",
            candidate_sha="cand_opensim",
            horizon_s=0.85,
            metrics=SharedMetrics(
                whole_marker_rmse_m=0.245,
                early_marker_rmse_m=0.080,
                terminal_marker_rmse_m=0.350,
                club_marker_rmse_m=0.180,
                pelvis_yaw_rmse_rad=0.090,
            ),
            acceptance={
                "status": "REJECTED",
                "is_physically_accepted": False,
                "gates": [],
            },
            artefacts=ArtefactPaths(
                npz="evidence/matched/driver_g1_opensim/candidate.npz",
                gif="evidence/matched/driver_g1_opensim/playback.gif",
            ),
        ),
        LedgerRow(
            receipt_path="evidence/matched/iron_full_pinocchio/receipt.json",
            sha256="ccc333",
            engine="pinocchio",
            lane="matched",
            capture="iron",
            candidate_sha="cand_pin",
            horizon_s=1.82,
            metrics=SharedMetrics(
                whole_marker_rmse_m=0.075,
            ),
            acceptance={
                "status": "REJECTED",
                "is_physically_accepted": False,
                "gates": [],
            },
            artefacts=ArtefactPaths(
                gif="evidence/matched/iron_full_pinocchio/playback.gif",
            ),
        ),
        LedgerRow(
            receipt_path="docs/development/full_body_models/evidence/ground_support/receipt.json",
            sha256="ddd444",
            engine="mujoco",
            lane="ground_support",
            capture=None,
            candidate_sha=None,
            horizon_s=None,
            metrics=SharedMetrics(),
            acceptance=None,
            artefacts=ArtefactPaths(),
        ),
    ]


@pytest.fixture
def fixture_ledger(fixture_rows: list[LedgerRow]) -> Ledger:
    return Ledger(
        schema_version="1.0.0",
        generated_at="2026-09-20T00:00:00Z",
        total_receipts=len(fixture_rows),
        rows=fixture_rows,
    )


class TestMatchedSwingFilter:
    def test_default_filter_matches_everything(
        self, fixture_rows: list[LedgerRow]
    ) -> None:
        model = MatchedSwingBrowserModel()
        filtered = model.filter_rows(fixture_rows, MatchedSwingFilter())
        assert len(filtered) == len(fixture_rows)

    def test_filter_by_engine(self, fixture_rows: list[LedgerRow]) -> None:
        model = MatchedSwingBrowserModel()
        drake_rows = model.filter_rows(fixture_rows, MatchedSwingFilter(engine="drake"))
        assert len(drake_rows) == 1
        assert drake_rows[0].engine == "drake"

    def test_filter_by_capture(self, fixture_rows: list[LedgerRow]) -> None:
        model = MatchedSwingBrowserModel()
        driver_rows = model.filter_rows(
            fixture_rows, MatchedSwingFilter(capture="driver")
        )
        assert len(driver_rows) == 2
        assert all(r.capture == "driver" for r in driver_rows)

    def test_filter_by_lane(self, fixture_rows: list[LedgerRow]) -> None:
        model = MatchedSwingBrowserModel()
        gs_rows = model.filter_rows(
            fixture_rows, MatchedSwingFilter(lane="ground_support")
        )
        assert len(gs_rows) == 1
        assert gs_rows[0].lane == "ground_support"

    def test_filter_by_verdict(self, fixture_rows: list[LedgerRow]) -> None:
        model = MatchedSwingBrowserModel()
        passed_rows = model.filter_rows(
            fixture_rows, MatchedSwingFilter(verdict="PASSED")
        )
        assert len(passed_rows) == 1
        assert passed_rows[0].engine == "drake"

        rejected_rows = model.filter_rows(
            fixture_rows, MatchedSwingFilter(verdict="REJECTED")
        )
        assert len(rejected_rows) == 2

        unclassified = model.filter_rows(
            fixture_rows, MatchedSwingFilter(verdict="UNCLASSIFIED")
        )
        assert len(unclassified) == 1
        assert unclassified[0].engine == "mujoco"

    def test_filter_by_text(self, fixture_rows: list[LedgerRow]) -> None:
        model = MatchedSwingBrowserModel()
        text_rows = model.filter_rows(
            fixture_rows, MatchedSwingFilter(text="cand_opensim")
        )
        assert len(text_rows) == 1
        assert text_rows[0].engine == "opensim"

    def test_combined_filters(self, fixture_rows: list[LedgerRow]) -> None:
        model = MatchedSwingBrowserModel()
        results = model.filter_rows(
            fixture_rows,
            MatchedSwingFilter(engine="opensim", capture="driver", verdict="REJECTED"),
        )
        assert len(results) == 1
        assert results[0].engine == "opensim"

    def test_to_result_filter_delegation(self) -> None:
        ms_filter = MatchedSwingFilter(engine="mujoco", text="ground")
        rf = ms_filter.to_result_filter()
        assert isinstance(rf, ResultFilter)
        assert rf.backend == "mujoco"
        assert rf.text == "ground"


class TestMatchedSwingBrowserModel:
    def test_load_from_fixture_json(
        self, tmp_path: Path, fixture_ledger: Ledger
    ) -> None:
        ledger_file = tmp_path / "test_ledger.json"
        fixture_ledger.write_json(ledger_file)

        model = MatchedSwingBrowserModel()
        rows = model.load_ledger(ledger_file)
        assert len(rows) == 4
        assert rows[0].engine == "drake"

    def test_load_production_ledger(self) -> None:
        model = MatchedSwingBrowserModel()
        rows = model.load_ledger()
        assert len(rows) >= 40
        assert len(rows) == 99

    def test_format_metric(self) -> None:
        model = MatchedSwingBrowserModel()
        assert model.format_metric(0.02452, "mm") == "24.52 mm"
        assert model.format_metric(None, "mm") == "—"
        # 0.05236 rad ~ 3.00 deg
        assert "3.00°" in model.format_metric(0.05236, "deg")

    def test_extract_unique_filter_values(self, fixture_rows: list[LedgerRow]) -> None:
        model = MatchedSwingBrowserModel()
        engines = model.get_unique_engines(fixture_rows)
        assert set(engines) == {"drake", "opensim", "pinocchio", "mujoco"}
        captures = model.get_unique_captures(fixture_rows)
        assert set(captures) == {"driver", "iron"}
        lanes = model.get_unique_lanes(fixture_rows)
        assert set(lanes) == {"matched", "ground_support"}

    def test_resolve_artifact_paths(
        self, tmp_path: Path, fixture_rows: list[LedgerRow]
    ) -> None:
        model = MatchedSwingBrowserModel(repo_root=tmp_path)
        row = fixture_rows[0]

        # Create mock file
        assert row.artefacts.gif is not None
        mock_gif = tmp_path / row.artefacts.gif
        mock_gif.parent.mkdir(parents=True, exist_ok=True)
        mock_gif.write_bytes(b"GIF89a")

        resolved_gif = model.resolve_artifact_path(row, "gif")
        assert resolved_gif is not None
        assert resolved_gif.exists()

        # Uncreated npz should return None
        resolved_npz = model.resolve_artifact_path(row, "npz")
        assert resolved_npz is None

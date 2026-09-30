"""Unit tests for comparable-candidate filtering and best-candidate ranking (MMR-16, #11102).

Validates:
1. MatchedSwingFilter supports drive_mode and profile filtering.
2. MatchedSwingBrowserModel ranks comparable candidates by whole_marker_rmse_m.
3. Candidate selection forwards honest acceptance verdicts, drive modes, and candidate SHAs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest

from src.shared.python.motion_matching.ledger_schema import (
    ArtefactPaths,
    LedgerRow,
    SharedMetrics,
)
from src.tools.matched_swing_browser.model import (
    MatchedSwingBrowserModel,
    MatchedSwingFilter,
)

pytestmark = pytest.mark.unit


def _make_row(
    receipt_path: str,
    engine: str,
    rmse: float,
    *,
    capture: str = "driver",
    lane: str = "matched",
    drive_mode: str = "torque_driven",
    status: str = "PASSED",
    candidate_sha: str = "abc12345",
) -> LedgerRow:
    return LedgerRow(
        receipt_path=receipt_path,
        sha256="f" * 64,
        engine=engine,
        lane=lane,
        capture=capture,
        candidate_sha=candidate_sha,
        metrics=SharedMetrics(whole_marker_rmse_m=rmse),
        artefacts=ArtefactPaths(npz=f"evidence/{engine}_replay.npz"),
        acceptance={"status": status, "is_physically_accepted": (status == "PASSED")},
        reason=f"drive_mode:{drive_mode}",
    )


class TestMatchedSwingComparableFilters:
    def test_filter_by_drive_mode(self) -> None:
        model = MatchedSwingBrowserModel()
        rows = [
            _make_row("r1.json", "mujoco", 0.012, drive_mode="torque_driven"),
            _make_row("r2.json", "pinocchio", 0.015, drive_mode="kinematic_prescribed"),
            _make_row("r3.json", "drake", 0.018, drive_mode="torque_driven"),
        ]

        crit = MatchedSwingFilter(drive_mode="torque_driven")
        filtered = model.filter_rows(rows, crit)
        assert len(filtered) == 2
        assert {r.engine for r in filtered} == {"mujoco", "drake"}

    def test_rank_comparable_candidates_ascending_rmse(self) -> None:
        model = MatchedSwingBrowserModel()
        rows = [
            _make_row("r1.json", "mujoco", 0.024, capture="driver"),
            _make_row("r2.json", "pinocchio", 0.011, capture="driver"),
            _make_row("r3.json", "drake", 0.017, capture="driver"),
            _make_row(
                "r4.json", "simscape", 0.009, capture="iron"
            ),  # different capture
        ]

        ranked = model.rank_candidates(rows, capture="driver")
        assert len(ranked) == 3
        # Best candidate first (smallest RMSE)
        assert ranked[0].engine == "pinocchio"
        assert ranked[0].metrics.whole_marker_rmse_m == 0.011
        assert ranked[1].engine == "drake"
        assert ranked[1].metrics.whole_marker_rmse_m == 0.017
        assert ranked[2].engine == "mujoco"
        assert ranked[2].metrics.whole_marker_rmse_m == 0.024

    def test_rank_candidates_preserves_rejected_verdict_honesty(self) -> None:
        model = MatchedSwingBrowserModel()
        rows = [
            _make_row("r1.json", "mujoco", 0.005, status="REJECTED"),
            _make_row("r2.json", "drake", 0.012, status="PASSED"),
        ]

        ranked = model.rank_candidates(rows, capture="driver")
        assert len(ranked) == 2
        # Lowest RMSE is top candidate, but its status is honestly preserved as REJECTED
        assert ranked[0].engine == "mujoco"
        assert model.extract_verdict_string(ranked[0]) == "REJECTED"
        assert ranked[1].engine == "drake"
        assert model.extract_verdict_string(ranked[1]) == "PASSED"


def _install_ledger(monkeypatch: pytest.MonkeyPatch, rows: list[LedgerRow]) -> None:
    monkeypatch.setattr(
        MatchedSwingBrowserModel,
        "load_ledger",
        lambda self, ledger_path=None: list(rows),
    )


def _table_verdict_at(widget: Any, row: int) -> str:
    return widget._table.item(row, 5).text()


class TestBrowserBuildPathRanking:
    """Codex P1: rank_candidates must be wired into the browser list-build path."""

    def _make_widget(
        self,
        qapp: Any,
        monkeypatch: pytest.MonkeyPatch,
        rows: list[LedgerRow],
        tmp_path: Path,
    ) -> Any:
        from src.tools.matched_swing_browser.gui import MatchedSwingBrowserWidget

        _install_ledger(monkeypatch, rows)
        return MatchedSwingBrowserWidget(ledger_path=tmp_path / "ledger.json")

    def test_best_comparable_candidate_first_for_out_of_order_rmse_ledger(
        self, qapp: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            _make_row("r1.json", "mujoco", 0.024),
            _make_row("r2.json", "pinocchio", 0.011),
            _make_row("r3.json", "drake", 0.017),
        ]
        widget = self._make_widget(qapp, monkeypatch, rows, tmp_path)  # type: ignore[arg-type]

        assert [r.engine for r in widget._current_filtered_rows] == [
            "pinocchio",
            "drake",
            "mujoco",
        ]
        assert widget._current_filtered_rows[0].metrics.whole_marker_rmse_m == 0.011

    def test_ranking_keeps_rejected_rows_visible_with_failure_verdict(
        self, qapp: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            _make_row("r1.json", "mujoco", 0.005, status="REJECTED"),
            _make_row("r2.json", "drake", 0.012, status="PASSED"),
        ]
        widget = self._make_widget(qapp, monkeypatch, rows, tmp_path)  # type: ignore[no-untyped-call]

        assert widget._table.rowCount() == 2
        assert _table_verdict_at(widget, 0) == "REJECTED"
        assert _table_verdict_at(widget, 1) == "PASSED"


class TestViewerOpenPathForwardsProvenance:
    """Codex P1: the open-viewer path must forward the selected row's provenance."""

    def test_selection_provenance_reaches_viewer_load(
        self, qapp: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from src.tools.matched_swing_browser import gui as browser_gui
        from src.tools.matched_swing_browser.gui import MatchedSwingBrowserWidget
        from src.tools.tour_matching_viewer import gui as viewer_gui

        rows = [
            _make_row(
                "r1.json",
                "drake",
                0.019,
                status="REJECTED",
                candidate_sha="rowsha000111",
            ),
        ]
        _install_ledger(monkeypatch, rows)
        widget = MatchedSwingBrowserWidget(ledger_path=tmp_path / "ledger.json")

        npz_file = tmp_path / "candidate_replay.npz"
        npz_file.write_bytes(b"npz")
        widget._model.resolve_artifact_path = (  # type: ignore[method-assign]
            lambda row, artifact_type: npz_file
        )
        widget._selected_row = rows[0]

        captured: dict[str, Any] = {}

        class _FakeViewerWidget:
            def load_file(self, path: Path, **kwargs: Any) -> None:
                captured["path"] = path
                captured.update(kwargs)

        class _FakeViewerWindow:
            def __init__(self, parent: Any = None) -> None:
                self.widget = _FakeViewerWidget()

            def show(self) -> None:
                pass

        monkeypatch.setattr(viewer_gui, "TourMatchingViewerWindow", _FakeViewerWindow)
        widget._on_open_tour_matching_viewer()

        assert captured["path"] == npz_file
        # Selected receipt provenance, not filename-derived placeholders
        assert captured["candidate_hash"] == "rowsha000111"
        assert captured["engine_name"] == "drake"
        assert captured["drive_mode"] == "torque_driven"
        # Rejection verdict travels with the row so the failure banner shows
        assert captured["is_accepted"] is False

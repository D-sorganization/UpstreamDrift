"""Tests for ``src.launchers.lift_baseline_panel`` (LIFT-8, #11748).

``LiftBaselinePanel`` renders the LIFT-1 cross-engine baseline receipt. These
tests exercise it against the real committed receipt and a synthetic one that
hits the "unavailable is never zero" and pair-metric status paths the
committed receipt does not conveniently cover.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PyQt6")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic receipt fixture (mirrors tests/unit/lifting/test_baseline_view.py).
# ---------------------------------------------------------------------------


def _engine_result(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "structure": {"n_bodies": 10, "nq": 20, "nv": 18},
        "segment_masses_kg": {},
        "start": {
            "total_mass_kg": 100.0,
            "bar_above_sole_m": 1.0,
            "hand_mid_above_sole_m": 0.9,
        },
        "start_contact": {
            "value_n": 0.0,
            "non_ground_normal_force_n": 5.0,
            "n_ground_contacts": 1,
            "n_non_ground_contacts": 0,
            "reason": None,
        },
        "smoke": {"loaded": True, "stepped": True, "max_abs_qvel": 0.1},
        "same_q": {},
        "reference_fk": [],
        "phases": [
            {
                "name": "start",
                "fraction": 0.0,
                "n_targets": 2,
                "summary": {
                    "hand_bar": {
                        "l": {"axis_distance_m": 0.3},
                        "r": {"axis_distance_m": 0.3},
                    }
                },
            }
        ],
    }
    base.update(overrides)
    return base


@pytest.fixture
def synthetic_receipt() -> dict[str, Any]:
    from src.shared.python.lifting.pack_audit.baseline import SCHEMA

    return {
        "schema": SCHEMA,
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "anthropometry": {"body_mass_kg": 80.0},
        "tolerances": {"position_m": 0.02, "mass_rel": 1e-6},
        "packs": {
            "mujoco": {
                "repo": "MuJoCo_Models",
                "commit": "abc123def",
                "licence": "MIT",
            },
            "opensim": {"repo": "OpenSim_Models", "commit": None, "licence": "MIT"},
        },
        "results": {
            "squat": {
                "mujoco": _engine_result(),
                "opensim": _engine_result(
                    start={
                        "total_mass_kg": None,
                        "bar_above_sole_m": math.nan,
                        "hand_mid_above_sole_m": 0.91,
                    }
                ),
            },
            "deadlift": {
                "mujoco": _engine_result(),
            },
        },
        "comparisons": {
            "squat": {
                "poses": {
                    "zero": {
                        "mujoco|opensim": {
                            "segments_max_m": 0.01,
                            "hands_max_m": 0.03,
                            "feet_max_m": None,
                            "bar_centre_max_m": math.nan,
                            "com_max_m": 0.0,
                            "lifter_com_max_m": 0.019999,
                        }
                    }
                },
                "mass": {},
            }
        },
        "gaps": [
            {
                "key": "no_grf",
                "title": "Right hand is not attached to the bar",
                "engines": ["mujoco"],
                "evidence": ["mujoco: right-hand attachment is not a bar weld"],
                "issues": ["MuJoCo_Models#1"],
                "lift_story": "LIFT-3",
                "new_issue": False,
            },
        ],
        "deferred": [],
    }


# ---------------------------------------------------------------------------
# Real committed receipt
# ---------------------------------------------------------------------------


class TestRealReceipt:
    def test_all_five_lifts_in_combo(self, qapp) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel()
        try:
            assert panel.error_message() is None
            assert panel.lift_count() == 5
        finally:
            panel.deleteLater()

    def test_squat_has_four_engine_rows(self, qapp) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(initial_lift="squat")
        try:
            assert panel.current_lift() == "squat"
            assert panel.engine_row_count() == 4
        finally:
            panel.deleteLater()

    def test_switching_lift_updates_engine_table(self, qapp) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(initial_lift="squat")
        try:
            before = panel.cell_text("engines", 0, 0)
            panel.select_lift("deadlift")
            assert panel.current_lift() == "deadlift"
            after = panel.cell_text("engines", 0, 0)
            # Same engine set (both lifts have all four engines), but the
            # table must have been rebuilt for the new lift, not left stale.
            assert panel.engine_row_count() == 4
            assert before == after  # engine names are identical across lifts
        finally:
            panel.deleteLater()


# ---------------------------------------------------------------------------
# Synthetic receipt: unavailable-is-never-zero + pair-metric status
# ---------------------------------------------------------------------------


class TestSyntheticReceipt:
    def test_missing_total_mass_renders_unavailable_not_zero(
        self, qapp, synthetic_receipt
    ) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(receipt=synthetic_receipt, initial_lift="squat")
        try:
            # engines sorted as returned by lift_view; find the opensim row.
            rows = range(panel.engine_row_count())
            opensim_row = next(
                r for r in rows if panel.cell_text("engines", r, 0) == "OpenSim"
            )
            mass_text = panel.cell_text("engines", opensim_row, 3)
            assert mass_text == "unavailable"
            assert mass_text != "0"
        finally:
            panel.deleteLater()

    def test_nan_bar_height_renders_unavailable_with_reason_tooltip(
        self, qapp, synthetic_receipt
    ) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(receipt=synthetic_receipt, initial_lift="squat")
        try:
            rows = range(panel.engine_row_count())
            opensim_row = next(
                r for r in rows if panel.cell_text("engines", r, 0) == "OpenSim"
            )
            assert panel.cell_text("engines", opensim_row, 4) == "unavailable"
            assert panel.cell_tooltip("engines", opensim_row, 4)
        finally:
            panel.deleteLater()

    def test_missing_pack_commit_renders_unavailable(
        self, qapp, synthetic_receipt
    ) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(receipt=synthetic_receipt, initial_lift="squat")
        try:
            rows = range(panel.engine_row_count())
            opensim_row = next(
                r for r in rows if panel.cell_text("engines", r, 0) == "OpenSim"
            )
            assert panel.cell_text("engines", opensim_row, 1) == "unavailable"
        finally:
            panel.deleteLater()

    def test_pair_metric_status_reflected(self, qapp, synthetic_receipt) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(receipt=synthetic_receipt, initial_lift="squat")
        try:
            panel.select_pose("zero")
            assert panel.pair_row_count() == 1
            # Column order mirrors _PAIR_METRIC_COLUMNS: segments, hands, feet,
            # bar_centre, com, lifter_com.
            assert panel.cell_text("pairs", 0, 0) == "10.0 mm"  # segments: pass
            assert panel.cell_text("pairs", 0, 1) == "30.0 mm"  # hands: fail
            assert panel.cell_text("pairs", 0, 2) == "unavailable"  # feet: None
            assert panel.cell_text("pairs", 0, 3) == "unavailable"  # bar_centre: NaN
        finally:
            panel.deleteLater()

    def test_fewer_than_two_engines_has_no_pair_rows(
        self, qapp, synthetic_receipt
    ) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(receipt=synthetic_receipt, initial_lift="deadlift")
        try:
            assert panel.pair_row_count() == 0
        finally:
            panel.deleteLater()

    def test_gaps_list_shows_title_and_evidence_tooltip(
        self, qapp, synthetic_receipt
    ) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(receipt=synthetic_receipt, initial_lift="squat")
        try:
            assert panel.gap_count() == 1
            assert "Right hand is not attached" in panel.gap_text(0)
            assert "MuJoCo" in panel.gap_text(0)
            assert "bar weld" in (panel.gap_tooltip(0) or "")
        finally:
            panel.deleteLater()

    def test_phase_table_for_selected_engine(self, qapp, synthetic_receipt) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(receipt=synthetic_receipt, initial_lift="squat")
        try:
            panel.select_phase_engine("mujoco")
            assert panel.phase_row_count() == 1
            assert panel.cell_text("phases", 0, 0) == "start"
            assert panel.cell_text("phases", 0, 2) == "300.0 mm"
        finally:
            panel.deleteLater()

    def test_provenance_shows_schema_and_generated(
        self, qapp, synthetic_receipt
    ) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(receipt=synthetic_receipt, initial_lift="squat")
        try:
            text = panel.provenance_text()
            assert synthetic_receipt["schema"] in text
            assert synthetic_receipt["generated_utc"] in text
        finally:
            panel.deleteLater()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestMissingReceipt:
    def test_missing_receipt_shows_message_not_crash(
        self, qapp, monkeypatch, tmp_path: Path
    ) -> None:
        import src.shared.python.lifting.baseline_view as baseline_view_mod
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        monkeypatch.setattr(
            baseline_view_mod, "DEFAULT_BASELINE_PATH", tmp_path / "missing.json"
        )
        panel = LiftBaselinePanel()
        try:
            assert panel.error_message() is not None
            assert panel.message_visible()
            assert panel.engine_row_count() == 0
        finally:
            panel.deleteLater()

    def test_invalid_json_receipt_shows_message(self, qapp, tmp_path: Path) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel
        from src.shared.python.lifting.baseline_view import load_baseline

        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_baseline(path)

        # The panel itself only loads the default path; directly verify the
        # ValueError branch is reachable by constructing with a receipt that
        # fails the dict/results precondition instead.
        panel = LiftBaselinePanel(receipt={"no": "results key"})
        try:
            assert panel.error_message() is not None
        finally:
            panel.deleteLater()

    def test_receipt_with_no_lifts_shows_message(self, qapp) -> None:
        from src.shared.python.lifting.pack_audit.baseline import SCHEMA
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(receipt={"schema": SCHEMA, "results": {}})
        try:
            assert panel.error_message() is not None
        finally:
            panel.deleteLater()


class TestConstructorPreconditions:
    def test_non_dict_receipt_rejected(self, qapp) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        with pytest.raises(TypeError):
            LiftBaselinePanel(receipt="not a dict")  # type: ignore[arg-type]

    def test_non_str_initial_lift_rejected(self, qapp) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        with pytest.raises(TypeError):
            LiftBaselinePanel(initial_lift=123)  # type: ignore[arg-type]

    def test_unknown_table_name_rejected(self, qapp) -> None:
        from src.launchers.lift_baseline_panel import LiftBaselinePanel

        panel = LiftBaselinePanel(initial_lift="squat")
        try:
            with pytest.raises(ValueError, match="unknown table"):
                panel.cell_text("bogus", 0, 0)
        finally:
            panel.deleteLater()

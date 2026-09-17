"""Unit tests for the Moco G1 horizon ladder helpers (MS-42 phase A, #10341).

Pure functions only: no ``opensim`` import. Covers horizon windowing, mesh
interval selection, marker gap policy, warm-start assembly, the G1 gate table
and the OS-7 receipt contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.moco_g1 import (
    G1_TARGETS,
    REQUIRED_RECEIPT_KEYS,
    assemble_warm_start,
    fill_marker_gaps,
    gate_table,
    horizon_ladder,
    mesh_intervals_for,
    read_sto,
    retain_markers,
    trim_trailing_invalid,
    validate_os7_receipt,
    window_capture,
    write_sto,
)
from src.shared.python.contracts import ContractViolationError
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.motion_matching.tour_metrics import SharedMetrics

pytestmark = pytest.mark.unit


def _capture(frames: int = 20, labels: tuple[str, ...] = ("A", "B")) -> TourCapture:
    time_s = np.arange(frames) / 100.0
    points = np.zeros((frames, len(labels), 3))
    points[:, :, 0] = time_s[:, None]
    points[:, :, 1] = np.arange(len(labels))[None, :]
    valid = np.ones((frames, len(labels)), dtype=bool)
    return TourCapture(time_s, labels, points, valid)


# ---------------------------------------------------------------- ladder


def test_horizon_ladder_ends_at_full_capture() -> None:
    assert horizon_ladder(1.8139) == (0.1, 0.3, 0.6, 0.85, 1.8139)


def test_horizon_ladder_truncates_to_short_capture() -> None:
    assert horizon_ladder(0.5) == (0.1, 0.3, 0.5)
    assert horizon_ladder(0.85) == (0.1, 0.3, 0.6, 0.85)


def test_horizon_ladder_rejects_nonpositive() -> None:
    with pytest.raises((ValueError, ContractViolationError)):
        horizon_ladder(0.0)


# ---------------------------------------------------------------- mesh


@pytest.mark.parametrize(
    ("duration", "expected"), [(0.1, 10), (0.3, 30), (0.85, 85), (1.8139, 181)]
)
def test_mesh_intervals_scale_with_horizon(duration: float, expected: int) -> None:
    assert mesh_intervals_for(duration) == expected


def test_mesh_intervals_floor_and_validation() -> None:
    assert mesh_intervals_for(0.01) == 10
    with pytest.raises((ValueError, ContractViolationError)):
        mesh_intervals_for(-1.0)


# ---------------------------------------------------------------- windowing


def test_window_capture_keeps_frames_up_to_horizon() -> None:
    cap = _capture(frames=20)
    win = window_capture(cap, 0.1)
    assert win.frames == 11
    assert win.time_s[0] == 0.0
    assert win.time_s[-1] == pytest.approx(0.1)
    assert win.labels == cap.labels


def test_window_capture_rejects_horizon_before_second_frame() -> None:
    cap = _capture(frames=20)
    with pytest.raises((ValueError, ContractViolationError)):
        window_capture(cap, 0.0)


def test_trim_trailing_invalid_drops_only_the_tail() -> None:
    cap = _capture(frames=10)
    valid = cap.valid.copy()
    valid[8:, 0] = False  # marker A missing on the last two frames
    valid[3, 1] = False  # interior hole must stay
    cap = TourCapture(cap.time_s, cap.labels, cap.points_m, valid)
    trimmed, dropped = trim_trailing_invalid(cap, ("A",))
    assert dropped == 2
    assert trimmed.frames == 8
    assert not trimmed.valid[3, 1]


# ---------------------------------------------------------------- gap policy


def test_fill_marker_gaps_interpolates_short_interior_gaps() -> None:
    cap = _capture(frames=10)
    pts = cap.points_m.copy()
    valid = cap.valid.copy()
    pts[4:6, 0] = np.nan
    valid[4:6, 0] = False
    cap = TourCapture(cap.time_s, cap.labels, pts, valid)
    filled, report = fill_marker_gaps(cap, max_gap_frames=3)
    assert report == {"A": 2}
    assert filled.valid[4:6, 0].all()
    assert filled.points_m[4, 0, 0] == pytest.approx(0.04)
    assert filled.points_m[5, 0, 0] == pytest.approx(0.05)
    # the source capture is untouched
    assert not cap.valid[4, 0]


def test_fill_marker_gaps_leaves_long_and_edge_gaps() -> None:
    cap = _capture(frames=10)
    pts = cap.points_m.copy()
    valid = cap.valid.copy()
    pts[2:7, 0] = np.nan  # five-frame hole, longer than allowed
    valid[2:7, 0] = False
    pts[0, 1] = np.nan  # leading edge
    valid[0, 1] = False
    cap = TourCapture(cap.time_s, cap.labels, pts, valid)
    filled, report = fill_marker_gaps(cap, max_gap_frames=3)
    assert report == {}
    assert not filled.valid[2:7, 0].any()
    assert not filled.valid[0, 1]


def test_retain_markers_by_valid_fraction() -> None:
    cap = _capture(frames=10, labels=("Good", "Sparse"))
    valid = cap.valid.copy()
    valid[:8, 1] = False  # 20 % valid
    cap = TourCapture(cap.time_s, cap.labels, cap.points_m, valid)
    kept, dropped = retain_markers(cap, min_valid_fraction=0.9)
    assert kept == ("Good",)
    assert dropped == {"Sparse": pytest.approx(0.2)}


# ---------------------------------------------------------------- warm start


def test_assemble_warm_start_chains_previous_solution_then_ik() -> None:
    grid = np.linspace(0.0, 0.3, 31)
    prev_t = np.linspace(0.0, 0.1, 11)
    prev_states = {
        "/j/q/value": 1.0 + prev_t,
        "/j/q/speed": np.ones_like(prev_t),
    }
    prev_controls = {"/f/tau": 0.5 * np.ones_like(prev_t)}
    ik_t = np.linspace(0.0, 0.3, 61)
    ik_values = {"/j/q/value": 2.0 * ik_t}
    states, controls = assemble_warm_start(
        grid,
        state_names=("/j/q/value", "/j/q/speed"),
        control_names=("/f/tau",),
        previous=(prev_t, prev_states, prev_controls),
        ik=(ik_t, ik_values),
    )
    assert set(states) == {"/j/q/value", "/j/q/speed"}
    assert states["/j/q/value"].shape == grid.shape
    # inside the previous horizon: the previous solution
    assert states["/j/q/value"][5] == pytest.approx(1.05)
    assert states["/j/q/speed"][5] == pytest.approx(1.0)
    assert controls["/f/tau"][5] == pytest.approx(0.5)
    # beyond it: IK values, finite-difference speeds, held control
    assert states["/j/q/value"][20] == pytest.approx(0.4)
    assert states["/j/q/speed"][20] == pytest.approx(2.0, abs=1e-6)
    assert controls["/f/tau"][20] == pytest.approx(0.5)


def test_assemble_warm_start_without_previous_uses_ik_and_zero_controls() -> None:
    grid = np.linspace(0.0, 0.1, 11)
    ik_t = np.linspace(0.0, 0.1, 37)
    states, controls = assemble_warm_start(
        grid,
        state_names=("/j/q/value", "/j/q/speed"),
        control_names=("/f/tau",),
        previous=None,
        ik=(ik_t, {"/j/q/value": np.sin(ik_t)}),
    )
    assert states["/j/q/value"][-1] == pytest.approx(np.sin(0.1), abs=1e-6)
    assert np.all(controls["/f/tau"] == 0.0)


def test_assemble_warm_start_rejects_missing_ik_column() -> None:
    grid = np.linspace(0.0, 0.1, 11)
    with pytest.raises((KeyError, ValueError, ContractViolationError)):
        assemble_warm_start(
            grid,
            state_names=("/j/q/value", "/j/q/speed"),
            control_names=(),
            previous=None,
            ik=(grid, {"/other/value": grid}),
        )


# ---------------------------------------------------------------- gates


def test_gate_table_reports_each_target_and_overall_verdict() -> None:
    passing = SharedMetrics(0.020, 0.010, 0.030, 0.050, np.radians(2.0))
    table = gate_table(passing)
    assert set(table["gates"]) == {
        "whole_marker_rmse_m",
        "early_marker_rmse_m",
        "terminal_marker_rmse_m",
        "club_marker_rmse_m",
        "pelvis_yaw_rmse_deg",
    }
    assert (
        table["gates"]["whole_marker_rmse_m"]["target"]
        == G1_TARGETS["whole_marker_rmse_m"]
    )
    assert table["gates"]["pelvis_yaw_rmse_deg"]["value"] == pytest.approx(2.0)
    assert all(g["passed"] for g in table["gates"].values())
    assert table["all_passed"] is True

    failing = SharedMetrics(0.078, 0.042, 0.171, 0.020, 0.180)
    table = gate_table(failing)
    assert table["all_passed"] is False
    assert table["gates"]["club_marker_rmse_m"]["passed"] is True
    assert table["gates"]["terminal_marker_rmse_m"]["passed"] is False


# ---------------------------------------------------------------- receipt


def _receipt() -> dict:
    return dict.fromkeys(REQUIRED_RECEIPT_KEYS, "x")


def test_validate_receipt_accepts_complete_document() -> None:
    assert validate_os7_receipt(_receipt()) is True


def test_validate_receipt_names_missing_keys() -> None:
    doc = _receipt()
    del doc["replay_metrics"]
    with pytest.raises(
        (KeyError, ValueError, ContractViolationError), match="replay_metrics"
    ):
        validate_os7_receipt(doc)


# ---------------------------------------------------------------- sto


def test_read_sto_returns_time_and_named_columns(tmp_path) -> None:
    sto = tmp_path / "x.sto"
    header = "inDegrees=no\nnRows=3\nendheader\n"
    body = "time\t/j/q/value\t/f/tau\n0\t1\t2\n0.5\t3\t4\n1\t5\t6\n"
    sto.write_text(header + body, encoding="utf-8")
    t, columns = read_sto(sto)
    assert t.tolist() == [0.0, 0.5, 1.0]
    assert columns["/j/q/value"].tolist() == [1.0, 3.0, 5.0]
    assert columns["/f/tau"].tolist() == [2.0, 4.0, 6.0]


def test_write_sto_roundtrips_through_read_sto(tmp_path) -> None:
    t = np.array([0.0, 0.25, 0.5])
    columns = {"/j/q/value": np.array([1.0, 2.0, 3.0]), "/j/q/speed": t * 2}
    path = write_sto(tmp_path / "rt.mot", t, columns, in_degrees=False)
    t_back, back = read_sto(path)
    assert t_back.tolist() == t.tolist()
    assert back["/j/q/speed"].tolist() == (t * 2).tolist()
    assert "inDegrees=no" in path.read_text(encoding="utf-8")


def test_read_sto_rejects_file_without_header(tmp_path) -> None:
    sto = tmp_path / "bad.sto"
    sto.write_text("time\tq\n0\t1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_sto(sto)

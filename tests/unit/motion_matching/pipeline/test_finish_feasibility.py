"""Finish-feasibility metrics on synthetic trajectories (Balance-1, #11668)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.finish_feasibility import (
    FinishHistory,
    FootTrack,
    foot_slide_and_yaw,
    friction_summary,
    friction_utilisation,
    pelvis_yaw_error_deg,
    summarise_history,
    supported_fraction,
    vertical_force_range,
)

pytestmark = pytest.mark.unit

TIMES = np.linspace(0.0, 2.0, 201)  # 100 Hz, finish window 1.0-2.0 s
MU = 0.8


def _foot(heel_xy: np.ndarray, yaw_deg: np.ndarray, length: float = 0.2) -> FootTrack:
    """Foot of ``length`` whose heel is at ``heel_xy`` and axis at ``yaw_deg``."""
    yaw = np.radians(yaw_deg)
    heel = np.column_stack([heel_xy, np.zeros(len(yaw))])
    toe = heel + length * np.column_stack(
        [np.cos(yaw), np.sin(yaw), np.zeros(len(yaw))]
    )
    return FootTrack(heel=heel, toe=toe, centre=0.5 * (heel + toe))


def _still_foot() -> FootTrack:
    n = len(TIMES)
    return _foot(np.zeros((n, 2)), np.zeros(n))


def _history(**overrides: object) -> FinishHistory:
    n = len(TIMES)
    base = {
        "time_s": TIMES,
        "vertical_bw": np.ones(n),
        "friction_ratio": np.full(n, 0.2),
        "supported": np.ones(n, dtype=bool),
        "feet": {"r": _still_foot(), "l": _still_foot()},
        "pelvis_yaw_error_rad": np.zeros(n),
    }
    base.update(overrides)
    return FinishHistory(**base)  # type: ignore[arg-type]


def test_supported_fraction_counts_only_the_window() -> None:
    supported = np.ones(len(TIMES), dtype=bool)
    supported[(TIMES >= 1.0) & (TIMES < 1.25)] = False  # outside for a quarter
    frac = supported_fraction(TIMES, supported, start_s=1.0, end_s=1.5)
    assert frac == pytest.approx(0.5, abs=0.02)
    # the unsupported span does not leak into the earlier window
    assert supported_fraction(TIMES, supported, start_s=0.0, end_s=1.0) == 1.0


def test_supported_fraction_rejects_empty_window() -> None:
    with pytest.raises(ValueError, match="no samples"):
        supported_fraction(TIMES, np.ones(len(TIMES), bool), start_s=5.0, end_s=6.0)


def test_friction_utilisation_is_ratio_over_mu_and_nan_when_unloaded() -> None:
    ratio = np.array([0.4, 0.8, 0.5])
    vertical = np.array([1.0, 1.0, 0.0])
    util = friction_utilisation(ratio, vertical, MU, min_vertical=0.1)
    assert util[0] == pytest.approx(0.5)
    assert util[1] == pytest.approx(1.0)
    assert np.isnan(util[2])


def test_friction_summary_max_and_saturated_fraction() -> None:
    util = np.array([0.1, 0.96, 0.99, 0.5, np.nan, 1.0, 0.2, 0.3, 0.4, 0.94])
    times = np.arange(util.size, dtype=float)
    peak, frac = friction_summary(times, util, start_s=0.0, saturation=0.95)
    assert peak == pytest.approx(1.0)
    assert frac == pytest.approx(3 / 9)  # NaN frame excluded from the denominator


def test_friction_summary_all_unloaded_is_not_saturated() -> None:
    peak, frac = friction_summary(
        np.arange(3.0), np.full(3, np.nan), start_s=0.0, saturation=0.95
    )
    assert peak == 0.0 and frac == 0.0


def test_foot_slide_and_yaw_ignore_motion_before_window() -> None:
    n = len(TIMES)
    x = np.where(TIMES < 1.0, 0.5 * TIMES, 0.5)  # moves 0.5 m, all before t = 1
    track = _foot(np.column_stack([x, np.zeros(n)]), np.zeros(n))
    slide_mm, yaw_deg = foot_slide_and_yaw({"r": track}, TIMES, start_s=1.0)
    assert slide_mm == pytest.approx(0.0, abs=1e-9)
    assert yaw_deg == pytest.approx(0.0, abs=1e-9)


def test_foot_slide_and_yaw_measure_displacement_and_pivot() -> None:
    n = len(TIMES)
    ramp = np.clip(TIMES - 1.0, 0.0, None)  # 0 at 1 s, 1 at 2 s
    slide = _foot(np.column_stack([0.1 * ramp, np.zeros(n)]), np.zeros(n))
    pivot = _foot(np.zeros((n, 2)), 40.0 * ramp)
    slide_mm, yaw_deg = foot_slide_and_yaw({"r": slide, "l": pivot}, TIMES, start_s=1.0)
    assert slide_mm == pytest.approx(100.0, rel=1e-6)  # worst foot, millimetres
    assert yaw_deg == pytest.approx(40.0, rel=1e-6)  # worst foot, degrees


def test_foot_yaw_is_unwrapped_through_pi() -> None:
    n = len(TIMES)
    yaw = 170.0 + 40.0 * np.clip(TIMES - 1.0, 0.0, None)  # crosses +-180
    track = _foot(np.zeros((n, 2)), yaw)
    _, yaw_deg = foot_slide_and_yaw({"r": track}, TIMES, start_s=1.0)
    assert yaw_deg == pytest.approx(40.0, rel=1e-6)


def test_pelvis_yaw_error_reports_peak_and_final() -> None:
    err = np.zeros(len(TIMES))
    err[TIMES >= 1.0] = np.radians(-np.linspace(0.0, 21.0, int((TIMES >= 1.0).sum())))
    peak, final = pelvis_yaw_error_deg(TIMES, err, start_s=1.0)
    assert peak == pytest.approx(21.0)
    assert final == pytest.approx(-21.0)


def test_vertical_force_range_uses_window_only() -> None:
    fz = np.ones(len(TIMES))
    fz[TIMES < 1.0] = 3.0  # impact-free spike before the window
    fz[(TIMES >= 1.5) & (TIMES < 1.6)] = 0.0
    lo, hi = vertical_force_range(TIMES, fz, start_s=1.0)
    assert (lo, hi) == (0.0, 1.0)


def test_summarise_history_feasible_trajectory() -> None:
    out = summarise_history(_history(), mu=MU)
    assert out["zmp_inside_fraction_1_0_to_1_5s"] == 1.0
    assert out["zmp_inside_fraction_finish"] == 1.0
    assert out["friction_utilisation_max"] == pytest.approx(0.2 / MU)
    assert out["friction_saturated_fraction"] == 0.0
    assert out["foot_slide_mm_max"] == pytest.approx(0.0, abs=1e-9)
    assert out["foot_yaw_pivot_deg_max"] == pytest.approx(0.0, abs=1e-9)
    assert out["pelvis_yaw_error_deg_max"] == 0.0
    assert out["vertical_force_bw_min"] == 1.0
    assert out["vertical_force_bw_max"] == 1.0


def test_summarise_history_infeasible_trajectory() -> None:
    n = len(TIMES)
    late = TIMES >= 1.0
    supported = np.ones(n, dtype=bool)
    supported[late] = False
    friction = np.where(late, 0.79, 0.1)
    vertical = np.where(late, 0.0, 1.0)
    vertical[(TIMES >= 1.0) & (TIMES < 1.1)] = 0.6  # loaded just after impact
    out = summarise_history(
        _history(
            supported=supported,
            friction_ratio=friction,
            vertical_bw=vertical,
            pelvis_yaw_error_rad=np.where(late, np.radians(-21.0), 0.0),
        ),
        mu=MU,
    )
    assert out["zmp_inside_fraction_1_0_to_1_5s"] == 0.0
    assert out["friction_utilisation_max"] == pytest.approx(0.79 / MU)
    assert out["friction_saturated_fraction"] == 1.0  # only loaded frames count
    assert out["vertical_force_bw_min"] == 0.0
    assert out["pelvis_yaw_error_deg_max"] == pytest.approx(21.0)


def test_summarise_history_without_pelvis_yaw_reports_none() -> None:
    out = summarise_history(_history(pelvis_yaw_error_rad=None), mu=MU)
    assert out["pelvis_yaw_error_deg_max"] is None
    assert out["pelvis_yaw_error_deg_final"] is None


@pytest.mark.parametrize(
    "bad",
    [
        {"vertical_bw": np.ones(5)},
        {"friction_ratio": np.ones(5)},
        {"supported": np.ones(5, dtype=bool)},
        {"pelvis_yaw_error_rad": np.zeros(5)},
        {"time_s": TIMES[::-1].copy()},
        {"vertical_bw": np.full(len(TIMES), np.nan)},
        {"feet": {}},
    ],
)
def test_history_rejects_malformed_inputs(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _history(**bad)


def test_summarise_history_rejects_nonpositive_friction() -> None:
    with pytest.raises(ValueError, match="mu"):
        summarise_history(_history(), mu=0.0)


def test_foot_track_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        FootTrack(heel=np.zeros((4, 3)), toe=np.zeros((5, 3)), centre=np.zeros((4, 3)))


def test_annotate_receipt_writes_block_in_place(tmp_path) -> None:
    import json

    from src.shared.python.motion_matching.pipeline.finish_feasibility_cli import (
        annotate_receipt,
    )

    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"dynamics": {"marker_rms_m": 0.08}}))
    annotate_receipt(path, {"reference": {}, "simulation": {}})
    written = json.loads(path.read_text())
    assert written["dynamics"]["marker_rms_m"] == 0.08  # untouched
    assert written["dynamics"]["finish_feasibility"] == {
        "reference": {},
        "simulation": {},
    }


def test_annotate_receipt_requires_a_dynamics_block(tmp_path) -> None:
    from src.shared.python.motion_matching.pipeline.finish_feasibility_cli import (
        annotate_receipt,
    )

    path = tmp_path / "receipt.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="dynamics"):
        annotate_receipt(path, {})


def test_load_record_names_missing_arrays(tmp_path) -> None:
    from src.shared.python.motion_matching.pipeline.finish_feasibility_cli import (
        load_record,
    )

    with pytest.raises(FileNotFoundError):
        load_record(tmp_path)
    np.savez(tmp_path / "dynamics_record.npz", time_s=np.zeros(3))
    with pytest.raises(ValueError, match="lacks arrays"):
        load_record(tmp_path)


def _block(inside: float, saturated: float) -> dict[str, dict[str, float]]:
    side = {
        "zmp_inside_fraction_1_0_to_1_5s": inside,
        "zmp_inside_fraction_finish": inside,
        "friction_saturated_fraction": saturated,
    }
    return {"reference": dict(side), "simulation": dict(side)}


def test_regressions_flags_a_fall_in_inside_fraction() -> None:
    from src.shared.python.motion_matching.pipeline.finish_feasibility import (
        regressions,
    )

    baseline = _block(0.9, 0.1)
    assert regressions(_block(0.9, 0.1), baseline) == []
    assert regressions(_block(0.95, 0.05), baseline) == []  # improvement is fine
    found = regressions(_block(0.8, 0.1), baseline)
    assert any("simulation.zmp_inside_fraction_1_0_to_1_5s" in m for m in found)


def test_regressions_flags_a_rise_in_saturated_fraction() -> None:
    from src.shared.python.motion_matching.pipeline.finish_feasibility import (
        regressions,
    )

    found = regressions(_block(0.9, 0.3), _block(0.9, 0.1))
    assert any("friction_saturated_fraction" in m for m in found)


def test_regressions_rejects_unknown_metric_and_side() -> None:
    from src.shared.python.motion_matching.pipeline.finish_feasibility import (
        regressions,
    )

    with pytest.raises(ValueError, match="not a ratcheted"):
        regressions(_block(1, 0), {"simulation": {"foot_slide_mm_max": 1.0}})
    with pytest.raises(ValueError, match="lacks"):
        regressions({"reference": {}}, {"simulation": {}})

"""Predictive-versus-tracked deadlift comparison (OSL-7, #13044)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.msk_lift_predictive_compare import (
    Trajectory,
    bar_rise_fraction,
    compare_group_activations,
    compare_point,
    compare_trajectories,
    compare_with_tracked,
    load_tracked_reference,
)
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit


def _line(lo: float, hi: float, n: int, offset: float = 0.0) -> Trajectory:
    progress = np.linspace(lo, hi, n)
    return Trajectory(progress, 2.0 * progress + offset, "m")


def test_bar_rise_fraction_and_its_contracts() -> None:
    frac = bar_rise_fraction(np.array([0.4, 0.6, 0.9]))
    np.testing.assert_allclose(frac, [0.0, 0.4, 1.0])
    assert bar_rise_fraction(np.array([0.5]), 0.4, 0.9) == pytest.approx([0.2])
    with pytest.raises(ValueError, match="bar must rise"):
        bar_rise_fraction(np.array([0.9, 0.4]))
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory(np.array([0.0, 0.5, 0.5]), np.zeros(3), "m")


def test_constant_offset_gives_that_rms_inside_the_overlap() -> None:
    result = compare_trajectories(_line(0.0, 1.0, 51, 0.01), _line(0.3, 0.5, 21))
    assert result["status"] == "compared"
    assert result["bar_rise_window"] == pytest.approx([0.3, 0.5])
    assert result["samples"] == 21
    assert result["rms_difference"] == pytest.approx(0.01)
    assert result["max_abs_difference"] == pytest.approx(0.01)


def test_only_the_shared_interval_is_compared() -> None:
    result = compare_trajectories(_line(0.0, 0.4, 41), _line(0.3, 0.5, 21))
    assert result["bar_rise_window"] == pytest.approx([0.3, 0.4])
    assert result["samples"] == 11
    assert result["rms_difference"] == pytest.approx(0.0, abs=1e-12)


def test_disjoint_intervals_are_unavailable_not_zero() -> None:
    result = compare_trajectories(_line(0.0, 0.2, 5), _line(0.3, 0.5, 5))
    assert result["status"] == "unavailable" and "share no" in result["reason"]


def test_point_comparison_interpolates_and_bounds() -> None:
    pred = _line(0.0, 1.0, 11)
    point = compare_point(pred, 0.5, 0.35)
    assert point["predictive"] == pytest.approx(0.7)
    assert point["difference"] == pytest.approx(0.2)
    assert compare_point(pred, None, 0.35)["status"] == "unavailable"
    assert compare_point(_line(0.5, 1.0, 3), 0.5, 0.35)["status"] == "unavailable"


def test_group_activations_use_the_osl4_statistics_in_the_window() -> None:
    progress = np.linspace(0.0, 1.0, 11)
    acts = {m: np.where(progress <= 0.5, 0.2, 0.9) for m in ("vaslat_r", "vasmed_r")}
    tracked = {
        "quadriceps": {"peak_activation": 1.0, "mean_activation": 0.25},
        "erector spinae": {"peak_activation": None, "mean_activation": None},
    }
    out = compare_group_activations(progress, acts, tracked, (0.3, 0.5))
    assert out["quadriceps"]["predictive_peak"] == pytest.approx(0.2)
    assert out["quadriceps"]["mean_difference"] == pytest.approx(-0.05)
    assert out["erector spinae"]["status"] == "unavailable"


def test_tracked_reference_reads_the_real_receipts() -> None:
    tracked = load_tracked_reference()
    lo, hi = tracked["window"]
    assert 0.0 < lo < hi < 1.0
    assert tracked["bar"].unit == "m"
    assert tracked["start_pose_deg"]["hip_flexion_r_deg"] > 0.0
    assert tracked["muscle_classification"] == "unqualified"
    assert "l5_s1" not in tracked
    assert "OSL-5" in tracked["l5_s1_reason"]


def test_comparison_runs_every_quantity(tmp_path: Path) -> None:
    tracked = load_tracked_reference()
    rise = float(tracked["bar"].values[-1] / tracked["bar"].progress[-1])
    height = 0.43 + rise * np.linspace(0.0, 1.0, 101)
    predictive = {
        "bar_height_m": height,
        "coordinates": {"hip_flexion_r": np.full(101, np.radians(90.0))},
    }
    result = compare_with_tracked(predictive, tracked)
    assert result["status"] == "compared"
    assert result["bar_path"]["status"] == "compared"
    assert result["bar_path"]["rms_difference"] == pytest.approx(0.0, abs=1e-9)
    hip = result["joint_angles"]["hip_flexion_r"]
    assert hip["predictive"] == pytest.approx(90.0)
    assert hip["tracked"] == tracked["start_pose_deg"]["hip_flexion_r_deg"]
    assert result["joint_angles"]["knee_angle_r"]["status"] == "unavailable"
    assert result["activations"]["status"] == "unavailable"
    assert result["l5_s1"]["status"] == "unavailable"
    receipt = tmp_path / "muscles.json"
    receipt.write_text(json.dumps({"lifts": {"deadlift": {}}}), "utf-8")
    missing = load_tracked_reference(muscle_receipt=receipt)
    assert compare_with_tracked(predictive, missing)["joint_angles"]["reason"]


def test_non_rising_predictive_bar_is_unavailable() -> None:
    predictive = {"bar_height_m": np.array([0.43, 0.42, 0.9]), "coordinates": {}}
    result = compare_with_tracked(predictive, load_tracked_reference())
    assert result["status"] == "unavailable"
    assert "alignment undefined" in result["reason"]


@pytest.mark.requires_opensim
@pytest.mark.slow
def test_trajectories_extract_from_a_real_coarse_solve() -> None:
    """A real 3-interval solve yields the comparison input with every column."""
    require_opensim()
    from src.engines.physics_engines.opensim.python.msk_lift_predictive import (
        PredictiveDeadliftConfig,
        build_predictive_deadlift_study,
    )
    from src.engines.physics_engines.opensim.python.msk_lift_predictive_compare import (
        predictive_trajectories,
    )

    study, _meta = build_predictive_deadlift_study(
        PredictiveDeadliftConfig(mesh_intervals=3, max_iterations=1)
    )
    solution = study.solve()
    solution.unseal()
    traj = predictive_trajectories(study, solution)
    n = solution.getNumTimes()
    assert traj["bar_height_m"].shape == (n,)
    assert np.all(np.isfinite(traj["bar_height_m"]))
    assert {"hip_flexion_r", "knee_angle_r", "pelvis_ty"} <= set(traj["coordinates"])
    assert traj["activations"] == {}

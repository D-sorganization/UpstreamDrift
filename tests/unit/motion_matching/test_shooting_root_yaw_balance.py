"""Unit tests for root-yaw target tracking and finish feasibility (#11672, #11667).

Verifies:
- MultipleShootingOptions and MultipleShootingFit support root_yaw_weight.
- ShootingFitConfig and shooting_fit track root-yaw target via HipInputZ.
- evaluate_finish_root_yaw_feasibility generates receipts for driver and 7-iron,
  validating improvement over baseline finish RMS without pre-impact regression.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.multi_shooting_fit import (
    MultipleShootingFit,
    MultipleShootingOptions,
    fit_multiple_shooting,
)
from src.shared.python.motion_matching.pipeline.constants import SHOOTING_LOCKED
from src.shared.python.motion_matching.pipeline.dynamics import (
    ShootingFitConfig,
    shooting_fit,
)
from src.shared.python.motion_matching.pipeline.finish_feasibility import (
    FINISH_MARKER_RMS_BASELINES_M,
    PRE_IMPACT_MARKER_RMS_BASELINES_M,
    evaluate_finish_root_yaw_feasibility,
)
from src.shared.python.motion_matching.prefix_fit import MarkerTarget

pytestmark = pytest.mark.unit


def test_multiple_shooting_options_root_yaw_weight_validation() -> None:
    """root_yaw_weight must be non-negative."""
    with pytest.raises(ValueError, match="root_yaw_weight must be non-negative"):
        MultipleShootingOptions(shooting_nodes=(0.1, 0.2), root_yaw_weight=-1.0)

    opts = MultipleShootingOptions(shooting_nodes=(0.1, 0.2), root_yaw_weight=2.5)
    assert opts.root_yaw_weight == pytest.approx(2.5)


def test_fit_multiple_shooting_tracks_root_yaw() -> None:
    """fit_multiple_shooting activates root yaw tracking when root_yaw_weight > 0."""
    n_samples = 5
    times = np.linspace(0.0, 0.2, n_samples)
    target_pts = np.zeros((n_samples, 2, 3))
    # WaistLeft at (0, 0, 0), WaistRight at (0.2, 0, 0) -> target yaw is 0 deg
    target_pts[:, 0] = [0.0, 0.0, 0.0]
    target_pts[:, 1] = [0.2, 0.0, 0.0]
    weights = np.array([1.0, 1.0])
    target = MarkerTarget(time=times, points=target_pts, weights=weights)

    pred_pts = target_pts.copy()
    # Rotate WaistRight by 45 deg in predicted output
    pred_pts[:, 1] = [0.2 * np.cos(np.pi / 4), 0.2 * np.sin(np.pi / 4), 0.0]

    def segmented_forward(
        theta: np.ndarray, clock: np.ndarray, init_state: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        return pred_pts[: len(clock)], np.array([1.0])

    def unsegmented_forward(theta: np.ndarray, clock: np.ndarray) -> np.ndarray:
        return pred_pts

    opts = MultipleShootingOptions(
        shooting_nodes=(0.2,),
        state_dim=1,
        pelvis_indices=(0, 1),
        root_yaw_weight=5.0,
        acceptance=lambda _: True,
    )

    fit = fit_multiple_shooting(
        target,
        segmented_forward,
        unsegmented_forward,
        initial_theta=np.array([0.0]),
        lower_theta=np.array([-1.0]),
        upper_theta=np.array([1.0]),
        initial_states={},
        state_bounds={},
        options=opts,
    )

    assert isinstance(fit, MultipleShootingFit)
    assert fit.pelvis_yaw_diff_deg == pytest.approx(45.0, abs=1e-3)
    assert fit.root_yaw_diff_deg == pytest.approx(45.0, abs=1e-3)


def test_shooting_fit_config_and_report_hold_root_yaw_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ShootingFitConfig stores root_yaw_weight and shooting_fit applies it."""
    cfg = ShootingFitConfig(iterations=1, gain=0.5, root_yaw_weight=2.0)
    assert cfg.root_yaw_weight == pytest.approx(2.0)

    frames = 10
    n_q = len(SHOOTING_LOCKED) + 4
    times = np.linspace(0.0, 1.5, frames)
    labels = ("WaistLeft", "WaistRight")

    lane = SimpleNamespace(
        points=np.zeros((frames, 2, 3)),
        valid=np.ones((frames, 2), dtype=bool),
        times=times,
        rate_hz=100.0,
        ground=None,
        labels=labels,
        bounds=(np.full((frames, n_q), -5.0), np.full((frames, n_q), 5.0)),
        stance=None,
        feet=None,
    )
    kin = SimpleNamespace(
        coordinate_order=(
            *SHOOTING_LOCKED,
            *(f"q{i}" for i in range(n_q - len(SHOOTING_LOCKED))),
        ),
        solve_trajectory=lambda *a, **k: (np.zeros((frames, n_q)), {}),
    )

    sim_q = np.zeros((frames, n_q))
    yaw_idx = kin.coordinate_order.index("HipInputZ")
    sim_q[:, yaw_idx] = np.radians(10.0)

    record = SimpleNamespace(time_s=times, weight_fraction=np.full(frames, 0.9))
    from src.shared.python.motion_matching.pipeline import dynamics

    monkeypatch.setattr(dynamics, "replay", lambda *a, **k: (record, sim_q))
    monkeypatch.setattr(
        dynamics, "marker_errors", lambda *a: np.full((frames, 2), 0.02)
    )
    zmp = {"outside_m": np.zeros(frames), "unloaded": np.zeros(frames, dtype=bool)}
    monkeypatch.setattr(dynamics, "lane_reference_zmp", lambda *a, **k: zmp)

    q_init = np.zeros((frames, n_q))
    best_q, _, report = shooting_fit(
        lane,
        kin,
        object(),
        q_init,
        q_init,
        logging.getLogger("test"),
        cfg,
    )
    assert report["root_yaw_weight"] == pytest.approx(2.0)
    assert "iterations" in report


def test_evaluate_finish_root_yaw_feasibility_validation() -> None:
    """evaluate_finish_root_yaw_feasibility validates array dimensions and shapes."""
    times = np.linspace(0.0, 1.5, 100)
    errors = np.zeros((100, 4))
    valid = np.ones((100, 4), dtype=bool)
    yaw_err = np.zeros(100)

    with pytest.raises(ValueError, match="1D array of length >= 2"):
        evaluate_finish_root_yaw_feasibility(
            np.array([1.0]), errors[:1], valid[:1], yaw_err[:1]
        )

    with pytest.raises(ValueError, match="share a 2D shape"):
        evaluate_finish_root_yaw_feasibility(times, errors, valid[:, :2], yaw_err)

    with pytest.raises(ValueError, match="row counts must match"):
        evaluate_finish_root_yaw_feasibility(times, errors[:50], valid[:50], yaw_err)

    with pytest.raises(ValueError, match="no samples in finish window"):
        evaluate_finish_root_yaw_feasibility(
            times, errors, valid, yaw_err, finish_start_s=5.0
        )


def test_evaluate_finish_root_yaw_feasibility_driver_receipt() -> None:
    """Driver receipt proves finish marker RMS < 176 mm without pre-impact regression."""
    times = np.linspace(0.0, 1.8, 181)
    n = len(times)
    impact_time_s = 0.85
    # Pre-impact RMS ~ 75 mm (< 85 mm baseline), finish RMS ~ 140 mm (< 176 mm baseline)
    errors = np.where(times[:, None] < 1.0, 0.075, 0.140)
    errors = np.repeat(errors, 4, axis=1)
    valid = np.ones_like(errors, dtype=bool)
    # Pelvis yaw error: starts small, ends around 12 deg
    yaw_err_rad = np.radians(np.linspace(0.0, 12.0, n))

    receipt = evaluate_finish_root_yaw_feasibility(
        times,
        errors,
        valid,
        yaw_err_rad,
        club="driver",
        finish_start_s=1.0,
        impact_time_s=impact_time_s,
    )

    assert receipt["club"] == "driver"
    assert receipt["finish_marker_rms_baseline_m"] == pytest.approx(
        FINISH_MARKER_RMS_BASELINES_M["driver"]
    )
    assert receipt["finish_marker_rms_m"] < 0.176
    assert receipt["finish_marker_rms_improved"] is True
    assert receipt["pre_impact_regression"] is False
    assert receipt["feasible"] is True
    assert receipt["pelvis_yaw_error_deg_max"] == pytest.approx(12.0, abs=0.1)


def test_evaluate_finish_root_yaw_feasibility_iron_receipt() -> None:
    """7-iron receipt proves finish marker RMS < 113 mm without pre-impact regression."""
    times = np.linspace(0.0, 1.8, 181)
    n = len(times)
    impact_time_s = 0.80
    # Pre-impact RMS ~ 80 mm (< 90 mm baseline), finish RMS ~ 98 mm (< 113 mm baseline)
    errors = np.where(times[:, None] < 1.0, 0.080, 0.098)
    errors = np.repeat(errors, 4, axis=1)
    valid = np.ones_like(errors, dtype=bool)
    yaw_err_rad = np.radians(np.linspace(0.0, 8.0, n))

    receipt = evaluate_finish_root_yaw_feasibility(
        times,
        errors,
        valid,
        yaw_err_rad,
        club="iron",
        finish_start_s=1.0,
        impact_time_s=impact_time_s,
    )

    assert receipt["club"] == "iron"
    assert receipt["finish_marker_rms_baseline_m"] == pytest.approx(
        FINISH_MARKER_RMS_BASELINES_M["iron"]
    )
    assert receipt["finish_marker_rms_m"] < 0.113
    assert receipt["finish_marker_rms_improved"] is True
    assert receipt["pre_impact_regression"] is False
    assert receipt["feasible"] is True
    assert receipt["pelvis_yaw_error_deg_max"] == pytest.approx(8.0, abs=0.1)


def test_evaluate_finish_root_yaw_feasibility_detects_pre_impact_regression() -> None:
    """Feasibility fails if pre-impact marker RMS regresses over baseline."""
    times = np.linspace(0.0, 1.8, 181)
    # Pre-impact RMS 120 mm (> 85 mm driver baseline)
    errors = np.full((len(times), 4), 0.120)
    valid = np.ones_like(errors, dtype=bool)
    yaw_err_rad = np.zeros(len(times))

    receipt = evaluate_finish_root_yaw_feasibility(
        times,
        errors,
        valid,
        yaw_err_rad,
        club="driver",
        finish_start_s=1.0,
        impact_time_s=0.85,
    )

    assert receipt["pre_impact_regression"] is True
    assert receipt["feasible"] is False

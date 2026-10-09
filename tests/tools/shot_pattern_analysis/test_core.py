"""Shot pattern comparison contracts and paired sampling tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.tools.shot_pattern_analysis.core import (
    AnalysisCancelled,
    AnalysisConfig,
    run_analysis,
)
from src.tools.shot_pattern_analysis.physics import ShotOutcome

pytestmark = pytest.mark.unit


class LinearPhysics:
    """Deterministic stand-in; isolates sampling and comparison logic."""

    def simulate(
        self,
        *,
        face_deg: float,
        path_deg: float,
        config: AnalysisConfig,
        sample_trajectory: bool = False,
    ) -> ShotOutcome:
        lateral = face_deg * 10.0 + (face_deg - path_deg) * 2.0
        return ShotOutcome(
            carry_x_m=200.0,
            carry_y_m=lateral,
            carry_m=math.hypot(200.0, lateral),
            launch_azimuth_deg=face_deg,
            spin_rpm=2500.0,
            spin_axis_tilt_deg=face_deg - path_deg,
            trajectory_xy_m=((0.0, 0.0), (100.0, lateral / 2.0), (200.0, lateral)),
        )


def test_paired_sampling_and_mirror_patterns() -> None:
    result = run_analysis(AnalysisConfig(n_shots=100, seed=42), physics=LinearPhysics())
    by_name = {
        name: [row for row in result.shots if row.pattern == name]
        for name in ("Straight", "Draw", "Fade")
    }
    assert all(len(rows) == 100 for rows in by_name.values())
    for straight, draw, fade in zip(*by_name.values(), strict=True):
        assert draw.face_deg - straight.face_deg == pytest.approx(1.5)
        assert fade.face_deg - straight.face_deg == pytest.approx(-1.5)
        assert (straight.path_deg, draw.path_deg, fade.path_deg) == (0.0, 3.0, -3.0)
        assert draw.shot_index == straight.shot_index == fade.shot_index


def test_doubled_curve_centers_keep_straight_baseline_and_paired_errors() -> None:
    result = run_analysis(
        AnalysisConfig(n_shots=12, curve_scale=2.0, seed=42), physics=LinearPhysics()
    )
    rows = {
        name: [r for r in result.shots if r.pattern == name]
        for name in ("Straight", "Draw", "Fade")
    }
    for straight, draw, fade in zip(*rows.values(), strict=True):
        assert straight.path_deg == 0.0
        assert draw.path_deg == 6.0
        assert fade.path_deg == -6.0
        assert draw.face_deg - straight.face_deg == pytest.approx(3.0)
        assert fade.face_deg - straight.face_deg == pytest.approx(-3.0)


def test_invalid_curve_scale_rejected() -> None:
    for value in (0.0, -1.0, float("nan"), float("inf"), 3.0):
        with pytest.raises((TypeError, ValueError)):
            AnalysisConfig(curve_scale=value)


def test_coupled_delivery_passes_pattern_nominal_face_reference() -> None:
    class CapturingPhysics(LinearPhysics):
        def __init__(self) -> None:
            self.calls: list[tuple[float, float]] = []

        def simulate(self, *, face_deg, path_deg, config, sample_trajectory=False, nominal_face_deg=0.0):
            self.calls.append((face_deg, nominal_face_deg))
            return super().simulate(
                face_deg=face_deg,
                path_deg=path_deg,
                config=config,
                sample_trajectory=sample_trajectory,
            )

    engine = CapturingPhysics()
    run_analysis(
        AnalysisConfig(n_shots=2, curve_scale=2.0, delivery_mode="shaft_rotation"),
        physics=engine,
    )
    assert engine.calls[0] == (0.0, 0.0)
    assert engine.calls[1] == (3.0, 3.0)
    assert engine.calls[2] == (-3.0, -3.0)
    assert engine.calls[6][1] == 3.0  # Draw sample deviations use draw reference.


def test_delivery_mode_contract() -> None:
    for mode in ("unknown", "", None):
        with pytest.raises((TypeError, ValueError)):
            AnalysisConfig(delivery_mode=mode)
    with pytest.raises(ValueError):
        AnalysisConfig(lie_deg=0)
    with pytest.raises(ValueError):
        AnalysisConfig(shaft_lean_deg=35)
    assert AnalysisConfig(club_id="driver").club_id == "driver"
    with pytest.raises(ValueError):
        AnalysisConfig(club_id="")


def test_aimed_and_raw_statistics_are_distinct() -> None:
    result = run_analysis(AnalysisConfig(n_shots=100, seed=42), physics=LinearPhysics())
    draw = result.summary["Draw"]
    straight = result.summary["Straight"]
    assert draw["raw_mean_lateral_m"] > straight["raw_mean_lateral_m"]
    assert abs(draw["aimed_mean_lateral_m"]) < 2.0
    assert draw["lateral_sd_m"] == pytest.approx(straight["lateral_sd_m"])
    assert draw["raw_target_hit_fraction"] < draw["aimed_target_hit_fraction"]


@pytest.mark.parametrize(
    "changes",
    [
        {"n_shots": 0},
        {"face_sd_deg": -1},
        {"seed": -1},
        {"club_speed_mps": 0},
        {"dt_s": 0},
        {"max_time_s": 0},
        {"target_radius_m": -1},
        {"loft_deg": float("nan")},
    ],
)
def test_invalid_inputs_rejected(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        AnalysisConfig(**changes)


def test_seed_reproduces_face_deviations() -> None:
    cfg = AnalysisConfig(n_shots=17, seed=731)
    first = run_analysis(cfg, physics=LinearPhysics())
    second = run_analysis(cfg, physics=LinearPhysics())
    np.testing.assert_allclose(
        [r.face_deg for r in first.shots], [r.face_deg for r in second.shots]
    )


def test_nonfinite_physics_result_fails_contract() -> None:
    class BrokenPhysics(LinearPhysics):
        def simulate(self, *, face_deg, path_deg, config, sample_trajectory=False):
            good = super().simulate(
                face_deg=face_deg,
                path_deg=path_deg,
                config=config,
                sample_trajectory=sample_trajectory,
            )
            return ShotOutcome(
                float("nan"),
                good.carry_y_m,
                good.carry_m,
                good.launch_azimuth_deg,
                good.spin_rpm,
                good.spin_axis_tilt_deg,
            )

    with pytest.raises(ValueError, match="finite"):
        run_analysis(AnalysisConfig(n_shots=2), physics=BrokenPhysics())


def test_progress_and_cooperative_cancellation() -> None:
    progress: list[tuple[int, int]] = []
    checks = 0

    def cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks > 7

    with pytest.raises(AnalysisCancelled):
        run_analysis(
            AnalysisConfig(n_shots=100),
            physics=LinearPhysics(),
            progress_callback=lambda done, total: progress.append((done, total)),
            cancel_check=cancel,
        )
    assert checks == 8
    assert progress[0][1] == 300

    finished: list[tuple[int, int]] = []
    run_analysis(
        AnalysisConfig(n_shots=10),
        physics=LinearPhysics(),
        progress_callback=lambda done, total: finished.append((done, total)),
    )
    assert finished[-1] == (30, 30)

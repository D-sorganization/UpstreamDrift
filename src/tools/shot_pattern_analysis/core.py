"""Paired Monte Carlo comparison of straight, draw, and fade shot patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .physics import ShotOutcome, ShotPhysics


@dataclass(frozen=True)
class PatternConfig:
    name: str
    face_deg: float
    path_deg: float


PATTERNS = (
    PatternConfig("Straight", 0.0, 0.0),
    PatternConfig("Draw", 1.5, 3.0),
    PatternConfig("Fade", -1.5, -3.0),
)


class AnalysisCancelled(RuntimeError):
    """Raised when a caller cancels a running Monte Carlo comparison."""


@dataclass(frozen=True)
class AnalysisConfig:
    club_id: str = "custom"
    n_shots: int = 10_000
    face_sd_deg: float = 1.0
    curve_scale: float = 1.0
    seed: int = 20_261_008
    club_speed_mps: float = 45.0
    loft_deg: float = 10.9
    attack_angle_deg: float = 0.0
    clubhead_mass_kg: float = 0.2
    delivery_mode: str = "fixed_loft"
    lie_deg: float = 58.0
    shaft_lean_deg: float = 0.0
    dt_s: float = 0.02
    max_time_s: float = 12.0
    target_radius_m: float = 15.0

    def __post_init__(self) -> None:
        if not isinstance(self.club_id, str) or not self.club_id.strip():
            raise ValueError("club_id must be a non-empty string")
        if not isinstance(self.n_shots, int) or self.n_shots < 2:
            raise ValueError("n_shots must be an integer >= 2")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "face_sd_deg",
            "curve_scale",
            "club_speed_mps",
            "loft_deg",
            "attack_angle_deg",
            "clubhead_mass_kg",
            "lie_deg",
            "shaft_lean_deg",
            "dt_s",
            "max_time_s",
            "target_radius_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.face_sd_deg < 0 or self.face_sd_deg > 5:
            raise ValueError("face_sd_deg must be in [0, 5]")
        if not 0 < self.curve_scale <= 2:
            raise ValueError("curve_scale must be in (0, 2]")
        if self.club_speed_mps <= 0 or not (0 < self.loft_deg < 45):
            raise ValueError("club speed must be positive and loft in (0, 45) degrees")
        if self.clubhead_mass_kg <= 0 or abs(self.attack_angle_deg) >= 30:
            raise ValueError(
                "clubhead mass must be positive and attack angle below 30 degrees"
            )
        if self.delivery_mode not in ("fixed_loft", "shaft_rotation"):
            raise ValueError("delivery_mode must be fixed_loft or shaft_rotation")
        from .delivery_geometry import delivery_from_face_angle

        delivery_from_face_angle(
            0.0,
            base_loft_deg=self.loft_deg,
            lie_deg=self.lie_deg,
            shaft_lean_deg=self.shaft_lean_deg,
        )
        if not (0 < self.dt_s <= 0.05) or self.max_time_s <= self.dt_s:
            raise ValueError("dt_s must be in (0, .05] and below max_time_s")
        if self.target_radius_m < 0:
            raise ValueError("target_radius_m must be non-negative")


@dataclass(frozen=True)
class ShotRecord:
    pattern: str
    shot_index: int
    face_deg: float
    path_deg: float
    launch_azimuth_deg: float
    spin_rpm: float
    spin_axis_tilt_deg: float
    carry_x_m: float
    carry_y_m: float
    carry_m: float
    aimed_x_m: float
    aimed_y_m: float


@dataclass(frozen=True)
class AnalysisResult:
    config: AnalysisConfig
    shots: tuple[ShotRecord, ...]
    summary: dict[str, dict[str, float]]
    flight_samples: dict[str, tuple[tuple[tuple[float, float], ...], ...]]
    target_x_m: float
    nominal_outcomes: dict[str, ShotOutcome]


class PhysicsProtocol(Protocol):
    def simulate(
        self,
        *,
        face_deg: float,
        path_deg: float,
        config: AnalysisConfig,
        sample_trajectory: bool = False,
        nominal_face_deg: float = 0.0,
    ) -> ShotOutcome: ...


def _validate_outcome(outcome: ShotOutcome) -> ShotOutcome:
    values = (
        outcome.carry_x_m,
        outcome.carry_y_m,
        outcome.carry_m,
        outcome.launch_azimuth_deg,
        outcome.spin_rpm,
        outcome.spin_axis_tilt_deg,
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("physics outcome must contain only finite values")
    if outcome.carry_x_m <= 0 or outcome.carry_m <= 0 or outcome.spin_rpm < 0:
        raise ValueError("physics outcome violates positive carry/spin contract")
    return outcome


def _rotate_to_aim(x: float, y: float, aim_angle: float) -> tuple[float, float]:
    ca, sa = math.cos(aim_angle), math.sin(aim_angle)
    return x * ca + y * sa, -x * sa + y * ca


def _summarize(
    rows: list[ShotRecord], target_x_m: float, radius_m: float
) -> dict[str, float]:
    lateral = np.array([r.carry_y_m for r in rows])
    aimed_lateral = np.array([r.aimed_y_m for r in rows])
    carries = np.array([r.carry_m for r in rows])
    raw_error = np.array(
        [math.hypot(r.carry_x_m - target_x_m, r.carry_y_m) for r in rows]
    )
    aimed_error = np.array(
        [math.hypot(r.aimed_x_m - target_x_m, r.aimed_y_m) for r in rows]
    )
    q5, q50, q95 = np.quantile(lateral, [0.05, 0.5, 0.95])
    aq5, aq50, aq95 = np.quantile(aimed_lateral, [0.05, 0.5, 0.95])
    n = len(rows)

    def wilson(success_fraction: float) -> tuple[float, float]:
        z = 1.959963984540054
        center = (success_fraction + z * z / (2 * n)) / (1 + z * z / n)
        spread = (
            z
            * math.sqrt(
                success_fraction * (1 - success_fraction) / n + z * z / (4 * n * n)
            )
            / (1 + z * z / n)
        )
        return center - spread, center + spread

    raw_hit = float(np.mean(raw_error <= radius_m))
    aimed_hit = float(np.mean(aimed_error <= radius_m))
    raw_hit_lo, raw_hit_hi = wilson(raw_hit)
    aimed_hit_lo, aimed_hit_hi = wilson(aimed_hit)
    return {
        "n": float(n),
        "raw_mean_lateral_m": float(np.mean(lateral)),
        "aimed_mean_lateral_m": float(np.mean(aimed_lateral)),
        "lateral_sd_m": float(np.std(lateral, ddof=1)),
        "lateral_variance_m2": float(np.var(lateral, ddof=1)),
        "lateral_mean_se_m": float(np.std(lateral, ddof=1) / math.sqrt(n)),
        "aimed_lateral_sd_m": float(np.std(aimed_lateral, ddof=1)),
        "aimed_lateral_variance_m2": float(np.var(aimed_lateral, ddof=1)),
        "raw_lateral_p05_m": float(q5),
        "raw_lateral_p50_m": float(q50),
        "raw_lateral_p95_m": float(q95),
        "aimed_lateral_p05_m": float(aq5),
        "aimed_lateral_p50_m": float(aq50),
        "aimed_lateral_p95_m": float(aq95),
        "raw_lateral_p05_p95_width_m": float(q95 - q5),
        "aimed_lateral_p05_p95_width_m": float(aq95 - aq5),
        "raw_target_hit_fraction": raw_hit,
        "raw_target_hit_wilson95_low": raw_hit_lo,
        "raw_target_hit_wilson95_high": raw_hit_hi,
        "aimed_target_hit_fraction": aimed_hit,
        "aimed_target_hit_wilson95_low": aimed_hit_lo,
        "aimed_target_hit_wilson95_high": aimed_hit_hi,
        "raw_target_rmse_m": float(np.sqrt(np.mean(raw_error**2))),
        "aimed_target_rmse_m": float(np.sqrt(np.mean(aimed_error**2))),
        "mean_carry_m": float(np.mean(carries)),
        "carry_sd_m": float(np.std(carries, ddof=1)),
        "mean_launch_azimuth_deg": float(np.mean([r.launch_azimuth_deg for r in rows])),
        "mean_spin_axis_tilt_deg": float(np.mean([r.spin_axis_tilt_deg for r in rows])),
        "opposite_curve_fraction": float(
            np.mean(
                [
                    r.face_deg > r.path_deg
                    if rows[0].pattern == "Draw"
                    else r.face_deg < r.path_deg
                    if rows[0].pattern == "Fade"
                    else False
                    for r in rows
                ]
            )
        ),
    }


def run_analysis(
    config: AnalysisConfig,
    *,
    physics: PhysicsProtocol | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> AnalysisResult:
    """Simulate three paired patterns using common normal face deviations.

    Nominal aim rotates each pattern's coordinates until its nominal carry
    lies on the target line. The target remains the straight nominal carry.
    """
    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be AnalysisConfig")
    patterns = tuple(
        PatternConfig(
            p.name, p.face_deg * config.curve_scale, p.path_deg * config.curve_scale
        )
        for p in PATTERNS
    )
    total_shots = len(patterns) * config.n_shots
    completed = 0
    report_every = max(1, total_shots // 100)
    if progress_callback is not None:
        progress_callback(completed, total_shots)
    engine = physics if physics is not None else ShotPhysics()
    rng = np.random.default_rng(config.seed)
    face_deviations = rng.normal(0.0, config.face_sd_deg, config.n_shots)
    sample_indices = {
        int(v)
        for v in np.argsort(face_deviations)[
            np.linspace(0, config.n_shots - 1, 9, dtype=int)
        ]
    }

    def simulate_pattern(
        pattern: PatternConfig, face_deg: float, *, sample_trajectory: bool
    ) -> ShotOutcome:
        if config.delivery_mode == "shaft_rotation":
            return engine.simulate(
                face_deg=face_deg,
                path_deg=pattern.path_deg,
                config=config,
                sample_trajectory=sample_trajectory,
                nominal_face_deg=pattern.face_deg,
            )
        return engine.simulate(
            face_deg=face_deg,
            path_deg=pattern.path_deg,
            config=config,
            sample_trajectory=sample_trajectory,
        )

    nominal = {
        pattern.name: _validate_outcome(
            simulate_pattern(pattern, pattern.face_deg, sample_trajectory=True)
        )
        for pattern in patterns
    }
    target_x_m = nominal["Straight"].carry_x_m
    rows: list[ShotRecord] = []
    samples: dict[str, tuple[tuple[tuple[float, float], ...], ...]] = {}
    stats: dict[str, dict[str, float]] = {}
    for pattern in patterns:
        aim = math.atan2(
            nominal[pattern.name].carry_y_m, nominal[pattern.name].carry_x_m
        )
        pattern_rows: list[ShotRecord] = []
        trajectories = [nominal[pattern.name].trajectory_xy_m]
        for index, deviation in enumerate(face_deviations):
            if cancel_check is not None and cancel_check():
                raise AnalysisCancelled("shot pattern analysis cancelled")
            face = pattern.face_deg + float(deviation)
            outcome = _validate_outcome(
                simulate_pattern(
                    pattern, face, sample_trajectory=index in sample_indices
                )
            )
            ax, ay = _rotate_to_aim(outcome.carry_x_m, outcome.carry_y_m, aim)
            pattern_rows.append(
                ShotRecord(
                    pattern=pattern.name,
                    shot_index=index,
                    face_deg=face,
                    path_deg=pattern.path_deg,
                    launch_azimuth_deg=outcome.launch_azimuth_deg,
                    spin_rpm=outcome.spin_rpm,
                    spin_axis_tilt_deg=outcome.spin_axis_tilt_deg,
                    carry_x_m=outcome.carry_x_m,
                    carry_y_m=outcome.carry_y_m,
                    carry_m=outcome.carry_m,
                    aimed_x_m=ax,
                    aimed_y_m=ay,
                )
            )
            if outcome.trajectory_xy_m:
                trajectories.append(outcome.trajectory_xy_m)
            completed += 1
            if progress_callback is not None and (
                completed % report_every == 0 or completed == total_shots
            ):
                progress_callback(completed, total_shots)
        stats[pattern.name] = _summarize(
            pattern_rows, target_x_m, config.target_radius_m
        )
        rows.extend(pattern_rows)
        samples[pattern.name] = tuple(trajectories)
    if len(rows) != 3 * config.n_shots:
        raise RuntimeError("each pattern must produce exactly n_shots outcomes")
    return AnalysisResult(config, tuple(rows), stats, samples, target_x_m, nominal)


__all__ = [
    "AnalysisCancelled",
    "AnalysisConfig",
    "AnalysisResult",
    "PATTERNS",
    "PatternConfig",
    "ShotRecord",
    "run_analysis",
]

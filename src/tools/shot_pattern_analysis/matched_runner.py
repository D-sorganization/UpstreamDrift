"""Paired matched-endpoint experiments with complete population accounting."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypedDict

import numpy as np
from scipy.stats import binomtest

from .core import PATTERN_NAMES, PATTERNS, AnalysisCancelled
from .dispersion_stats import landing_dispersion
from .matched_config import MAX_CLUB_SPEED_MPS, MatchedConfig
from .matched_diagnostics import joint_diagnostics
from .matched_physics import MatchedDeliveryOutcome, MatchedShotPhysics, UnsupportedShot
from .noise import NOISE_FIELDS, NoiseSpec, sample_paired_noise
from .nominal_matching import MatchBounds, MatchedNominal, match_nominal_landing
from .physics import ShotOutcome
from .scenario_scoring import score_course_endpoints
from .uncertainty import PUBLICATION_RESAMPLES, paired_mean_difference


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen experiment request; target None selects within-lean matching."""

    club: MatchedConfig
    noise: NoiseSpec
    n_shots: int = 10000
    seed: int = 20261009
    curve_scale: float = 1.0
    target_x_m: float | None = None
    match_tolerance_m: float = 0.05
    target_radius_m: float = 15.0
    match_face_bounds_deg: tuple[float, float] | None = None
    match_speed_factors: tuple[float, float] = (0.75, 1.25)

    def __post_init__(self) -> None:
        if not isinstance(self.club, MatchedConfig):
            raise TypeError("club must be a MatchedConfig")
        if not isinstance(self.noise, NoiseSpec):
            raise TypeError("noise must be a NoiseSpec")
        if (
            isinstance(self.n_shots, bool)
            or not isinstance(self.n_shots, int)
            or self.n_shots < 2
        ):
            raise ValueError("n_shots must be an integer >=2")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        for value in (self.curve_scale, self.match_tolerance_m, self.target_radius_m):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "curve scale, match tolerance and target radius must be positive"
                )
        if self.curve_scale > 2:
            raise ValueError("curve scale must not exceed2")
        if self.target_x_m is not None and (
            not math.isfinite(self.target_x_m) or self.target_x_m <= 0
        ):
            raise ValueError("target distance must be finite and positive")
        if self.match_face_bounds_deg is not None:
            if len(self.match_face_bounds_deg) != 2:
                raise ValueError("match face bounds require two values")
            MatchBounds(*self.match_face_bounds_deg, 1.0, 2.0)
        factors = self.match_speed_factors
        if (
            len(factors) != 2
            or not all(math.isfinite(value) for value in factors)
            or factors[0] <= 0
            or factors[0] >= factors[1]
        ):
            raise ValueError(
                "match speed bounds require finite ordered positive factors"
            )
        if factors[1] * self.club.club_speed_mps >= MAX_CLUB_SPEED_MPS:
            raise ValueError("match speed bounds must stay below 100 m/s club speed")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        """Rebuild a request from its ``asdict`` JSON form (lists become tuples).

        Raises KeyError, TypeError or ValueError for an incomplete or invalid
        record; the round trip ``from_dict(asdict(config)) == config`` holds.
        """
        club = dict(values["club"])
        club["inertia_diagonal_kg_m2"] = tuple(club["inertia_diagonal_kg_m2"])
        noise = values["noise"]
        correlation = noise.get("correlation")
        options: dict[str, Any] = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in values.items()
            if key not in ("club", "noise")
        }
        return cls(
            club=MatchedConfig(**club),
            noise=NoiseSpec(
                tuple(noise["standard_deviations"]),
                None if correlation is None else tuple(map(tuple, correlation)),
            ),
            **options,
        )

    def bounds_for(self, reference_face_deg: float) -> MatchBounds:
        """Resolve the recorded request into an absolute physical search box."""
        faces = self.match_face_bounds_deg
        if faces is None:
            faces = (reference_face_deg - 3.0, reference_face_deg + 3.0)
        speed = self.club.club_speed_mps
        return MatchBounds(
            faces[0],
            faces[1],
            self.match_speed_factors[0] * speed,
            self.match_speed_factors[1] * speed,
        )


@dataclass(frozen=True)
class MatchedShotRecord:
    pattern: str
    shot_index: int
    status: str
    shaft_rotation_deg: float
    path_deg: float
    speed_mps: float
    contact_horizontal_m: float
    contact_vertical_m: float
    shaft_lean_deg: float
    attack_angle_deg: float
    face_deg: float | None = None
    loft_deg: float | None = None
    contact_face_deg: float | None = None
    contact_loft_deg: float | None = None
    carry_x_m: float | None = None
    carry_y_m: float | None = None
    spin_rpm: float | None = None
    launch_azimuth_deg: float | None = None
    spin_axis_tilt_deg: float | None = None
    launch_elevation_deg: float | None = None
    ball_speed_mps: float | None = None


@dataclass(frozen=True)
class MatchedAnalysisResult:
    config: ExperimentConfig
    target_x_m: float
    nominals: dict[str, MatchedNominal]
    shots: tuple[MatchedShotRecord, ...]
    summary: dict[str, dict]
    input_diagnostics: dict
    course_scoring: dict
    paired_comparisons: dict
    flight_samples: dict[str, tuple]


class _ShotInputs(TypedDict):
    pattern: str
    shot_index: int
    status: str
    shaft_rotation_deg: float
    path_deg: float
    speed_mps: float
    contact_horizontal_m: float
    contact_vertical_m: float
    shaft_lean_deg: float
    attack_angle_deg: float


def _point(row: MatchedShotRecord) -> tuple[float, float]:
    if row.status != "supported" or row.carry_x_m is None or row.carry_y_m is None:
        raise ValueError("supported endpoints require finite coordinates")
    return row.carry_x_m, row.carry_y_m


class MatchedPhysicsProtocol(Protocol):
    def simulate(
        self,
        *,
        face_deg: float,
        path_deg: float,
        config: MatchedConfig,
        sample_trajectory: bool = False,
        nominal_face_deg: float | None = None,
    ) -> ShotOutcome: ...
    def simulate_delivery(
        self,
        *,
        nominal_face_deg: float,
        path_deg: float,
        config: MatchedConfig,
        shaft_rotation_deg: float = 0.0,
        contact_horizontal_m: float = 0.0,
        contact_vertical_m: float = 0.0,
        sample_trajectory: bool = False,
    ) -> MatchedDeliveryOutcome: ...


def calibrate_patterns(
    request: ExperimentConfig, physics: MatchedPhysicsProtocol
) -> tuple[float, dict[str, MatchedNominal]]:
    """Verify all nominal raw landings before drawing any random deliveries."""
    club = request.club
    target = request.target_x_m
    if target is None:
        target = physics.simulate(face_deg=0, path_deg=0, config=club).carry_x_m
    nominals = {}
    for pattern in PATTERNS:
        reference = pattern.face_deg * request.curve_scale
        nominals[pattern.name] = match_nominal_landing(
            physics,
            config=club,
            target_x_m=target,
            reference_face_deg=reference,
            face_to_path_deg=(pattern.face_deg - pattern.path_deg)
            * request.curve_scale,
            bounds=request.bounds_for(reference),
            tolerance_m=request.match_tolerance_m,
        )
    return target, nominals


def _summarize_records(
    rows: list[MatchedShotRecord], target: float, radius: float
) -> dict:
    good = [r for r in rows if r.status == "supported"]
    interval = binomtest(len(good), len(rows)).proportion_ci(confidence_level=0.95)
    summary: dict[str, Any] = {
        "total_n": len(rows),
        "supported_n": len(good),
        "unsupported_reasons": dict(
            Counter(r.status for r in rows if r.status != "supported")
        ),
        "supported_fraction": len(good) / len(rows),
        "supported_fraction_exact95": [float(interval.low), float(interval.high)],
        "statistics_conditioning": "all_samples"
        if len(good) == len(rows)
        else "supported_airborne_only",
    }
    if len(good) < 2:
        summary["statistics"] = None
        return summary
    summary["joint_delivery_outcome_diagnostics"] = joint_diagnostics(good)
    points = [_point(r) for r in good]
    summary["statistics"] = landing_dispersion(
        points, target, nominal_point=(target, 0)
    )
    xy = np.asarray(points)
    radial = np.hypot(xy[:, 0] - target, xy[:, 1])
    summary.update(
        lateral_sd_m=float(np.std(xy[:, 1], ddof=1)),
        downrange_sd_m=float(np.std(xy[:, 0], ddof=1)),
        mean_carry_m=float(np.mean(np.hypot(xy[:, 0], xy[:, 1]))),
        target_rmse_m=float(np.sqrt(np.mean(radial**2))),
        target_containment_fraction=float(np.mean(radial <= radius)),
    )
    faces = np.asarray([r.face_deg for r in good], dtype=float)
    paths = np.asarray([r.path_deg for r in good], dtype=float)
    summary["realized_face_path_covariance_deg2"] = float(
        np.cov(faces, paths, ddof=1)[0, 1]
    )
    summary["realized_face_path_correlation"] = (
        float(np.corrcoef(faces, paths)[0, 1])
        if np.std(faces) > 1e-14 and np.std(paths) > 1e-14
        else None
    )
    return summary


def _paired_results(
    rows: tuple[MatchedShotRecord, ...],
    request: ExperimentConfig,
    target: float,
    include_scoring: bool,
    resamples: int = PUBLICATION_RESAMPLES,
) -> tuple[dict, dict]:
    by = {
        p.name: {
            r.shot_index: r
            for r in rows
            if r.pattern == p.name and r.status == "supported"
        }
        for p in PATTERNS
    }
    indices = sorted(set.intersection(*(set(v) for v in by.values())))
    endpoints = {name: [(i, *_point(v[i])) for i in indices] for name, v in by.items()}
    return paired_endpoint_statistics(
        endpoints,
        club_id=request.club.club_id,
        target_x_m=target,
        target_radius_m=request.target_radius_m,
        n_shots=request.n_shots,
        seed=request.seed,
        include_scoring=include_scoring,
        resamples=resamples,
    )


def paired_endpoint_statistics(
    endpoints: dict[str, list[tuple[int, float, float]]],
    *,
    club_id: str,
    target_x_m: float,
    target_radius_m: float,
    n_shots: int,
    seed: int,
    include_scoring: bool = True,
    resamples: int = PUBLICATION_RESAMPLES,
) -> tuple[dict, dict]:
    """Return ``(course_scoring, paired_comparisons)`` for common endpoints.

    ``endpoints`` maps every pattern name to the same ordered shot indices:
    the shots each pattern supported. Intervals are paired bootstraps over
    those shared indices; they cover Monte Carlo sampling only.
    """
    if set(endpoints) != set(PATTERN_NAMES):
        raise ValueError("endpoints must cover exactly the Straight/Draw/Fade patterns")
    indices = [i for i, _, _ in endpoints["Straight"]]
    if any([i for i, _, _ in points] != indices for points in endpoints.values()):
        raise ValueError("every pattern must use the same ordered shot indices")
    if len(indices) > n_shots:
        raise ValueError("paired count cannot exceed the attempted shot count")
    info = {
        "paired_n": len(indices),
        "total_n": n_shots,
        "conditioning": (
            "all_samples"
            if len(indices) == n_shots
            else "common_supported_indices_only"
        ),
        "bootstrap_resamples": resamples,
    }
    if len(indices) < 2:
        return {
            **info,
            "status": "unavailable",
            "reason": "fewer_than_two_common_supported_shots",
        }, {**info, "status": "unavailable"}
    radii = {
        name: np.asarray([math.hypot(x - target_x_m, y) for _, x, y in points])
        for name, points in endpoints.items()
    }
    comparisons: dict[str, Any] = {**info}
    for key, transform in (
        ("mean_radial_error_difference_vs_straight", lambda r: r),
        ("mean_squared_target_error_difference_m2", lambda r: r**2),
        (
            "containment_fraction_difference",
            lambda r: (r <= target_radius_m).astype(float),
        ),
    ):
        comparisons[key] = {
            name: paired_mean_difference(
                transform(radii[name]),
                transform(radii["Straight"]),
                seed=seed,
                resamples=resamples,
            )
            for name in ("Draw", "Fade")
        }
    comparisons["uncertainty_scope"] = (
        "paired bootstrap; sampling only; excludes model and course uncertainty"
    )
    if not include_scoring:
        return {**info, "status": "not_requested"}, comparisons
    try:
        scoring = score_course_endpoints(
            endpoints,
            club_id=club_id,
            target_x_m=target_x_m,
            green_radius_m=target_radius_m,
            seed=seed,
            resamples=resamples,
        )
    except ValueError as exc:
        return {**info, "status": "unavailable", "reason": str(exc)}, comparisons
    scoring.update(info)
    if len(indices) < n_shots:
        scoring["status"] = "conditional_only"
        scoring["full_population_status"] = "unavailable_unsupported_terminal_lies"
    else:
        scoring["full_population_status"] = (
            "available_under_hypothetical_course_assumptions"
        )
    return scoring, comparisons


def _supported_record(
    inputs: _ShotInputs,
    delivered: MatchedDeliveryOutcome,
) -> MatchedShotRecord:
    """Assemble a supported record while keeping numerical failures fatal."""
    outcome = delivered.outcome
    if not all(
        math.isfinite(v)
        for v in (outcome.carry_x_m, outcome.carry_y_m, outcome.spin_rpm)
    ):
        raise RuntimeError("matched flight returned nonfinite values")
    return MatchedShotRecord(
        **inputs,
        face_deg=delivered.delivered_face_deg,
        loft_deg=delivered.delivered_loft_deg,
        contact_face_deg=delivered.contact_face_deg,
        contact_loft_deg=delivered.contact_loft_deg,
        carry_x_m=outcome.carry_x_m,
        carry_y_m=outcome.carry_y_m,
        spin_rpm=outcome.spin_rpm,
        launch_azimuth_deg=outcome.launch_azimuth_deg,
        spin_axis_tilt_deg=(
            outcome.spin_axis_tilt_deg if outcome.spin_axis_defined else None
        ),
        launch_elevation_deg=outcome.launch_elevation_deg,
        ball_speed_mps=outcome.ball_speed_mps,
    )


def _simulate_noisy_shot(
    engine: MatchedPhysicsProtocol,
    request: ExperimentConfig,
    name: str,
    nominal: MatchedNominal,
    nominal_club: MatchedConfig,
    index: int,
    error: np.ndarray,
) -> tuple[MatchedShotRecord, tuple[tuple[float, float], ...] | None]:
    """Simulate one perturbed delivery; return its record and sampled flight.

    A delivery outside the club's valid range or an unsupported contact is
    recorded with its reason, never dropped. The flight is returned only for
    the first five supported shots of a pattern.
    """
    rotation, path, speed, horizontal, vertical, lean, attack = (
        float(v) for v in error
    )
    inputs: _ShotInputs = {
        "pattern": name,
        "shot_index": index,
        "status": "supported",
        "shaft_rotation_deg": rotation,
        "path_deg": nominal.path_deg + path,
        "speed_mps": nominal.speed_mps + speed,
        "contact_horizontal_m": horizontal,
        "contact_vertical_m": vertical,
        "shaft_lean_deg": request.club.shaft_lean_deg + lean,
        "attack_angle_deg": request.club.attack_angle_deg + attack,
    }
    try:
        club = replace(
            nominal_club,
            club_speed_mps=inputs["speed_mps"],
            shaft_lean_deg=inputs["shaft_lean_deg"],
            attack_angle_deg=inputs["attack_angle_deg"],
        )
    except ValueError:
        inputs["status"] = "delivery_out_of_range"
        return MatchedShotRecord(**inputs), None
    sample = index < 5
    try:
        delivered = engine.simulate_delivery(
            nominal_face_deg=nominal.face_deg,
            path_deg=inputs["path_deg"],
            config=club,
            shaft_rotation_deg=rotation,
            contact_horizontal_m=horizontal,
            contact_vertical_m=vertical,
            sample_trajectory=sample,
        )
    except UnsupportedShot as exc:
        inputs["status"] = exc.reason
        return MatchedShotRecord(**inputs), None
    trajectory = delivered.outcome.trajectory_xy_m if sample else None
    return _supported_record(inputs, delivered), trajectory


def _input_diagnostics(request: ExperimentConfig, errors: np.ndarray) -> dict:
    """Summarize the original paired, untruncated input matrix in field order."""
    return {
        "fields": list(NOISE_FIELDS),
        "requested_standard_deviations": list(request.noise.standard_deviations),
        "requested_correlation": request.noise.correlation_matrix().tolist(),
        "realized_standard_deviations": np.std(errors, axis=0, ddof=1).tolist(),
        "realized_covariance": np.cov(errors, rowvar=False, ddof=1).tolist(),
        "distribution": "untruncated_multivariate_normal_assumption",
        "paired_inputs": True,
    }


def run_matched_analysis(
    request: ExperimentConfig,
    *,
    physics: MatchedPhysicsProtocol | None = None,
    include_scoring: bool = True,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    resamples: int = PUBLICATION_RESAMPLES,
) -> MatchedAnalysisResult:
    """Retain every seeded input once per pattern; never clip or resample failures.

    ``resamples`` sets the paired-bootstrap count for the reported intervals.
    """
    engine = physics or MatchedShotPhysics()
    target, nominals = calibrate_patterns(request, engine)
    errors = sample_paired_noise(
        request.noise, n_shots=request.n_shots, seed=request.seed
    )
    rows = []
    flights = {}
    for name, nominal in nominals.items():
        nominal_club = replace(request.club, club_speed_mps=nominal.speed_mps)
        paths = [
            engine.simulate(
                face_deg=nominal.face_deg,
                path_deg=nominal.path_deg,
                config=nominal_club,
                sample_trajectory=True,
            ).trajectory_xy_m
        ]
        for index, error in enumerate(errors):
            if cancelled is not None and cancelled():
                raise AnalysisCancelled("matched shot analysis cancelled")
            record, trajectory = _simulate_noisy_shot(
                engine, request, name, nominal, nominal_club, index, error
            )
            rows.append(record)
            if trajectory is not None:
                paths.append(trajectory)
            if progress is not None and (
                index % 100 == 0 or index == request.n_shots - 1
            ):
                progress(len(rows), 3 * request.n_shots)
        flights[name] = tuple(paths)
    records = tuple(rows)
    summary = {
        name: _summarize_records(
            [r for r in rows if r.pattern == name], target, request.target_radius_m
        )
        for name in nominals
    }
    scoring, comparisons = _paired_results(
        records, request, target, include_scoring, resamples
    )
    diagnostics = _input_diagnostics(request, errors)
    return MatchedAnalysisResult(
        request,
        target,
        nominals,
        records,
        summary,
        diagnostics,
        scoring,
        comparisons,
        flights,
    )

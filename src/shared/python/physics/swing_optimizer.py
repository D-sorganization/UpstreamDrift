"""Inverse swing optimizer for the swing -> ball-flight pipeline.

This module implements the bounded core API for #7220. It composes the
existing ``SwingBallFlightPipeline`` instead of reaching through impact or
flight internals.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.optimize import minimize

from src.shared.python.contracts import ensure, require
from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.swing_ball_flight_pipeline import (
    PipelineResult,
    SwingBallFlightPipeline,
    SwingState,
)


@dataclass(frozen=True)
class FlightTarget:
    """Target ball-flight metrics in metres."""

    carry_m: float
    max_height_m: float | None = None
    lateral_m: float | None = None

    def __post_init__(self) -> None:
        require(math.isfinite(self.carry_m), "carry_m must be finite", self.carry_m)
        require(self.carry_m >= 0.0, "carry_m must be non-negative", self.carry_m)
        if self.max_height_m is not None:
            require(
                math.isfinite(self.max_height_m),
                "max_height_m must be finite",
                self.max_height_m,
            )
            require(
                self.max_height_m >= 0.0,
                "max_height_m must be non-negative",
                self.max_height_m,
            )
        if self.lateral_m is not None:
            require(
                math.isfinite(self.lateral_m),
                "lateral_m must be finite",
                self.lateral_m,
            )


@dataclass(frozen=True)
class ClubPreset:
    """Swing-parameter bounds for a club family."""

    name: str
    speed_bounds_mps: tuple[float, float]
    loft_bounds_deg: tuple[float, float]
    attack_angle_bounds_deg: tuple[float, float]
    face_to_path_bounds_deg: tuple[float, float]
    initial_guess: tuple[float, float, float, float]
    clubhead_mass: float = 0.200
    clubhead_moi: float = 5e-3

    def __post_init__(self) -> None:
        _require_ordered_bounds(self.speed_bounds_mps, "speed_bounds_mps")
        _require_ordered_bounds(self.loft_bounds_deg, "loft_bounds_deg")
        _require_ordered_bounds(self.attack_angle_bounds_deg, "attack_angle_bounds_deg")
        _require_ordered_bounds(self.face_to_path_bounds_deg, "face_to_path_bounds_deg")
        require(self.clubhead_mass > 0.0, "clubhead_mass must be positive")
        require(self.clubhead_moi > 0.0, "clubhead_moi must be positive")
        for value, bounds, name in zip(
            self.initial_guess,
            self.bounds,
            ("speed", "loft", "attack_angle", "face_to_path"),
            strict=True,
        ):
            require(
                bounds[0] <= value <= bounds[1],
                f"{name} initial_guess must be within bounds",
                value,
            )

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Return scipy-compatible bounds in optimizer parameter order."""
        return (
            self.speed_bounds_mps,
            self.loft_bounds_deg,
            self.attack_angle_bounds_deg,
            self.face_to_path_bounds_deg,
        )

    @classmethod
    def driver(cls) -> ClubPreset:
        """Return conservative driver bounds."""
        return cls(
            name="driver",
            speed_bounds_mps=(30.0, 62.0),
            loft_bounds_deg=(7.0, 15.0),
            attack_angle_bounds_deg=(-6.0, 8.0),
            face_to_path_bounds_deg=(-10.0, 10.0),
            initial_guess=(44.0, 10.5, 2.0, 0.0),
            clubhead_mass=0.200,
            clubhead_moi=5e-3,
        )

    @classmethod
    def iron_7(cls) -> ClubPreset:
        """Return conservative 7-iron bounds."""
        return cls(
            name="7i",
            speed_bounds_mps=(22.0, 45.0),
            loft_bounds_deg=(26.0, 36.0),
            attack_angle_bounds_deg=(-8.0, 4.0),
            face_to_path_bounds_deg=(-10.0, 10.0),
            initial_guess=(34.0, 32.0, -4.0, 0.0),
            clubhead_mass=0.268,
            clubhead_moi=4e-3,
        )


@dataclass(frozen=True)
class OptimizationControls:
    """Iteration, timeout, and objective controls for ``SwingOptimizer``."""

    max_iterations: int = 64
    timeout_s: float = 2.0
    absolute_tolerance_m: float = 1.0
    carry_tolerance_fraction: float = 0.02
    weights: Mapping[str, float] = field(
        default_factory=lambda: {"carry": 1.0, "height": 0.5, "lateral": 0.5}
    )

    def __post_init__(self) -> None:
        require(self.max_iterations > 0, "max_iterations must be positive")
        require(self.timeout_s >= 0.0, "timeout_s must be non-negative")
        require(
            self.absolute_tolerance_m >= 0.0,
            "absolute_tolerance_m must be non-negative",
        )
        require(
            self.carry_tolerance_fraction >= 0.0,
            "carry_tolerance_fraction must be non-negative",
        )
        for key in ("carry", "height", "lateral"):
            require(key in self.weights, f"weights must include {key!r}")
            require(self.weights[key] >= 0.0, f"{key} weight must be non-negative")


@dataclass(frozen=True)
class SwingOptimizationDiagnostics:
    """Convergence and reachability diagnostics."""

    converged: bool
    unreachable: bool
    timed_out: bool
    message: str
    objective: float
    carry_error_m: float
    height_error_m: float | None
    lateral_error_m: float | None
    target_error_m: float
    evaluations: int
    iterations: int
    elapsed_s: float
    optimizer_status: int | None = None
    optimizer_message: str = ""


@dataclass(frozen=True)
class SwingOptimizationResult:
    """Best optimizer result and diagnostics."""

    swing_state: SwingState | None
    pipeline_result: PipelineResult | None
    diagnostics: SwingOptimizationDiagnostics


class _ForwardPipeline(Protocol):
    def run(self, swing: SwingState) -> PipelineResult:
        """Run a swing through the forward pipeline."""
        ...


class _OptimizationTimeout(RuntimeError):
    """Internal sentinel for bounded optimizer runtime."""


@dataclass
class _Evaluation:
    parameters: tuple[float, float, float, float]
    swing: SwingState
    result: PipelineResult
    objective: float
    errors: tuple[float, float | None, float | None]


class SwingOptimizer:
    """Solve swing parameters for a target ball flight."""

    def __init__(self, pipeline: _ForwardPipeline | None = None) -> None:
        self._pipeline = pipeline or SwingBallFlightPipeline()

    def solve(
        self,
        target: FlightTarget,
        club: ClubPreset,
        controls: OptimizationControls | None = None,
    ) -> SwingOptimizationResult:
        """Return the best swing state for ``target`` and ``club``.

        The solver is bounded by both ``max_iterations`` and ``timeout_s``.
        Non-achievable targets return the best bounded attempt with
        ``diagnostics.unreachable=True`` rather than an apparently successful
        garbage swing.
        """
        require(isinstance(target, FlightTarget), "target must be a FlightTarget")
        require(isinstance(club, ClubPreset), "club must be a ClubPreset")
        controls = controls or OptimizationControls()

        started = time.monotonic()
        context = _OptimizationContext(target, club, controls, started, self)
        initial = np.asarray(club.initial_guess, dtype=float)
        context.evaluate(initial, check_timeout=False)

        timed_out = False
        optimizer_status: int | None = None
        optimizer_message = ""
        try:
            response = minimize(
                context.objective,
                initial,
                method="SLSQP",
                bounds=club.bounds,
                options={
                    "maxiter": controls.max_iterations,
                    "ftol": 1e-9,
                    "disp": False,
                },
            )
            optimizer_status = int(response.status)
            optimizer_message = str(response.message)
            context.evaluate(response.x)
        except _OptimizationTimeout:
            timed_out = True
            optimizer_message = "optimizer timed out"

        return _build_optimization_result(
            context=context,
            timed_out=timed_out,
            optimizer_status=optimizer_status,
            optimizer_message=optimizer_message,
        )

    def build_swing_state(
        self,
        *,
        speed_mps: float,
        loft_deg: float,
        attack_angle_deg: float,
        face_to_path_deg: float,
        club: ClubPreset,
    ) -> SwingState:
        """Build a ``SwingState`` from optimizer-facing parameters."""
        require(speed_mps > 0.0, "speed_mps must be positive", speed_mps)
        require(math.isfinite(loft_deg), "loft_deg must be finite", loft_deg)
        require(
            math.isfinite(attack_angle_deg),
            "attack_angle_deg must be finite",
            attack_angle_deg,
        )
        require(
            math.isfinite(face_to_path_deg),
            "face_to_path_deg must be finite",
            face_to_path_deg,
        )

        attack_rad = math.radians(attack_angle_deg)
        face_rad = math.radians(face_to_path_deg)
        loft_rad = math.radians(loft_deg)
        velocity = speed_mps * np.array(
            [math.cos(attack_rad), 0.0, math.sin(attack_rad)],
            dtype=float,
        )
        orientation = np.array(
            [
                math.cos(loft_rad) * math.cos(face_rad),
                math.cos(loft_rad) * math.sin(face_rad),
                math.sin(loft_rad),
            ],
            dtype=float,
        )
        orientation = orientation / float(
            math.hypot(*orientation)
        )  # ⚡ Bolt: math.hypot is ~1.5x faster than np.linalg.norm
        launch_conditions = LaunchConditions(
            velocity=speed_mps,
            launch_angle=loft_rad,
            azimuth_angle=face_rad,
        )
        return SwingState(
            clubhead_velocity=velocity,
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=orientation,
            clubhead_mass=club.clubhead_mass,
            clubhead_loft_deg=loft_deg,
            clubhead_moi=club.clubhead_moi,
            engine_name="swing_optimizer",
            metadata={
                "optimizer_parameters": {
                    "speed_mps": float(speed_mps),
                    "loft_deg": float(loft_deg),
                    "attack_angle_deg": float(attack_angle_deg),
                    "face_to_path_deg": float(face_to_path_deg),
                    "club": club.name,
                },
                "launch_conditions": launch_conditions,
            },
        )


class _OptimizationContext:
    def __init__(
        self,
        target: FlightTarget,
        club: ClubPreset,
        controls: OptimizationControls,
        started: float,
        optimizer: SwingOptimizer,
    ) -> None:
        self.target = target
        self.club = club
        self.controls = controls
        self.started = started
        self.optimizer = optimizer
        self.cache: dict[tuple[float, float, float, float], _Evaluation] = {}
        self.best: _Evaluation | None = None
        self.evaluations = 0

    def objective(self, values: np.ndarray) -> float:
        self._raise_if_timed_out()
        return self.evaluate(values).objective

    def evaluate(
        self,
        values: np.ndarray,
        *,
        check_timeout: bool = True,
    ) -> _Evaluation:
        key = _cache_key(values)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if check_timeout:
            self._raise_if_timed_out()
        swing = self.optimizer.build_swing_state(
            speed_mps=key[0],
            loft_deg=key[1],
            attack_angle_deg=key[2],
            face_to_path_deg=key[3],
            club=self.club,
        )
        forward = self.optimizer._pipeline.run(swing)
        errors = _target_errors(self.target, forward)
        objective = _weighted_objective(errors, self.controls)
        evaluation = _Evaluation(
            parameters=key,
            swing=swing,
            result=forward,
            objective=objective,
            errors=errors,
        )
        self.cache[key] = evaluation
        self.evaluations += 1
        if self.best is None or evaluation.objective < self.best.objective:
            self.best = evaluation
        return evaluation

    def _raise_if_timed_out(self) -> None:
        if time.monotonic() - self.started >= self.controls.timeout_s:
            raise _OptimizationTimeout


def _require_ordered_bounds(bounds: tuple[float, float], name: str) -> None:
    lower, upper = bounds
    require(math.isfinite(lower), f"{name} lower bound must be finite", lower)
    require(math.isfinite(upper), f"{name} upper bound must be finite", upper)
    require(lower < upper, f"{name} lower bound must be < upper bound", bounds)


def _cache_key(values: np.ndarray) -> tuple[float, float, float, float]:
    rounded = tuple(round(float(value), 12) for value in values)
    if len(rounded) != 4:
        raise ValueError("optimizer values must contain four parameters")
    return rounded  # type: ignore[return-value]


def _target_errors(
    target: FlightTarget,
    result: PipelineResult,
) -> tuple[float, float | None, float | None]:
    carry_error = result.carry_m - target.carry_m
    height_error = (
        result.max_height_m - target.max_height_m
        if target.max_height_m is not None
        else None
    )
    lateral_error = (
        _terminal_lateral_m(result) - target.lateral_m
        if target.lateral_m is not None
        else None
    )
    return carry_error, height_error, lateral_error


def _terminal_lateral_m(result: PipelineResult) -> float:
    if "terminal_lateral_m" in result.metadata:
        return float(result.metadata["terminal_lateral_m"])
    if result.trajectory:
        return float(result.trajectory[-1].position[1])
    return 0.0


def _weighted_objective(
    errors: tuple[float, float | None, float | None],
    controls: OptimizationControls,
) -> float:
    carry_error, height_error, lateral_error = errors
    total = controls.weights["carry"] * carry_error**2
    if height_error is not None:
        total += controls.weights["height"] * height_error**2
    if lateral_error is not None:
        total += controls.weights["lateral"] * lateral_error**2
    return float(total)


def _target_error_m(errors: tuple[float, float | None, float | None]) -> float:
    components = [errors[0]]
    if errors[1] is not None:
        components.append(errors[1])
    if errors[2] is not None:
        components.append(errors[2])
    return float(math.sqrt(sum(error * error for error in components)))


def _target_tolerance_m(target: FlightTarget, controls: OptimizationControls) -> float:
    carry_tolerance = controls.carry_tolerance_fraction * max(target.carry_m, 1.0)
    return max(controls.absolute_tolerance_m, carry_tolerance)


def _build_optimization_result(
    *,
    context: _OptimizationContext,
    timed_out: bool,
    optimizer_status: int | None,
    optimizer_message: str,
) -> SwingOptimizationResult:
    best = context.best
    ensure(best is not None, "optimizer must evaluate at least one swing")
    if best is None:
        diagnostics = SwingOptimizationDiagnostics(
            converged=False,
            unreachable=True,
            timed_out=timed_out,
            message="No swing evaluations completed.",
            objective=math.inf,
            carry_error_m=math.inf,
            height_error_m=None,
            lateral_error_m=None,
            target_error_m=math.inf,
            evaluations=context.evaluations,
            iterations=0,
            elapsed_s=time.monotonic() - context.started,
            optimizer_status=optimizer_status,
            optimizer_message=optimizer_message,
        )
        return SwingOptimizationResult(None, None, diagnostics)

    target_error = _target_error_m(best.errors)
    tolerance = _target_tolerance_m(context.target, context.controls)
    converged = (not timed_out) and target_error <= tolerance
    unreachable = (not timed_out) and not converged
    if timed_out:
        message = (
            f"Swing optimization timed out after {context.controls.timeout_s:.3f}s; "
            f"best target error is {target_error:.3f} m."
        )
    elif unreachable:
        message = (
            "Target is unreachable within club bounds; "
            f"best target error is {target_error:.3f} m "
            f"(tolerance {tolerance:.3f} m)."
        )
    else:
        message = (
            f"Swing optimization converged with target error {target_error:.3f} m "
            f"(tolerance {tolerance:.3f} m)."
        )

    diagnostics = SwingOptimizationDiagnostics(
        converged=converged,
        unreachable=unreachable,
        timed_out=timed_out,
        message=message,
        objective=best.objective,
        carry_error_m=best.errors[0],
        height_error_m=best.errors[1],
        lateral_error_m=best.errors[2],
        target_error_m=target_error,
        evaluations=context.evaluations,
        iterations=min(context.controls.max_iterations, context.evaluations),
        elapsed_s=time.monotonic() - context.started,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
    )
    return SwingOptimizationResult(best.swing, best.result, diagnostics)

# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""CLI for headless batch optimization.

Usage:
    python3 -m movement_optimizer.cli --exercise squat --body-mass 80 --bar-mass 100
    python3 -m movement_optimizer.cli --exercise deadlift --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .exercises import make_clean_config, make_jerk_config, make_snatch_config
from .models import (
    BodyModel,
    make_bench_press_config,
    make_deadlift_config,
    make_full_squat_config,
    make_squat_config,
)
from .trajectory import OptimizationResult, TrajectoryOptimizer
from .validation import (
    BAR_MASS_RANGE,
    BODY_MASS_RANGE,
    DURATION_RANGE,
    HEIGHT_RANGE,
    SMOOTHNESS_RANGE,
    validate_all,
)

logger = logging.getLogger(__name__)

EXERCISE_FACTORIES = {
    "squat": make_squat_config,
    "full_squat": make_full_squat_config,
    "deadlift": make_deadlift_config,
    "bench_press": make_bench_press_config,
    "clean": make_clean_config,
    "jerk": make_jerk_config,
    "snatch": make_snatch_config,
}


def _add_body_args(parser: argparse.ArgumentParser) -> None:
    """Register anthropometric and load arguments on *parser*."""
    parser.add_argument(
        "--body-mass",
        type=float,
        default=75.0,
        help=(f"Body mass in kg (range {BODY_MASS_RANGE[0]}-{BODY_MASS_RANGE[1]}, default: 75.0)."),
    )
    parser.add_argument(
        "--height",
        type=float,
        default=1.75,
        help=(f"Height in metres (range {HEIGHT_RANGE[0]}-{HEIGHT_RANGE[1]}, default: 1.75)."),
    )
    parser.add_argument(
        "--bar-mass",
        type=float,
        default=60.0,
        help=(
            f"Barbell mass in kg (range {BAR_MASS_RANGE[0]}-{BAR_MASS_RANGE[1]}, default: 60.0)."
        ),
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Register optimisation and output arguments on *parser*."""
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help=(
            f"Movement duration in seconds "
            f"(range {DURATION_RANGE[0]}-{DURATION_RANGE[1]}, default: 2.0)."
        ),
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=1.0,
        help=(
            f"Smoothness weight (range {SMOOTHNESS_RANGE[0]}-{SMOOTHNESS_RANGE[1]}, default: 1.0)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON. If omitted, prints summary to stdout.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the fully configured CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="movement-optimizer-cli",
        description="Headless batch optimization for barbell exercises.",
    )
    parser.add_argument(
        "--exercise",
        choices=list(EXERCISE_FACTORIES.keys()),
        required=True,
        help="Exercise type to optimize.",
    )
    _add_body_args(parser)
    _add_run_args(parser)
    return parser


def _result_to_summary(result: OptimizationResult, exercise: str) -> dict:
    """Convert an OptimizationResult to a summary dict."""
    peak_torques = np.max(np.abs(result.torques), axis=0)
    return {
        "exercise": exercise,
        "success": result.success,
        "cost": float(result.cost),
        "n_evals": result.n_evals,
        "elapsed_s": float(result.elapsed_s),
        "com_horizontal_range_cm": float(result.com_horizontal_range_cm),
        "peak_torques_Nm": {
            "ankle": float(peak_torques[0]),
            "knee": float(peak_torques[1]),
            "hip": float(peak_torques[2]),
        },
    }


def _result_to_full_dict(result: OptimizationResult, exercise: str) -> dict:
    """Convert to full JSON-serializable dict including arrays."""
    summary = _result_to_summary(result, exercise)
    # Extract arrays first (LoD: avoid chaining .attribute.method())
    t_list = result.t.tolist()
    q_list = result.q.tolist()
    qd_list = result.qd.tolist()
    qdd_list = result.qdd.tolist()
    torques_list = result.torques.tolist()
    power_list = result.power.tolist()
    com_list = result.com.tolist()
    bar_list = result.bar.tolist()
    summary["arrays"] = {
        "t": t_list,
        "q": q_list,
        "qd": qd_list,
        "qdd": qdd_list,
        "torques": torques_list,
        "power": power_list,
        "com": com_list,
        "bar": bar_list,
    }
    return summary


def _emit_cli_summary(summary: dict) -> None:
    """Write the human-facing CLI summary to stdout.

    The CLI contract is a plain JSON payload on stdout, so this uses
    ``sys.stdout.write`` instead of logging to avoid log prefixes that would
    break downstream parsing.
    """
    sys.stdout.write(f"{json.dumps(summary, indent=2)}\n")


def _configure_logging(verbose: bool) -> None:
    """Configure root logging based on verbosity flag.

    Args:
        verbose: If True, sets level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_duration(exercise: str, requested: float) -> float:
    """Enforce minimum durations for multi-phase exercises.

    Args:
        exercise: Exercise key from EXERCISE_FACTORIES.
        requested: User-requested duration in seconds.

    Returns:
        Effective duration (>= minimum for the exercise type).
    """
    if exercise in ("full_squat", "snatch"):
        return max(requested, 3.0)
    if exercise in ("clean", "jerk"):
        return max(requested, 2.0)
    return requested


def _unpack_exercise_config(
    config: tuple,
) -> tuple[
    Any,
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[tuple[float, float], ...],
    NDArray[np.float64] | None,
]:
    """Unpack a 4- or 5-tuple factory config into ``(dyn, qs, qe, qb, q_via)``.

    Exercise factories return either a 4-tuple ``(dyn, qs, qe, qb)`` for
    single-phase lifts or a 5-tuple ``(dyn, qs, qe, qb, q_via)`` for
    multi-phase ones.  The ``q_via`` field is ``None`` when absent.

    Args:
        config: Raw tuple returned by an exercise factory.

    Returns:
        5-tuple ``(dyn, q_start, q_end, q_bounds, q_via)`` where ``q_via``
        may be ``None``.
    """
    if len(config) == 5:
        dyn, qs, qe, qb, q_via = config
    else:
        dyn, qs, qe, qb = config
        q_via = None
    return dyn, qs, qe, qb, q_via


def _build_optimizer(
    body: BodyModel,
    exercise: str,
    bar_mass: float,
    duration: float,
    smoothness: float,
) -> tuple[TrajectoryOptimizer, object]:
    """Construct the TrajectoryOptimizer for *exercise*.

    Returns a ``(optimizer, dynamics)`` pair ready for ``opt.optimize()``.
    """
    factory = EXERCISE_FACTORIES[exercise]
    config = factory(body, bar_mass)
    dyn, qs, qe, qb, q_via = _unpack_exercise_config(config)
    opt = TrajectoryOptimizer(
        body,
        dyn,  # type: ignore[arg-type]
        exercise,
        bar_mass,
        qs,  # type: ignore[arg-type]
        qe,  # type: ignore[arg-type]
        qb,  # type: ignore[arg-type]
        q_via=q_via,  # type: ignore[arg-type]
        duration=duration,
        n_waypoints=12,
        smoothness=smoothness,
    )
    return opt, dyn


def _save_or_emit(result: OptimizationResult, exercise: str, output: str | None) -> None:
    """Write result to file or emit summary to stdout.

    Args:
        result: Completed optimization result.
        exercise: Exercise label for the summary dict.
        output: File path to write JSON, or None to emit summary to stdout.
    """
    if output:
        full_data = _result_to_full_dict(result, exercise)
        Path(output).write_text(json.dumps(full_data, indent=2), encoding="utf-8")
        logger.info("Results saved to %s", output)
    else:
        _emit_cli_summary(_result_to_summary(result, exercise))


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid numeric CLI arguments via parser.error.

    Delegates to :func:`movement_optimizer.validation.validate_all` so the
    GUI and CLI share a single source of truth for valid ranges. Any
    :class:`ValueError` raised by the validator is converted into a
    user-friendly ``parser.error`` exit (which prints usage and exits 2).
    """
    try:
        validate_all(
            body_mass=args.body_mass,
            height=args.height,
            bar_mass=args.bar_mass,
            duration=args.duration,
            smoothness=args.smoothness,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))


def _log_optimization_start(
    exercise: str, body_mass: float, height: float, bar_mass: float, duration: float
) -> None:
    """Log a human-readable summary of the optimization parameters."""
    logger.info(
        "Optimizing %s: body=%.0fkg, height=%.2fm, bar=%.0fkg, dur=%.1fs",
        exercise,
        body_mass,
        height,
        bar_mass,
        duration,
    )


def _log_optimization_done(elapsed: float, cost: float, success: bool) -> None:
    """Log a summary of the completed optimization."""
    logger.info(
        "Optimization completed in %.1fs (cost=%.2f, success=%s)",
        elapsed,
        cost,
        success,
    )


def main(argv: list[str] | None = None) -> int:
    """Run CLI optimization.

    Args:
        argv: Argument list (uses sys.argv if None).

    Returns:
        0 on successful optimization, 1 on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    _configure_logging(args.verbose)
    body = BodyModel(body_mass=args.body_mass, height=args.height)
    duration = _resolve_duration(args.exercise, args.duration)
    _log_optimization_start(args.exercise, args.body_mass, args.height, args.bar_mass, duration)
    t_start = time.perf_counter()
    opt, _dyn = _build_optimizer(body, args.exercise, args.bar_mass, duration, args.smoothness)
    result = opt.optimize()
    _log_optimization_done(time.perf_counter() - t_start, result.cost, result.success)
    _save_or_emit(result, args.exercise, args.output)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

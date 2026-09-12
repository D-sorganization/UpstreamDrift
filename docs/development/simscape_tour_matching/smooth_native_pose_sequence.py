"""Smooth static native poses and reproject pose/rate weld closure at samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from src.engines.physics_engines.pinocchio.python.native_model import NativePinocchioModel


def _mapping(names: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("Expected finite native state values")
    return dict(zip(names, values.tolist(), strict=True))


def _project(
    initial: np.ndarray,
    residual: Callable[[np.ndarray], np.ndarray],
    *,
    tolerance: float,
) -> np.ndarray:
    result = minimize(
        lambda value: float(np.sum((value - initial) ** 2)),
        initial,
        method="SLSQP",
        constraints={"type": "eq", "fun": residual},
        options={"maxiter": 100, "ftol": 1e-14},
    )
    value = np.asarray(result.x, dtype=float)
    error = np.asarray(residual(value), dtype=float)
    if not np.isfinite(value).all() or not np.isfinite(error).all() or np.max(abs(error)) > tolerance:
        raise ValueError("Closure projection failed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--closure-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.closure_tolerance) or args.closure_tolerance <= 0:
        raise ValueError("Output must be new and closure tolerance positive")
    raw = args.model.read_bytes()
    specification = json.loads(raw)
    names = tuple(specification["coordinate_order"])
    source = json.loads(args.source.read_text())
    records = source["records"]
    times = np.asarray([record["time_s"] for record in records], dtype=float)
    positions = np.asarray([record["coordinates"] for record in records], dtype=float)
    if times.ndim != 1 or len(times) < 4 or not np.all(np.diff(times) > 0) or positions.shape != (len(times), len(names)):
        raise ValueError("Expected four or more strictly timed native pose samples")
    spline = CubicSpline(times, positions, axis=0)
    model = NativePinocchioModel(specification)
    corrected_positions = []
    corrected_rates = []
    for time in times:
        position = _project(
            np.asarray(spline(time), dtype=float),
            lambda value: model.closure_residuals(_mapping(names, value))[0],
            tolerance=args.closure_tolerance,
        )

        def rate_closure(
            value: np.ndarray, fixed_position: np.ndarray = position
        ) -> np.ndarray:
            return model.closure_residuals(
                _mapping(names, fixed_position), _mapping(names, value)
            )[1]

        rate = _project(
            np.asarray(spline(time, 1), dtype=float),
            rate_closure,
            tolerance=args.closure_tolerance,
        )
        corrected_positions.append(position)
        corrected_rates.append(rate)
    q = np.asarray(corrected_positions)
    qd = np.asarray(corrected_rates)
    qdd = np.gradient(qd, times, axis=0, edge_order=2)
    rate_residual = np.asarray(
        [model.closure_residuals(_mapping(names, q_i), _mapping(names, qd_i))[1] for q_i, qd_i in zip(q, qd, strict=True)]
    )
    rate_change = np.gradient(rate_residual, times, axis=0, edge_order=2)
    payload = {
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "times_s": times.tolist(),
        "coordinates": q.tolist(),
        "rates": qd.tolist(),
        "finite_difference_accelerations": qdd.tolist(),
        "pose_closure_max_abs": float(
            np.max(
                [np.max(abs(model.closure_residuals(_mapping(names, q_i))[0])) for q_i in q]
            )
        ),
        "rate_closure_max_abs": float(np.max(abs(rate_residual))),
        "rate_residual_change_max_abs": float(np.max(abs(rate_change))),
        "scope": "Static cubic smoothing with sampled pose/rate closure projections; finite-difference acceleration is not a constrained-dynamics acceleration qualification.",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

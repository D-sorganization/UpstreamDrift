"""Reproduce time-only run45 inputs and compare bounded integration tolerances.

Diagnostic only: retains saved feedback as a time-dependent input source, never
resets the integrated state, and does not certify a polynomial match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.shared.python.motion_matching.acceleration_effort import (
    allocate_acceleration_effort,
)
from src.shared.python.motion_matching.continuous_forward import integrate_forward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "path", "feedback", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=1.1)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    path = json.loads(args.path.read_bytes())
    if path["model_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Reference and model identities disagree")
    saved = np.load(args.feedback, allow_pickle=False)
    names = spec["coordinate_order"]
    n = len(names)
    times, states = saved["time"], saved["state"]
    if not np.isfinite(args.horizon) or not 0 < args.horizon <= times[-1]:
        raise ValueError("Horizon must lie within the saved capture")
    initial = states[0].copy()
    reference = CubicSpline(
        [r["time_s"] for r in path["records"]],
        [r["coordinates"] for r in path["records"]],
        axis=0,
        bc_type=((1, initial[n:]), (2, np.zeros(n))),
    )
    reference_state = CubicHermiteSpline(
        times, states, saved["state_derivatives"], axis=0
    )
    model = NativePinocchioModel(spec)
    zero = dict.fromkeys(names, 0.0)

    def mapping(values: np.ndarray) -> dict[str, float]:
        return dict(zip(names, map(float, values), strict=True))

    def effort(t: float) -> np.ndarray:
        x = reference_state(t)
        q, v = mapping(x[:n]), mapping(x[n:])
        free_map = model.accelerations(q, v, zero)
        response = model.acceleration_derivatives(q, v, zero).deffort
        return allocate_acceleration_effort(
            response,
            np.array([free_map[name] for name in names]),
            reference(t, 2)
            + 40 * (reference(t, 1) - x[n:])
            + 400 * (reference(t) - x[:n]),
            acceleration_scales=np.r_[np.full(3, 0.1), np.ones(n - 3)],
            effort_scales=np.r_[np.full(3, 100.0), np.full(n - 3, 20.0)],
        ).effort

    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        u = effort(t)
        values = model.accelerations(mapping(x[:n]), mapping(x[n:]), mapping(u))
        return np.r_[x[n:], [values[name] for name in names]]

    args.output.mkdir()
    clock = np.unique(np.r_[times[times <= args.horizon], args.horizon])
    report = {
        "scope": "Fixed run45 time-only input convergence; not match acceptance",
        "inputs_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.model, args.path, args.feedback)
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": [],
    }
    previous = None
    for label, rtol, atol, max_step in (
        ("baseline", 1e-10, 1e-12, 0.00025),
        ("tight", 1e-12, 1e-14, 0.000125),
    ):
        result = integrate_forward(
            initial, clock, derivative, rtol=rtol, atol=atol, max_step=max_step
        )
        reference_values = reference_state(clock)
        delta = result.state - reference_values
        rows = []
        for t in (0.6, 0.7, 0.8, 0.9, 1.0, args.horizon):
            if t > args.horizon:
                continue
            i = int(np.argmin(abs(clock - t)))
            rows.append(
                {
                    "time_s": float(clock[i]),
                    "q_max_abs": float(np.max(abs(delta[i, :n]))),
                    "v_max_abs": float(np.max(abs(delta[i, n:]))),
                    "largest_coordinate": names[int(np.argmax(abs(delta[i, :n])))],
                }
            )
        item = {
            "label": label,
            "rtol": rtol,
            "atol": atol,
            "max_step": max_step,
            "evaluations": result.evaluations,
            "elapsed_s": result.elapsed_s,
            "errors": rows,
        }
        if previous is not None:
            item["terminal_state_difference_from_baseline"] = float(
                np.max(abs(result.state[-1] - previous[-1]))
            )
        report["runs"].append(item)
        np.savez_compressed(
            args.output / (label + ".npz"), time=clock, state=result.state
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        previous = result.state


if __name__ == "__main__":
    main()

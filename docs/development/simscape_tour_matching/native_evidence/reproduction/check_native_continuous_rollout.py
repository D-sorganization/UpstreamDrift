"""Replay saved native polynomial inputs with no intermediate state injection."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("module", "integrator", "spec", "fixture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--max-step", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text())
    fixture = json.loads(args.fixture.read_text())
    model = runpy.run_path(str(args.module))["NativePinocchioModel"](spec)
    integrate = runpy.run_path(str(args.integrator))["integrate_forward"]
    names = spec["coordinate_order"]
    if fixture["coordinate_names"] != names or fixture["release"] != "2025b":
        raise ValueError("Native coordinate or release mismatch")
    trajectory = fixture["trajectory"]
    clock = np.asarray(trajectory["time_s"])
    if not np.isfinite(args.duration) or not 0 < args.duration <= clock[-1]:
        raise ValueError("Duration must lie within native reference coverage")
    mask = clock <= args.duration
    clock = clock[mask]
    reference_q = np.asarray(trajectory["q"])[mask]
    reference_v = np.asarray(trajectory["qd"])[mask]
    coefficients = np.asarray(fixture["native_coefficients"])
    rotation = np.asarray(spec["joints"][0]["parent_to_base"])[:3, :3].T
    n = len(names)

    def efforts(t: float) -> dict[str, float]:
        values = np.polyval(coefficients.T, t)
        values[:3] = rotation @ values[:3]
        return dict(zip(names, values, strict=True))

    def derivative(t: float, state: np.ndarray) -> np.ndarray:
        acceleration = model.accelerations(
            dict(zip(names, state[:n], strict=True)),
            dict(zip(names, state[n:], strict=True)),
            efforts(t),
        )
        return np.concatenate((state[n:], [acceleration[name] for name in names]))

    result = integrate(
        np.concatenate((reference_q[0], reference_v[0])),
        clock,
        derivative,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.max_step,
    )
    q = result.state[:, :n]
    v = result.state[:, n:]
    pose_error, velocity_error = [], []
    for t, position, velocity in zip(clock, q, v, strict=True):
        model.accelerations(
            dict(zip(names, position, strict=True)),
            dict(zip(names, velocity, strict=True)),
            efforts(t),
        )
        contact = model.constraint_data[0]
        pose_error.append(float(np.max(np.abs(contact.contact_placement_error.vector))))
        velocity_error.append(
            float(np.max(np.abs(contact.contact_velocity_error.vector)))
        )
    receipt = {
        "qualification": "continuous same-input diagnostic; full swing unqualified",
        "duration_s": float(clock[-1]),
        "evaluations": result.evaluations,
        "elapsed_s": result.elapsed_s,
        "max_step": args.max_step,
        "rtol": args.rtol,
        "atol": args.atol,
        "q_max_abs_error_by_coordinate": np.max(
            np.abs(q - reference_q), axis=0
        ).tolist(),
        "qd_max_abs_error_by_coordinate": np.max(
            np.abs(v - reference_v), axis=0
        ).tolist(),
        "closure_pose_max_abs": max(pose_error),
        "closure_velocity_max_abs": max(velocity_error),
        "coordinate_names": names,
        "sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("module", "integrator", "spec", "fixture")
        },
        "time_s": clock.tolist(),
        "q": q.tolist(),
        "qd": v.tolist(),
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

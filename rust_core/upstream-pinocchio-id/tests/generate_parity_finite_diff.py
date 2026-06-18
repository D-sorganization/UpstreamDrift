#!/usr/bin/env python3
"""Generate golden finite-difference vectors for `tests/parity_finite_diff.rs`.

Reproduces the non-uniform-dt qdot/qddot scheme of
`src/shared/python/motion_pipeline/matching/inverse_dyn_pinocchio.py`
(`_finite_difference`, the documented source-of-truth for the Rust port in
`upstream-pinocchio-id::finite_diff`). NumPy is the only dependency.

Run from this directory to (re)generate the CSV goldens:

    python3 generate_parity_finite_diff.py

Each CSV row is one case:
    case,n_frames,n_dof,times...,q(row-major)...,qdot(row-major)...,qddot(row-major)...
with the variable-length sections delimited by the leading n_frames/n_dof.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


def finite_diff(times: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror inverse_dyn_pinocchio._finite_difference (non-uniform dt)."""
    qdot = np.zeros_like(q)
    for i in range(1, len(times) - 1):
        dt = times[i + 1] - times[i - 1]
        if dt > 0:
            qdot[i] = (q[i + 1] - q[i - 1]) / dt
    if len(times) >= 2:
        qdot[0] = (q[1] - q[0]) / max(times[1] - times[0], 1e-9)
        qdot[-1] = (q[-1] - q[-2]) / max(times[-1] - times[-2], 1e-9)

    qddot = np.zeros_like(q)
    for i in range(1, len(times) - 1):
        dt_b = times[i] - times[i - 1]
        dt_f = times[i + 1] - times[i]
        if dt_b > 0 and dt_f > 0:
            qddot[i] = (
                2.0
                * (q[i + 1] * dt_b - q[i] * (dt_b + dt_f) + q[i - 1] * dt_f)
                / (dt_b * dt_f * (dt_b + dt_f))
            )
    if len(times) >= 3:
        qddot[0] = qddot[1]
        qddot[-1] = qddot[-2]
    return qdot, qddot


def cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(7660)
    out: list[tuple[str, np.ndarray, np.ndarray]] = []

    # Uniform-dt quadratic, single DOF.
    t = np.linspace(0.0, 0.4, 5)
    q = (0.5 * 2.0 * t**2).reshape(-1, 1)
    out.append(("uniform_quadratic_1dof", t, q))

    # Non-uniform timestamps, two DOF, smooth trig.
    t = np.array([0.0, 0.05, 0.13, 0.22, 0.40, 0.55])
    q = np.column_stack([np.sin(2.0 * t), np.cos(1.3 * t)])
    out.append(("nonuniform_trig_2dof", t, q))

    # Larger random trajectory, three DOF, uniform dt.
    t = np.linspace(0.0, 1.0, 20)
    q = np.cumsum(rng.standard_normal((20, 3)) * 0.01, axis=0)
    out.append(("random_walk_3dof", t, q))

    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    out_path = here / "parity_finite_diff.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "n_frames", "n_dof", "data"])
        for name, t, q in cases():
            qdot, qddot = finite_diff(t, q)
            n, d = q.shape
            flat = (
                list(t)
                + list(q.flatten())
                + list(qdot.flatten())
                + list(qddot.flatten())
            )
            w.writerow([name, n, d] + [repr(float(x)) for x in flat])
    sys.stderr.write(f"wrote {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

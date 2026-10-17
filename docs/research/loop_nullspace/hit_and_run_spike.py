"""Reproduce the NSX-70 hit-and-run spike (research, exploratory).

Run from the repository root::

    PYTHONPATH=. python3 docs/research/loop_nullspace/hit_and_run_spike.py [--out results.json]

Scenario: the planar two-arm golfer fixture (NSX-00 generic configuration,
full actuation) decomposed once; the same ``x = x_p + N z`` is reused at every
frame of a synthetic 21-frame window with a *declared* time-varying symmetric
torque capacity.  The capacity profile is invented for the spike; no golfer is
measured.  Output is a JSON record of every diagnostic quoted in the report.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import math
from pathlib import Path

import numpy as np

from src.shared.python.loop_nullspace.decomposition import decompose
from src.shared.python.loop_nullspace.envelopes import (
    joint_torque_quantities,
    static_envelopes,
    x_functional_quantities,
)
from src.shared.python.loop_nullspace.feasible_rows import FeasibleRow
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.hit_and_run import (
    TrajectoryDomain,
    bspline_temporal_basis,
    run_chains,
)
from src.shared.python.loop_nullspace.mixing_diagnostics import (
    autocorrelation,
    effective_sample_size,
    split_rhat,
)
from src.shared.python.loop_nullspace.sampled_marginals import marginal_reach

N_FRAMES = 21
N_CHAINS = 8
N_DRAWS = 4000
PILOT_DRAWS = 3000
BURN_IN = 1000
SEED = 12588
CAP_BASE = 25.0  # N m, declared symmetric capacity per actuated coordinate
CAP_SWING = 0.4  # fractional modulation over the window


def build_frames() -> tuple[list[FeasibleSet], list[list], np.ndarray]:
    """Per-frame feasible sets, their quantities and the sample times."""
    snap = planar_golfer_snapshot(generic_planar_config(), actuation="full").snapshot
    dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
    times = np.linspace(0.0, 1.0, N_FRAMES)
    frames = []
    for t in times:
        cap = CAP_BASE * (1.0 + CAP_SWING * math.sin(2.0 * math.pi * t))
        rows = []
        for i in range(dec.n_actuated):
            rows.append(
                FeasibleRow(
                    dec.N[i],
                    cap - dec.x_p[i],
                    f"T{i}:up",
                    "H-TORQUE-CAPACITY",
                    "torque",
                )
            )
            rows.append(
                FeasibleRow(
                    -dec.N[i],
                    cap + dec.x_p[i],
                    f"T{i}:lo",
                    "H-TORQUE-CAPACITY",
                    "torque",
                )
            )
        fs = FeasibleSet.from_rows(dec.nullity, rows)
        frames.append(dataclasses.replace(fs, decomposition=dec))
    lam = np.zeros((2, dec.dim_x))
    lam[0, dec.n_actuated] = 1.0
    lam[1, dec.n_actuated + 1] = 1.0
    factories = [
        joint_torque_quantities(),
        x_functional_quantities(
            ("lam_1", "lam_2"), lam, units="N", scale=dec.scales.force
        ),
    ]
    quantities = []
    for fs in frames:
        quantities.append([q for f in factories for q in f(fs)])
    return frames, quantities, times


def pilot_scale(domain: TrajectoryDomain) -> np.ndarray:
    """Cholesky factor of a pilot-chain covariance (fixed before the real run)."""
    pilot = run_chains(
        domain, n_chains=1, n_draws=PILOT_DRAWS, burn_in=BURN_IN, seed=SEED + 1
    )[0]
    cov = np.cov(pilot.coefficients, rowvar=False)
    cov += 1e-9 * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return np.linalg.cholesky(cov)


def run_case(
    frames, quantities, times, n_coeff: int | None, lower, upper, *, precondition: bool
) -> dict:
    basis = (
        np.eye(N_FRAMES)
        if n_coeff is None
        else bspline_temporal_basis(times, n_coeff, degree=3)
    )
    domain = TrajectoryDomain(frames, basis)
    scale = pilot_scale(domain) if precondition else None
    chains = run_chains(
        domain,
        n_chains=N_CHAINS,
        n_draws=N_DRAWS,
        burn_in=BURN_IN,
        seed=SEED,
        direction_scale=scale,
    )
    arr = np.stack([c.coefficients for c in chains])
    z = domain.trajectories(arr.reshape(-1, domain.dimension))
    reach = marginal_reach(z, quantities, lower, upper)
    ess = [effective_sample_size(arr[:, :, d]) for d in range(domain.dimension)]
    rhat = [split_rhat(arr[:, :, d]) for d in range(domain.dimension)]
    # ESS/R-hat of the physical functionals at the mid frame (all quantities)
    mid = N_FRAMES // 2
    zz = domain.trajectories(arr.reshape(-1, domain.dimension))[:, mid, :]
    coeff = np.array([q.coeff for q in quantities[mid]])
    const = np.array([q.const for q in quantities[mid]])
    func = (zz @ coeff.T + const).reshape(N_CHAINS, N_DRAWS, -1)
    f_ess = [effective_sample_size(func[:, :, k]) for k in range(func.shape[2])]
    f_rhat = [split_rhat(func[:, :, k]) for k in range(func.shape[2])]
    acf = autocorrelation(arr[0, :, 0], max_lag=50)
    return {
        "preconditioned": precondition,
        "n_basis": n_coeff if n_coeff is not None else N_FRAMES,
        "dimension": domain.dimension,
        "min_ess": float(np.nanmin(ess)),
        "median_ess": float(np.nanmedian(ess)),
        "max_rhat": float(np.nanmax(rhat)),
        "functional_min_ess_mid_frame": float(np.nanmin(f_ess)),
        "functional_max_rhat_mid_frame": float(np.nanmax(f_rhat)),
        "acf_lag1": float(acf[1]),
        "acf_lag10": float(acf[10]),
        "acf_lag50": float(acf[50]),
        "containment_violation": reach.containment_violation,
        "names": list(reach.names),
        "mean_reach": [float(v) for v in reach.mean_reach()],
        "mean_central_reach": [
            float(v) for v in np.nanmean(reach.central_reach, axis=0)
        ],
        "min_reach": [float(v) for v in np.nanmin(reach.reach, axis=0)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    frames, quantities, times = build_frames()
    by_frame = {id(fs): qs for fs, qs in zip(frames, quantities, strict=True)}
    series = static_envelopes(frames, lambda fs: by_frame[id(fs)])
    out = {
        "scenario": {
            "n_frames": N_FRAMES,
            "n_chains": N_CHAINS,
            "n_draws": N_DRAWS,
            "burn_in": BURN_IN,
            "seed": SEED,
            "cap_base_nm": CAP_BASE,
            "cap_swing": CAP_SWING,
            "n_z": frames[0].n_z,
        },
        "envelope_width_mean": [float(v) for v in np.nanmean(series.width(), axis=0)],
        "cases": [
            run_case(
                frames,
                quantities,
                times,
                k,
                series.lower,
                series.upper,
                precondition=pc,
            )
            for k in (4, 6, 8, None)
            for pc in (False, True)
        ],
    }
    text = json.dumps(out, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()

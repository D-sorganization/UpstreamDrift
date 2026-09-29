"""Differentiable trajectory optimisation of the tracked reference with MuJoCo
MJX (FB-5 / MM-7b, #10109).

Runs in the MJX environment (JAX plus MuJoCo >= 3.13; see HANDOFF) on the
package written by ``export_mjx_package.py``. The plant is a JAX port of the
shared replay: the run's MJCF, the Hunt-Crossley plus regularised-Coulomb
sphere contact law (same parameters, applied as world wrenches at the
calcanei), the dual-grip weld (MJX soft constraint) and the computed-torque
tracking controller with the root free. The rollout is a ``lax.scan`` over
capture frames (several integrator steps each) that emits the marker
positions, so the cost is the marker error of the *replayed* motion against
the capture, plus a small regulariser on the reference change. The decision
variables are knot values of a correction added to the actuated coordinates
of the tracked reference (linear interpolation between knots); Adam on the
gradient through the whole rollout.

    python mjx_trajectory_optimisation.py --run anthro_driver --iterations 40
    -> <run>/mjx_optimised_reference.npz (q, frames x coordinates, spec order)
       <run>/mjx_optimisation_receipt.json

``--iterations 0`` only replays the unmodified reference in MJX, which is the
port check against the shared simulator's receipt.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import jax
import numpy as np

# Ensure repo root is on sys.path
_here = Path(__file__).resolve()
REPO_ROOT = next(
    (p for p in _here.parents if (p / "pyproject.toml").is_file()), _here.parents[5]
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The MJX environment has no PyYAML; the src import chain only needs the name
# to exist, and any real use of the stub fails loudly.
try:
    import yaml  # noqa: F401
except ImportError:
    import types

    sys.modules.setdefault("yaml", types.ModuleType("yaml"))

from src.engines.physics_engines.mujoco.python.motion_matching.mjx_knot_optimiser import (
    ARMATURE_KG_M2,
    ROOT_VERTICAL_COORDINATE,
    WELD_DAMPING_N_S_M,
    WELD_ROT_DAMPING_N_M_S,
    WELD_ROT_STIFFNESS_N_M_RAD,
    WELD_STIFFNESS_N_M,
    KnotOptimisationSettings,
    build_reference,
    diagnose_reference,
    load_mjx_package,
    optimise_reference,
    optimisation_receipt,
)

LOG = logging.getLogger("mjx_opt")
load_package = load_mjx_package

__all__ = [
    "ARMATURE_KG_M2",
    "ROOT_VERTICAL_COORDINATE",
    "WELD_DAMPING_N_S_M",
    "WELD_ROT_DAMPING_N_M_S",
    "WELD_ROT_STIFFNESS_N_M_RAD",
    "WELD_STIFFNESS_N_M",
    "build_reference",
    "load_package",
    "main",
]


def main(argv: Sequence[str] | None = None) -> None:
    jax.config.update("jax_enable_x64", False)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--substeps", type=int, default=6)
    parser.add_argument("--knot-spacing", type=float, default=0.04)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--regularisation", type=float, default=1e-3)
    parser.add_argument("--name", default="mjx_optimised_reference")
    parser.add_argument("--weld-stiffness", type=float, default=WELD_STIFFNESS_N_M)
    parser.add_argument("--weld-damping", type=float, default=WELD_DAMPING_N_S_M)
    parser.add_argument(
        "--horizon",
        type=float,
        default=1.65,
        help="frames beyond this time (s) are left out of the cost: the soft-weld "
        "plant departs from the rigid-weld one in the last follow-through",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="also save the reference every N iterations as <name>_iterN.npz so "
        "the shared-law plant can select the iteration that transfers best",
    )
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="warm-start the knot correction from a previous optimised reference npz",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="forward rollout only: first non-finite frame, peak joint speed, replay error",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    here = Path(__file__).resolve().parent
    run = args.run if args.run.is_absolute() else here / args.run
    package = load_mjx_package(run)

    settings = KnotOptimisationSettings(
        iterations=args.iterations,
        substeps=args.substeps,
        knot_spacing_s=args.knot_spacing,
        learning_rate=args.learning_rate,
        regularisation=args.regularisation,
        horizon_s=args.horizon,
        weld_stiffness=args.weld_stiffness,
        weld_damping=args.weld_damping,
    )

    times = package.arrays["time_s"]
    q_track = package.arrays["q_track"]
    targets = package.arrays["targets_m"]

    if args.diagnose:
        t0 = time.perf_counter()
        res = diagnose_reference(package, settings)
        np.savez(
            run / "mjx_diagnose.npz",
            markers_m=res.markers,
            q_sim=res.q_sim,
            peak_qvel=res.peak_qvel,
        )
        valid = res.valid
        finite = np.isfinite(res.markers).all(axis=(1, 2))
        err = np.sqrt(np.sum((res.markers - np.asarray(targets)) ** 2, axis=2))
        root_err = np.linalg.norm(res.q_sim[:, :3] - q_track[:, :3], axis=1)
        LOG.info(
            "diagnose: rollout %.0f s; first non-finite frame %d (t=%.3f s); replay markers over finite frames %.1f mm",
            time.perf_counter() - t0,
            res.first_bad_frame,
            times[res.first_bad_frame] if res.first_bad_frame >= 0 else float("nan"),
            res.replay_marker_rms_m * 1e3,
        )
        for k in range(0, len(times), 36):
            LOG.info(
                "  t=%.2f: peak |qvel| %.1f rad/s, root err %.0f mm, marker rms %.0f mm",
                times[k],
                res.peak_qvel[k],
                root_err[k] * 1e3,
                float(np.sqrt(np.mean(err[k][valid[k]] ** 2))) * 1e3
                if finite[k]
                else float("nan"),
            )
        return

    init_delta: np.ndarray | None = None
    if args.init is not None:
        previous = np.load(args.init)
        if float(previous["knot_spacing_s"]) != args.knot_spacing:
            raise ValueError("--init was optimised with a different knot spacing")
        init_delta = np.asarray(previous["delta_knots"])

    t0 = time.perf_counter()
    best_cost = float("inf")

    def on_iteration(
        k: int, delta: np.ndarray, total: float, objective: float, q: np.ndarray
    ) -> None:
        nonlocal best_cost
        rms = float(np.sqrt(max(0.0, objective)))
        finite = bool(np.isfinite(total))
        delta_max = float(np.max(np.abs(delta)))
        if k == 0:
            LOG.info(
                "port check: MJX replay of the unmodified reference %.1f mm "
                "(shared simulator %.1f mm); first gradient in %.0f s",
                rms * 1e3,
                package.meta["baseline"]["replay_marker_rms_m"] * 1e3,
                time.perf_counter() - t0,
            )
        else:
            LOG.info(
                "iteration %d: replay markers %.1f mm, delta max %.2f deg, %s",
                k,
                rms * 1e3,
                float(np.degrees(delta_max)),
                "ok" if finite else "NON-FINITE",
            )
            if args.snapshot_every and k % args.snapshot_every == 0:
                np.savez(
                    run / f"{args.name}_iter{k}.npz",
                    q=q,
                    delta_knots=delta,
                    knot_spacing_s=args.knot_spacing,
                )
        if finite and objective < best_cost:
            best_cost = objective
            np.savez(
                run / f"{args.name}.npz",
                q=q,
                delta_knots=delta,
                knot_spacing_s=args.knot_spacing,
            )

    result = optimise_reference(
        package,
        settings,
        init_delta=init_delta,
        on_iteration=on_iteration,
    )
    if result.stop_reason == "non_finite_gradient":
        LOG.info("gradient not finite; stopping")

    np.savez(
        run / f"{args.name}.npz",
        q=result.q_best,
        delta_knots=result.delta_best,
        knot_spacing_s=args.knot_spacing,
    )

    receipt = optimisation_receipt(
        result,
        vars(args),
        package=package,
        run=run,
        elapsed_s=round(time.perf_counter() - t0, 1),
    )
    (run / "mjx_optimisation_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    LOG.info(
        "best MJX replay %.1f mm after %d iterations; wrote %s",
        result.best_replay_marker_rms_m * 1e3,
        args.iterations,
        run / f"{args.name}.npz",
    )


if __name__ == "__main__":
    main()

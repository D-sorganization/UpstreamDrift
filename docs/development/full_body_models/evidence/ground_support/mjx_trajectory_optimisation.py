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
import xml.etree.ElementTree as ET  # serialisation only; parsing is defused

import defusedxml.ElementTree as DET
import jax
import jax.numpy as jnp
import mujoco
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

from src.engines.physics_engines.mujoco.python.motion_matching.mjx_tracking_plant import (
    TrackingPlantSpec,
    build_tracking_plant,
    reference_derivatives,
    substep_tables,
)
from src.shared.python.motion_matching.jax_contact import WeldGains
from src.shared.python.motion_matching.knot_gradient_optimiser import (
    AdamSettings,
    adam_minimise,
    horizon_knot_mask,
    knot_basis,
    knot_grid,
)

LOG = logging.getLogger("mjx_opt")

ARMATURE_KG_M2 = 5e-3
WELD_STIFFNESS_N_M = 2.0e5
WELD_DAMPING_N_S_M = 400.0
WELD_ROT_STIFFNESS_N_M_RAD = 2.0e3
WELD_ROT_DAMPING_N_M_S = 4.0

# Root vertical slide of these documents (world z); read by name, never guessed.
ROOT_VERTICAL_COORDINATE = "TranslationInputZ"


def load_package(run: Path) -> tuple[dict, dict, mujoco.MjModel]:
    """Package plus the model with every equality removed: MJX's constraint
    solver is an iterative loop JAX cannot reverse-differentiate, so the grip
    weld is applied here as a stiff spring-damper wrench instead."""
    meta = json.loads((run / "mjx_package.json").read_text(encoding="utf-8"))
    pkg = dict(np.load(run / "mjx_package.npz"))
    root = DET.fromstring((run / "mjx_package.xml").read_text(encoding="utf-8"))
    for equality in root.findall("equality"):
        root.remove(equality)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # The rigid weld of the shared simulator couples the near-massless hand
    # standoff to the club; with a spring weld those dofs need an armature
    # (rotor inertia) floor or they explode. Applied to every dof.
    model.dof_armature[:] = np.maximum(model.dof_armature, ARMATURE_KG_M2)
    return meta, pkg, model


def build_reference(
    q_track: jnp.ndarray | np.ndarray,
    act_indices: jnp.ndarray | np.ndarray,
    basis: jnp.ndarray | np.ndarray,
    delta: jnp.ndarray,
    knot_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Tracked reference plus the knot correction; masked knots are frozen."""
    q = jnp.asarray(q_track)
    act = jnp.asarray(act_indices)
    b = jnp.asarray(basis)
    d = delta if knot_mask is None else delta * knot_mask[:, None]
    return q.at[:, act].add(b @ d)


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
    meta, pkg, model = load_package(run)

    weld_gains = WeldGains(
        k=args.weld_stiffness,
        c=args.weld_damping,
        rot_k=WELD_ROT_STIFFNESS_N_M_RAD,
        rot_c=WELD_ROT_DAMPING_N_M_S,
    )
    root_v_idx = meta["coordinate_order"].index(ROOT_VERTICAL_COORDINATE)
    spec = TrackingPlantSpec.from_package(
        meta,
        pkg,
        substeps=args.substeps,
        root_vertical_index=root_v_idx,
        weld_gains=weld_gains,
    )
    plant = build_tracking_plant(model, spec)

    times = pkg["time_s"]
    q_track = pkg["q_track"]
    targets = jnp.asarray(np.nan_to_num(pkg["targets_m"]))
    in_horizon = (times <= args.horizon)[:, None]
    valid = jnp.asarray(
        pkg["valid"] & np.isfinite(pkg["targets_m"]).all(axis=2) & in_horizon
    )
    count = float(valid.sum())

    knots = knot_grid(times, args.knot_spacing)
    basis = jnp.asarray(knot_basis(times, knots))
    knot_mask = jnp.asarray(horizon_knot_mask(knots, args.horizon), dtype=jnp.float32)
    n_knots = knots.size
    act = np.asarray(spec.act_indices)
    mass = float(meta["mass_kg"])

    def reference(delta: jnp.ndarray) -> jnp.ndarray:
        """Tracked reference plus the knot correction; knots beyond the cost
        horizon are frozen so the uncosted tail keeps the original reference."""
        return build_reference(q_track, act, basis, delta, knot_mask)

    def replay(delta: jnp.ndarray) -> jnp.ndarray:
        q_np = reference(delta)
        v_np, a_np = reference_derivatives(q_np, times)
        d0 = plant.initial_state(q_np[0], v_np[0], mass)
        return plant.rollout(
            d0,
            substep_tables(q_np, args.substeps),
            substep_tables(v_np, args.substeps),
            substep_tables(a_np, args.substeps),
        )

    def cost(delta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        m = replay(delta)
        err2 = jnp.sum((m - targets) ** 2, axis=2)
        marker_cost = jnp.sum(jnp.where(valid, err2, 0.0)) / count
        reg = args.regularisation * jnp.mean(delta**2)
        return marker_cost + reg, marker_cost

    if args.diagnose:
        q_np = reference(jnp.zeros((n_knots, act.size)))
        v_np, a_np = reference_derivatives(q_np, times)
        d0 = plant.initial_state(q_np[0], v_np[0], mass)
        t0 = time.perf_counter()
        m, speed, q_sim = plant.rollout_diagnostic(
            d0,
            substep_tables(q_np, args.substeps),
            substep_tables(v_np, args.substeps),
            substep_tables(a_np, args.substeps),
        )
        m, speed, q_sim = np.asarray(m), np.asarray(speed), np.asarray(q_sim)
        finite = np.isfinite(m).all(axis=(1, 2))
        first_bad = int(np.argmin(finite)) if not finite.all() else -1
        err = np.sqrt(np.sum((m - np.asarray(targets)) ** 2, axis=2))
        ok = np.asarray(valid) & finite[:, None]
        rms = float(np.sqrt(np.mean(err[ok] ** 2))) if ok.any() else float("nan")
        root_err = np.linalg.norm(q_sim[:, :3] - q_track[:, :3], axis=1)
        np.savez(run / "mjx_diagnose.npz", markers_m=m, q_sim=q_sim, peak_qvel=speed)
        LOG.info(
            "diagnose: rollout %.0f s; first non-finite frame %d (t=%.3f s); replay markers over finite frames %.1f mm",
            time.perf_counter() - t0,
            first_bad,
            times[first_bad] if first_bad >= 0 else float("nan"),
            rms * 1e3,
        )
        for k in range(0, len(times), 36):
            LOG.info(
                "  t=%.2f: peak |qvel| %.1f rad/s, root err %.0f mm, marker rms %.0f mm",
                times[k],
                speed[k],
                root_err[k] * 1e3,
                float(np.sqrt(np.mean(err[k][np.asarray(valid)[k]] ** 2))) * 1e3
                if finite[k]
                else float("nan"),
            )
        return

    value_and_grad = jax.jit(jax.value_and_grad(cost, has_aux=True))
    delta = jnp.zeros((n_knots, act.size))
    if args.init is not None:
        previous = np.load(args.init)
        if float(previous["knot_spacing_s"]) != args.knot_spacing:
            raise ValueError("--init was optimised with a different knot spacing")
        delta = jnp.asarray(previous["delta_knots"])

    t0 = time.perf_counter()
    best_cost = float("inf")

    def on_iteration(k: int, x: Any, total: float, objective: float) -> None:
        nonlocal best_cost
        rms = float(np.sqrt(max(0.0, objective)))
        finite = bool(np.isfinite(total))
        delta_arr = np.asarray(x)
        delta_max = float(np.max(np.abs(delta_arr)))
        if k == 0:
            LOG.info(
                "port check: MJX replay of the unmodified reference %.1f mm "
                "(shared simulator %.1f mm); first gradient in %.0f s",
                rms * 1e3,
                meta["baseline"]["replay_marker_rms_m"] * 1e3,
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
                    q=np.asarray(reference(x)),
                    delta_knots=delta_arr,
                    knot_spacing_s=args.knot_spacing,
                )
        if finite and objective < best_cost:
            best_cost = objective
            np.savez(
                run / f"{args.name}.npz",
                q=np.asarray(reference(x)),
                delta_knots=delta_arr,
                knot_spacing_s=args.knot_spacing,
            )

    settings = AdamSettings(
        learning_rate=args.learning_rate,
        max_iterations=args.iterations,
    )
    result = adam_minimise(
        value_and_grad,
        delta,
        settings,
        xp=jnp,
        on_iteration=on_iteration,
    )
    if result.stop_reason == "non_finite_gradient":
        LOG.info("gradient not finite; stopping")

    q_best = np.asarray(reference(result.best_x))
    np.savez(
        run / f"{args.name}.npz",
        q=q_best,
        delta_knots=np.asarray(result.best_x),
        knot_spacing_s=args.knot_spacing,
    )

    history = []
    for row in result.history:
        entry: dict[str, Any] = {
            "iteration": int(row["iteration"]),
            "replay_marker_rms_m": float(np.sqrt(max(0.0, row["objective"]))),
            "total_cost": float(row["total"]),
        }
        if row["iteration"] > 0:
            entry["delta_max_rad"] = float(row["max_abs_x"])
        history.append(entry)

    receipt = {
        "run": run.name,
        "settings": vars(args) | {"run": str(run)},
        "knots": int(n_knots),
        "actuated_coordinates": int(act.size),
        "port_check_replay_marker_rms_m": history[0]["replay_marker_rms_m"],
        "shared_simulator_replay_marker_rms_m": meta["baseline"]["replay_marker_rms_m"],
        "best_replay_marker_rms_m": float(np.sqrt(max(0.0, result.best_objective))),
        "history": history,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "dtype": str(jnp.array(1.0).dtype),
    }
    (run / "mjx_optimisation_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    LOG.info(
        "best MJX replay %.1f mm after %d iterations; wrote %s",
        float(np.sqrt(max(0.0, result.best_objective))) * 1e3,
        args.iterations,
        run / f"{args.name}.npz",
    )


if __name__ == "__main__":
    main()

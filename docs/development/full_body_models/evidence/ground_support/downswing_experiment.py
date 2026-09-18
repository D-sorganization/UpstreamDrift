"""Downswing tracking experiments on a finished ground-support run (MM-7, #10109).

The forward-dynamics stage of ``run_ground_support.py`` tracks the
consistency-resolved reference ``q_ref`` with computed torque. On the driver
that reference carries root accelerations near 500 m/s^2 around impact (the
re-solve steps 20 mm in one frame where markers drop out), which the
feedforward injects as torque and the feet leave the ground. This script
replays only the dynamics stage from a run directory with the reference
conditioned in a chosen way, so candidates can be compared in minutes with
one receipt each (``<run>/downswing_<name>.json``):

    python downswing_experiment.py --run anthro_driver --name lp8 --cutoff-hz 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.engines.physics_engines.mujoco.python.full_body_ik import (  # noqa: E402
    FullBodyMarkerKinematics,
)
from src.engines.physics_engines.mujoco.python.full_body_model import (  # noqa: E402
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching import (  # noqa: E402
    full_body_forward_dynamics as fs,
)
from src.shared.python.motion_matching.pipeline import (  # noqa: E402
    BALANCE,
    CAPTURES,
    DT_S,
    OMEGA_RAD_S,
    RATE_HZ,
    Lane,
    marker_errors,
    segment_rms,
)

HERE = Path(__file__).resolve().parent


def condition(
    q_ref: np.ndarray, times: np.ndarray, cutoff_hz: float | None
) -> np.ndarray:
    """Zero-phase low-pass of the reference (all coordinates) or the reference."""
    if cutoff_hz is None:
        return q_ref
    if cutoff_hz <= 0:
        raise ValueError("cutoff must be positive")
    rate = 1.0 / float(np.mean(np.diff(times)))
    b, a = butter(4, cutoff_hz / (0.5 * rate))
    return np.asarray(filtfilt(b, a, q_ref, axis=0, padlen=min(60, len(q_ref) - 1)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("downswing")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cutoff-hz", type=float, default=None)
    parser.add_argument("--omega", type=float, default=OMEGA_RAD_S)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--feedforward", type=float, default=1.0)
    parser.add_argument(
        "--legs-omega", type=float, default=None, help="compliant lower-limb frequency"
    )
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument(
        "--root-regulation",
        nargs=2,
        type=float,
        metavar=("STIFFNESS", "DAMPING"),
        default=None,
        help="PD on the root pose through the planted legs (1/s^2, 1/s)",
    )
    parser.add_argument(
        "--transition-velocity",
        type=float,
        default=None,
        help="override the contact law's friction transition velocity (m/s)",
    )
    parser.add_argument(
        "--friction",
        nargs=2,
        type=float,
        metavar=("STATIC", "DYNAMIC"),
        default=None,
        help="override the contact law's friction coefficients (spiked shoes)",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=None,
        help="override the contact stiffness (N/m); softer soles share load "
        "between heel and toe less sensitively",
    )
    parser.add_argument("--dissipation", type=float, default=None)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="npz with a 'q' array (frames, coordinates) to track instead of the "
        "run's own reference (an optimised reference); errors stay against the "
        "run's q_ref and markers",
    )
    parser.add_argument("--dt", type=float, default=DT_S)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else HERE / args.run
    receipt = json.loads((run / "receipt.json").read_text())
    spec_bytes = (run / "full_body_spec_hipcal_scaled.json").read_bytes()
    document = json.loads(spec_bytes)
    ik = np.load(run / "ik_trajectory.npz")
    times, q_ref, valid = ik["time_s"], ik["q_ref"], ik["valid"]
    labels = tuple(receipt["labels"])
    lane = Lane(labels, CAPTURES[receipt["capture"]])
    adapter = NativeMujocoFullBodyModel(spec_bytes)
    attachments = {
        label: (a["body"], a["offset_m"])
        for label, a in document["marker_attachments"].items()
        if a.get("offset_m") is not None
    }
    # Final placements (static trial and leg calibration) live in the receipt.
    placements = receipt["ik"].get("attachments_m")
    if placements is None:  # older receipts: legs only, upper body as seeded
        placements = receipt["ik"]["calibration"]["offsets_m"]
    for label, a in placements.items():
        attachments[label] = (a["body"], a["offset_m"])
    kin = FullBodyMarkerKinematics(adapter, {k: attachments[k] for k in labels})
    comparable = receipt["ik"].get("attachments_m") is not None
    if args.transition_velocity is not None:
        adapter.contact_parameters = replace(
            adapter.contact_parameters, transition_velocity_m_s=args.transition_velocity
        )
    if args.friction is not None:
        adapter.contact_parameters = replace(
            adapter.contact_parameters,
            static_friction=args.friction[0],
            dynamic_friction=args.friction[1],
        )
    if args.stiffness is not None:
        adapter.contact_parameters = replace(
            adapter.contact_parameters, stiffness_n_m=args.stiffness
        )
    if args.dissipation is not None:
        adapter.contact_parameters = replace(
            adapter.contact_parameters, dissipation_s_m=args.dissipation
        )
    sim = fs.FullBodySimulator(adapter)

    tracked = q_ref
    if args.reference is not None:
        tracked = np.load(args.reference)["q"]
        if tracked.shape != q_ref.shape:
            raise ValueError("--reference must match the run's (frames, coordinates)")
    reference = condition(tracked, times, args.cutoff_hz)
    v_ref = np.gradient(reference, times, axis=0)
    a_ref = np.gradient(v_ref, times, axis=0)
    omega: float | np.ndarray = args.omega
    if args.legs_omega is not None:
        omega = fs.joint_natural_frequencies(
            sim, upper_body=args.omega, lower_limb=args.legs_omega
        )
    controller = fs.tracking_controller(
        sim,
        times,
        reference,
        omega_rad_s=omega,
        zeta=args.zeta,
        balance=None if args.no_balance else BALANCE,
        root_regulation=(
            None
            if args.root_regulation is None
            else (args.root_regulation[0], args.root_regulation[1])
        ),
        acceleration_feedforward=args.feedforward,
    )
    duration = float(args.duration or times[-1])
    q0 = fs.preload_feet(sim, reference[0])
    v0 = v_ref[0]
    t0 = time.perf_counter()
    record = sim.run(
        q0, v0, controller, duration_s=duration, dt_s=args.dt,
        record_every=int(round(1.0 / (RATE_HZ * args.dt))),
    )  # fmt: skip
    elapsed = time.perf_counter() - t0
    keep = times <= duration
    sim_q = np.array(
        [
            [np.interp(t, record.time_s, record.q[:, k]) for k in range(sim.nv)]
            for t in times[keep]
        ]
    )
    root_err = np.linalg.norm(sim_q[:, :3] - q_ref[keep, :3], axis=1)
    root_tilt = np.degrees(np.abs(sim_q[:, 3:6] - q_ref[keep, 3:6]).max(axis=1))
    m = (times >= 1.0) & (times < 1.6) & keep
    out = {
        "run": run.name,
        "name": args.name,
        "settings": {
            "cutoff_hz": args.cutoff_hz,
            "reference": None if args.reference is None else str(args.reference),
            "omega_rad_s": args.omega,
            "legs_omega_rad_s": args.legs_omega,
            "zeta": args.zeta,
            "acceleration_feedforward": args.feedforward,
            "balance": None if args.no_balance else BALANCE,
            "root_regulation": args.root_regulation,
            "transition_velocity_m_s": args.transition_velocity,
            "friction": args.friction,
            "stiffness_n_m": args.stiffness,
            "dissipation_s_m": args.dissipation,
            "dt_s": args.dt,
            "duration_s": duration,
        },
        "reference": {
            "root_acceleration_max_m_s2": float(np.abs(a_ref[:, :3]).max()),
            "root_acceleration_downswing_max_m_s2": (
                float(np.abs(a_ref[m][:, :3]).max()) if m.any() else None
            ),
            "joint_acceleration_max_deg_s2": float(
                np.degrees(np.abs(a_ref[:, 6:]).max())
            ),
            "deviation_from_q_ref_root_max_m": float(
                np.abs(reference[:, :3] - q_ref[:, :3]).max()
            ),
            "deviation_from_q_ref_joints_max_deg": float(
                np.degrees(np.abs(reference[:, 6:] - q_ref[:, 6:]).max())
            ),
        },
        "elapsed_s": round(elapsed, 1),
        "root_error_timeline_m": {
            f"{t:.2f}": float(root_err[int(round(t * RATE_HZ))])
            for t in (0.0, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75)
            if t <= duration
        },
        "root_error_max_m": float(root_err.max()),
        "root_tilt_timeline_deg": {
            f"{t:.2f}": float(root_tilt[int(round(t * RATE_HZ))])
            for t in (0.5, 1.0, 1.2, 1.3, 1.4, 1.5, 1.75)
            if t <= duration
        },
        "joint_tracking_rms_deg": float(
            np.degrees(np.sqrt(np.mean((sim_q[:, 6:] - q_ref[keep, 6:]) ** 2)))
        ),
        "weight_fraction": {
            "min": float(record.weight_fraction.min()),
            "max": float(record.weight_fraction.max()),
            "mean": float(record.weight_fraction.mean()),
        },
        "inside_support_polygon_fraction": float(record.inside_support_polygon.mean()),
        "peak_joint_torque_n_m": float(np.abs(record.tau).max()),
        "lowest_sphere_height_m": [
            float(record.lowest_sphere_height_m.min()),
            float(record.lowest_sphere_height_m.max()),
        ],
    }
    errors = marker_errors(kin, sim_q, lane.points[keep])
    out["marker_rms_m"] = float(np.sqrt(np.mean(errors[valid[keep]] ** 2)))
    out["marker_rms_comparable_to_receipt"] = comparable
    out["segment_rms_m"] = segment_rms(labels, errors, valid[keep])
    ref_markers = np.stack([kin.marker_positions(row) for row in q_ref[keep]])
    sim_markers = np.stack([kin.marker_positions(row) for row in sim_q])
    out["tracking_rms_vs_reference_m"] = float(
        np.sqrt(np.mean(np.linalg.norm(sim_markers - ref_markers, axis=2) ** 2))
    )
    out["backswing_to_1s"] = {
        "root_error_max_m": float(root_err[times[keep] <= 1.0].max()),
        "marker_rms_m": float(
            np.sqrt(
                np.mean(
                    errors[times[keep] <= 1.0][valid[keep][times[keep] <= 1.0]] ** 2
                )
            )
        ),
    }
    window = times[keep] <= 1.5
    out["marker_rms_to_1_5s_m"] = float(
        np.sqrt(np.mean(errors[window][valid[keep][window]] ** 2))
    )
    np.savez(run / f"downswing_{args.name}.npz", sim_q=sim_q, errors_m=errors)
    (run / f"downswing_{args.name}.json").write_text(json.dumps(out, indent=2) + "\n")
    log.info(
        "%s: root err max %.0f mm, timeline %s, wf %.2f..%.2f, torque %.0f N m, %s",
        args.name, out["root_error_max_m"] * 1e3,
        {k: round(v * 1e3) for k, v in out["root_error_timeline_m"].items()},
        out["weight_fraction"]["min"], out["weight_fraction"]["max"],
        out["peak_joint_torque_n_m"],
        f"markers {out['marker_rms_m'] * 1e3:.1f} mm" if "marker_rms_m" in out else "",
    )  # fmt: skip


if __name__ == "__main__":
    main()

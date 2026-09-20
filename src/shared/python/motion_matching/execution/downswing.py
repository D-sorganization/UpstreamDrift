"""Downswing tracking experiments on a finished ground-support run (MM-7, #10109, #10520).

Replays the forward dynamics stage from a run directory with the reference
conditioned in a chosen way, comparing root error and marker RMS metrics.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

import numpy as np

from scipy.signal import butter, filtfilt

from src.shared.python.motion_matching import (
    full_body_forward_dynamics as fs,
)
from src.shared.python.motion_matching.pipeline import (
    BALANCE,
    CAPTURES,
    DT_S,
    OMEGA_RAD_S,
    RATE_HZ,
    Lane,
    marker_errors,
    segment_rms,
)

log = logging.getLogger("downswing")


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


def build_parser() -> argparse.ArgumentParser:
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
        help="PD on root pose through planted legs (1/s^2, 1/s)",
    )
    parser.add_argument(
        "--transition-velocity",
        type=float,
        default=None,
        help="override contact law friction transition velocity (m/s)",
    )
    parser.add_argument(
        "--friction",
        nargs=2,
        type=float,
        metavar=("STATIC", "DYNAMIC"),
        default=None,
        help="override contact law friction coefficients (spiked shoes)",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=None,
        help="override contact stiffness (N/m)",
    )
    parser.add_argument("--dissipation", type=float, default=None)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="npz with 'q' array to track instead of run's reference",
    )
    parser.add_argument("--dt", type=float, default=DT_S)
    parser.add_argument("--duration", type=float, default=None)
    return parser


def _load_downswing_run(
    run: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    tuple[str, ...],
    Lane,
    NativeMujocoFullBodyModel,
    FullBodyMarkerKinematics,
    bool,
]:
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    spec_bytes = (run / "full_body_spec_hipcal_scaled.json").read_bytes()
    document = json.loads(spec_bytes.decode("utf-8"))
    ik = np.load(run / "ik_trajectory.npz")
    labels = tuple(receipt["labels"])
    lane = Lane(labels, CAPTURES[receipt["capture"]])
    adapter = NativeMujocoFullBodyModel(spec_bytes)
    attachments = {
        label: (a["body"], a["offset_m"])
        for label, a in document["marker_attachments"].items()
        if a.get("offset_m") is not None
    }
    placements = receipt["ik"].get("attachments_m")
    if placements is None:
        placements = receipt["ik"]["calibration"]["offsets_m"]
    for label, a in placements.items():
        attachments[label] = (a["body"], a["offset_m"])
    kin = FullBodyMarkerKinematics(adapter, {k: attachments[k] for k in labels})
    comparable = receipt["ik"].get("attachments_m") is not None
    return document, receipt, ik, labels, lane, adapter, kin, comparable


def _configure_contact_parameters(
    adapter: NativeMujocoFullBodyModel, args: argparse.Namespace
) -> None:
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


def _build_tracking_controller(
    sim: fs.FullBodySimulator,
    times: np.ndarray,
    q_ref: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
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
    return controller, reference, v_ref, a_ref


@dataclass(frozen=True)
class _RunContext:
    run: Path
    args: argparse.Namespace
    times: np.ndarray
    q_ref: np.ndarray
    valid: np.ndarray
    labels: tuple[str, ...]
    lane: Lane
    kin: FullBodyMarkerKinematics
    comparable: bool
    sim: fs.FullBodySimulator


def _build_settings_and_ref(
    args: argparse.Namespace,
    duration: float,
    reference: np.ndarray,
    q_ref: np.ndarray,
    a_ref: np.ndarray,
    m: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = {
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
    }
    ref = {
        "root_acceleration_max_m_s2": float(np.abs(a_ref[:, :3]).max()),
        "root_acceleration_downswing_max_m_s2": (
            float(np.abs(a_ref[m][:, :3]).max()) if m.any() else None
        ),
        "joint_acceleration_max_deg_s2": float(np.degrees(np.abs(a_ref[:, 6:]).max())),
        "deviation_from_q_ref_root_max_m": float(
            np.abs(reference[:, :3] - q_ref[:, :3]).max()
        ),
        "deviation_from_q_ref_joints_max_deg": float(
            np.degrees(np.abs(reference[:, 6:] - q_ref[:, 6:]).max())
        ),
    }
    return settings, ref


def _save_downswing_results(
    run: Path,
    name: str,
    sim_q: np.ndarray,
    errors: np.ndarray,
    out: dict[str, Any],
) -> None:
    np.savez(run / f"downswing_{name}.npz", sim_q=sim_q, errors_m=errors)
    (run / f"downswing_{name}.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    log.info(
        "%s: root err max %.0f mm, timeline %s, wf %.2f..%.2f, torque %.0f N m, %s",
        name,
        out["root_error_max_m"] * 1e3,
        {k: round(v * 1e3) for k, v in out["root_error_timeline_m"].items()},
        out["weight_fraction"]["min"],
        out["weight_fraction"]["max"],
        out["peak_joint_torque_n_m"],
        f"markers {out['marker_rms_m'] * 1e3:.1f} mm" if "marker_rms_m" in out else "",
    )


def _compute_downswing_metrics(
    ctx: _RunContext,
    record: Any,
    reference: np.ndarray,
    a_ref: np.ndarray,
    duration: float,
    elapsed: float,
) -> dict[str, Any]:
    keep = ctx.times <= duration
    sim_q = np.array(
        [
            [np.interp(t, record.time_s, record.q[:, k]) for k in range(ctx.sim.nv)]
            for t in ctx.times[keep]
        ]
    )
    root_err = np.linalg.norm(sim_q[:, :3] - ctx.q_ref[keep, :3], axis=1)
    root_tilt = np.degrees(np.abs(sim_q[:, 3:6] - ctx.q_ref[keep, 3:6]).max(axis=1))
    m = (ctx.times >= 1.0) & (ctx.times < 1.6) & keep

    settings, ref_metrics = _build_settings_and_ref(
        ctx.args, duration, reference, ctx.q_ref, a_ref, m
    )

    out: dict[str, Any] = {
        "run": ctx.run.name,
        "name": ctx.args.name,
        "settings": settings,
        "reference": ref_metrics,
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
            np.degrees(np.sqrt(np.mean((sim_q[:, 6:] - ctx.q_ref[keep, 6:]) ** 2)))
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
    errors = marker_errors(ctx.kin, sim_q, ctx.lane.points[keep])
    out["marker_rms_m"] = float(np.sqrt(np.mean(errors[ctx.valid[keep]] ** 2)))
    out["marker_rms_comparable_to_receipt"] = ctx.comparable
    out["segment_rms_m"] = segment_rms(ctx.labels, errors, ctx.valid[keep])
    ref_markers = np.stack([ctx.kin.marker_positions(row) for row in ctx.q_ref[keep]])
    sim_markers = np.stack([ctx.kin.marker_positions(row) for row in sim_q])
    out["tracking_rms_vs_reference_m"] = float(
        np.sqrt(np.mean(np.linalg.norm(sim_markers - ref_markers, axis=2) ** 2))
    )
    out["backswing_to_1s"] = {
        "root_error_max_m": float(root_err[ctx.times[keep] <= 1.0].max()),
        "marker_rms_m": float(
            np.sqrt(
                np.mean(
                    errors[ctx.times[keep] <= 1.0][
                        ctx.valid[keep][ctx.times[keep] <= 1.0]
                    ]
                    ** 2
                )
            )
        ),
    }
    window = ctx.times[keep] <= 1.5
    out["marker_rms_to_1_5s_m"] = float(
        np.sqrt(np.mean(errors[window][ctx.valid[keep][window]] ** 2))
    )
    _save_downswing_results(ctx.run, ctx.args.name, sim_q, errors, out)
    return out


def run_downswing_experiment(args: argparse.Namespace) -> dict[str, Any]:
    run = Path(args.run)
    (
        document,
        receipt,
        ik,
        labels,
        lane,
        adapter,
        kin,
        comparable,
    ) = _load_downswing_run(run)
    times, q_ref, valid = ik["time_s"], ik["q_ref"], ik["valid"]

    _configure_contact_parameters(adapter, args)
    sim = fs.FullBodySimulator(adapter)

    controller, reference, v_ref, a_ref = _build_tracking_controller(
        sim, times, q_ref, args
    )
    duration = float(args.duration or times[-1])
    q0 = fs.preload_feet(sim, reference[0])
    v0 = v_ref[0]

    t0 = time.perf_counter()
    record = sim.run(
        q0,
        v0,
        controller,
        duration_s=duration,
        dt_s=args.dt,
        record_every=int(round(1.0 / (RATE_HZ * args.dt))),
    )
    elapsed = time.perf_counter() - t0

    ctx = _RunContext(
        run=run,
        args=args,
        times=times,
        q_ref=q_ref,
        valid=valid,
        labels=labels,
        lane=lane,
        kin=kin,
        comparable=comparable,
        sim=sim,
    )
    return _compute_downswing_metrics(
        ctx=ctx,
        record=record,
        reference=reference,
        a_ref=a_ref,
        duration=duration,
        elapsed=elapsed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_downswing_experiment(args)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

"""Full-swing (0-1.8 s) bushing-grip kinetics from the closure-consistent fits.

Issue #11739 (OSV-7), impact phase, epic #11726.  Reproduce from the
repository root (the measured wrist speeds need the private captures)::

    CAPTURE_DATA_DIR=... MPLBACKEND=Agg PYTHONPATH=.:src python \\
        docs/development/full_body_models/evidence/grip_kinetics/run_full_swing_grip_kinetics.py \\
        --club driver --convergence

Input: the OSV-10 club-face fixtures (``tests/fixtures/club_face/swing_q_<club>.npz``,
the 1 kHz same-input reference of the shared ground-support fit, every second
sample, IK marker RMS about 33 mm), mapped by coordinate name onto the
committed ``full_body_spec_anthro_<club>.json`` (:func:`load_coordinate_swing`)
and prescribed unfiltered.  The club is a free body held by the two grip
bushings (modal damping zeta = 0.7).  The receipt records the input
kinematics first (two-hand loop closure, model hand speeds against the
measured wrist-marker speeds), the integrator and its convergence, the full
window and impact metrics, and the published magnitudes used as a sanity
check.  Outputs are overwritten by name; nothing is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_grip_kinetics import (  # noqa: E402
    plot_series,
    render_clip,
    summarize,
    window,
)

from src.engines.physics_engines.opensim.python.grip_bushing_sim import (  # noqa: E402
    BushingGripSimulator,
    BushingRun,
    analyze_run,
    hand_power_w,
    input_kinematics_report,
)
from src.shared.python.grip_contact import (  # noqa: E402
    ClubDynamics,
    ClubKinematics,
    GripInterface,
    couple_consistency,
    decompose_hand_forces,
    load_coordinate_swing,
    modal_damping,
    peak_squeeze_n,
    required_hand_moment_nm,
)
from src.shared.python.model_appearance import club_face as cf  # noqa: E402

ROOT = HERE.parents[4]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
DEFAULT_OUT = Path(
    "/home/dieterolson/Videos/Parity Audit/golfer_realism/grip_kinetics/full_swing"
)
CAPTURES = {"driver": ("driver", "A"), "iron7": ("iron", "B")}
IMPACT_HALF_WINDOW_S = 0.010
OLD_XFAIL_WINDOW_S = 1.30  # t_end of the phase-1 candidate xfail test
LITERATURE = [
    {
        "source": "Nesbit S M (2005) A three dimensional kinematic and kinetic "
        "study of the golf swing. J Sports Sci Med 4(4):499-519, Table 3",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3899667/",
        "quantity": "golfer-club linear (net) force at impact, 84 male + 1 female",
        "value": "mean 397.5 N, median 400 N, SD 87.7 N, range 300-490 N",
        "compare_to": "net force at impact",
    },
    {
        "source": "Grober R D (2020) Forces and torques near to impact in the "
        "golf swing. arXiv:2006.11778, section VIII, quoting MacKenzie's "
        "instrumented-grip data (one golfer, 116.5 mph, last frame before impact)",
        "url": "https://arxiv.org/abs/2006.11778",
        "quantity": "net force, moment of force and couple on the club",
        "value": "F0 = 456 N, M0 = 55.8 N m, K0 = -59.1 N m; a 50 N m couple "
        "needs about 300 N per hand at 1/6 m spacing",
        "compare_to": "net force and equivalent couple at impact",
    },
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak(v: np.ndarray, t: np.ndarray) -> dict[str, float]:
    k = int(np.argmax(v))
    return {"peak_m_s": float(v[k]), "time_s": float(t[k])}


def measured_wrist_speeds(capture: str) -> dict[str, Any]:
    """Peak speed of the measured wrist-top markers (private capture)."""
    try:
        from src.shared.python.motion_matching.pipeline.constants import (
            capture_path,
        )
        from src.shared.python.motion_matching.tour_capture_contract import (
            load_tour_capture,
        )

        cap = load_tour_capture(capture_path(capture))
    except (OSError, ValueError, KeyError, ImportError) as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}
    t = np.asarray(cap.time_s, float) - float(cap.time_s[0])
    out: dict[str, Any] = {}
    for label, name in (("left", "LWristTop"), ("right", "RWristTop")):
        pts = np.asarray(cap.points_m)[:, list(cap.labels).index(name)]
        valid = np.asarray(cap.valid)[:, list(cap.labels).index(name)]
        speed = np.linalg.norm(np.gradient(pts, axis=0), axis=1) / np.gradient(t)
        speed[~valid.astype(bool)] = np.nan
        k = int(np.nanargmax(speed))
        out[label] = {
            "peak_m_s": float(speed[k]),
            "time_s": float(t[k]),
            "p95_m_s": float(np.nanpercentile(speed, 95)),
        }
    return out


def input_report(club: str, spec_bytes: bytes, swing: Any) -> dict[str, Any]:
    """Closure and hand-speed credibility of the prescribed motion."""
    rep = input_kinematics_report(spec_bytes, swing.names, swing.time_s, swing.q)
    t = swing.time_s
    prov = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    rec = prov["clubs"][club]
    return {
        "fixture": f"tests/fixtures/club_face/swing_q_{club}.npz",
        "fixture_sha256": swing.sha256,
        "ik_marker_rms_m": rec["marker_rms_m"]["ik"],
        "reference_marker_rms_m": rec["marker_rms_m"]["reference"],
        "closure_distance_mm": {
            "max": float(rep["closure_distance_m"].max() * 1e3),
            "median": float(np.median(rep["closure_distance_m"]) * 1e3),
        },
        "closure_angle_deg": {"max": float(rep["closure_angle_deg"].max())},
        "model_left_grip_point_speed": _peak(rep["left_hand_speed_m_s"], t),
        "model_right_closure_frame_speed": _peak(rep["right_hand_speed_m_s"], t),
        "measured_wrist_marker_speed": measured_wrist_speeds(CAPTURES[club][0]),
        "note": "model speeds are of the grip points, measured of the wrist-top "
        "markers; the grip points lie further from the swing centre",
    }


def impact_metrics(run: BushingRun) -> dict[str, Any]:
    """Per-hand, net, internal and couple values at the impact frame."""
    k = cf.impact_frame(run.time_s, run.club_position_m)
    analyses = analyze_run(run)
    fl, fr = run.force_on_club_n["L"][k], run.force_on_club_n["R"][k]
    dec = decompose_hand_forces(
        fl, fr, run.grip_point_m["L"][k], run.grip_point_m["R"][k]
    )
    near = np.abs(run.time_s - run.time_s[k]) <= IMPACT_HALF_WINDOW_S
    dec_all = decompose_hand_forces(
        run.force_on_club_n["L"],
        run.force_on_club_n["R"],
        run.grip_point_m["L"],
        run.grip_point_m["R"],
    )
    mag = {s: np.linalg.norm(run.force_on_club_n[s], axis=1) for s in "LR"}
    couple = np.array([np.linalg.norm(a.couple_at_midpoint_nm) for a in analyses])
    return {
        "impact_time_s": float(run.time_s[k]),
        "detector": "model_appearance.club_face.impact_frame on the club body origin",
        "at_impact": {
            "force_n": {
                "L_lead": float(np.linalg.norm(fl)),
                "R_trail": float(np.linalg.norm(fr)),
            },
            "lead_share": float(
                np.linalg.norm(fl) / (np.linalg.norm(fl) + np.linalg.norm(fr))
            ),
            "net_n": float(np.linalg.norm(dec.net_n)),
            "internal_n": float(np.linalg.norm(dec.internal_n)),
            "internal_axial_squeeze_n": float(dec.internal_axial_n),
            "internal_transverse_n": float(np.linalg.norm(dec.internal_transverse_n)),
            "force_pair_couple_nm": float(dec.couple_moment_nm),
            "equivalent_couple_at_midpoint_nm": float(couple[k]),
            "free_torque_nm": {
                s: float(np.linalg.norm(run.torque_on_club_nm[s][k])) for s in "LR"
            },
        },
        f"peak_within_{IMPACT_HALF_WINDOW_S * 1e3:.0f}_ms": {
            "force_n": {s: float(mag[s][near].max()) for s in "LR"},
            "net_n": float(np.linalg.norm(dec_all.net_n, axis=1)[near].max()),
            "internal_n": float(np.linalg.norm(dec_all.internal_n, axis=1)[near].max()),
            "equivalent_couple_at_midpoint_nm": float(couple[near].max()),
        },
        "peak_full_window": {
            "equivalent_couple_at_midpoint_nm": float(couple.max()),
            "couple_peak_time_s": float(run.time_s[int(np.argmax(couple))]),
            "internal_peak_time_s": float(
                run.time_s[int(np.argmax(np.linalg.norm(dec_all.internal_n, axis=1)))]
            ),
        },
    }


def internal_force_checks(run: BushingRun, spec: dict) -> dict[str, Any]:
    """Squeeze and couple-consistency checks (#11739 owner decision).

    The realised check uses the club's own engine accelerations (a
    Newton-Euler closure); the rigid-input entry uses the club welded to the
    prescribed lead hand and shows the bushing's dynamic amplification.
    """
    forces = (run.force_on_club_n["L"], run.force_on_club_n["R"])
    points = (run.grip_point_m["L"], run.grip_point_m["R"])
    torques = (run.torque_on_club_nm["L"], run.torque_on_club_nm["R"])
    mid = 0.5 * (points[0] + points[1])
    dyn = ClubDynamics.from_spec(spec)
    g = np.asarray(spec["gravity_m_s2"], dtype=float)
    realised = ClubKinematics(
        run.club_rotation,
        run.club_omega_rad_s,
        run.club_alpha_rad_s2,
        run.club_com_m,
        run.club_com_acceleration_m_s2,
    )
    m_real = required_hand_moment_nm(realised, dyn, g, mid)
    m_rigid = required_hand_moment_nm(run.rigid_club, dyn, g, mid)
    res = couple_consistency(forces, points, torques, m_real)
    k = int(np.argmax(res.actual_transverse_n))
    t = run.time_s
    spacing = np.linalg.norm(points[1] - points[0], axis=1)
    return {
        "squeeze_bound_n": 50.0,
        "couple_relative_error_bound": 0.05,
        "noise_floor_nm": res.noise_floor_nm,
        "peak_squeeze_n": peak_squeeze_n(*forces, *points),
        "peak_internal_transverse_n": float(res.actual_transverse_n[k]),
        "peak_internal_time_s": float(t[k]),
        "couple_over_d_prediction_at_peak_n": float(res.predicted_transverse_n[k]),
        "max_relative_error": res.max_relative_error(),
        "checked_fraction": float(res.checked.mean()),
        "hand_spacing_m": {"min": float(spacing.min()), "max": float(spacing.max())},
        "realised_hand_moment_peak_nm": _peak(np.linalg.norm(m_real, axis=1), t),
        "rigid_input_hand_moment_peak_nm": _peak(np.linalg.norm(m_rigid, axis=1), t),
    }


def simulate(club: str, accuracy: float, method: str) -> tuple[BushingRun, dict]:
    """Run the full window; returns the run and integrator metadata."""
    import opensim as osim

    spec_path = MODELS / f"full_body_spec_anthro_{club}.json"
    spec_bytes = spec_path.read_bytes()
    order = json.loads(spec_bytes)["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz", FIXTURES / "address_poses.json", club, order
    )
    sim = BushingGripSimulator(spec_bytes, swing.names, swing.time_s, swing.q)
    original = osim.Manager.__init__

    def init(manager: Any, *args: Any) -> None:  # select the integrator
        original(manager, *args)
        manager.setIntegratorMethod(getattr(osim.Manager, f"IntegratorMethod_{method}"))

    osim.Manager.__init__ = init
    try:
        start = time.perf_counter()
        run = sim.run(None, accuracy=accuracy)
        elapsed = time.perf_counter() - start
    finally:
        osim.Manager.__init__ = original
    meta = {"method": method, "accuracy": accuracy, "wall_time_s": elapsed}
    return run, {"integrator": meta, "swing": swing, "spec_bytes": spec_bytes}


def _previous_convergence(club: str) -> dict[str, Any]:
    """Keep the convergence block of an earlier receipt (rerun without it)."""
    path = HERE / f"receipt_full_swing_{club}.json"
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text()).get("convergence", {}))


def _bounds(run: BushingRun) -> dict[str, float]:
    s = summarize(run)
    return {
        "peak_force_L_n": s["peak_force_n"]["L"],
        "peak_force_R_n": s["peak_force_n"]["R"],
        "peak_net_n": s["internal_force"]["peak_net_n"],
        "peak_internal_n": s["internal_force"]["peak_internal_n"],
        "max_deflection_mm": max(s["max_translation_deflection_mm"].values()),
        "max_rotation_deg": max(s["max_rotation_deflection_deg"].values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--club", choices=sorted(CAPTURES), required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--accuracy", type=float, default=1e-3)
    ap.add_argument("--method", default="RungeKuttaMerson")
    ap.add_argument("--convergence", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run, info = simulate(args.club, args.accuracy, args.method)
    spec = json.loads(info["spec_bytes"])
    gi = GripInterface.from_spec(spec)
    freqs, zetas = modal_damping(gi, ClubDynamics.from_spec(spec))
    impact = impact_metrics(run)
    receipt: dict[str, Any] = {
        "issue": "#11739",
        "epic": "#11726",
        "club": args.club,
        "capture": CAPTURES[args.club][1],
        "spec_sha256": _sha(MODELS / f"full_body_spec_anthro_{args.club}.json"),
        "input_kinematics": input_report(args.club, info["spec_bytes"], info["swing"]),
        "prescription": "fixture coordinates as committed, SimmSpline, no filter",
        "integrator": info["integrator"],
        "modal_frequencies_hz": freqs.tolist(),
        "modal_damping_ratios": zetas.tolist(),
        "bounds": {
            "deflection_mm": 3.0,
            "rotation_deg": 2.0,
            "squeeze_n": 50.0,
            "couple_relative_error": 0.05,
        },
        "full_window": summarize(run),
        "full_window_bounds": _bounds(run),
        "window_0_to_1p30_s_bounds": _bounds(window(run, OLD_XFAIL_WINDOW_S)),
        "impact": impact,
        "internal_force_checks": internal_force_checks(run, spec),
        "literature_sanity_check": LITERATURE,
        "convergence": _previous_convergence(args.club),
    }
    if args.convergence:
        for method, acc in (("RungeKuttaMerson", 1e-5), ("CPodes", 1e-4)):
            other, oinfo = simulate(args.club, acc, method)
            receipt["convergence"][f"{method}_{acc:g}"] = {
                **oinfo["integrator"],
                **_bounds(other),
            }
    series = {"time_s": run.time_s}
    for s in "LR":
        series[f"force_on_club_{s}_n"] = run.force_on_club_n[s]
        series[f"torque_on_club_{s}_nm"] = run.torque_on_club_nm[s]
        series[f"grip_point_{s}_m"] = run.grip_point_m[s]
        series[f"deflection_{s}_m"] = run.deflection_m[s]
        series[f"rotation_deflection_{s}_rad"] = run.rotation_deflection_rad[s]
        series[f"power_{s}_w"] = hand_power_w(run, s)
    np.savez_compressed(HERE / f"{args.club}_bushing_series_full_swing.npz", **series)
    (HERE / f"receipt_full_swing_{args.club}.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    label = "Driver" if args.club == "driver" else "7-Iron"
    plot_series(
        run,
        args.out / f"{args.club}_bushing_grip_wrench_series_full_swing.png",
        title=f"{label} Full Swing, Bushing Grip: Wrenches Exerted by Each Hand on the Club",
        events={"impact": impact["impact_time_s"]},
    )
    if not args.no_video:
        render_clip(
            run,
            args.out / f"{args.club}_opensim_bushing_grip_hands_closeup_0p5x.mp4",
            length_m=None,
            title=f"Bushing Grip, {label} (0.5x)",
        )
    sys.stdout.write(
        json.dumps(
            {k: receipt[k] for k in ("full_window_bounds", "impact", "convergence")},
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

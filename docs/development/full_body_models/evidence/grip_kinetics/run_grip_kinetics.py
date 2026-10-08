"""Driver bushing-grip kinetics: series, receipt, force/couple plot, close-up clip.

Issue #11739 (OSV-7) phase 1, epic #11726.  Reproduce from the repository root::

    MPLBACKEND=Agg PYTHONPATH=.:src python \
        docs/development/full_body_models/evidence/grip_kinetics/run_grip_kinetics.py

The 44 body coordinates of the matched OpenSim driver candidate are low-pass
filtered (25 Hz) and prescribed; the club is a free body held by the two grip
bushings.  Existing output files are overwritten by name; nothing is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.engines.physics_engines.opensim.python.grip_bushing_sim import (  # noqa: E402
    BushingGripSimulator,
    BushingRun,
    analyze_run,
    hand_power_w,
    lowpass_trajectory,
)
from src.shared.python.biomechanics.grip_wrench import (  # noqa: E402
    allocate_min_norm,
    to_overlay_wrenches,
)
from src.shared.python.force_overlay.contracts import ForceTorqueFrame  # noqa: E402
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs  # noqa: E402
from src.shared.python.force_overlay.renderers.matplotlib_glyphs import (  # noqa: E402
    draw_glyphs_3d,
)
from src.shared.python.grip_contact import GripInterface  # noqa: E402

ROOT = Path(__file__).resolve().parents[5]
MODELS = ROOT / "docs/development/full_body_models"
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = Path("/home/dieterolson/Videos/Parity Audit/golfer_realism/grip_kinetics")
GRAVITY = 9.80665


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_swing(
    scale: float, accuracy: float, t_end: float | None = None
) -> tuple[BushingRun, dict]:
    """Integrate the filtered driver motion with stiffness scaled by ``scale``."""
    spec_path = MODELS / "full_body_spec_anthro_driver.json"
    cand = MODELS / "evidence/ground_support/anthro_driver_opensim/candidate.npz"
    data = np.load(cand, allow_pickle=True)
    names = json.loads(str(data["manifest_json"]))["coordinate_names"]
    t, q = (
        np.asarray(data["time_s"]),
        lowpass_trajectory(np.asarray(data["time_s"]), np.asarray(data["q"]), 25.0),
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    gi = GripInterface.from_spec(spec)
    gi = GripInterface(gi.left, gi.right, gi.bushing.scaled(scale), gi.contact_material)
    sim = BushingGripSimulator(spec_path.read_bytes(), names, t, q, interface=gi)
    run = sim.run(t_end, accuracy=accuracy)
    meta = {
        "spec_sha256": _sha(spec_path),
        "candidate_sha256": _sha(cand),
        "stiffness_scale": scale,
        "bushing": {
            "translational_stiffness_n_m": gi.bushing.translational_stiffness_n_m,
            "rotational_stiffness_nm_rad": gi.bushing.rotational_stiffness_nm_rad,
            "translational_damping_ns_m": gi.bushing.translational_damping_ns_m,
            "rotational_damping_nms_rad": gi.bushing.rotational_damping_nms_rad,
            "source": gi.bushing.source,
        },
        "accuracy": accuracy,
        "t_end_s": float(run.time_s[-1]),
        "prefilter": "zero-phase 4th-order Butterworth, 25 Hz",
    }
    return run, meta


def summarize(run: BushingRun) -> dict:
    """Deflection, load, split and allocation-comparison statistics."""
    analyses = analyze_run(run)
    fl, fr = run.force_on_club_n["L"], run.force_on_club_n["R"]
    mag_l, mag_r = np.linalg.norm(fl, axis=1), np.linalg.norm(fr, axis=1)
    net = np.linalg.norm(fl + fr, axis=1)
    alloc_l = np.array([allocate_min_norm(a)[0] for a in analyses])
    alloc_r = np.array([allocate_min_norm(a)[1] for a in analyses])
    peak = int(np.argmax(mag_l + mag_r))
    loaded = (mag_l + mag_r) > 0.25 * (mag_l + mag_r).max()
    share = lambda a, b: a / np.maximum(a + b, 1e-12)  # noqa: E731
    out = {
        "samples": int(run.time_s.size),
        "peak_time_s": float(run.time_s[peak]),
        "peak_force_n": {"L": float(mag_l.max()), "R": float(mag_r.max())},
        "peak_net_force_n": float(net.max()),
        "max_translation_deflection_mm": {
            s: float(np.linalg.norm(run.deflection_m[s], axis=1).max() * 1e3)
            for s in "LR"
        },
        "max_rotation_deflection_deg": {
            s: float(np.degrees(run.rotation_deflection_rad[s].max())) for s in "LR"
        },
        "left_force_share": {
            "bushing_at_peak": float(share(mag_l, mag_r)[peak]),
            "bushing_median_loaded": float(np.median(share(mag_l, mag_r)[loaded])),
            "min_norm_at_peak": float(
                share(np.linalg.norm(alloc_l, axis=1), np.linalg.norm(alloc_r, axis=1))[
                    peak
                ]
            ),
            "min_norm_median_loaded": float(
                np.median(
                    share(
                        np.linalg.norm(alloc_l, axis=1),
                        np.linalg.norm(alloc_r, axis=1),
                    )[loaded]
                )
            ),
        },
        "left_force_difference_vs_min_norm_n": {
            "max": float(np.linalg.norm(fl - alloc_l, axis=1).max()),
            "rms_loaded": float(
                np.sqrt(np.mean(np.linalg.norm(fl - alloc_l, axis=1)[loaded] ** 2))
            ),
        },
        "peak_hand_power_w": {
            s: float(np.abs(hand_power_w(run, s)).max()) for s in "LR"
        },
    }
    return out


def static_hold(spec_path: Path) -> dict:
    """Settled static hold at address: sum of hand forces versus club weight."""
    cand = MODELS / "evidence/ground_support/anthro_driver_opensim/candidate.npz"
    data = np.load(cand, allow_pickle=True)
    names = json.loads(str(data["manifest_json"]))["coordinate_names"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    n = 31
    sim = BushingGripSimulator(
        spec_path.read_bytes(),
        names,
        np.linspace(0.0, 0.3, n),
        np.repeat(np.asarray(data["q"])[:1], n, axis=0),
    )
    run = sim.run(accuracy=1e-6)
    total = run.force_on_club_n["L"][-1] + run.force_on_club_n["R"][-1]
    weight = spec["club"]["total_mass_kg"] * GRAVITY
    up = -np.asarray(spec["gravity_m_s2"]) / GRAVITY
    return {
        "club_weight_n": weight,
        "sum_hand_force_up_n": float(total @ up),
        "relative_error": float(abs(total @ up - weight) / weight),
        "left_force_n": run.force_on_club_n["L"][-1].tolist(),
        "right_force_n": run.force_on_club_n["R"][-1].tolist(),
    }


def plot_series(run: BushingRun, path: Path) -> None:
    """Per-hand force and couple (free torque + moment of force) versus time."""
    analyses = analyze_run(run)
    t = run.time_s
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for side, colour in (("L", "#1f77b4"), ("R", "#d62728")):
        axes[0].plot(
            t,
            np.linalg.norm(run.force_on_club_n[side], axis=1),
            colour,
            label=f"{side} hand force on club",
        )
        axes[1].plot(
            t,
            np.linalg.norm(run.torque_on_club_nm[side], axis=1),
            colour,
            label=f"{side} hand free torque on club",
        )
        axes[3].plot(
            t,
            np.linalg.norm(run.deflection_m[side], axis=1) * 1e3,
            colour,
            label=f"{side} translation",
        )
        axes[3].plot(
            t,
            np.degrees(run.rotation_deflection_rad[side]),
            colour,
            ls="--",
            label=f"{side} rotation (deg)",
        )
    couple = np.array([np.linalg.norm(a.couple_at_midpoint_nm) for a in analyses])
    axes[2].plot(t, couple, "k", label="equivalent couple at midpoint")
    axes[2].plot(
        t, [np.linalg.norm(a.net_force_n) for a in analyses], "g", label="net force (N)"
    )
    axes[3].axhline(3.0, color="grey", lw=0.8)
    axes[3].axhline(2.0, color="grey", lw=0.8, ls=":")
    for ax, label in zip(
        axes,
        (
            "Force (N)",
            "Torque (N*m)",
            "Net force (N) / couple (N*m)",
            "Deflection (mm, deg)",
        ),
        strict=True,
    ):
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Driver Swing, Bushing Grip: Wrenches Exerted by Each Hand on the Club"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_clip(run: BushingRun, path: Path, length_m: float = 1.156) -> int:
    """0.5x hands close-up with per-hand force arrows (GCV-4/GCV-10 glyphs)."""
    analyses = analyze_run(run)
    step = max(1, int(round(run.time_s.size / (run.time_s[-1] * 60.0))))  # ~60 sim-fps
    idx = list(range(0, run.time_s.size, step))
    style = ForceGlyphStyle(force_scale_m_per_n=1.0 / 800.0, max_length_m=0.25)
    ffmpeg = subprocess.run(
        [sys.executable, "-c", "import imageio_ffmpeg as i;print(i.get_ffmpeg_exe())"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    with tempfile.TemporaryDirectory() as tmp:
        for n, i in enumerate(idx):
            ax.clear()
            mid = 0.5 * (run.grip_point_m["L"][i] + run.grip_point_m["R"][i])
            r = run.club_rotation[i]
            butt = run.club_position_m[i] + r @ np.array([0.0, -length_m, 0.064])
            head = run.club_position_m[i]
            ax.plot(*zip(butt, head, strict=True), color="#444", lw=3)
            for side, colour in (("L", "#1f77b4"), ("R", "#d62728")):
                ax.scatter(*run.grip_point_m[side][i], color=colour, s=60)
            for fore, hand, colour in (
                ("LF", "LGrip", "#1f77b4"),
                ("RF", "Grip", "#d62728"),
            ):
                ax.plot(
                    *zip(
                        run.body_positions_m[fore][i],
                        run.body_positions_m[hand][i],
                        strict=True,
                    ),
                    color=colour,
                    lw=6,
                    alpha=0.5,
                )
            frame = ForceTorqueFrame(
                time_s=float(run.time_s[i]),
                engine="opensim",
                wrenches=tuple(
                    to_overlay_wrenches(analyses[i], source="opensim:bushing")
                ),
            )
            draw_glyphs_3d(ax, build_glyphs(frame, style))
            half = 0.3
            ax.set_xlim(mid[0] - half, mid[0] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_zlim(mid[2] - half, mid[2] + half)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=15, azim=-70)
            ax.set_title(
                f"Bushing Grip, Driver (0.5x): t = {run.time_s[i]:.3f} s\nL blue, R red; arrows = hand force on club",
                fontsize=9,
            )
            fig.savefig(Path(tmp) / f"f{n:05d}.png", dpi=100)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                "30",
                "-i",
                str(Path(tmp) / "f%05d.png"),
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                str(path),
            ],
            check=True,
        )
    plt.close(fig)
    return len(idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--accuracy", type=float, default=1e-5)
    ap.add_argument("--sensitivity", type=float, nargs="*", default=[])
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run, meta = run_swing(1.0, args.accuracy, args.t_end)
    receipt = {
        "issue": "#11739",
        "epic": "#11726",
        "meta": meta,
        "summary": summarize(run),
        "static_hold": static_hold(MODELS / "full_body_spec_anthro_driver.json"),
        "sensitivity": {},
    }
    for scale in args.sensitivity:
        r2, _ = run_swing(scale, args.accuracy, args.t_end)
        receipt["sensitivity"][f"{scale:g}"] = summarize(r2)
    series = {"time_s": run.time_s}
    for s in "LR":
        series[f"force_on_club_{s}_n"] = run.force_on_club_n[s]
        series[f"torque_on_club_{s}_nm"] = run.torque_on_club_nm[s]
        series[f"grip_point_{s}_m"] = run.grip_point_m[s]
        series[f"deflection_{s}_m"] = run.deflection_m[s]
        series[f"rotation_deflection_{s}_rad"] = run.rotation_deflection_rad[s]
        series[f"power_{s}_w"] = hand_power_w(run, s)
    np.savez_compressed(HERE / "driver_bushing_series.npz", **series)
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    plot_series(run, args.out / "driver_bushing_grip_wrench_series.png")
    if not args.no_video:
        render_clip(
            run, args.out / "driver_opensim_bushing_grip_hands_closeup_0p5x.mp4"
        )
    sys.stdout.write(json.dumps(receipt["summary"], indent=2) + "\n")
    sys.stdout.write(json.dumps(receipt["static_hold"], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

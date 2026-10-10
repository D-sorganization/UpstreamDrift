"""Same-input bushing-grip parity across OpenSim, MuJoCo, Drake and Pinocchio.

Issue #11739 (OSV-7 phase 2), epic #11726.  Reproduce from the repository
root, one heavy simulation at a time::

    MPLBACKEND=Agg PYTHONPATH=.:src python3 \\
        docs/development/full_body_models/evidence/grip_kinetics/run_grip_parity.py \\
        --club driver --reference          # OpenSim reference series (slow)
    ... run_grip_parity.py --club driver --engines mujoco drake pinocchio
    ... run_grip_parity.py --club driver --report   # metrics json and plots

Input: the OSV-10 fixture ``tests/fixtures/club_face/swing_q_<club>.npz``
mapped by name onto ``full_body_spec_anthro_<club>.json``.  Every engine
prescribes the same coordinate spline (``grip_contact.CoordinateSpline``,
identical to OpenSim's ``SimmSpline``), computes the hand frames from its own
forward kinematics of the weld model and integrates a free club held by two
bushings.  Outputs are overwritten by name; nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from src.shared.python.grip_contact import load_coordinate_swing  # noqa: E402
from src.shared.python.grip_contact.parity import (  # noqa: E402
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    GripKineticsSeries,
    parity_errors,
)

MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
OUT = HERE / "parity"
PLOTS = Path(
    "/home/dieterolson/Videos/Parity Audit/golfer_realism/grip_kinetics/parity"
)
ENGINE_MODULES = {
    "mujoco": "src.engines.physics_engines.mujoco.python.grip_bushing",
    "drake": "src.engines.physics_engines.drake.python.grip_bushing",
    "pinocchio": "src.engines.physics_engines.pinocchio.python.grip_bushing",
    "myosuite": "src.engines.physics_engines.myosuite.python.grip_bushing",
}
REFERENCE_ACCURACY = 1e-5


def _spec_bytes(club: str) -> bytes:
    return (MODELS / f"full_body_spec_anthro_{club}.json").read_bytes()


def _swing(club: str) -> Any:
    names = json.loads(_spec_bytes(club))["coordinate_order"]
    return load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz", FIXTURES / "address_poses.json", club, names
    )


def _series_path(engine: str, club: str) -> Path:
    return OUT / f"{engine}_{club}_series.npz"


def run_reference(club: str, accuracy: float) -> dict[str, Any]:
    """OpenSim BushingForce reference (RK-Merson, error controlled)."""
    from src.engines.physics_engines.opensim.python.grip_bushing_sim import (
        BushingGripSimulator,
    )

    swing = _swing(club)
    start = time.perf_counter()
    sim = BushingGripSimulator(_spec_bytes(club), swing.names, swing.time_s, swing.q)
    run = sim.run(accuracy=accuracy)
    wall = time.perf_counter() - start
    series = GripKineticsSeries.from_bushing_run("opensim", run)
    series.save_npz(_series_path("opensim", club))
    return {"engine": "opensim", "accuracy": accuracy, "wall_time_s": wall}


def run_engine(engine: str, club: str) -> dict[str, Any]:
    """Run one engine on the same input and save its series."""
    module = import_module(ENGINE_MODULES[engine])
    start = time.perf_counter()
    series = module.simulate_grip_bushing(_spec_bytes(club), _swing(club))
    wall = time.perf_counter() - start
    series.save_npz(_series_path(engine, club))
    return {"engine": engine, "wall_time_s": wall, **dict(series.metadata)}


ENGINE_STYLE = {
    "opensim": ("#000000", "-", 2.4),
    "mujoco": ("#E69F00", "--", 1.4),
    "drake": ("#56B4E9", "-.", 1.4),
    "pinocchio": ("#009E73", ":", 1.8),
    "myosuite": ("#CC79A7", "--", 1.2),
}
PANELS = (
    ("force_L", "Left Hand Force", "N", 1.0),
    ("force_R", "Right Hand Force", "N", 1.0),
    ("net_force", "Net Force at Grip Midpoint", "N", 1.0),
    ("couple", "Equivalent Couple at Midpoint", "N*m", 1.0),
    ("internal_force", "Internal (Antagonistic) Force", "N", 1.0),
    ("squeeze", "Squeeze (Axial Internal Force)", "N", 1.0),
    ("deflection_L", "Left Bushing Deflection", "mm", 1e3),
    ("deflection_R", "Right Bushing Deflection", "mm", 1e3),
    ("rotation_L", "Left Bushing Rotation", "deg", float(np.degrees(1.0))),
    ("rotation_R", "Right Bushing Rotation", "deg", float(np.degrees(1.0))),
)


def _impact_time(club: str) -> float:
    receipt = HERE / f"receipt_full_swing_{club}.json"
    data = json.loads(receipt.read_text(encoding="utf-8"))
    return float(data["impact"]["impact_time_s"])


def _available_series(club: str) -> dict[str, GripKineticsSeries]:
    out = {}
    for engine in ("opensim", *ENGINE_MODULES):
        path = _series_path(engine, club)
        if path.is_file():
            out[engine] = GripKineticsSeries.load_npz(path)
    if "opensim" not in out:
        raise FileNotFoundError(f"no OpenSim reference for {club}; run --reference")
    return out


def _signed_or_magnitude(name: str, values: np.ndarray) -> np.ndarray:
    # The squeeze is a signed scalar; everything else is plotted as magnitude.
    return values[:, 0] if name == "squeeze" else np.linalg.norm(values, axis=1)


def _overlay_figure(
    club: str,
    series: dict[str, GripKineticsSeries],
    metrics: dict[str, dict[str, Any]],
    window: tuple[float, float] | None,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    impact = _impact_time(club)
    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=True)
    quantities = {e: s.quantities() for e, s in series.items()}
    for ax, (name, title, unit, scale) in zip(axes.flat, PANELS, strict=True):
        notes = []
        for engine, q in quantities.items():
            color, style, width = ENGINE_STYLE[engine]
            y = scale * _signed_or_magnitude(name, q[name])
            ax.plot(
                series[engine].time_s, y, style, color=color, lw=width, label=engine
            )
            if engine in metrics:
                err = metrics[engine][name]
                flag = "ok" if err["passes"] else "FAIL"
                notes.append(
                    f"{engine}: peak {100 * err['peak_error']:.2f}% "
                    f"rms {100 * err['rms_error']:.2f}% {flag}"
                )
        ax.axvline(impact, color="#F87171", ls="--", lw=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(True, alpha=0.3)
        if notes:
            ax.text(
                0.01,
                0.97,
                "\n".join(notes),
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.8, "lw": 0},
            )
        if window is not None:
            ax.set_xlim(*window)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    span = "Impact Window" if window else "Full Swing"
    fig.suptitle(
        f"Grip Kinetics Parity vs OpenSim Bushing Reference: {club.capitalize()}, {span} "
        f"(tolerance peak {100 * PEAK_TOLERANCE:.0f}%, "
        f"RMS {100 * RMS_TOLERANCE:.0f}% of peak)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _error_figure(club: str, series: dict[str, GripKineticsSeries], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    ref = series["opensim"].quantities()
    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=True)
    for ax, (name, title, _unit, _scale) in zip(axes.flat, PANELS, strict=True):
        peak = float(np.linalg.norm(ref[name], axis=1).max())
        for engine, s in series.items():
            if engine == "opensim":
                continue
            color, style, width = ENGINE_STYLE[engine]
            diff = np.linalg.norm(s.quantities()[name] - ref[name], axis=1)
            ax.plot(
                s.time_s, 100 * diff / peak, style, color=color, lw=width, label=engine
            )
        ax.axvline(_impact_time(club), color="#F87171", ls="--", lw=1.0)
        ax.set_title(f"{title}: |Engine - OpenSim| / Peak", fontsize=10)
        ax.set_ylabel("% of reference peak", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Grip Kinetics Parity Error Traces: {club.capitalize()}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _gcv10_figure(club: str, series: GripKineticsSeries, path: Path) -> None:
    """The engine's series through the GCV-10 plot model and renderer."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from src.shared.python.biomechanics.grip_plot_model import (
        build_grip_plot_series,
    )
    from src.shared.python.plotting.renderers.grip_wrench import plot_grip_wrench

    plot = build_grip_plot_series(
        series.time_s.tolist(),
        series.analyses(),
        events={"impact": _impact_time(club)},
    )
    fig = plt.figure(figsize=(10, 11))
    plot_grip_wrench(fig, plot)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(club: str) -> dict[str, Any]:
    """Parity metrics JSON and overlay/error/GCV-10 plots for ``club``."""
    series = _available_series(club)
    reference = series["opensim"]
    metrics = {
        engine: {k: v.to_dict() for k, v in parity_errors(s, reference).items()}
        for engine, s in series.items()
        if engine != "opensim"
    }
    summary = {
        "club": club,
        "tolerance": {"peak": PEAK_TOLERANCE, "rms_of_peak": RMS_TOLERANCE},
        "impact_time_s": _impact_time(club),
        "engines": metrics,
        "passes": {e: all(q["passes"] for q in m.values()) for e, m in metrics.items()},
    }
    (OUT / f"metrics_{club}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    PLOTS.mkdir(parents=True, exist_ok=True)
    impact = summary["impact_time_s"]
    _overlay_figure(club, series, metrics, None, PLOTS / f"overlay_{club}.png")
    _overlay_figure(
        club,
        series,
        metrics,
        (impact - 0.12, min(impact + 0.08, float(reference.time_s[-1]))),
        PLOTS / f"overlay_{club}_impact.png",
    )
    _error_figure(club, series, PLOTS / f"error_{club}.png")
    for engine, s in series.items():
        _gcv10_figure(club, s, PLOTS / f"gcv10_{engine}_{club}.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club", choices=("driver", "iron7"), required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--accuracy", type=float, default=REFERENCE_ACCURACY)
    parser.add_argument("--engines", nargs="*", default=[])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"runs_{args.club}.json"
    runs = json.loads(log.read_text(encoding="utf-8")) if log.is_file() else {}
    if args.reference:
        runs["opensim"] = run_reference(args.club, args.accuracy)
    for engine in args.engines:
        runs[engine] = run_engine(engine, args.club)
    if args.reference or args.engines:
        log.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
        sys.stdout.write(json.dumps(runs, indent=2) + "\n")
    if args.report:
        sys.stdout.write(json.dumps(report(args.club)["passes"], indent=2) + "\n")


if __name__ == "__main__":
    main()

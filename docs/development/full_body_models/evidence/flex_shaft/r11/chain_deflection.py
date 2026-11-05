"""R11 (#12691) evidence: chain-shaft deflection through impact, per engine.

``series`` replays the impact window of a chain bundle (``same_input_bundle
export --shaft-model chain``) open loop in one engine, from the bundle's
reference state at the window start with the bundle's efforts and ball force
(as ``segmented_replay`` does), and evaluates, with that engine's own forward
kinematics, the Clubhead deflection

    d(t) = P R0^T (p(q) - p(q with the chain coordinates zeroed)),

the Clubhead displacement caused by the bent shaft, in the club body frame
of the straight shaft (shaft along -y, head at the origin): R0 is the
straight shaft's Clubhead orientation and P the Clubhead frame's rotation in
its body (spec ``frames`` placement).  ``plot`` overlays the engines.

    PYTHONPATH=.:src python3 docs/development/full_body_models/evidence/\\
flex_shaft/r11/chain_deflection.py series --bundle B.npz --engine opensim \\
        --out series_opensim.npz
    ... plot --series series_*.npz --out deflection.png

Figures are evidence of identical dynamics, not a native render: the native
viewers draw the club on the club body, so they show a straight shaft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.shared.python.contracts import require

#: Window around the bundle's impact split time; the chain run diverges at +10 ms.
WINDOW_S = (-0.08, 0.005)


def _impact_s(bundle: object) -> float:
    provenance = getattr(bundle, "provenance", {}) or {}
    value = provenance.get("impact_split_time_s")
    if value is None:
        raise ValueError("bundle provenance has no impact_split_time_s")
    return float(value)


def deflection_series(
    plant: object, q: np.ndarray, shaft: list[int], placement: np.ndarray
) -> np.ndarray:
    """``(len(q), 3)`` Clubhead deflection (m) in the straight club frame."""
    rotation = np.asarray(placement, dtype=float)[:3, :3]
    out = np.empty((q.shape[0], 3))
    for k, row in enumerate(q):
        straight = row.copy()
        straight[shaft] = 0.0
        bent = plant.kinematic_frames(row)["Clubhead"]  # type: ignore[attr-defined]
        ref = plant.kinematic_frames(straight)["Clubhead"]  # type: ignore[attr-defined]
        out[k] = rotation @ ref[:3, :3].T @ (bent[:3, 3] - ref[:3, 3])
    return out


def series(bundle_path: Path, engine: str, out: Path) -> None:
    from src.shared.python.motion_matching.club_shaft_chain import passive_coordinates
    from src.shared.python.motion_matching.same_input import (
        InputBundle,
        VectorPlant,
        open_loop,
    )

    bundle = InputBundle.load(bundle_path)
    plant = VectorPlant(engine, bundle.spec_bytes)
    spec = json.loads(bundle.spec_bytes)
    names = passive_coordinates(spec)
    placement = next(f["placement"] for f in spec["frames"] if f["name"] == "Clubhead")
    shaft = [plant.coordinate_order.index(n) for n in names]
    dt = float(bundle.dt_s)
    impact_s = _impact_s(bundle)
    first = max(0, int(round((impact_s + WINDOW_S[0]) / dt)))
    stop = min(bundle.steps, int(round((impact_s + WINDOW_S[1]) / dt)))
    ball = bundle.ball_impact()
    rollout = open_loop(
        plant,
        bundle.reference_q[first],
        bundle.reference_v[first],
        bundle.replay_efforts(stop)[first * bundle.effort_holds :],
        dt_s=dt,
        policy=bundle.step_policy(),
        impact=None if ball is None else ball.shifted(-first * dt),
    )
    q = np.asarray(rollout.q)
    reference = np.asarray(bundle.reference_q[first : stop + 1])
    np.savez_compressed(
        out,
        engine=engine,
        time_s=(first + np.arange(q.shape[0])) * dt - impact_s,
        deflection_m=deflection_series(plant, q, shaft, placement),
        shaft_q=q[:, shaft],
        max_coordinate_error_rad=float(np.abs(q - reference).max()),
    )
    print(out, "max |q - reference| =", float(np.abs(q - reference).max()))  # noqa: T201


def _to_impact(run: dict) -> np.ndarray:
    """Mask of samples up to contact; the chain run diverges just after it (R11)."""
    return np.asarray(run["time_s"]) <= 0.0


def plot(paths: list[Path], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [dict(np.load(p)) for p in paths]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    labels = ("Club x (mm)", "Club y, along the Shaft (mm)", "Club z (mm)")
    styles = {
        "mujoco": "-",
        "opensim": "--",
        "myosuite": ":",
        "drake": "-.",
        "pinocchio": (0, (5, 1, 1, 1, 1, 1)),
    }
    for run in runs:
        engine = str(run["engine"])
        for axis, ax in enumerate(axes):
            keep = _to_impact(run)
            ax.plot(
                1e3 * run["time_s"][keep],
                1e3 * run["deflection_m"][keep, axis],
                ls=styles.get(engine, "-."),
                lw=2,
                label=engine,
            )
    for ax, label in zip(axes, labels, strict=True):
        ax.axvline(0.0, color="0.6", lw=1)
        ax.set_xlabel("Time from Impact (ms)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(
        "Clubhead Deflection of the Chain Shaft, Same Input per Engine (Up to Impact)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)  # noqa: T201 - evidence script progress


#: Top-level receipt sections measured on the previous bundle only.
_PREVIOUS_RUN_KEYS = ("full_horizon_open_loop", "rigid_baseline_opensim")


def _bundle_manifest(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["manifest"]))


def update_receipt(
    paths: list[Path],
    receipt: Path,
    bundle: Path,
    code_commit: str,
    libraries: dict[str, str],
) -> None:
    """Rewrite the receipt's run-level numbers and per-engine block.

    ``paths`` are engine series files of ONE bundle (``bundle``); MuJoCo is the
    reference.  The previous headline run is kept under ``previous_run`` the
    first time, never replaced silently.  Deflection differences are scored
    from the window start to contact; coordinate errors cover the whole window
    (to +5 ms, including the post-contact divergence).
    """
    runs = {str(r["engine"]): r for r in (dict(np.load(p)) for p in paths)}
    require(
        "mujoco" in runs,
        "a mujoco series is required as the reference",
    )
    grid = runs["mujoco"]["time_s"]
    for name, run in runs.items():
        require(
            np.array_equal(run["time_s"], grid),
            f"{name} series is on another time grid than mujoco (mixed bundles?)",
        )
    keep = _to_impact(runs["mujoco"])
    ref = runs["mujoco"]["deflection_m"][keep]
    document = json.loads(receipt.read_text(encoding="utf-8"))
    window = document["impact_window"]
    if "previous_run" not in document:
        document["previous_run"] = {
            "code_commit": document["code_commit"],
            "bundle": document["bundle"],
            "impact_window": {
                k: window[k]
                for k in (
                    "max_clubhead_deflection_m",
                    "clubhead_deflection_at_impact_m",
                    "max_abs_shaft_q_rad",
                )
            },
        }
    for key in _PREVIOUS_RUN_KEYS:  # measured on the previous bundle only
        if key in document:
            document["previous_run"][key] = document.pop(key)
    provenance = _bundle_manifest(bundle)["provenance"]
    document["code_commit"] = code_commit
    document["bundle"] = {
        "command": document["bundle"]["command"],
        "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "note": "bundle hashes depend on platform and library versions",
        "steps": int(_bundle_manifest(bundle)["steps"]),
        "dt_s": float(_bundle_manifest(bundle)["dt_s"]),
        "impact_split_time_s": float(provenance["impact_split_time_s"]),
        "max_pose_drift_m": float(provenance["max_pose_drift_m"]),
        "libraries": libraries,
    }
    window["engines"] = {
        name: {
            "max_coordinate_error_vs_reference_rad": float(
                run["max_coordinate_error_rad"]
            ),
            "max_deflection_difference_vs_mujoco_m": float(
                np.abs(run["deflection_m"][keep] - ref).max()
            ),
        }
        for name, run in runs.items()
    }
    at_impact = int(np.argmin(np.abs(grid[keep])))
    window["clubhead_deflection_at_impact_m"] = [float(v) for v in ref[at_impact]]
    window["max_clubhead_deflection_m"] = float(np.linalg.norm(ref, axis=1).max())
    window["max_abs_shaft_q_rad"] = float(np.abs(runs["mujoco"]["shaft_q"][keep]).max())
    norms = np.linalg.norm(ref, axis=1)
    grid_keep = grid[keep]
    at_minus_10 = float(norms[int(np.argmin(np.abs(grid_keep + 0.010)))])
    window["engines_run_note"] = (
        "run-level numbers, engine entries and the overlay are all from "
        "`bundle` above (impact_split_time_s identical to previous_run); "
        "full_horizon_open_loop and rigid_baseline_opensim moved into "
        "previous_run because they were measured on that bundle. "
        "max_deflection_difference_vs_mujoco_m is scored from the window start "
        "to contact; max_coordinate_error_vs_reference_rad covers the whole "
        "window to +5 ms including the post-contact divergence (R11). The "
        "at-impact deflection differs from previous_run because it is read in "
        "the last milliseconds before contact, where the chain deflection grows "
        f"fast (norm {at_minus_10:.3f} m at -10 ms to {float(norms[at_impact]):.3f}"
        " m at contact in this run) and differs between exports; the cause was "
        "not isolated."
    )
    text = json.dumps(document, indent=2) + chr(10)
    receipt.write_text(text, encoding="utf-8")
    print(receipt)  # noqa: T201 - evidence script progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("series")
    one.add_argument("--bundle", type=Path, required=True)
    one.add_argument("--engine", required=True)
    one.add_argument("--out", type=Path, required=True)
    many = sub.add_parser("plot")
    many.add_argument("--series", type=Path, nargs="+", required=True)
    many.add_argument("--out", type=Path, required=True)
    upd = sub.add_parser("receipt")
    upd.add_argument("--series", type=Path, nargs="+", required=True)
    upd.add_argument("--receipt", type=Path, required=True)
    upd.add_argument("--bundle", type=Path, required=True)
    upd.add_argument("--code-commit", required=True)
    upd.add_argument("--library", action="append", default=[], metavar="NAME=VERSION")
    args = parser.parse_args()
    if args.command == "series":
        series(args.bundle, args.engine, args.out)
    elif args.command == "receipt":
        libraries = dict(item.split("=", 1) for item in args.library)
        update_receipt(
            args.series, args.receipt, args.bundle, args.code_commit, libraries
        )
    else:
        plot(args.series, args.out)


if __name__ == "__main__":
    main()

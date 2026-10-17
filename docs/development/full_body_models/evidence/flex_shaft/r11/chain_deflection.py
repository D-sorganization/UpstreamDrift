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
import json
from pathlib import Path

import numpy as np

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


def plot(paths: list[Path], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [dict(np.load(p)) for p in paths]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    labels = ("Club x (mm)", "Club y, along the Shaft (mm)", "Club z (mm)")
    styles = {"mujoco": "-", "opensim": "--", "myosuite": ":"}
    for run in runs:
        engine = str(run["engine"])
        for axis, ax in enumerate(axes):
            ax.plot(
                1e3 * run["time_s"],
                1e3 * run["deflection_m"][:, axis],
                styles.get(engine, "-."),
                lw=2,
                label=engine,
            )
    for ax, label in zip(axes, labels, strict=True):
        ax.axvline(0.0, color="0.6", lw=1)
        ax.set_xlabel("Time from Impact (ms)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Clubhead Deflection of the Chain Shaft, Same Input per Engine")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)  # noqa: T201 - evidence script progress


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
    args = parser.parse_args()
    if args.command == "series":
        series(args.bundle, args.engine, args.out)
    else:
        plot(args.series, args.out)


if __name__ == "__main__":
    main()

"""Render a MyoFullBody swing with muscles coloured by activation (issue #11646).

    MUJOCO_GL=egl MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen PYTHONPATH=.:src \
        python3 scripts/render_myofullbody_swing.py --solution driver.npz \
        --receipt driver.json --label driver --out-dir OUT

Reads the solution written by ``scripts/run_myofullbody_swing.py`` and writes, per
swing: four single-view videos (face-on, down-the-line, overhead, oblique), a 2x2
composite, ``<label>_activation.png`` and ``<label>_reserve.png``, plus a copy of
the receipt.  Videos play at 0.25x speed.  Pass ``--readme`` (after rendering
every swing) to write the README of ``--out-dir`` from the receipts there.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any

import numpy as np

from src.shared.python.myofullbody import render

logger = logging.getLogger(__name__)
FPS = 50.0
SLOWDOWN = 0.25
TENDON_WIDTH_SCALE = 1.2
PIPELINE = """export MUJOCO_GL=egl MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen PYTHONPATH=.:src
python3 scripts/run_myofullbody_swing.py --bundle driver.npz --stride 5 \\
  --receipt driver.json --solution driver.npz.solution.npz
python3 scripts/render_myofullbody_swing.py --solution driver.npz.solution.npz \\
  --receipt driver.json --label driver --out-dir OUT
python3 scripts/render_myofullbody_swing.py --out-dir OUT --readme"""
GROUP_COLOURS = {"legs": "tab:green", "trunk": "tab:orange", "arms": "tab:blue"}


def _heading(
    model: Any, data: Any, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Facing direction, target line (golfer's left, right-handed) and pelvis."""
    import mujoco

    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    body = model.body("pelvis").id
    rot = np.asarray(data.xmat[body]).reshape(3, 3)
    forward = rot @ np.array([0.0, -1.0, 0.0])  # MyoFullBody faces -y at zero pose
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    pelvis = np.asarray(data.xpos[body]).copy()
    return forward, np.cross([0.0, 0.0, 1.0], forward), pelvis


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio

    writer: Any
    with imageio.get_writer(
        path, fps=FPS, codec="libx264", quality=8, macro_block_size=None
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def render_swing(solution: dict[str, np.ndarray], out_dir: Path, label: str) -> None:
    """Four single-view videos and the 2x2 composite of one swing."""
    import mujoco

    from src.shared.python.myofullbody import assets

    tree = assets.cached_tree()
    if tree is None:
        raise SystemExit("MyoFullBody cache missing: run scripts/fetch_myofullbody.py")
    model, data = assets.load_myofullbody(tree)
    render.style_bones(model)
    model.tendon_width[:] *= TENDON_WIDTH_SCALE
    qpos, act = solution["qpos"], solution["activation"]
    forward, target, pelvis = _heading(model, data, qpos[0])
    centre = pelvis + np.array([0.0, 0.0, 0.1])
    cameras = render.view_cameras(forward, target, centre, distance=2.6)
    shown = render.select_frames(solution["time_s"], FPS, SLOWDOWN)
    renderer = mujoco.Renderer(model, 480, 640)
    views: dict[str, list[np.ndarray]] = {v: [] for v in render.VIEW_NAMES}
    for k in shown:
        for view in render.VIEW_NAMES:
            views[view].append(
                render.render_view(
                    model, data, qpos[k], act[k], cameras[view], renderer
                )
            )
    for view, frames in views.items():
        _write_video(out_dir / f"{label}_{view}.mp4", frames)
    grid = [
        np.vstack(
            [np.hstack([views[v][j] for v in render.VIEW_NAMES[:2]]),
             np.hstack([views[v][j] for v in render.VIEW_NAMES[2:]])]
        )
        for j in range(len(shown))
    ]  # fmt: skip
    _write_video(out_dir / f"{label}_four_views.mp4", grid)
    logger.info("%s: %d frames per view", label, len(shown))


def plot_activation(solution: dict[str, np.ndarray], path: Path, label: str) -> None:
    """Heat map of the 40 most active muscles over time, key frames marked."""
    import matplotlib.pyplot as plt

    act, t = solution["activation"], solution["time_s"]
    order = np.argsort(act.max(axis=0))[::-1][:40]
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(
        act[:, order].T, aspect="auto", cmap="coolwarm", vmin=0.0, vmax=1.0,
        extent=(t[0], t[-1], len(order) - 0.5, -0.5),
    )  # fmt: skip
    ax.set_yticks(
        range(len(order)), [str(solution["muscles"][i]) for i in order], fontsize=6
    )
    steps = solution["steps"]
    for name, k in zip(
        ("Address", "Top", "Impact", "Finish"), solution["key_frames"], strict=True
    ):
        x = steps.searchsorted(k)
        ax.axvline(t[min(x, len(t) - 1)], color="k", lw=0.8, ls="--")
        ax.text(t[min(x, len(t) - 1)], -1.2, name, fontsize=7, ha="center")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Muscle Activation, {label.title()} Swing (40 Most Active Muscles)")
    fig.colorbar(image, label="Activation")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reserve(solution: dict[str, np.ndarray], path: Path, label: str) -> None:
    """Reserve and spec effort magnitude per body group over time."""
    import matplotlib.pyplot as plt

    from src.shared.python.myofullbody import redundancy

    names = [str(c) for c in solution["coordinates"]]
    groups = redundancy.coordinate_groups(tuple(names))  # indices into `names`
    t = solution["time_s"]
    fig, axes = plt.subplots(len(GROUP_COLOURS), 1, figsize=(10, 8), sharex=True)
    for ax, (group, colour) in zip(axes, GROUP_COLOURS.items(), strict=True):
        idx = groups[group]
        reserve = np.linalg.norm(solution["reserve"][:, idx], axis=1)
        effort = np.linalg.norm(solution["tau"][:, idx], axis=1)
        ax.plot(t, effort, color="0.5", label="Spec effort (norm)")
        ax.plot(t, reserve, color=colour, label="Reserve actuator (norm)")
        ax.set_ylabel("N m")
        ax.set_title(group.title(), fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Reserve Actuators, {label.title()} Swing")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_readme(out_dir: Path, invocation: str) -> None:
    """README of ``out_dir`` built from the receipt copies found there."""
    lines = [
        "# Musculoskeletal Full-Body Swing Renders",
        "",
        "MyoFullBody (MyoSuite `myo_sim`, pinned commit; 416 muscles) driven by the",
        "spec swings of the same-input bundles.  Muscle colour is the solved",
        "activation, blue (0) to red (1).  Videos play at 0.25x real time.",
        "",
        "## What This Is Not",
        "",
        "- The motion is the spec skeleton's, mapped onto MyoFullBody by segment",
        "  orientation; the muscles do not generate it.",
        "- The activations come from per-frame static optimisation of the spec joint",
        "  efforts with reserve actuators.  Where the reserve is large the muscles",
        "  cannot supply the effort and the colours understate the real demand.",
        "- Colours show activation (0 to 1), not force.",
        "",
    ]
    for path in sorted(out_dir.glob("*_receipt.json")):
        r = json.loads(path.read_text())
        q = r["qualification"]
        lines += [
            f"## {path.name.split('_')[0].title()}",
            "",
            f"- Status: {q['status']}",
        ]
        lines += [f"- Reason: {why}" for why in q["reasons"]]
        for group, m in r["reserves_by_group"].items():
            lines.append(
                f"- Reserve {group}: RMS {m['reserve_rms_nm']:.1f} N m "
                f"({m['reserve_over_effort_rms']:.0%} of effort RMS), "
                f"peak {m['reserve_peak_nm']:.1f} N m"
            )
        lines += [f"- Receipt digest: `{r['receipt_digest']}`", ""]
    lines += [
        "## Invocation",
        "",
        "Per swing (run from the repository root; needs `scripts/fetch_myofullbody.py`",
        "to have filled the cache and a same-input bundle `driver.npz` or `iron.npz`):",
        "",
        "```",
        PIPELINE,
        "```",
        "",
        "README written by:",
        "",
        "```",
        invocation,
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--label", default="swing")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--readme", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    invocation = shlex.join(["python3", *sys.argv])
    if args.readme:
        write_readme(args.out_dir, invocation)
        return 0
    if args.solution is None or args.receipt is None:
        parser.error("--solution and --receipt are required unless --readme is given")
    with np.load(args.solution, allow_pickle=False) as z:
        solution = {k: z[k] for k in z.files}
    shutil.copyfile(args.receipt, args.out_dir / f"{args.label}_receipt.json")
    plot_activation(solution, args.out_dir / f"{args.label}_activation.png", args.label)
    plot_reserve(solution, args.out_dir / f"{args.label}_reserve.png", args.label)
    render_swing(solution, args.out_dir, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

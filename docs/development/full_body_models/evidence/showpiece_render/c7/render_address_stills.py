"""SHOW-C7 (#14563): address-pose stills, capture-fit grip vs neutral showpiece grip.

Takes two address-only pipeline outputs (``address_report.json`` and
``full_body_spec_hipcal_scaled.json`` each) and writes one PNG with face-on and
down-the-line projections of each: the body-frame skeleton, the club and the
hand frames (the grip roll shows in the hand frames and the zoomed inset).  PNG stills only: the run directories, any
npz and any video stay off-repo.

    python render_address_stills.py --before <capture_run> --after <showpiece_run> --out <png>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (  # noqa: E402
    export_full_body_mjcf,
)
from src.shared.python.motion_matching.address_pose import address_qpos  # noqa: E402

#: View name -> (horizontal, vertical) world axes; world Z is up.  The feet are
#: apart along Y, so Y-Z is the face-on view and X-Z the down-the-line view.
VIEWS = {"face-on (world Y-Z)": (1, 2), "down-the-line (world X-Z)": (0, 2)}
CLUB_SITES = ("native_frame_16", "native_frame_15")  # Grip, Clubhead
HAND_BODIES = {  # hand frame bodies of the lead and trail hand
    "lead": "Clubface Vector",
    "trail": "RHandStandoff",
}
AXIS_COLOURS = ("tab:red", "tab:green", "tab:blue")
AXIS_LENGTH_M = 0.06


@dataclass(frozen=True)
class Pose:
    """Geometry of one address pose, world frame, metres."""

    points: np.ndarray  # body-frame origins
    links: np.ndarray  # (parent, child) index pairs into ``points``
    club: np.ndarray  # (2, 3) grip and clubhead
    hands: dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (origin, 3x3 frame)


def _body_id(model: mujoco.MjModel, suffix: str) -> int:
    for b in range(model.nbody):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith(
            suffix
        ):
            return b
    raise ValueError(f"model has no body ending in {suffix!r}")


def pose_of(run: Path) -> Pose:
    """Address pose of an address-only run directory."""
    xml, _ = export_full_body_mjcf(
        (run / "full_body_spec_hipcal_scaled.json").read_bytes()
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = address_qpos(
        model, json.loads((run / "address_report.json").read_text())
    )
    mujoco.mj_forward(model, data)
    links = np.array(
        [(model.body_parentid[b], b) for b in range(2, model.nbody)], dtype=int
    )
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in CLUB_SITES]
    hands = {}
    for name, suffix in HAND_BODIES.items():
        b = _body_id(model, suffix)
        hands[name] = (np.array(data.xpos[b]), np.array(data.xmat[b]).reshape(3, 3))
    return Pose(np.array(data.xpos), links, np.array(data.site_xpos[sites]), hands)


def _draw(ax: plt.Axes, pose: Pose, h: int, v: int, *, detail: bool) -> None:
    """Skeleton (or just the club and hand frames when ``detail``)."""
    if not detail:
        for parent, child in pose.links:
            ax.plot(
                pose.points[[parent, child], h],
                pose.points[[parent, child], v],
                color="tab:blue",
                lw=1.2,
            )
    ax.plot(pose.club[:, h], pose.club[:, v], color="black", lw=2.5 if detail else 1.5)
    for name, (origin, frame) in pose.hands.items():
        for axis, colour in enumerate(AXIS_COLOURS):
            tip = origin + AXIS_LENGTH_M * frame[:, axis]
            ax.plot([origin[h], tip[h]], [origin[v], tip[v]], color=colour, lw=1.5)
        ax.annotate(name, (origin[h], origin[v]), fontsize=7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def render_figure(rows: dict[str, Pose], out: Path) -> Path:
    """One row per pose: both views, each with a zoomed hands inset."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 5.5 * len(rows)), squeeze=False)
    for r, (label, pose) in enumerate(rows.items()):
        for c, (view, (h, v)) in enumerate(VIEWS.items()):
            ax = axes[r, c]
            _draw(ax, pose, h, v, detail=False)
            ax.set_title(f"{label}: {view}", fontsize=9)
            ax.set_xlabel("m")
            ax.set_ylabel("m")
            inset = ax.inset_axes((0.62, 0.55, 0.36, 0.42))
            _draw(inset, pose, h, v, detail=True)
            centre = pose.club[0]
            inset.set_xlim(centre[h] - 0.3, centre[h] + 0.3)
            inset.set_ylim(centre[v] - 0.3, centre[v] + 0.3)
            inset.set_title("grip: club, hand frames (x/y/z = r/g/b)", fontsize=6)
            inset.tick_params(labelsize=6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def render(before: Path, after: Path, out: Path) -> Path:
    """Write the before/after comparison ``out`` and return it."""
    return render_figure(
        {
            "capture-fit grip (before)": pose_of(before),
            "neutral showpiece grip (after)": pose_of(after),
        },
        out,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.before, args.after, args.out))  # noqa: T201

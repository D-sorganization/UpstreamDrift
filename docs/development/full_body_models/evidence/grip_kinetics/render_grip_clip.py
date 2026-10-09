"""0.5x close-up clip of a grip series with per-hand force arrows (#11739).

Reproduce from the repository root::

    MPLBACKEND=Agg PYTHONPATH=.:src nice -n 10 python3 \\
        docs/development/full_body_models/evidence/grip_kinetics/render_grip_clip.py \\
        --club driver --variant contact

Frames are drawn with the GCV-10 overlay glyphs (``force_overlay``), the same
code as ``run_grip_kinetics.render_clip``.  Output is 1920x1080 at 60 fps; at
0.5x the clip steps 30 simulated seconds per 60 video frames, so frame ``k``
shows the series sample nearest to ``k / 120`` s.  Pad forces are not drawn per
pad; the arrows are the per-hand resultants and the discs mark the grip points.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from src.shared.python.biomechanics.grip_wrench import (  # noqa: E402
    to_overlay_wrenches,
)
from src.shared.python.force_overlay.contracts import ForceTorqueFrame  # noqa: E402
from src.shared.python.force_overlay.glyphs import (  # noqa: E402
    ForceGlyphStyle,
    build_glyphs,
)
from src.shared.python.force_overlay.renderers.matplotlib_glyphs import (  # noqa: E402
    draw_glyphs_3d,
)
from src.shared.python.grip_contact.contact_run import ContactRun  # noqa: E402
from src.shared.python.grip_contact.interface import GripInterface  # noqa: E402
from src.shared.python.grip_contact.parity import GripKineticsSeries  # noqa: E402

MODELS = ROOT / "docs/development/full_body_models"
OUT_DIR = Path("/home/dieterolson/Videos/Parity Audit/golfer_realism/grip_kinetics")
FPS = 60
SPEED = 0.5
SIZE = (1920, 1080)
COLOURS = {"L": "#1f77b4", "R": "#d62728"}


def load(club: str, variant: str) -> tuple[GripKineticsSeries, str]:
    ev = HERE
    if variant == "contact":
        return ContactRun.load_npz(ev / f"contact/mujoco_{club}_series.npz").series, (
            "Contact Grip (MuJoCo)"
        )
    if variant == "bushing":
        return GripKineticsSeries.load_npz(ev / f"parity/opensim_{club}_series.npz"), (
            "Bushing Grip (OpenSim)"
        )
    if variant == "myosuite":
        return GripKineticsSeries.load_npz(ev / f"parity/myosuite_{club}_series.npz"), (
            "Bushing Grip (MyoSuite)"
        )
    raise ValueError(f"unknown variant {variant!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--club", choices=("driver", "iron7"), required=True)
    ap.add_argument(
        "--variant", choices=("contact", "bushing", "myosuite"), required=True
    )
    ap.add_argument("--t-end", type=float, default=None)
    args = ap.parse_args()
    spec = json.loads(
        (MODELS / f"full_body_spec_anthro_{args.club}.json").read_text(encoding="utf-8")
    )
    interface = GripInterface.from_spec(spec)
    length = float(spec["club"]["length_m"])
    series, label = load(args.club, args.variant)
    analyses = series.analyses()
    t = series.time_s
    t_end = t[-1] if args.t_end is None else min(args.t_end, t[-1])
    n_frames = int(t_end * FPS / SPEED)
    index = np.minimum(
        np.searchsorted(t, np.arange(n_frames) * SPEED / FPS), t.size - 1
    )
    style = ForceGlyphStyle(force_scale_m_per_n=1.0 / 800.0, max_length_m=0.25)
    pos_r = np.asarray(interface.right.position_m, float)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.club}_{args.variant}_grip_hands_closeup_0p5x_60fps.mp4"
    ffmpeg = subprocess.run(
        [sys.executable, "-c", "import imageio_ffmpeg as i;print(i.get_ffmpeg_exe())"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{SIZE[0]}x{SIZE[1]}", "-framerate", str(FPS), "-i", "-",
            "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "18", str(path),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for i in index:
        ax.clear()
        rot = series.club_rotation[i]
        mid = 0.5 * (series.grip_point_m["L"][i] + series.grip_point_m["R"][i])
        origin = series.grip_point_m["R"][i] - rot @ pos_r
        head = origin
        butt = head + rot @ np.array([0.0, -length, 0.064])
        ax.plot(*zip(butt, head, strict=True), color="#444", lw=4)
        for side in "LR":
            ax.scatter(*series.grip_point_m[side][i], color=COLOURS[side], s=160)
        frame = ForceTorqueFrame(
            time_s=float(t[i]),
            engine=args.variant,
            wrenches=tuple(
                to_overlay_wrenches(analyses[i], source=f"{args.variant}:grip")
            ),
        )
        draw_glyphs_3d(ax, build_glyphs(frame, style))
        half = 0.3
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid, strict=True):
            setter(c - half, c + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-70)
        fl = float(np.linalg.norm(series.force_on_club_n["L"][i]))
        fr = float(np.linalg.norm(series.force_on_club_n["R"][i]))
        ax.set_title(
            f"{label}, {args.club.capitalize()}, 0.5x: t = {t[i]:.3f} s\n"
            f"L (blue) {fl:6.0f} N   R (red) {fr:6.0f} N; arrows = hand force on club",
            fontsize=16,
        )
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        assert buf.shape[:2] == (SIZE[1], SIZE[0]), buf.shape
        proc.stdin.write(buf.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    plt.close(fig)
    sys.stdout.write(f"{path} {n_frames} frames\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""SHOW-A7 (#14731): face-on native-viewer stills of the address grip, before and after.

For one spec and its address coordinates this builds a two-state same-input bundle,
runs ``python -m src.tools.native_viewer_export`` (appearance meshes, force vectors,
torque arcs and ``--segment-shading`` on) for the ``face_on`` view, takes the first
frame of the clip and draws the owner's clock inset on it: 12 o'clock is the back of
the wrist on top of the shaft, + is counterclockwise (toward the target on the left),
the green tick is the 15 deg default, red is the lead wrist and blue the trail wrist.

Run from the repo root (``MUJOCO_GL=egl`` where there is a GPU)::

    python docs/development/full_body_models/evidence/grip_roll/render_a7_stills.py \
        --spec SPEC.json --q ADDRESS.npz --tag after --out DIR

PNG only; the bundle and the clip stay in a temporary directory.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.shared.python.motion_matching.grip_roll import (  # noqa: E402
    address_grip_roll_deg,
    grip_pose_state_from_spec,
)
from src.shared.python.motion_matching.same_input.reference import (  # noqa: E402
    generate_reference_bundle,
)

WHITE = (255, 255, 255)
COLORS = {"lead": (255, 60, 60), "trail": (80, 120, 255)}
TARGET_DEG = 15.0


def address_rolls(spec_bytes: bytes, q: np.ndarray) -> dict[str, float]:
    """Lead and trail roll of ``spec_bytes`` posed at ``q`` (club frame from ``LW``)."""
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, meta = export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for value, name in zip(q, meta["coordinate_order"], strict=True):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"]["LW"])
    state = grip_pose_state_from_spec(
        json.loads(spec_bytes), data.site_xmat[site].reshape(3, 3)
    )
    lead, trail = address_grip_roll_deg(state)
    return {"lead": lead, "trail": trail}


def draw_clock(image, rolls: dict[str, float], cx: int, cy: int, r: int = 80) -> None:
    """Clock inset in the golfer's face-on view: 12 o'clock up, target to the left."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)

    def at(deg: float, scale: float = 1.0) -> tuple[float, float]:
        rad = math.radians(deg)  # 0 = 12 o'clock, + counterclockwise (to the left)
        return cx - scale * r * math.sin(rad), cy - scale * r * math.cos(rad)

    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(200, 200, 200))
    draw.line((cx - r, cy, cx + r, cy), fill=(90, 90, 90))
    draw.line((cx, cy - r, cx, cy + r), fill=(90, 90, 90))
    draw.line((at(TARGET_DEG, 0.8), at(TARGET_DEG)), fill=(60, 220, 60), width=4)
    draw.text((cx - r - 40, cy - 5), "target", fill=WHITE)
    draw.text((cx - 20, cy - r - 14), "12 = 0 deg", fill=WHITE)
    for key, rgb, width in (("lead", COLORS["lead"], 7), ("trail", COLORS["trail"], 3)):
        draw.line((cx, cy, *at(rolls[key])), fill=rgb, width=width)
    draw.text((cx - r, cy + r + 6), "wrist roll: red lead, blue trail", fill=WHITE)
    draw.text((cx - r, cy + r + 20), "green tick = 15 deg default", fill=WHITE)


class _FixedTarget:
    """Stands in for ``solve_neutral_grip.Rig``: reports one fixed club pose and closure."""

    def __init__(self, sites) -> None:
        self._sites = sites

    def sites(self, _q: np.ndarray):
        return self._sites


def to_spec_order(q: np.ndarray, order_spec: Path, spec_bytes: bytes) -> np.ndarray:
    """Address coordinates of ``spec_bytes`` (v2) holding the club pose of ``order_spec`` (v1).

    The v2 best-match model adds the thoracic and lumbar coordinates to the v1
    driver's: they start at zero (neutral), the shared coordinates are copied, and
    the arm coordinates are re-solved (``solve_neutral_grip.resolve_swing``) so the
    club pose and the grip closure equal the v1 address's, so the v2 still shows the
    same address the v1 numbers describe.
    """
    from solve_neutral_grip import Rig, resolve_swing

    src_rig = Rig(order_spec.read_bytes())
    dst_rig = Rig(spec_bytes)
    by_name = dict(zip(src_rig.order, q, strict=True))
    q_dst = np.array([by_name.get(n, 0.0) for n in dst_rig.order])
    return resolve_swing(_FixedTarget(src_rig.sites(q)), dst_rig, q_dst[None, :])[0]


def render(
    spec_path: Path,
    q_path: Path,
    tag: str,
    out: Path,
    q_spec: Path | None = None,
) -> dict[str, object]:
    """Write ``face_on_a7_<tag>.png`` and return its receipt record."""
    import imageio.v3 as iio
    from PIL import Image, ImageDraw

    spec_bytes = spec_path.read_bytes()
    q = np.load(q_path)["q"][0]
    if q_spec is not None:
        q = to_spec_order(q, q_spec, spec_bytes)
    rolls = address_rolls(spec_bytes, q)
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        bundle = generate_reference_bundle(
            spec_bytes,
            np.array([0.0, 0.01]),
            np.stack([q, q]),
            provenance={"issue": "14731", "purpose": f"address grip still ({tag})"},
        )
        bundle_path = tmp / "address.npz"
        bundle.save(bundle_path)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.tools.native_viewer_export",
                "--bundle",
                str(bundle_path),
                "--swing",
                f"addr_{tag}",
                "--club",
                "Driver",
                "--out",
                str(tmp),
                "--engines",
                "mujoco",
                "--speeds",
                "1",
                "--preset",
                "hq",
                "--no-ball",
                "--force-vectors",
                "--torque-vectors",
                "--segment-shading",
            ],
            check=True,
            cwd=ROOT,
        )
        clips = [c for c in sorted(tmp.glob("*face_on*.mp4")) if "grid" not in c.name]
        if not clips:
            raise RuntimeError("the exporter wrote no face_on clip")
        frame = next(iter(iio.imiter(clips[0])))
    image = Image.fromarray(np.asarray(frame))
    ImageDraw.Draw(image).text(
        (12, 44),
        f"address grip ({tag}): lead {rolls['lead']:+.1f} deg, "
        f"trail {rolls['trail']:+.1f} deg (0 = back of wrist up, + = counterclockwise)",
        fill=WHITE,
    )
    draw_clock(image, rolls, image.width - 120, image.height - 130)
    path = out / f"face_on_a7_{tag}.png"
    image.save(path)
    return {"tag": tag, "png": path.name, **{k: round(v, 2) for k, v in rolls.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--q", type=Path, required=True, help="npz with q[0] = address")
    parser.add_argument(
        "--q-spec", type=Path, help="spec whose coordinate order the npz uses (v1)"
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    record = render(args.spec, args.q, args.tag, args.out, args.q_spec)
    print(json.dumps(record, indent=2))  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

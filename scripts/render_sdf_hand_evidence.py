"""Write the SDF hand evidence sheet (lead, trail, both; three views; three rolls).

Stdout is the list of PNG paths, one per line. The default directory is
``docs/development/full_body_models/evidence/sdf_hand``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.shared.python.grip_spec import GripSpec, HandGripSpec  # noqa: E402
from src.shared.python.model_appearance.hand_backends.contract import (  # noqa: E402
    GripArticulation,
)
from src.shared.python.model_appearance.hand_backends.sdf_hand import (  # noqa: E402
    build_sdf_hand,
)
from src.shared.python.model_appearance.hand_backends.sdf_render import (  # noqa: E402
    cylinder_mesh,
    default_evidence_dir,
    render_meshes,
    render_standard_views,
    skin_srgb,
)
from src.shared.python.model_appearance.hand_backends.sdf_settings import (  # noqa: E402
    load_sdf_hand_settings,
)

SIZE_SKIN_VARIANTS = (
    ("small", 0.88, "skin_1"),
    ("base", 1.0, "skin_7"),
    ("large", 1.15, "skin_4"),
)


def render_size_skin_variants(output_dir: Path | None = None) -> Path:
    """Lead hand at 0.88x / 1x / 1.15x size, each in a different skin tone.

    All panels share the large hand's framing so the size change stays visible.
    """
    settings = load_sdf_hand_settings()
    base = GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=0.0))
    hands = []
    for _tag, scale, skin in SIZE_SKIN_VARIANTS:
        size = base.hand_size
        spec = replace(
            base,
            hand_size=replace(
                size, length_m=size.length_m * scale, breadth_m=size.breadth_m * scale
            ),
        )
        articulation = GripArticulation.from_grip_spec(spec, "lead")
        hands.append(
            (build_sdf_hand(spec, "lead", articulation, settings=settings), skin)
        )
    focus = hands[-1][0].mesh.vertices
    club = cylinder_mesh(
        float(focus[:, 2].min()) - 0.05,
        float(focus[:, 2].max()) + 0.05,
        hands[-1][0].grip_radius_m,
    )
    shaft = np.array([0.0, 0.0, 1.0])
    panels = []
    for hand, skin in hands:
        dorsal = np.asarray(hand.skeleton.dorsal_hint, dtype=np.float64)
        cam = dorsal * 0.72 + np.cross(dorsal, shaft) * 0.38 + shaft * 0.42
        panels.append(
            render_meshes(
                [
                    (club, np.array(settings.grip_rgb)),
                    (hand.mesh, np.array(skin_srgb(skin))),
                ],
                focus,
                cam / np.linalg.norm(cam),
                shaft,
                settings,
                settings.frame_px // 2,
            )
        )
    sheet = np.clip(np.concatenate(panels, axis=1), 0.0, 1.0)
    destination = default_evidence_dir() if output_dir is None else Path(output_dir)
    out = destination / "size_skin_variants_lead_oblique.png"
    Image.fromarray((sheet * 255).astype(np.uint8)).save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--frame-px", type=int, default=None, help="frame size")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="marching-cubes samples along the longest axis",
    )
    args = parser.parse_args()
    for path in render_standard_views(
        args.out, frame_px=args.frame_px, resolution=args.resolution
    ):
        print(path)
    print(render_size_skin_variants(args.out))


if __name__ == "__main__":
    main()

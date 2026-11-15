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

from src.shared.python.model_appearance.hand_backends.sdf_render import (  # noqa: E402
    render_standard_views,
)


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


if __name__ == "__main__":
    main()

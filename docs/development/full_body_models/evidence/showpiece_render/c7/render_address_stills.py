"""SHOW-C7 (#14563): address-pose stills, capture-fit grip vs neutral showpiece grip.

Takes two address-only pipeline outputs (``address_report.json`` and
``full_body_spec_hipcal_scaled.json`` each) and writes one PNG with two skeleton
projections (body-frame origins at the address pose) of each.  PNG stills only: the run directories, any
npz and any video stay off-repo.

    python render_address_stills.py --before <capture_run> --after <showpiece_run> --out <png>
"""

from __future__ import annotations

import argparse
import json
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

#: View name -> (horizontal, vertical) world axes of the projection; world Z is up.
VIEWS = {
    "world Y-Z (across the stance)": (1, 2),
    "world X-Z (along the stance)": (0, 2),
}


def _skeleton(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame origins at the address pose and the (parent, child) links."""
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
    return np.array(data.xpos), links


def render(before: Path, after: Path, out: Path) -> Path:
    """Write the 2x2 comparison ``out`` and return it."""
    rows = {
        "capture-fit grip (before)": _skeleton(before),
        "neutral showpiece grip (after)": _skeleton(after),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for r, (label, (points, links)) in enumerate(rows.items()):
        for c, (view, (h, v)) in enumerate(VIEWS.items()):
            ax = axes[r, c]
            for parent, child in links:
                ax.plot(
                    points[[parent, child], h],
                    points[[parent, child], v],
                    color="tab:blue",
                    lw=1.5,
                )
            ax.scatter(points[1:, h], points[1:, v], s=8, color="tab:red")
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_title(f"{label}: {view}", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.before, args.after, args.out))  # noqa: T201

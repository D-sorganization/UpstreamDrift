"""Standalone keyframe graphics from saved model/reference bundles."""

import json
from pathlib import Path
from typing import cast

import numpy as np

from .model import ReferenceMotion
from .storage import ReferenceLibrary


def render_fit_preview(bundle: Path, output: Path) -> Path:
    """Render four source-clock keyframes as PNG; no camera calibration implied.

    Preconditions: a complete fit bundle and .png output. Postcondition: saved
    model geometry and observed proxies share one fixed Z-up view and scale.
    The source C3D need not be available; the bundle retains observations.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from src.shared.python.motion_matching.diagnostics._skeleton_render import (
        draw_segments,
        equalize_3d_axes,
    )

    if output.suffix.lower() != ".png":
        raise ValueError("Reference preview output must be PNG")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    asset = ReferenceLibrary(bundle / "references").load(manifest["reference_id"])
    if not isinstance(asset, ReferenceMotion):
        raise ValueError("Fit bundle must contain a motion reference")
    points = np.asarray(asset.points_m, dtype=float)
    observed = np.load(bundle / "observed_m.npy", allow_pickle=False)
    observed = observed[..., [0, 2, 1]] * np.array([1, -1, 1])
    indices = np.linspace(0, len(points) - 1, 4).astype(int)
    figure = plt.figure(figsize=(16, 5))
    try:
        for slot, frame in enumerate(indices):
            axis = cast(Axes3D, figure.add_subplot(1, 4, slot + 1, projection="3d"))
            for edge_index, (a, b) in enumerate(asset.edges):
                draw_segments(
                    axis,
                    [points[frame, a], points[frame, b]],
                    color="#087e8b",
                    linewidth=3,
                    label="Fitted Model" if edge_index == 0 else None,
                )
            valid = observed[frame][np.isfinite(observed[frame]).all(axis=1)]
            if len(valid):
                axis.scatter(*valid.T, color="#cc5500", s=16, label="C3D Proxies")
            equalize_3d_axes(axis, points.reshape(-1, 3))
            axis.view_init(elev=15, azim=-65)
            axis.set_title(f"t = {asset.time_s[frame]:.3f} s")
            axis.set_xlabel("X (m)")
            axis.set_ylabel("Y (m)")
            axis.set_zlabel("Z (m)")
            if slot == 0:
                axis.legend(loc="upper left", fontsize=8)
        figure.suptitle(
            f"{asset.title}\nKinematic Fit — Surface Markers Approximate Joint Centers"
        )
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=140)
    finally:
        plt.close(figure)
    return output

"""Render actual saved R2025b states with shared FK and visual primitives.

Run from repository root with python -m docs.development.simscape_tour_matching.render_saved_simscape.
This is retained-state visualization, not another physics simulation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.io import loadmat

from src.engines.physics_engines.mujoco.python.native_mjcf import transform
from src.shared.python.motion_matching.visual_skeleton import (
    derive_visual_skeleton,
    skeleton_world_segments,
)
from src.tools.tour_matching_viewer.core import body_poses_from_state


def cylinder_faces(
    start: np.ndarray, end: np.ndarray, radius: float
) -> list[np.ndarray]:
    """Build drawing-only cylinder faces; physical geometry is unchanged."""
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return []
    axis = direction / length
    helper = np.eye(3)[np.argmin(abs(axis))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.linspace(0, 2 * np.pi, 13)
    ring = radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    a, b = start + ring, end + ring
    return [np.array([a[k], a[k + 1], b[k + 1], b[k]]) for k in range(12)]


def main() -> None:
    root = Path(__file__).resolve().parent
    evidence = root / "native_evidence"
    run = evidence / "two_window_fit_9967_102"
    output = root / "visuals_returned102"
    output.mkdir(exist_ok=True)
    spec_path = evidence / "native_geometry_spec_9967.json"
    candidate_path = run / "returned-candidate.json"
    mat_path = run / "qualified_candidate_replay.mat"
    spec = json.loads(spec_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    mat = loadmat(mat_path)
    with np.load(run / "returned-replay.npz") as archive:
        target, valid = archive["target_m"].copy(), archive["valid"].copy()
    time = mat["time_s"].ravel()
    q = mat["q"]
    assert q.shape == (len(time), len(candidate["coordinate_names"]))
    assert np.isfinite(q).all() and np.all(np.diff(time) > 0)
    visual = derive_visual_skeleton(spec)
    offsets = {
        j["child"]: np.linalg.inv(transform(j["child_to_follower"]))
        for j in spec["joints"]
    }
    frames = {f["name"]: f for f in spec["frames"]}
    indices = np.unique(np.r_[np.arange(0, len(time), 6), len(time) - 1])
    meshes, max_error = [], 0.0
    for i in indices:
        follower = body_poses_from_state(spec, q[i], candidate["coordinate_names"])
        physical = {body: follower[body] @ offset for body, offset in offsets.items()}
        reconstructed = []
        for name, point in zip(
            candidate["marker_bodies"], candidate["marker_offsets_m"], strict=True
        ):
            frame = frames[name]
            pose = physical[frame["body"]] @ transform(frame["placement"])
            reconstructed.append(pose[:3, :3] @ point + pose[:3, 3])
        error = np.max(
            np.linalg.norm(np.asarray(reconstructed) - mat["prediction"][i], axis=1)
        )
        max_error = max(max_error, float(error))
        if error >= 1e-4:
            raise ValueError("Saved-state visual reconstruction exceeds 0.1 mm")
        polygons, colors = [], []
        for segment in skeleton_world_segments(visual, physical):
            faces = cylinder_faces(segment.start_m, segment.end_m, segment.radius_m)
            polygons.extend(faces)
            colors.extend(["#437fbb"] * len(faces))
        meshes.append((polygons, colors))
    fig = plt.figure(figsize=(11, 5.5), facecolor="#f7f8fa")
    axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    all_points = np.concatenate(
        [mat["prediction"].reshape(-1, 3), target[valid]], axis=0
    )
    lo, hi = all_points.min(0) - 0.12, all_points.max(0) + 0.12
    title = fig.suptitle("")

    def draw(k: int) -> list:
        i = indices[k]
        for ax, azimuth in zip(axes, (-55, 35), strict=True):
            ax.clear()
            ax.add_collection3d(
                Poly3DCollection(
                    meshes[k][0],
                    facecolors=meshes[k][1],
                    edgecolors="#244764",
                    linewidths=0.15,
                    alpha=0.95,
                )
            )
            obs = target[i, valid[i]]
            ax.scatter(*obs.T, s=13, c="#df7837", label="C3D Target")
            ax.scatter(
                *mat["prediction"][i].T, s=6, c="#172d40", label="Simscape Markers"
            )
            ax.set(
                xlim=(lo[0], hi[0]),
                ylim=(lo[1], hi[1]),
                zlim=(lo[2], hi[2]),
                xlabel="X (m)",
                ylabel="Y (m)",
                zlabel="Z (m)",
            )
            ax.set_box_aspect(hi - lo)
            ax.view_init(elev=18, azim=azimuth)
        title.set_text(
            f"Actual R2025b Saved Replay — Run 102 — t={time[i]:.3f} s\nExploratory / Rejected Prefix Only: 0–0.85 s | Cylinders Are Visual Geometry"
        )
        return []

    draw(len(indices) - 1)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output / "terminal_views.png", dpi=120)
    animation = FuncAnimation(fig, draw, frames=len(indices), interval=50, blit=False)
    animation.save(
        output / "simscape_returned102_cylinders.gif",
        writer=PillowWriter(fps=20),
        dpi=90,
    )
    plt.close(fig)
    receipt = {
        "source": "actual MATLAB R2025b retained q; no new simulation",
        "run": "returned102",
        "status": "rejected prefix",
        "horizon_s": float(time[-1]),
        "playback": "approximately one-third real speed; last frame retained",
        "sample_indices": indices.tolist(),
        "max_rendered_marker_reconstruction_m": max_error,
        "visual_geometry": "shared capsules rendered as cylinders, no added legs/head joints",
        "hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (spec_path, candidate_path, mat_path, Path(__file__))
        },
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

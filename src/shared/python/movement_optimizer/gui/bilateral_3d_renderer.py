# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""3D Bilateral rendering for the animation canvas.

Renders a :class:`~movement_optimizer.models.Bilateral3DModel` pose onto
a matplotlib 3D axis as a stick figure: joints as scatter points, bones
as line segments.

The renderer is intentionally minimal -- no shading, no barbell, no trace
-- because the 3D Bilateral model is a kinematics-only MVP (see issue
#225).  Richer visualisation (plates, trajectory trace, COM marker) can
be added incrementally without changing this module's public API.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import Bilateral3DModel, Bilateral3DPose
from ..rendering import Palette


def draw_bilateral_3d_pose(
    ax: Any,
    model: Bilateral3DModel,
    pose: Bilateral3DPose,
    *,
    title: str | None = None,
) -> None:
    """Render ``pose`` on the 3D axis ``ax`` as a stick figure.

    Preconditions:
        ax is a matplotlib ``Axes3D`` (projection='3d')
        model is a ``Bilateral3DModel``
        pose is a ``Bilateral3DPose``
    """
    if not isinstance(model, Bilateral3DModel):
        raise TypeError("model must be a Bilateral3DModel instance")
    if not isinstance(pose, Bilateral3DPose):
        raise TypeError("pose must be a Bilateral3DPose instance")

    fk = model.forward_kinematics(pose)
    ax.clear()

    # Draw bones.
    for proximal, distal in model.segment_pairs():
        p0 = fk[proximal]
        p1 = fk[distal]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=Palette.SEG_COLORS[1],
            lw=4,
            solid_capstyle="round",
        )

    # Draw joints.
    xs = [p[0] for p in fk.values()]
    ys = [p[1] for p in fk.values()]
    zs = [p[2] for p in fk.values()]
    ax.scatter(xs, ys, zs, color=Palette.FG, s=30, depthshade=True, zorder=5)

    # Ground plane hint: a thin disc at z=0.
    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    r = max(0.6, 0.75 * (model.stance_width_m + 0.5))
    ax.plot(r * np.cos(theta), r * np.sin(theta), 0.0, color=Palette.FG_DIM, lw=1, alpha=0.3)

    # Reasonable default view.
    total_h = model.L_shin + model.L_thigh + model.L_torso
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(0.0, max(1.0, 1.1 * total_h))
    ax.set_xlabel("x (forward)")
    ax.set_ylabel("y (left)")
    ax.set_zlabel("z (up)")
    if title:
        ax.set_title(title, color=Palette.FG, fontsize=10, fontweight="bold")


def is_3d_axis(ax: Any) -> bool:
    """Return True if ``ax`` is a matplotlib 3D axis."""
    return getattr(ax, "name", "") == "3d"

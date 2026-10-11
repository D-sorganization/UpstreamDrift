"""Skeleton stills of one engine's FK at the pack start pose (offscreen, Agg)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CHAINS = (
    ("pelvis", "torso", "head"),
    ("torso", "upper_arm_l", "forearm_l", "hand_l"),
    ("torso", "upper_arm_r", "forearm_r", "hand_r"),
    ("pelvis", "thigh_l", "shank_l", "foot_l"),
    ("pelvis", "thigh_r", "shank_r", "foot_r"),
    ("barbell_right_sleeve", "barbell_shaft", "barbell_left_sleeve"),
)
VIEWS = (("Side (X forward, Z up)", 0, 2), ("Front (Y left, Z up)", 1, 2))


def render_still(
    positions: Mapping[str, np.ndarray], title: str, path: Path, dpi: int = 70
) -> Path:
    """Write a two-view skeleton PNG of canonical-frame *positions*.

    Raises ``ValueError`` if *positions* lacks the pelvis or the barbell shaft.
    """
    for key in ("pelvis", "barbell_shaft"):
        if key not in positions:
            raise ValueError(f"positions must contain {key!r}")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
    for ax, (label, i, j) in zip(axes, VIEWS, strict=True):
        for chain in CHAINS:
            pts = [positions[n] for n in chain if n in positions]
            if len(pts) > 1:
                bar = chain[0].startswith("barbell")
                ax.plot(
                    [p[i] for p in pts],
                    [p[j] for p in pts],
                    "-o",
                    color="tab:red" if bar else "tab:blue",
                    lw=3 if bar else 1.5,
                    ms=3,
                )
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path

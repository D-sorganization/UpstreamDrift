"""Quality checks for hand visual meshes (settings-driven tolerances)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh

_REPO = Path(__file__).resolve().parents[5]
_QUALITY_PATH = _REPO / "scripts" / "config" / "hand_visual_quality.json"


@lru_cache(maxsize=1)
def load_hand_visual_quality() -> dict[str, float]:
    if not _QUALITY_PATH.is_file():
        raise ValueError(f"missing hand visual quality settings: {_QUALITY_PATH}")
    raw = json.loads(_QUALITY_PATH.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in raw.items()}


def max_grip_cylinder_penetration_m(
    vertices: NDArray[np.float64], radius_m: float
) -> float:
    """Positive depth when any vertex lies inside the grip cylinder of ``radius_m``."""
    rho = np.hypot(vertices[:, 0], vertices[:, 1])
    return float(max(0.0, radius_m - rho.min()))


def max_self_penetration_fraction(mesh: Mesh, *, sample_stride: int = 8) -> float:
    """Heuristic: fraction of sampled vertices closer than a bone-radius gap."""
    verts = mesh.vertices[::sample_stride]
    if len(verts) < 16:
        return 0.0
    min_dists: list[float] = []
    chunk = 48
    for i in range(0, len(verts), chunk):
        block = verts[i : i + chunk]
        diff = block[:, None, :] - verts[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        finite = d[np.isfinite(d) & (d > 0.0008)]
        min_dists.append(float(finite.min()) if len(finite) else 0.01)
    hits = sum(1 for d in min_dists if d < 0.0035)
    return hits / len(min_dists)

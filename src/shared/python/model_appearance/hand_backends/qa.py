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
    """Heuristic inter-digit overlap on a dense skinned mesh (not same-surface pairs)."""
    verts = mesh.vertices[::sample_stride]
    if len(verts) < 16:
        return 0.0
    y_bins = np.quantile(verts[:, 1], [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(verts[:, 1], y_bins)
    hits = 0
    checks = 0
    gap_m = 0.0032
    min_sep_m = 0.0045
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if labels[i] == labels[j]:
                continue
            d = float(np.linalg.norm(verts[i] - verts[j]))
            if d < min_sep_m:
                continue
            checks += 1
            if d < gap_m:
                hits += 1
    if checks < 8:
        return 0.0
    return hits / checks

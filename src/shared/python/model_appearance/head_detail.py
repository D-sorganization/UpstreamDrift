"""Bounded CC0 local nose and cheek shape deltas for the existing head mesh.

The optional artistic controls are not measurements or individual identity fits.
The source and immutable asset hash are recorded with the derivation in
``assets/head_cc0/face_detail_provenance.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.shared.python.model_appearance.geometry import Mesh

_ASSET_DIR = Path(__file__).resolve().parents[4] / "assets/head_cc0"


@dataclass(frozen=True)
class HeadDetail:
    """Signed artistic nose-curve and bilateral cheekbone controls in [-1, 1]."""

    nose_curve: float = 0.0
    cheek_bones: float = 0.0

    def __post_init__(self) -> None:
        """Reject non-finite, nonnumeric or unsupported target weights."""
        for value in (self.nose_curve, self.cheek_bones):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("face detail values must be finite and within [-1, 1]")
            try:
                valid = math.isfinite(value) and abs(value) <= 1
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("face detail values must be finite and within [-1, 1]")


@lru_cache(maxsize=1)
def _targets() -> dict[str, np.ndarray]:
    """Load pinned CC0 target deltas and fail closed on provenance mismatch."""
    path = _ASSET_DIR / "face_detail_cc0.npz"
    provenance = json.loads((_ASSET_DIR / "face_detail_provenance.json").read_text())
    if provenance.get("licence") != "CC0-1.0" or hashlib.sha256(
        path.read_bytes()
    ).hexdigest() != provenance.get("asset_sha256"):
        raise ValueError("face detail asset provenance mismatch")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].astype(float) for name in data.files}


def apply_detail(skin: Mesh, detail: HeadDetail) -> Mesh:
    """Apply shape deltas to the head surface while retaining its topology."""
    if detail == HeadDetail():
        return skin
    targets = _targets()
    vertices = skin.vertices.copy()
    operations = (
        (detail.nose_curve, "nose-curve-convex", "nose-curve-concave"),
        (detail.cheek_bones, "l-cheek-bones-incr", "l-cheek-bones-decr"),
        (detail.cheek_bones, "r-cheek-bones-incr", "r-cheek-bones-decr"),
    )
    for weight, positive, negative in operations:
        if weight:
            delta = targets[positive if weight > 0 else negative]
            if delta.shape != vertices.shape or not np.isfinite(delta).all():
                raise ValueError("face detail target does not match the head mesh")
            vertices += abs(weight) * delta
    return Mesh(vertices, skin.faces)

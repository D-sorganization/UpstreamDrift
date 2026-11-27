"""CC0 hand asset loader (MakeHuman mesh, skeleton and weights; epic #15150)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.shared.python.model_appearance.hand_backends.skinning import SkinnedHandAsset

REPO_ROOT = Path(__file__).resolve().parents[5]
ASSET_DIR = REPO_ROOT / "assets" / "hand_cc0"
ASSET_PATH = ASSET_DIR / "hand_cc0.npz"
PROVENANCE_PATH = ASSET_DIR / "PROVENANCE.json"
LICENCE_ID = "CC0-1.0"


@dataclass(frozen=True)
class HandAssetRecord:
    asset: SkinnedHandAsset
    sha256: str


def _load_npz(path: Path) -> SkinnedHandAsset:
    with np.load(path, allow_pickle=False) as data:
        return SkinnedHandAsset(
            rest_vertices=data["vertices"].astype(np.float64),
            faces=data["faces"].astype(np.int64),
            joint_indices=data["joint_indices"].astype(np.int64),
            joint_weights=data["joint_weights"].astype(np.float64),
            bone_names=tuple(str(x) for x in data["bone_names"].tolist()),
            bone_parents=data["bone_parents"].astype(np.int64),
            bone_heads=data["bone_heads"].astype(np.float64),
            bone_tails=data["bone_tails"].astype(np.float64),
            flex_axes=data["flex_axes"].astype(np.float64),
            rest_flexion_rad=data["rest_flexion_rad"].astype(np.float64),
        )


@lru_cache(maxsize=1)
def load_hand_cc0_asset() -> HandAssetRecord:
    """Load the committed asset. Raises ``ValueError`` if it is missing, is not
    recorded as CC0, or does not match the hash in ``PROVENANCE.json``."""
    if not PROVENANCE_PATH.is_file():
        raise ValueError(f"hand provenance missing: {PROVENANCE_PATH}")
    if not ASSET_PATH.is_file():
        raise ValueError(f"hand asset missing: {ASSET_PATH}")
    record = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if record.get("licence") != LICENCE_ID:
        raise ValueError(f"hand licence must be {LICENCE_ID}")
    digest = hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest()
    if digest != record.get("asset_sha256"):
        raise ValueError("hand asset sha256 does not match provenance")
    asset = _load_npz(ASSET_PATH)
    weights = asset.joint_weights
    asset = SkinnedHandAsset(
        **{
            **asset.__dict__,
            "joint_weights": weights / weights.sum(axis=1, keepdims=True),
        }
    )
    return HandAssetRecord(asset=asset, sha256=digest)

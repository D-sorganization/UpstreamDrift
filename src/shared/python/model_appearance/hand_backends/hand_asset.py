"""CC0 hand mesh asset loader (MakeHuman derivative, epic #15122)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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


def _load_npz(path: Path, record: dict[str, object]) -> SkinnedHandAsset:
    with np.load(path, allow_pickle=False) as data:
        quantum = float(record["quantum_metres"])
        verts = data["vertices"].astype(np.float64) * quantum
        faces = data["faces"].astype(np.int64)
        joint_indices = data["joint_indices"].astype(np.uint16)
        joint_weights = data["joint_weights"].astype(np.float32)
        rest_poses = data["rest_poses"].astype(np.float64)
        rest_poses[:, :3, 3] *= quantum
        bone_names = tuple(str(x) for x in np.asarray(data["bone_names"]).tolist())
        palm_ids = data["palm_arch_vertex_ids"].astype(np.int64)
        grip_ids = data["grip_bind_vertex_ids"].astype(np.int64)
    return SkinnedHandAsset(
        rest_vertices=verts,
        faces=faces,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
        rest_poses=rest_poses,
        bone_names=bone_names,
        palm_arch_vertex_ids=palm_ids,
        grip_bind_vertex_ids=grip_ids,
    )


@lru_cache(maxsize=1)
def load_hand_cc0_asset() -> HandAssetRecord:
    if not PROVENANCE_PATH.is_file():
        raise ValueError(f"hand provenance missing: {PROVENANCE_PATH}")
    if not ASSET_PATH.is_file():
        raise ValueError(f"hand asset missing: {ASSET_PATH}")
    record = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if record.get("licence") != LICENCE_ID:
        raise ValueError(f"hand licence must be {LICENCE_ID}")
    raw = ASSET_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record.get("asset_sha256"):
        raise ValueError("hand asset sha256 does not match provenance")
    asset = _load_npz(ASSET_PATH, record)
    return HandAssetRecord(asset=asset, sha256=digest)

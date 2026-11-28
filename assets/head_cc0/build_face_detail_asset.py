#!/usr/bin/env python3
"""Derive optional CC0 face deltas: python build_face_detail_asset.py INPUT_DIR.

INPUT_DIR contains the pinned base.obj and the six pinned .target files named
below. All inputs are MakeHuman bundled CC0 assets; see face_detail_provenance.json.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASSET_DIR.parent / "makehuman_cc0"))
from mh_rig_io import read_obj_groups  # noqa: E402
from mh_targets import read_target  # noqa: E402
from build_head_asset import NECK_JOINT, build, triangulate  # noqa: E402

BASE_HASH = "8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c"
TARGETS = {
    "nose-curve-convex": "3426e9a53c65962bc75b25aae04759a03e818d708bc5445ffc4dfb812497a7ca",
    "nose-curve-concave": "30dd70cfdf1ce1c77eabff12b49d238b05a3b36f95f2ba6f64d4b5b059292e58",
    "l-cheek-bones-incr": "244ffa39d2d2db94ff15130719346b4cae2236bf2a309c5682eac352da06be50",
    "r-cheek-bones-incr": "a393f67acce21568ab832091daadab267685d464492647cb0c4e8cb5af58c4c9",
    "l-cheek-bones-decr": "0da776725aabc2cbc602d805e50d0dfbc98b0519a0d65f21b7e6f133562e56f9",
    "r-cheek-bones-decr": "a336b46fe613d85763fb01ca79758039b15daef1930600e36dc420e799b683d4",
}


def checked(path: Path, expected: str) -> None:
    """Reject any source differing from the pinned CC0 source bytes."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Unexpected source hash for {path.name}")


def main(source: Path) -> None:
    """Map original MakeHuman vertex deltas into canonical head coordinates."""
    obj = source / "base.obj"
    checked(obj, BASE_HASH)
    vertices, groups = read_obj_groups(obj)
    faces = [f for f in groups["body"] if vertices[f][:, 1].mean() >= NECK_JOINT[1]]
    used = np.unique(triangulate(faces))
    top = vertices[used, 1].max() - NECK_JOINT[1]
    with np.load(ASSET_DIR / "head_cc0.npz", allow_pickle=False) as original:
        rebuilt = build(obj)
        if not all(np.array_equal(rebuilt[k], original[k]) for k in rebuilt):
            raise ValueError("Pinned base does not reconstruct this head asset")
    ids = {int(v): i for i, v in enumerate(used)}
    arrays = {}
    for name, digest in TARGETS.items():
        path = source / f"{name}.target"
        checked(path, digest)
        target = read_target(path)
        delta = np.zeros((len(used) + 1, 3), dtype=np.float32)
        for index, xyz in zip(target.indices, target.deltas, strict=True):
            if int(index) in ids:
                delta[ids[int(index)]] = xyz[[2, 0, 1]] / top
        arrays[name] = delta
    np.savez_compressed(ASSET_DIR / "face_detail_cc0.npz", **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

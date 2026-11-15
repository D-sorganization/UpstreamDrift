"""Load MakeHuman CC0 rig joints and weights (build-time only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HAND_BONES = (
    "wrist.R",
    "metacarpal1.R",
    "metacarpal2.R",
    "metacarpal3.R",
    "metacarpal4.R",
    "finger1-1.R",
    "finger1-2.R",
    "finger1-3.R",
    "finger2-1.R",
    "finger2-2.R",
    "finger2-3.R",
    "finger3-1.R",
    "finger3-2.R",
    "finger3-3.R",
    "finger4-1.R",
    "finger4-2.R",
    "finger4-3.R",
    "finger5-1.R",
    "finger5-2.R",
    "finger5-3.R",
)


def load_mhskel(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mhw_weights(path: Path) -> dict[str, list[list[float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("weights", {})
    return {k: [list(pair) for pair in v] for k, v in raw.items()}


def joint_position(full_verts: np.ndarray, joints: dict, name: str) -> np.ndarray:
    idxs = joints.get(name)
    if not idxs:
        raise ValueError(f"missing joint {name}")
    pts = full_verts[np.asarray(idxs, dtype=np.int64)]
    return pts.mean(axis=0)


def build_hand_weights(
    mhw: dict[str, list[list[float]]],
    bone_names: tuple[str, ...],
    local_index_of_global: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pack up to 4 bone influences per hand vertex from MH default weights."""
    n_local = len(local_index_of_global)
    accum: list[list[tuple[float, int]]] = [[] for _ in range(n_local)]
    bone_to_id = {n: i for i, n in enumerate(bone_names)}
    for bone in bone_names:
        if bone not in mhw:
            continue
        bid = bone_to_id[bone]
        for g_idx, w in mhw[bone]:
            li = local_index_of_global.get(int(g_idx))
            if li is None or w <= 1e-6:
                continue
            accum[li].append((float(w), bid))
    idx = np.zeros((n_local, 4), dtype=np.uint16)
    w = np.zeros((n_local, 4), dtype=np.float32)
    for vi, pairs in enumerate(accum):
        if not pairs:
            w[vi, 0] = 1.0
            continue
        pairs.sort(key=lambda t: -t[0])
        for k, (wt, bid) in enumerate(pairs[:4]):
            idx[vi, k] = bid
            w[vi, k] = wt
        w[vi] /= w[vi].sum()
    return idx, w
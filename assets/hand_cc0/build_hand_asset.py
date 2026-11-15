#!/usr/bin/env python3
"""Derive the CC0 right-hand skinned asset from MakeHuman base.obj (epic #15122).

Usage::

    python3 assets/hand_cc0/build_hand_asset.py PATH/TO/base.obj [--check]

Input: MakeHuman ``hm08`` ``base.obj`` (CC0, see ``PROVENANCE.json``).
Output: ``hand_cc0.npz`` with quantised vertices and bone rest poses in the
hand-grip frame (``z`` along fingers toward tips, ``x`` toward the palm side of
the grip cylinder, ``y`` completes a right-handed frame).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

QUANTUM = 5e-5
OUT = Path(__file__).with_name("hand_cc0.npz")
PROVENANCE = Path(__file__).with_name("PROVENANCE.json")

BONE_NAMES = (
    "wrist",
    "meta_index",
    "prox_index",
    "mid_index",
    "dist_index",
    "meta_middle",
    "prox_middle",
    "mid_middle",
    "dist_middle",
    "meta_ring",
    "prox_ring",
    "mid_ring",
    "dist_ring",
    "meta_little",
    "prox_little",
    "mid_little",
    "dist_little",
    "thumb_meta",
    "thumb_prox",
    "thumb_dist",
    "palm_cup",
)

# MakeHuman finger index: 1 thumb, 2 index, ..., 5 little
_MH_FINGER = {2: "index", 3: "middle", 4: "ring", 5: "little", 1: "thumb"}


def read_obj(path: Path) -> tuple[np.ndarray, dict[str, list[list[int]]]]:
    verts: list[list[float]] = []
    groups: dict[str, list[list[int]]] = {}
    current = "body"
    groups[current] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("g "):
            current = line.split()[1]
            groups.setdefault(current, [])
        elif line.startswith("f "):
            groups.setdefault(current, []).append(
                [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            )
    return np.asarray(verts, dtype=float), groups


def triangulate(quads: list[list[int]]) -> np.ndarray:
    tris = []
    for f in quads:
        for k in range(1, len(f) - 1):
            tris.append([f[0], f[k], f[k + 1]])
    return np.asarray(tris, dtype=np.int64)


def joint_centroid(
    verts: np.ndarray, groups: dict[str, list[list[int]]], name: str
) -> np.ndarray:
    faces = groups.get(name, [])
    if not faces:
        return np.zeros(3)
    tris = triangulate(faces)
    tri_centres = verts[tris].mean(axis=1)
    return tri_centres.mean(axis=0)


def mh_to_grip(p: np.ndarray) -> np.ndarray:
    """MakeHuman (y up, z forward, +x left) -> hand-grip frame."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, 3)
    x_mh, y_mh, z_mh = p[:, 0], p[:, 1], p[:, 2]
    length = 0.19  # target hand length scale
    scale = length / 0.25
    # Origin at wrist; y_mh up -> -z_grip along fingers; x_mh -> -y_grip; palm normal ~ -x_grip
    x_g = (-y_mh + 5.55) * scale
    y_g = (-x_mh) * scale
    z_g = (-z_mh - 0.05) * scale
    return np.column_stack([x_g, y_g, z_g])


def extract_hand_mesh(
    verts: np.ndarray, groups: dict[str, list[list[int]]]
) -> tuple[np.ndarray, np.ndarray]:
    body_faces = groups.get("body", [])
    tris = triangulate(body_faces)
    cen = verts[tris].mean(axis=1)
    # Right hand in MH: x in [-0.22, -0.03], y (height) 5.18–6.12, z forward of torso.
    mask = (cen[:, 0] < -0.02) & (cen[:, 1] > 5.35) & (cen[:, 1] < 6.05)
    hand_tris = tris[mask]
    used = np.unique(hand_tris)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    keep_mh = verts[used, 0] < -0.08
    remap3 = -np.ones(len(used), dtype=np.int64)
    remap3[np.where(keep_mh)[0]] = np.arange(int(keep_mh.sum()))
    used2 = used[keep_mh]
    hand_v = mh_to_grip(verts[used2])
    hand_f = remap[hand_tris]
    valid_tris = keep_mh[hand_f].all(axis=1)
    hand_f = remap3[hand_f[valid_tris]]
    if len(hand_f) > 5500:
        step = max(1, len(hand_f) // 4500)
        hand_f = hand_f[::step]
    return hand_v, hand_f


def bone_rest_positions(
    verts: np.ndarray, groups: dict[str, list[list[int]]]
) -> dict[str, np.ndarray]:
    wrist = joint_centroid(verts, groups, "joint-r-hand")
    wrist_g = mh_to_grip(wrist.reshape(1, -1))[0]
    positions: dict[str, np.ndarray] = {"wrist": wrist_g}
    for mh_idx, finger in _MH_FINGER.items():
        if finger == "thumb":
            j1 = mh_to_grip(
                joint_centroid(verts, groups, f"joint-r-finger-{mh_idx}-1").reshape(
                    1, -1
                )
            )[0]
            j2 = mh_to_grip(
                joint_centroid(verts, groups, f"joint-r-finger-{mh_idx}-2").reshape(
                    1, -1
                )
            )[0]
            j3 = mh_to_grip(
                joint_centroid(verts, groups, f"joint-r-finger-{mh_idx}-3").reshape(
                    1, -1
                )
            )[0]
            positions["thumb_meta"] = j1
            positions["thumb_prox"] = j2
            positions["thumb_dist"] = j3
            continue
        for seg, suffix in ((1, "meta"), (2, "prox"), (3, "mid"), (4, "dist")):
            name = f"joint-r-finger-{mh_idx}-{seg}"
            pos = mh_to_grip(joint_centroid(verts, groups, name).reshape(1, -1))[0]
            positions[f"{suffix}_{finger}"] = pos
    palm = wrist_g + np.array([-0.02, 0.0, 0.04])
    positions["palm_cup"] = palm
    return positions


def assign_weights(
    hand_v: np.ndarray, positions: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    n = len(hand_v)
    idx = np.zeros((n, 4), dtype=np.uint16)
    w = np.zeros((n, 4), dtype=np.float32)
    bone_ids = {name: i for i, name in enumerate(BONE_NAMES)}
    for i, v in enumerate(hand_v):
        dists = []
        for name in BONE_NAMES:
            if name not in positions:
                continue
            dists.append((np.linalg.norm(v - positions[name]), bone_ids[name]))
        dists.sort(key=lambda t: t[0])
        for k in range(min(4, len(dists))):
            d, bid = dists[k]
            idx[i, k] = bid
            w[i, k] = 1.0 / (d + 1e-4)
        w[i] /= w[i].sum()
    return idx, w


def rest_poses(positions: dict[str, np.ndarray]) -> np.ndarray:
    mats = np.zeros((len(BONE_NAMES), 4, 4))
    for i, name in enumerate(BONE_NAMES):
        mats[i] = np.eye(4)
        if name in positions:
            mats[i, :3, 3] = positions[name]
    return mats


def palm_and_grip_vertex_ids(
    hand_v: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    palm = np.where((hand_v[:, 0] < 0.01) & (hand_v[:, 2] < 0.08))[0]
    grip = np.where(
        (hand_v[:, 0] < -0.015) & (np.hypot(hand_v[:, 0], hand_v[:, 1]) < 0.095)
    )[0]
    return palm.astype(np.int64), grip.astype(np.int64)


def _normalize_hand_frame(
    hand_v: np.ndarray, hand_f: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    wrist = hand_v[np.argmin(hand_v[:, 2])].copy()
    hand_v = hand_v - wrist
    hand_v[:, 2] -= hand_v[:, 2].min()
    span = float(hand_v[:, 2].max())
    scale = 0.19 / span if span > 1e-6 else 1.0
    hand_v *= scale
    keep = (np.abs(hand_v[:, 1]) < 0.11) & (hand_v[:, 0] < 0.09)
    remap = -np.ones(len(hand_v), dtype=np.int64)
    remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
    hand_v = hand_v[keep]
    valid = keep[hand_f].all(axis=1)
    hand_f = remap[hand_f[valid]]
    return hand_v, hand_f, wrist, scale


def build(base_obj: Path) -> dict[str, np.ndarray]:
    verts, groups = read_obj(base_obj)
    hand_v, hand_f = extract_hand_mesh(verts, groups)
    hand_v, hand_f, wrist_anchor, scale = _normalize_hand_frame(hand_v, hand_f)
    raw_positions = bone_rest_positions(verts, groups)
    positions: dict[str, np.ndarray] = {}
    for name, raw in raw_positions.items():
        grip = mh_to_grip(raw.reshape(1, -1))[0]
        positions[name] = (grip - wrist_anchor) * scale
    positions["wrist"] = np.zeros(3)
    idx, w = assign_weights(hand_v, positions)
    poses = rest_poses(positions)
    palm_ids, grip_ids = palm_and_grip_vertex_ids(hand_v, hand_f)
    qv = np.round(hand_v / QUANTUM).astype(np.int32)
    qp = np.round(poses[:, :3, 3] / QUANTUM).astype(np.int32)
    poses_q = np.tile(np.eye(4), (len(BONE_NAMES), 1, 1))
    poses_q[:, :3, 3] = qp
    return {
        "vertices": qv,
        "faces": hand_f.astype(np.int32),
        "joint_indices": idx.astype(np.uint16),
        "joint_weights": w.astype(np.float32),
        "rest_poses": poses_q.astype(np.int32),
        "bone_names": np.asarray(
            BONE_NAMES, dtype=f"U{max(len(n) for n in BONE_NAMES)}"
        ),
        "palm_arch_vertex_ids": palm_ids.astype(np.int32),
        "grip_bind_vertex_ids": grip_ids.astype(np.int32),
    }


def write_provenance(digest: str) -> None:
    record = {
        "licence": "CC0-1.0",
        "source_name": "MakeHuman base mesh hm08 (hand extract)",
        "source_url": "https://github.com/makehumancommunity/makehuman/blob/3c701a8e52f09e69922e8b598d23be2d7dfc49e3/makehuman/data/3dobjs/base.obj",
        "quantum_metres": QUANTUM,
        "asset_sha256": digest,
        "research_only_alternatives": (
            "MANO and SMPL-X hand models are research/non-commercial licences "
            "and are not bundled; they remain research-only alternatives."
        ),
    }
    PROVENANCE.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_obj", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    arrays = build(args.base_obj)
    if args.check and OUT.is_file():
        with np.load(OUT, allow_pickle=True) as existing:
            for key in arrays:
                if key == "bone_names":
                    continue
                np.testing.assert_array_equal(arrays[key], existing[key])
        print("hand_cc0.npz unchanged")  # noqa: T201
        return
    np.savez_compressed(OUT, **arrays)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    write_provenance(digest)
    vol = float(
        np.einsum(
            "ij,ij->",
            arrays["vertices"][arrays["faces"][:, 0]] * QUANTUM,
            np.cross(
                arrays["vertices"][arrays["faces"][:, 1]] * QUANTUM
                - arrays["vertices"][arrays["faces"][:, 0]] * QUANTUM,
                arrays["vertices"][arrays["faces"][:, 2]] * QUANTUM
                - arrays["vertices"][arrays["faces"][:, 0]] * QUANTUM,
            ),
        )
        / 6.0
    )
    print(  # noqa: T201
        f"wrote {OUT} faces={len(arrays['faces'])} volume~{vol:.6f} sha256={digest[:12]}…"
    )


if __name__ == "__main__":
    main()

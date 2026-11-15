#!/usr/bin/env python3
"""Derive the CC0 right-hand skinned asset from MakeHuman base.obj (epic #15122).

Usage::

    python3 assets/hand_cc0/build_hand_asset.py PATH/TO/base.obj [--check]

Input: MakeHuman ``hm08`` ``base.obj`` (CC0, see ``PROVENANCE.json`` and ``NOTICE``).
Output: ``hand_cc0.npz`` with quantised vertices and bone rest poses in the
hand-grip frame (``z`` along fingers toward tips, ``y`` across knuckles,
``x`` toward the palm / grip contact side; grip cylinder ``x²+y²=R²`` along ``z``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

QUANTUM = 5e-5
OUT = Path(__file__).with_name("hand_cc0.npz")
PROVENANCE = Path(__file__).with_name("PROVENANCE.json")
NOTICE = Path(__file__).with_name("NOTICE")

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
_TARGET_LENGTH_M = 0.19
_MAX_FACES = 8500


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
) -> np.ndarray | None:
    faces = groups.get(name, [])
    if not faces:
        return None
    tris = triangulate(faces)
    tri_centres = verts[tris].mean(axis=1)
    return tri_centres.mean(axis=0)


def _hand_joint_cloud(
    verts: np.ndarray, groups: dict[str, list[list[int]]]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    joints: dict[str, np.ndarray] = {}
    wrist = joint_centroid(verts, groups, "joint-r-hand")
    if wrist is None:
        raise ValueError("base.obj missing joint-r-hand")
    joints["wrist"] = wrist
    for mh_idx in range(1, 6):
        for seg in range(1, 5):
            name = f"joint-r-finger-{mh_idx}-{seg}"
            c = joint_centroid(verts, groups, name)
            if c is not None:
                joints[name] = c
    pts = np.vstack(list(joints.values()))
    return pts, joints


def hand_grip_basis(joints: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal hand-grip axes (x palm toward club, y spread, z along fingers)."""
    wrist = joints["wrist"]
    tips = [
        joints[k]
        for k in joints
        if k.endswith("-4") or k == "joint-r-finger-1-3"
    ]
    tip_mean = np.mean(np.vstack(tips), axis=0)
    z_hat = _unit(tip_mean - wrist)
    idx = joints.get("joint-r-finger-2-1")
    lit = joints.get("joint-r-finger-5-1")
    if idx is None or lit is None:
        raise ValueError("missing index/little MCP joints")
    y_hat = _unit(idx - lit)
    x_hat = _unit(np.cross(y_hat, z_hat))
    y_hat = _unit(np.cross(z_hat, x_hat))
    # Palm bulk sits on -x for the right hand in MakeHuman; flip if needed.
    palm_probe = joints.get("joint-r-finger-3-2", tip_mean)
    if np.dot(palm_probe - wrist, x_hat) > 0.0:
        x_hat *= -1.0
        y_hat *= -1.0
    basis = np.column_stack([x_hat, y_hat, z_hat])
    return wrist, basis


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("degenerate vector")
    return v / n


def mh_to_grip_points(
    points: np.ndarray, wrist: np.ndarray, basis: np.ndarray, scale: float
) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, 3)
    local = (p - wrist) @ basis
    return local * scale


def extract_hand_mesh(
    verts: np.ndarray,
    groups: dict[str, list[list[int]]],
    joint_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    body_faces = groups.get("body", [])
    if not body_faces:
        raise ValueError("base.obj missing body group")
    tris = triangulate(body_faces)
    cen = verts[tris].mean(axis=1)
    pad = np.array([0.045, 0.055, 0.055])
    mn = joint_pts.min(axis=0) - pad
    mx = joint_pts.max(axis=0) + pad
    mask = (cen >= mn).all(axis=1) & (cen <= mx).all(axis=1)
    hand_tris = tris[mask]
    used = np.unique(hand_tris)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    hand_v_mh = verts[used]
    hand_f = remap[hand_tris]
    return hand_v_mh, hand_f, used


def largest_connected_component(
    hand_v: np.ndarray, hand_f: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop wrist-disconnected floater geometry."""
    n_v = len(hand_v)
    adj: list[set[int]] = [set() for _ in range(n_v)]
    for a, b, c in hand_f:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    wrist_seed = int(np.argmin(np.linalg.norm(hand_v, axis=1)))
    seen = {wrist_seed}
    stack = [wrist_seed]
    while stack:
        v = stack.pop()
        for nb in adj[v]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    keep = np.zeros(n_v, dtype=bool)
    keep[list(seen)] = True
    remap = -np.ones(n_v, dtype=np.int64)
    remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
    hand_v = hand_v[keep]
    valid = keep[hand_f].all(axis=1)
    hand_f = remap[hand_f[valid]]
    return hand_v, hand_f


def bone_rest_positions(
    joints: dict[str, np.ndarray],
    wrist: np.ndarray,
    basis: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {
        "wrist": np.zeros(3, dtype=float),
    }
    for mh_idx, finger in _MH_FINGER.items():
        if finger == "thumb":
            for seg, suffix in ((1, "thumb_meta"), (2, "thumb_prox"), (3, "thumb_dist")):
                key = f"joint-r-finger-1-{seg}"
                if key in joints:
                    positions[suffix] = mh_to_grip_points(
                        joints[key], wrist, basis, scale
                    )[0]
            continue
        for seg, suffix in (
            (1, f"meta_{finger}"),
            (2, f"prox_{finger}"),
            (3, f"mid_{finger}"),
            (4, f"dist_{finger}"),
        ):
            key = f"joint-r-finger-{mh_idx}-{seg}"
            if key in joints:
                positions[suffix] = mh_to_grip_points(joints[key], wrist, basis, scale)[
                    0
                ]
    palm = mh_to_grip_points(
        0.5 * (joints.get("joint-r-finger-2-1", wrist) + wrist),
        wrist,
        basis,
        scale,
    )[0]
    palm[0] -= 0.012
    positions["palm_cup"] = palm
    return positions


def _bone_segment_closest(
    point: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[float, np.ndarray]:
    ab = b - a
    t = float(np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-12))
    t = float(np.clip(t, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest)), closest


def assign_weights(
    hand_v: np.ndarray, positions: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    n = len(hand_v)
    idx = np.zeros((n, 4), dtype=np.uint16)
    w = np.zeros((n, 4), dtype=np.float32)
    bone_ids = {name: i for i, name in enumerate(BONE_NAMES)}
    chains: list[tuple[str, str]] = [
        ("meta_index", "prox_index"),
        ("prox_index", "mid_index"),
        ("mid_index", "dist_index"),
        ("meta_middle", "prox_middle"),
        ("prox_middle", "mid_middle"),
        ("mid_middle", "dist_middle"),
        ("meta_ring", "prox_ring"),
        ("prox_ring", "mid_ring"),
        ("mid_ring", "dist_ring"),
        ("meta_little", "prox_little"),
        ("prox_little", "mid_little"),
        ("mid_little", "dist_little"),
        ("thumb_meta", "thumb_prox"),
        ("thumb_prox", "thumb_dist"),
        ("wrist", "meta_middle"),
        ("wrist", "palm_cup"),
    ]
    for i, v in enumerate(hand_v):
        dists: list[tuple[float, int]] = []
        for name, pos in positions.items():
            if name not in bone_ids:
                continue
            d = float(np.linalg.norm(v - pos))
            dists.append((d, bone_ids[name]))
        for a_name, b_name in chains:
            if a_name not in positions or b_name not in positions:
                continue
            d, _ = _bone_segment_closest(v, positions[a_name], positions[b_name])
            dists.append((d, bone_ids[a_name]))
            dists.append((d * 1.02, bone_ids[b_name]))
        dists.sort(key=lambda t: t[0])
        pick: list[tuple[float, int]] = []
        seen: set[int] = set()
        for d, bid in dists:
            if bid in seen:
                continue
            seen.add(bid)
            pick.append((d, bid))
            if len(pick) >= 4:
                break
        for k, (d, bid) in enumerate(pick):
            idx[i, k] = bid
            w[i, k] = 1.0 / (d + 2e-4)
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
    hand_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rho = np.hypot(hand_v[:, 0], hand_v[:, 1])
    palm = np.where((hand_v[:, 0] < 0.008) & (hand_v[:, 2] > -0.02))[0]
    grip = np.where((hand_v[:, 0] < 0.02) & (rho < 0.11) & (hand_v[:, 2] < 0.12))[0]
    return palm.astype(np.int64), grip.astype(np.int64)


def _mesh_scale_factor(joints: dict[str, np.ndarray], wrist: np.ndarray) -> float:
    tip = joints.get("joint-r-finger-3-4")
    if tip is None:
        raise ValueError("missing middle fingertip joint")
    length_mh = float(np.linalg.norm(tip - wrist))
    return _TARGET_LENGTH_M / length_mh


def build(base_obj: Path) -> dict[str, np.ndarray]:
    verts, groups = read_obj(base_obj)
    joint_pts, joints = _hand_joint_cloud(verts, groups)
    wrist, basis = hand_grip_basis(joints)
    scale = _mesh_scale_factor(joints, wrist)
    hand_v_mh, hand_f, _ = extract_hand_mesh(verts, groups, joint_pts)
    hand_v_mh, hand_f = largest_connected_component(hand_v_mh, hand_f)
    hand_v = mh_to_grip_points(hand_v_mh, wrist, basis, scale)
    positions = bone_rest_positions(joints, wrist, basis, scale)
    if len(hand_f) > _MAX_FACES:
        step = int(np.ceil(len(hand_f) / _MAX_FACES))
        hand_f = hand_f[::step]
    idx, w = assign_weights(hand_v, positions)
    poses = rest_poses(positions)
    palm_ids, grip_ids = palm_and_grip_vertex_ids(hand_v)
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


def write_provenance(digest: str, *, source_commit: str, source_url: str) -> None:
    record = {
        "licence": "CC0-1.0",
        "source_name": "MakeHuman base mesh hm08 (right hand extract)",
        "source_url": source_url,
        "source_commit": source_commit,
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
    notice = NOTICE.read_text(encoding="utf-8") if NOTICE.is_file() else ""
    if "source_commit" not in notice:
        NOTICE.write_text(
            notice.rstrip()
            + f"\n\nSource mesh: {source_url}\nSource commit: {source_commit}\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_obj", type=Path)
    parser.add_argument(
        "--source-url",
        type=str,
        default=(
            "https://github.com/makehumancommunity/makehuman/blob/"
            "a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj"
        ),
    )
    parser.add_argument(
        "--source-commit",
        type=str,
        default="a8bc2d54ff0ac92e78ff71431b1023eda42bf482",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.base_obj.is_file():
        raise SystemExit(f"missing base.obj: {args.base_obj}")
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
    write_provenance(
        digest, source_commit=args.source_commit, source_url=args.source_url
    )
    print(  # noqa: T201
        f"wrote {OUT} verts={len(arrays['vertices'])} "
        f"faces={len(arrays['faces'])} sha256={digest[:12]}…"
    )


if __name__ == "__main__":
    main()
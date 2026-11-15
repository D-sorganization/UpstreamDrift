#!/usr/bin/env python3
"""Derive the CC0 right-hand skinned asset from MakeHuman (epic #15150, #15148).

Usage::

    python3 assets/hand_cc0/build_hand_asset.py PATH/TO/makehuman-checkout [--check]

The hand is cut out of MakeHuman's ``base.obj`` by **skin weight**: every vertex
that MakeHuman's own ``default_weights.mhw`` binds mostly to the right-hand
bones of ``default.mhskel``, plus a short forearm stub. Skeleton and weights are
MakeHuman's, unchanged apart from keeping the strongest influences per vertex.

Canonical rest frame of the asset (a **right** hand, metres):

* origin at the wrist joint;
* ``x`` radial (little finger toward index finger and thumb);
* ``y`` distal (wrist toward the middle knuckle);
* ``z`` palmar (out of the palm).

``--check`` rebuilds in memory and compares with the committed ``hand_cc0.npz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mh_rig_io import (  # noqa: E402
    FOREARM_BONES,
    HAND_BONES,
    MakeHumanRig,
    dense_weights,
    load_makehuman_rig,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "hand_cc0.npz"
PROVENANCE = HERE / "PROVENANCE.json"
SETTINGS = HERE / "build_settings.json"

#: Asset bone name -> MakeHuman bone (``None`` = the merged forearm stub root).
BONE_MAP: tuple[tuple[str, str | None], ...] = (
    ("forearm", None),
    ("wrist", "wrist.R"),
    ("meta_index", "metacarpal1.R"),
    ("meta_middle", "metacarpal2.R"),
    ("meta_ring", "metacarpal3.R"),
    ("meta_little", "metacarpal4.R"),
    ("thumb_meta", "finger1-1.R"),
    ("thumb_prox", "finger1-2.R"),
    ("thumb_dist", "finger1-3.R"),
    ("index_prox", "finger2-1.R"),
    ("index_mid", "finger2-2.R"),
    ("index_dist", "finger2-3.R"),
    ("middle_prox", "finger3-1.R"),
    ("middle_mid", "finger3-2.R"),
    ("middle_dist", "finger3-3.R"),
    ("ring_prox", "finger4-1.R"),
    ("ring_mid", "finger4-2.R"),
    ("ring_dist", "finger4-3.R"),
    ("little_prox", "finger5-1.R"),
    ("little_mid", "finger5-2.R"),
    ("little_dist", "finger5-3.R"),
)
BONE_NAMES = tuple(name for name, _ in BONE_MAP)
DIGIT_CHAINS: dict[str, tuple[str, ...]] = {
    "thumb": ("thumb_meta", "thumb_prox", "thumb_dist"),
    "index": ("index_prox", "index_mid", "index_dist"),
    "middle": ("middle_prox", "middle_mid", "middle_dist"),
    "ring": ("ring_prox", "ring_mid", "ring_dist"),
    "little": ("little_prox", "little_mid", "little_dist"),
}
METACARPALS = ("meta_index", "meta_middle", "meta_ring", "meta_little")


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("degenerate direction while building the hand frame")
    return v / norm


def load_settings() -> dict[str, object]:
    return json.loads(SETTINGS.read_text(encoding="utf-8"))


def canonical_frame(rig: MakeHumanRig) -> tuple[np.ndarray, np.ndarray]:
    """``(origin, R)`` with rows of ``R`` the radial, distal and palmar axes."""
    wrist = rig.bone_head("wrist.R")
    index_mcp = rig.bone_head("finger2-1.R")
    middle_mcp = rig.bone_head("finger3-1.R")
    little_mcp = rig.bone_head("finger5-1.R")
    palmar = _unit(np.cross(index_mcp - little_mcp, middle_mcp - wrist))
    # The rest fingers are slightly curled; the curl must point out of the palm.
    pip = rig.bone_head("finger3-2.R")
    tip = rig.bone_tail("finger3-3.R")
    along = _unit(pip - middle_mcp)
    curl = (tip - middle_mcp) - ((tip - middle_mcp) @ along) * along
    if float(curl @ palmar) <= 0.0:
        raise ValueError("MakeHuman '.R' hand is not a right hand in this frame")
    distal = middle_mcp - wrist
    distal = _unit(distal - (distal @ palmar) * palmar)
    radial = np.cross(distal, palmar)
    if float(radial @ (index_mcp - little_mcp)) <= 0.0:
        raise ValueError("radial axis does not point at the index finger")
    return wrist, np.stack([radial, distal, palmar])


def _digit_flex_axis(
    points: np.ndarray, toward: np.ndarray, ratio: float
) -> np.ndarray:
    """Hinge axis of a digit: normal of the plane its rest joints lie in.

    ``toward`` is the direction flexion must move the tip (palmar for a finger).
    Falls back to ``along x toward`` when the rest digit is too straight for a
    plane fit (second singular value below ``ratio`` of the first).
    """
    along = _unit(points[-1] - points[0])
    centred = points - points.mean(axis=0)
    _, sing, vt = np.linalg.svd(centred)
    axis = vt[2] if sing[1] >= ratio * sing[0] else np.cross(along, toward)
    axis = _unit(axis - (axis @ along) * along)
    if float(np.cross(axis, along) @ toward) < 0.0:
        axis = -axis
    return axis


def bone_tables(
    rig: MakeHumanRig, origin: np.ndarray, frame: np.ndarray, settings: dict
) -> dict[str, np.ndarray]:
    """Heads, tails, parents, flexion axes and rest flexion angles."""
    scale = float(settings["makehuman_unit_m"])

    def to_local(p: np.ndarray) -> np.ndarray:
        return (frame @ (p - origin)) * scale

    heads, tails, parents = [], [], []
    mh_to_asset = {mh: name for name, mh in BONE_MAP if mh is not None}
    for _name, mh in BONE_MAP:
        if mh is None:
            head = to_local(rig.bone_head(FOREARM_BONES[0]))
            tail = to_local(rig.bone_tail(FOREARM_BONES[-1]))
            parents.append(-1)
        else:
            head, tail = to_local(rig.bone_head(mh)), to_local(rig.bone_tail(mh))
            parent = rig.bone_parent(mh)
            parents.append(BONE_NAMES.index(mh_to_asset.get(parent or "", "forearm")))
        heads.append(head)
        tails.append(tail)
    heads_a, tails_a = np.asarray(heads), np.asarray(tails)
    palmar = np.array([0.0, 0.0, 1.0])
    ulnar_palmar = _unit(np.array([-1.0, 0.0, 1.0]))
    axes = np.zeros_like(heads_a)
    ratio = float(settings["min_plane_fit_ratio"])
    for digit, chain in DIGIT_CHAINS.items():
        ids = [BONE_NAMES.index(b) for b in chain]
        pts = np.vstack([heads_a[ids], tails_a[ids[-1]][None]])
        toward = ulnar_palmar if digit == "thumb" else palmar
        # A finger's rest joints are too nearly in line to fix its plane of
        # flexion (the fit rolls neighbouring fingers apart), so fingers hinge
        # about the palm-plane axis square to the finger; the thumb is fitted.
        fit_ratio = ratio if digit == "thumb" else float("inf")
        axes[ids] = _digit_flex_axis(pts, toward, fit_ratio)
    for meta in METACARPALS:
        i = BONE_NAMES.index(meta)
        axes[i] = _unit(np.cross(_unit(tails_a[i] - heads_a[i]), palmar))
    rest_flex = np.zeros(len(BONE_NAMES))
    for chain in DIGIT_CHAINS.values():
        for bone in chain:
            i = BONE_NAMES.index(bone)
            p = parents[i]
            if BONE_NAMES[p] in ("wrist", "forearm"):
                continue
            d_parent = _unit(tails_a[p] - heads_a[p])
            d_bone = _unit(tails_a[i] - heads_a[i])
            rest_flex[i] = float(
                np.arctan2(np.cross(d_parent, d_bone) @ axes[i], d_parent @ d_bone)
            )
    return {
        "bone_heads": heads_a,
        "bone_tails": tails_a,
        "bone_parents": np.asarray(parents, dtype=np.int16),
        "flex_axes": axes,
        "rest_flexion_rad": rest_flex,
    }


def _largest_component(quads: np.ndarray) -> np.ndarray:
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for quad in quads:
        root = find(int(quad[0]))
        for v in quad[1:]:
            parent[find(int(v))] = root
    roots = np.array([find(int(q[0])) for q in quads])
    values, counts = np.unique(roots, return_counts=True)
    return quads[roots == values[np.argmax(counts)]]


def _triangulate(quads: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Split each quad along its shorter diagonal (keeps the winding)."""
    tris = []
    for a, b, c, d in quads:
        if np.linalg.norm(verts[a] - verts[c]) <= np.linalg.norm(verts[b] - verts[d]):
            tris += [(a, b, c), (a, c, d)]
        else:
            tris += [(a, b, d), (b, c, d)]
    return np.asarray(tris, dtype=np.int64)


def _cap_boundaries(
    verts: np.ndarray, tris: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Close each open boundary loop with a centroid fan. Returns loops too."""
    edges: dict[tuple[int, int], int] = {}
    for tri in tris:
        for k in range(3):
            edge = (int(tri[k]), int(tri[(k + 1) % 3]))
            edges[edge] = edges.get(edge, 0) + 1
    nxt = {a: b for (a, b) in edges if (b, a) not in edges}
    loops: list[np.ndarray] = []
    new_verts, new_tris = [verts], [tris]
    count = len(verts)
    while nxt:
        start = next(iter(nxt))
        loop = [start]
        while (step := nxt.pop(loop[-1])) != start:
            loop.append(step)
        loop_a = np.asarray(loop, dtype=np.int64)
        loops.append(loop_a)
        new_verts.append(verts[loop_a].mean(axis=0)[None])
        fan = [
            (loop_a[(k + 1) % len(loop_a)], loop_a[k], count)
            for k in range(len(loop_a))
        ]
        new_tris.append(np.asarray(fan, dtype=np.int64))
        count += 1
    return np.vstack(new_verts), np.vstack(new_tris), loops


def build(makehuman_root: Path) -> dict[str, np.ndarray]:
    settings = load_settings()
    rig = load_makehuman_rig(makehuman_root)
    origin, frame = canonical_frame(rig)
    scale = float(settings["makehuman_unit_m"])
    local = (rig.vertices - origin) @ frame.T * scale

    hand_w = dense_weights(rig, HAND_BONES)
    arm_w = dense_weights(rig, FOREARM_BONES).sum(axis=1)
    in_hand = hand_w.sum(axis=1) > float(settings["hand_weight_threshold"])
    in_stub = (arm_w + hand_w.sum(axis=1) > 0.0) & (
        local[:, 1] > -float(settings["forearm_stub_length_m"])
    )
    keep = in_hand | in_stub
    quads = rig.body_quads[keep[rig.body_quads].all(axis=1)]
    quads = _largest_component(quads)
    used = np.unique(quads)
    remap = np.full(len(rig.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts = local[used]
    tris = _triangulate(remap[quads], verts)
    verts, tris, loops = _cap_boundaries(verts, tris)
    if len(loops) != 1:
        raise ValueError(f"expected one forearm opening, found {len(loops)}")

    weights = np.zeros((len(verts), len(BONE_NAMES)))
    weights[: len(used), 0] = arm_w[used]
    for column, bone in enumerate(HAND_BONES):
        asset_name = next(name for name, mh in BONE_MAP if mh == bone)
        weights[: len(used), BONE_NAMES.index(asset_name)] = hand_w[used, column]
    weights[len(used) :, 0] = 1.0  # cap centroid rides on the forearm
    empty = weights.sum(axis=1) <= 0.0
    weights[empty, 0] = 1.0
    top = int(settings["max_influences"])
    order = np.argsort(-weights, axis=1)[:, :top]
    top_w = np.take_along_axis(weights, order, axis=1)
    top_w /= top_w.sum(axis=1, keepdims=True)

    tables = bone_tables(rig, origin, frame, settings)
    volume = float(
        np.einsum(
            "ij,ij->i",
            verts[tris[:, 0]],
            np.cross(verts[tris[:, 1]], verts[tris[:, 2]]),
        ).sum()
        / 6.0
    )
    if volume <= 0.0:
        raise ValueError("hand mesh is not outward wound")
    return {
        "vertices": verts.astype(np.float32),
        "faces": tris.astype(np.int32),
        "joint_indices": order.astype(np.uint8),
        "joint_weights": top_w.astype(np.float32),
        "bone_names": np.asarray(BONE_NAMES),
        "bone_heads": tables["bone_heads"].astype(np.float32),
        "bone_tails": tables["bone_tails"].astype(np.float32),
        "bone_parents": tables["bone_parents"],
        "flex_axes": tables["flex_axes"].astype(np.float32),
        "rest_flexion_rad": tables["rest_flexion_rad"].astype(np.float32),
    }


def content_digest(arrays: dict[str, np.ndarray]) -> str:
    """Hash of the array contents (independent of zip metadata)."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()


def write_provenance(arrays: dict[str, np.ndarray]) -> None:
    settings = load_settings()
    base = f"{settings['source_repository']}/blob/{settings['source_commit']}"
    record = {
        "asset_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "content_sha256": content_digest(arrays),
        "licence": "CC0-1.0",
        "source_name": "MakeHuman hm08 base mesh, default skeleton and default "
        "weights (right hand extract)",
        "source_commit": settings["source_commit"],
        "source_urls": [f"{base}/{rel}" for rel in settings["source_files"]],
        "vertex_count": int(len(arrays["vertices"])),
        "face_count": int(len(arrays["faces"])),
        "bone_count": int(len(arrays["bone_names"])),
        "research_only_alternatives": "MANO and SMPL-X hand models are "
        "research/non-commercial licences and are not bundled; they remain "
        "research-only alternatives.",
    }
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("makehuman_root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    arrays = build(args.makehuman_root)
    if args.check:
        with np.load(OUT, allow_pickle=False) as committed:
            same = content_digest({k: committed[k] for k in committed.files}) == (
                content_digest(arrays)
            )
        print("hand_cc0.npz matches the MakeHuman source" if same else "MISMATCH")  # noqa: T201
        return 0 if same else 1
    np.savez_compressed(OUT, **arrays)
    write_provenance(arrays)
    print(  # noqa: T201
        f"wrote {OUT.name}: {len(arrays['vertices'])} vertices, "
        f"{len(arrays['faces'])} faces, {len(arrays['bone_names'])} bones"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

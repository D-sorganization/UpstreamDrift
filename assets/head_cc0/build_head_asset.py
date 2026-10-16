#!/usr/bin/env python3
"""Derive the shared CC0 head asset from the MakeHuman base mesh (R7, #12687).

Usage::

    python assets/head_cc0/build_head_asset.py PATH/TO/base.obj [--check]

Input: the MakeHuman ``base.obj`` (``hm08``, CC0 since 2020-09, see
``PROVENANCE.json``). The script is deterministic: the same input gives the
byte-identical ``head_cc0.npz`` array content (quantised integers).

What it does
    1. Keeps the ``body`` faces above the neck plane (the ``joint-neck`` height),
       skull, face, ears and neck, and caps the neck with a fan so the skin is a
       closed surface.
    2. Cuts a SHORT hair shell out of the scalp: the faces above a hairline that
       sits high on the forehead and above the ears and tapers low at the nape,
       offset outward along the skin normals (3 mm at the hairline to 7 mm on the
       crown) and closed with an inner shell and a rim.
    3. Re-expresses everything in the canonical head frame of
       ``model_appearance.head`` (origin at the neck point, x forward, y left,
       z up) in units of one head length (neck plane to vertex = 1.0), stored
       as int16 multiples of ``QUANTUM``.

The MakeHuman frame is y up, z forward, +x the character's left; one unit is a
decimetre.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

NL = chr(10)
QUANTUM = 1e-4  # head lengths per integer step (about 24 micrometres)
NECK_JOINT = np.array([0.0, 5.89025, 0.06805])  # MakeHuman ``joint-neck`` centre
HAIR_THICKNESS_DM = (0.03, 0.07)  # hairline .. crown, in MakeHuman decimetres
OUT = Path(__file__).with_name("head_cc0.npz")


def read_obj(path: Path) -> tuple[np.ndarray, dict[str, list[list[int]]]]:
    verts: list[list[float]] = []
    groups: dict[str, list[list[int]]] = {}
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("g "):
            current = line.split()[1]
            groups.setdefault(current, [])
        elif line.startswith("f "):
            groups[current].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.asarray(verts, dtype=float), groups


def triangulate(quads: list[list[int]]) -> np.ndarray:
    tris = []
    for f in quads:
        for k in range(1, len(f) - 1):
            tris.append([f[0], f[k], f[k + 1]])
    return np.asarray(tris, dtype=np.int64)


def boundary_edges(tris: np.ndarray) -> list[tuple[int, int]]:
    """Directed edges used by exactly one triangle (in that triangle's sense)."""
    count: collections.Counter[tuple[int, int]] = collections.Counter()
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            count[(min(u, v), max(u, v))] += 1
    out = []
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            if count[(min(u, v), max(u, v))] == 1:
                out.append((int(u), int(v)))
    return out


def vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    tri = verts[tris]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area weighted
    n = np.zeros_like(verts)
    for k in range(3):
        np.add.at(n, tris[:, k], face_n)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(norm > 0, norm, 1.0)


def signed_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    t = verts[tris]
    return float(np.einsum("ij,ij->", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6.0)


def hairline_y(z: np.ndarray) -> np.ndarray:
    """Hairline height (decimetres) versus the forward coordinate ``z``."""
    return np.interp(
        z,
        [-0.7, -0.35, -0.1, 0.1, 0.9, 1.05, 1.5],
        [6.55, 6.85, 7.55, 7.62, 7.62, 7.98, 8.05],
    )


def build(base_obj: Path) -> dict[str, np.ndarray]:
    v, groups = read_obj(base_obj)
    head_faces = [f for f in groups["body"] if v[f][:, 1].mean() >= NECK_JOINT[1]]
    tris = triangulate(head_faces)
    used = np.unique(tris)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    skin_v = v[used].copy()
    skin_t = remap[tris]
    # Neck cap: snap the boundary ring onto one plane and fan it to its centre.
    ring = boundary_edges(skin_t)
    ring_ids = np.unique(np.asarray(ring))
    skin_v[ring_ids, 1] = NECK_JOINT[1]
    centre = skin_v[ring_ids].mean(axis=0)
    skin_v = np.vstack([skin_v, centre])
    c = len(skin_v) - 1
    skin_t = np.vstack([skin_t, [[b, a, c] for a, b in ring]])
    if signed_volume(skin_v, skin_t) < 0:
        skin_t = skin_t[:, ::-1]
    # Short hair: scalp faces above the hairline.
    cen = skin_v[skin_t].mean(axis=1)
    scalp = skin_t[(cen[:, 1] > hairline_y(cen[:, 2])) & (cen[:, 1] > 7.3)]
    normals = vertex_normals(skin_v, skin_t)
    u = np.unique(scalp)
    hm = -np.ones(len(skin_v), dtype=np.int64)
    hm[u] = np.arange(len(u))
    base = skin_v[u]
    ramp = np.clip((base[:, 1] - 7.4) / 1.0, 0.0, 1.0)
    thick = HAIR_THICKNESS_DM[0] + (HAIR_THICKNESS_DM[1] - HAIR_THICKNESS_DM[0]) * ramp
    outer = base + normals[u] * thick[:, None]
    n_in = len(base)
    st = hm[scalp]
    side = []
    for a, b in boundary_edges(st):
        side += [[b + n_in, a + n_in, a], [b + n_in, a, b]]
    hair_v = np.vstack([base, outer])
    hair_t = np.vstack([st[:, ::-1], st + n_in, np.asarray(side)])
    if signed_volume(hair_v, hair_t) < 0:
        hair_t = hair_t[:, ::-1]

    def canonical(p: np.ndarray) -> np.ndarray:
        q = p - NECK_JOINT
        return np.column_stack([q[:, 2], q[:, 0], q[:, 1]])  # x fwd, y left, z up

    top = canonical(skin_v)[:, 2].max()
    out: dict[str, np.ndarray] = {}
    for name, verts, faces in (("skin", skin_v, skin_t), ("hair", hair_v, hair_t)):
        q = np.rint(canonical(verts) / top / QUANTUM)
        assert np.abs(q).max() < 32767
        out[f"{name}_vertices"] = q.astype("<i2")
        out[f"{name}_faces"] = np.asarray(faces, dtype="<u4")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_obj", type=Path)
    ap.add_argument("--check", action="store_true", help="compare with the asset")
    args = ap.parse_args(argv)
    arrays = build(args.base_obj)
    if args.check:
        have = np.load(OUT)
        same = all(np.array_equal(have[k], arrays[k]) for k in arrays)
        sys.stdout.write(("asset content matches" if same else "DIFFERS") + NL)
        return 0 if same else 1
    np.savez_compressed(OUT, **arrays)
    sys.stdout.write(f"wrote {OUT} ({OUT.stat().st_size} bytes)" + NL)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

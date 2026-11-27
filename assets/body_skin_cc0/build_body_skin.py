"""Derive the continuous CC0 body skin from MakeHuman (#15299).

Usage: ``MAKEHUMAN_ROOT=CHECKOUT python assets/body_skin_cc0/build_body_skin.py
[--check]``, where the checkout is the MakeHuman tree at the commit pinned in
``skin_settings.json``. Its location is never written to the outputs. The
build is deterministic (quantised integers).

The athletic male mix shared with the body sections is scaled to the reference
stature; the ``body`` faces that are not hands, feet or head, up to ``neck_cut_m`` above
the neck joint, are
kept as one triangle surface; MakeHuman's skin weights are summed into the bone
groups of the settings (a bone outside every group joins its nearest listed
ancestor); the joint landmarks of the runtime fit are stored. The same vertices
of the slim mix (``slim`` in the settings) are stored as offsets from the
athletic ones, for the runtime fit to mix in; its joint landmarks must be the
athletic ones (the muscle and weight targets move no joint). Stored frame:
hip-midpoint origin, x forward, y left, z up. Appearance only, no physics.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "makehuman_cc0"))

from mh_rig_io import (  # noqa: E402
    MakeHumanRig,
    dense_weights,
    load_makehuman_rig,
    sha256_file,
)
from mh_targets import macro_mix, morphed_rig  # noqa: E402

Array = NDArray[np.float64]
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SETTINGS_PATH = HERE / "skin_settings.json"
OUT = HERE / "body_skin_cc0.npz"
PROVENANCE_PATH = HERE / "PROVENANCE.json"


def group_bones(settings: dict[str, Any]) -> dict[str, list[str]]:
    """Group name to its listed MakeHuman bones, central groups first."""
    groups = {name: list(bones) for name, bones in settings["central_groups"].items()}
    for side in ("L", "R"):
        for name, bones in settings["side_groups"].items():
            groups[f"{name}_{side.lower()}"] = [f"{bone}.{side}" for bone in bones]
    return groups


def bone_to_group(rig: MakeHumanRig, groups: dict[str, list[str]]) -> dict[str, int]:
    """Every rig bone to a group index: its own or its nearest listed ancestor's.

    Raises ``ValueError`` when a bone has no listed ancestor.
    """
    listed = {b: i for i, bones in enumerate(groups.values()) for b in bones}
    out: dict[str, int] = {}
    for bone in rig.bones:
        walk: str | None = bone
        while walk is not None and walk not in listed:
            walk = rig.bone_parent(walk)
        if walk is None:
            raise ValueError(f"bone {bone!r} has no listed ancestor")
        out[bone] = listed[walk]
    return out


def to_model_frame(rig: MakeHumanRig, settings: dict[str, Any]) -> tuple[Array, Array]:
    """Vertices and landmarks in metres: hip midpoint origin, x fwd, y left, z up."""
    body = np.unique(rig.body_quads)
    scale = settings["stature_m"] / float(np.ptp(rig.vertices[body, 1]))
    marks = np.array([rig.bone_head(b) for b in settings["landmarks"].values()])
    names = list(settings["landmarks"])
    origin = 0.5 * (marks[names.index("hip_l")] + marks[names.index("hip_r")])

    def convert(points: Array) -> Array:
        local = (points - origin) * scale
        return np.column_stack([local[:, 2], local[:, 0], local[:, 1]])

    return convert(rig.vertices), convert(marks)


def group_weights(rig: MakeHumanRig, groups: dict[str, list[str]]) -> Array:
    """``(N, groups)`` weights on the full mesh, rows normalised where non-zero."""
    bones = tuple(rig.bones)
    column = bone_to_group(rig, groups)
    dense = dense_weights(rig, bones)
    out = np.zeros((len(rig.vertices), len(groups)))
    for index, bone in enumerate(bones):
        out[:, column[bone]] += dense[:, index]
    total = out.sum(axis=1, keepdims=True)
    return np.divide(out, total, out=np.zeros_like(out), where=total > 0)


def quantise_weights(weights: Array, settings: dict[str, Any]) -> tuple[Any, Any]:
    """Strongest ``max_influences`` per vertex as uint8 summing to ``weight_levels``."""
    keep, levels = settings["max_influences"], settings["weight_levels"]
    order = np.argsort(-weights, axis=1, kind="stable")[:, :keep]
    value = np.take_along_axis(weights, order, axis=1)
    if np.any(value[:, 0] <= 0):
        raise ValueError("a kept vertex has no skin weight")
    steps = np.floor(value / value.sum(axis=1, keepdims=True) * levels).astype(int)
    steps[:, 0] += levels - steps.sum(axis=1)  # the remainder goes to the strongest
    order[steps == 0] = order[:, :1].repeat(keep, axis=1)[steps == 0]
    return order.astype("u1"), steps.astype("u1")


def build(makehuman_root: Path, settings: dict[str, Any]) -> tuple[dict, list]:
    """The asset arrays and the morph targets read.

    Precondition: ``makehuman_root`` holds ``makehuman/data`` at the pinned
    commit. Postcondition: ``faces`` index every stored vertex, the surface is
    outward facing and each weight row sums to ``weight_levels``.
    """
    mixes = json.loads((REPO / settings["mix_settings"]).read_text())["mixes"]
    base = load_makehuman_rig(makehuman_root)
    mix = mixes[settings["mix"]]
    rig, used = morphed_rig(base, makehuman_root, macro_mix(mix))
    vertices, marks = to_model_frame(rig, settings)
    lean, lean_used = morphed_rig(
        base, makehuman_root, macro_mix({**mix, **settings["slim"]["mix"]})
    )
    lean_vertices, lean_marks = to_model_frame(lean, settings)
    owner = bone_to_group(rig, {"cut": settings["cut_roots"], "kept": ["root"]})
    cut_bones = tuple(bone for bone, index in owner.items() if index == 0)
    cut = dense_weights(rig, cut_bones).sum(axis=1)
    quads = rig.body_quads
    neck_z = marks[list(settings["landmarks"]).index(settings["neck_plane"]), 2]
    neck_z += settings["neck_cut_m"]
    keep = (vertices[quads][:, :, 2].mean(axis=1) < neck_z) & (
        cut[quads].mean(axis=1) < settings["cut_weight"]
    )
    quads = quads[keep]
    faces = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    ids, inverse = np.unique(faces, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    points = vertices[ids]
    a, b, c = (points[faces[:, k]] for k in range(3))
    if np.einsum("ij,ij->", a, np.cross(b, c)) < 0:  # signed volume: face outward
        faces = faces[:, ::-1]
    index, value = quantise_weights(
        group_weights(rig, group_bones(settings))[ids], settings
    )
    quantum = settings["quantum_m"]
    stored = np.rint(points / quantum)
    if np.abs(stored).max() >= 32767 or len(points) >= 65536:
        raise ValueError("mesh does not fit the int16/uint16 storage")
    arrays = {
        "vertices": stored.astype("<i2"),
        "faces": faces.astype("<u2"),
        "weight_index": index,
        "weight_value": value,
        "landmarks": np.rint(marks / quantum).astype("<i2"),
    }
    slim = np.rint(lean_vertices[ids] / quantum) - stored
    arrays["slim_offset"] = slim.astype("<i2")
    if not np.array_equal(np.rint(lean_marks / quantum), arrays["landmarks"]):
        raise ValueError("the slim mix moves a joint landmark")
    return arrays, [used, lean_used]


def provenance(root: Path, settings: dict[str, Any], used: list) -> dict[str, Any]:
    """The provenance record for the asset now on disk (no local paths)."""
    return {
        "asset": OUT.name,
        "asset_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "asset_description": "Athletic male MakeHuman body skin without head, hands and feet, as one triangle surface with group skin weights and joint landmarks, and the offsets of the same vertices to a slim build; int16 multiples of quantum_m, origin at the hip midpoint, x forward, y left, z up.",
        "quantum_m": settings["quantum_m"],
        "licence": "CC0-1.0",
        "licence_file": "../head_cc0/LICENSE.CC0.txt",
        "licence_statement_file": "../head_cc0/SOURCE_LICENSE_STATEMENT.txt",
        "source_repository": settings["source_repository"],
        "source_commit": settings["source_commit"],
        "source_sha256": {
            name: sha256_file(root.joinpath(*name.split("/")))
            for name in settings["source_files"]
        },
        "morph_targets": {
            t["path"]: f"{t['weight']} x sha256 {t['sha256']}" for t in used[0]
        },
        "slim_morph_targets": {
            t["path"]: f"{t['weight']} x sha256 {t['sha256']}" for t in used[1]
        },
        "author": "MakeHuman Team (copyright holders at the CC0 release: Data Collection AB, Joel Palmius, Jonas Hauquier)",
        "download_date": settings["download_date"],
        "derivation": "assets/body_skin_cc0/build_body_skin.py with skin_settings.json (deterministic).",
    }


def main(argv: list[str]) -> int:
    check = argv == ["--check"]
    root_text = os.environ.get("MAKEHUMAN_ROOT")
    if not root_text:
        raise SystemExit("set MAKEHUMAN_ROOT to the MakeHuman checkout")
    root = Path(root_text)
    settings = json.loads(SETTINGS_PATH.read_text())
    arrays, used = build(root, settings)
    if check:
        with np.load(OUT) as have:
            same = set(have.files) == set(arrays) and all(
                np.array_equal(have[k], arrays[k]) for k in arrays
            )
        same = same and json.loads(PROVENANCE_PATH.read_text()) == provenance(
            root, settings, used
        )
        sys.stdout.write(("asset content matches" if same else "DIFFERS") + "\n")
        return 0 if same else 1
    np.savez_compressed(OUT, **arrays)
    record = provenance(root, settings, used)
    PROVENANCE_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {OUT.name} ({OUT.stat().st_size} bytes)" + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

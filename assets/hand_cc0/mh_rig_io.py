"""Read MakeHuman's CC0 base mesh, default skeleton and skin weights (build-time only).

Nothing here runs at import time of the application: ``build_hand_asset.py`` uses
it once to derive ``hand_cc0.npz``. All three inputs are CC0 (see ``NOTICE``):

* ``makehuman/data/3dobjs/base.obj`` (hm08 base mesh, quads);
* ``makehuman/data/rigs/default.mhskel`` (bones, joint helper-vertex lists);
* ``makehuman/data/rigs/default_weights.mhw`` (per-bone ``[vertex, weight]``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: MakeHuman right-hand bones, parents first. ``finger1`` is the thumb.
FOREARM_BONES: tuple[str, ...] = ("lowerarm01.R", "lowerarm02.R")
HAND_BONES: tuple[str, ...] = (
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
BODY_GROUP = "body"


@dataclass(frozen=True)
class MakeHumanRig:
    """The parts of the MakeHuman default rig that the hand build needs."""

    vertices: Array  # (N, 3) base mesh, MakeHuman units
    body_quads: NDArray[np.int64]  # (Q, 4) faces of the ``body`` group
    bones: dict[str, dict[str, object]]
    joints: dict[str, list[int]]
    weights: dict[str, list[tuple[int, float]]]
    skeleton_licence: str
    weights_licence: str

    def joint_position(self, name: str) -> Array:
        """Centre of the joint helper vertices called ``name``."""
        ids = self.joints.get(name)
        if not ids:
            raise ValueError(f"MakeHuman skeleton has no joint {name!r}")
        return self.vertices[np.asarray(ids, dtype=np.int64)].mean(axis=0)

    def bone_head(self, bone: str) -> Array:
        return self.joint_position(str(self._bone(bone)["head"]))

    def bone_tail(self, bone: str) -> Array:
        return self.joint_position(str(self._bone(bone)["tail"]))

    def bone_parent(self, bone: str) -> str | None:
        parent = self._bone(bone).get("parent")
        return None if parent is None else str(parent)

    def _bone(self, bone: str) -> dict[str, object]:
        if bone not in self.bones:
            raise ValueError(f"MakeHuman skeleton has no bone {bone!r}")
        return self.bones[bone]


def read_base_obj(path: Path) -> tuple[Array, NDArray[np.int64]]:
    """Vertices and the quads of the ``body`` group of ``base.obj``."""
    vertices: list[list[float]] = []
    quads: list[list[int]] = []
    group = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("g "):
                group = line.split()[1]
            elif line.startswith("f ") and group == BODY_GROUP:
                corners = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                if len(corners) != 4:
                    raise ValueError("base.obj body group must be all quads")
                quads.append(corners)
    if not vertices or not quads:
        raise ValueError(f"{path} holds no body mesh")
    return np.asarray(vertices, dtype=float), np.asarray(quads, dtype=np.int64)


def load_makehuman_rig(makehuman_root: Path) -> MakeHumanRig:
    """Load mesh, skeleton and weights from a MakeHuman source tree.

    Precondition: ``makehuman_root`` is the repository root that contains
    ``makehuman/data``. Raises ``ValueError`` if a file is missing or is not
    marked CC0.
    """
    data = makehuman_root / "makehuman" / "data"
    obj_path = data / "3dobjs" / "base.obj"
    skel_path = data / "rigs" / "default.mhskel"
    weights_path = data / "rigs" / "default_weights.mhw"
    for path in (obj_path, skel_path, weights_path):
        if not path.is_file():
            raise ValueError(f"missing MakeHuman input: {path}")
    skeleton = json.loads(skel_path.read_text(encoding="utf-8"))
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    for name, record in (("skeleton", skeleton), ("weights", weights)):
        if record.get("license") != "CC0":
            raise ValueError(f"MakeHuman {name} is not marked CC0")
    vertices, quads = read_base_obj(obj_path)
    return MakeHumanRig(
        vertices=vertices,
        body_quads=quads,
        bones=skeleton["bones"],
        joints=skeleton["joints"],
        weights={
            bone: [(int(v), float(w)) for v, w in pairs]
            for bone, pairs in weights["weights"].items()
        },
        skeleton_licence=str(skeleton["license"]),
        weights_licence=str(weights["license"]),
    )


def dense_weights(rig: MakeHumanRig, bones: tuple[str, ...]) -> Array:
    """``(N, len(bones))`` weight matrix on the full base mesh."""
    out = np.zeros((len(rig.vertices), len(bones)))
    for column, bone in enumerate(bones):
        for vertex, weight in rig.weights.get(bone, ()):
            out[vertex, column] = weight
    return out

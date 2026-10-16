"""CC0 head mesh behind the shared ``model_appearance.head`` slot (R7, #12687).

``head.build_head_parts`` hands over to :func:`build_mesh_parts` when
``HeadSettings.model == "cc0_mesh"``, so every engine (MuJoCo today; Drake,
Pinocchio, OpenSim and Simscape through the shared OBJ export) gets the same
realistic head without any per-engine code (DRY, Law of Demeter).

The asset (``assets/head_cc0/head_cc0.npz``) is a derivative of the MakeHuman
base mesh, released CC0 by its authors; ``assets/head_cc0/PROVENANCE.json``
records source URL, author, licence text, download date and sha256 values.
The loader fails closed: a missing or mismatching provenance record, a bad
array layout, a non-closed shell or a head outside its size envelope raises
``ValueError`` and no mesh is returned.

Frame contract: the stored vertices are integer multiples of ``quantum`` head
lengths in the canonical head frame of :mod:`head` (origin at the neck plane,
x forward, y left, z up; the skin vertex is exactly one head length up).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh

REPO_ROOT = Path(__file__).resolve().parents[4]
ASSET_DIR = REPO_ROOT / "assets" / "head_cc0"
ASSET_PATH = ASSET_DIR / "head_cc0.npz"
PROVENANCE_PATH = ASSET_DIR / "PROVENANCE.json"
LICENCE_ID = "CC0-1.0"
MESH_PARTS = ("skin", "hair")
# Envelope (head lengths) a valid asset must respect; guards a corrupted file.
_MAX_HALF_WIDTH, _MAX_DEPTH = 0.75, 1.1
_MAX_FACES = 12_000


@dataclass(frozen=True)
class HeadAsset:
    """The validated head asset in unit head-length coordinates."""

    skin: Mesh
    hair: Mesh
    path: str
    sha256: str


def _closed(faces: NDArray[np.int64]) -> bool:
    """True when every directed edge has exactly one opposite edge."""
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    ).astype(np.int64)
    n = int(edges.max()) + 1
    fwd = np.sort(edges[:, 0] * n + edges[:, 1])
    rev = np.sort(edges[:, 1] * n + edges[:, 0])
    return len(np.unique(fwd)) == len(fwd) and np.array_equal(fwd, rev)


def _check_mesh(name: str, mesh: Mesh) -> None:
    v, f = mesh.vertices, mesh.faces
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or len(f) == 0:
        raise ValueError(f"head mesh {name!r}: bad vertex or face arrays")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"head mesh {name!r}: face index outside the vertices")
    if len(f) > _MAX_FACES:
        raise ValueError(f"head mesh {name!r}: {len(f)} faces exceed the budget")
    if not _closed(f):
        raise ValueError(f"head mesh {name!r} is not a closed manifold")
    if mesh.volume() <= 0:
        raise ValueError(f"head mesh {name!r} is not outward oriented")
    if (
        np.abs(v[:, 1]).max() > _MAX_HALF_WIDTH
        or v[:, 0].max() > _MAX_DEPTH
        or v[:, 0].min() < -_MAX_DEPTH
        or v[:, 2].min() < -0.05
        or v[:, 2].max() > 1.3
    ):
        raise ValueError(f"head mesh {name!r} lies outside the head envelope")


def _load(asset: Path, provenance: Path) -> HeadAsset:
    if not provenance.is_file():
        raise ValueError(f"head provenance record missing: {provenance}")
    record = json.loads(provenance.read_text(encoding="utf-8"))
    if record.get("licence") != LICENCE_ID:
        raise ValueError(f"head asset licence must be {LICENCE_ID}")
    raw = asset.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record.get("asset_sha256"):
        raise ValueError(f"head asset sha256 {digest} does not match its provenance")
    with np.load(asset, allow_pickle=False) as data:
        quantum = float(record["quantum_head_lengths"])
        meshes = {}
        for name in MESH_PARTS:
            vertices = data[f"{name}_vertices"].astype(np.float64) * quantum
            faces = data[f"{name}_faces"].astype(np.int64)
            meshes[name] = Mesh(vertices, faces)
    for name, mesh in meshes.items():
        _check_mesh(name, mesh)
    return HeadAsset(meshes["skin"], meshes["hair"], str(asset), digest)


def load_head_asset(
    asset: Path = ASSET_PATH, provenance: Path = PROVENANCE_PATH
) -> HeadAsset:
    """Load and validate the CC0 head (postcondition: closed, outward, in bounds)."""
    return _load(Path(asset), Path(provenance))


@lru_cache(maxsize=1)
def _default_asset() -> HeadAsset:
    return _load(ASSET_PATH, PROVENANCE_PATH)


def build_mesh_parts(length_m: float, headwear: str) -> list[tuple[str, Mesh, str]]:
    """``(name, mesh, material role)`` of the mesh head scaled to ``length_m``.

    Preconditions: positive finite length; ``headwear`` is ``"none"`` or
    ``"hair"`` (the asset has short hair only, a cap would not fit). The skin
    vertex lands exactly ``length_m`` above the neck point.
    """
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("Head length must be a positive finite number")
    if headwear not in ("none", "hair"):
        raise ValueError(
            f"the CC0 mesh head supports headwear 'none' or 'hair', not {headwear!r}"
            " (a cap needs the parametric head)"
        )
    asset = _default_asset()
    out = [
        ("skin_mesh", Mesh(asset.skin.vertices * length_m, asset.skin.faces), "skin")
    ]
    if headwear == "hair":
        hair = Mesh(asset.hair.vertices * length_m, asset.hair.faces)
        out.append(("hair_mesh", hair, "headwear"))
    return out


def mesh_reference() -> tuple[str, str, str]:
    """``(repo-relative path, licence, provenance path)`` of the asset."""
    rel = ASSET_PATH.relative_to(REPO_ROOT).as_posix()
    return rel, LICENCE_ID, PROVENANCE_PATH.relative_to(REPO_ROOT).as_posix()

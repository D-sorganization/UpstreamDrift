"""Read MakeHuman ``.target`` morphs and mix the macro targets (build-time only).

A target is a list of ``vertex dx dy dz`` offsets of the base mesh (#15299).
MakeHuman's macro sliders (gender, muscle, weight, height, ethnic group) each
weight a family of targets; the weight of one target is the product of the
factors named in its file name, which is what :func:`macro_mix` reproduces for
a young adult. Targets move the joint helper vertices too, so the skeleton of
a :func:`morphed_rig` follows the body.

Targets are CC0 (``LICENSE.md`` section C, and the header of each file); a
target whose header does not say so is refused.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mh_rig_io import Array, MakeHumanRig, sha256_file
from numpy.typing import NDArray

LICENCE_MARK = "CC0"
TARGETS_DIR = ("makehuman", "data", "targets")
MACRO_DIR = "macrodetails"
AGE = "young"


@dataclass(frozen=True)
class MorphTarget:
    """Offsets of some base-mesh vertices, MakeHuman units."""

    indices: NDArray[np.int64]  # (K,) vertex ids, each once
    deltas: Array  # (K, 3)


def read_target(path: Path) -> MorphTarget:
    """Read one ``.target`` file.

    Raises ``ValueError`` if the header is not marked CC0, a line is not
    ``index dx dy dz`` or a vertex repeats. A target may move no vertex:
    MakeHuman's average-muscle, average-weight macro target is the base mesh.
    """
    header: list[str] = []
    indices: list[int] = []
    deltas: list[list[float]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                header.append(line)
                continue
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 4:
                raise ValueError(f"{path.name}:{number} is malformed: {line!r}")
            indices.append(int(fields[0]))
            deltas.append([float(x) for x in fields[1:]])
    if not any(LICENCE_MARK in line for line in header):
        raise ValueError(f"MakeHuman target {path.name} is not marked CC0")
    if len(set(indices)) != len(indices):
        raise ValueError(f"MakeHuman target {path.name} repeats a vertex")
    moved = np.asarray(deltas, float).reshape(-1, 3)
    return MorphTarget(np.asarray(indices, np.int64), moved)


def apply_targets(
    vertices: Array, targets: Sequence[tuple[MorphTarget, float]]
) -> Array:
    """``vertices`` plus each target's offsets times its weight (a new array).

    Precondition: finite weights; every target index is a row of ``vertices``.
    """
    out = np.array(vertices, dtype=float, copy=True)
    for target, weight in targets:
        if not math.isfinite(weight):
            raise ValueError("target weights must be finite")
        if not len(target.indices):
            continue
        if target.indices.min() < 0 or target.indices.max() >= len(out):
            raise ValueError("target moves a vertex outside the mesh")
        out[target.indices] += weight * target.deltas
    return out


def _three_way(value: float, name: str) -> dict[str, float]:
    """Slider in 0..1 as weights of its ``min``, ``average`` and ``max`` ends."""
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"macro setting {name!r} must be in 0..1, got {value!r}")
    return {
        f"min{name}": max(0.0, 1.0 - 2.0 * value),
        f"average{name}": 1.0 - abs(2.0 * value - 1.0),
        f"max{name}": max(0.0, 2.0 * value - 1.0),
    }


def macro_mix(settings: Mapping[str, Any]) -> dict[str, float]:
    """Target path (under ``data/targets``) to weight for a young adult.

    ``settings``: ``male``, ``muscle``, ``weight`` and ``height`` in 0..1 (0.5
    is MakeHuman's average) and ``ethnic``, a table of group to share that is
    empty or sums to one. Postcondition: only targets of non-zero weight; the
    ``universal`` weights sum to one.
    """
    male = float(settings["male"])
    if not (math.isfinite(male) and 0.0 <= male <= 1.0):
        raise ValueError(f"macro setting 'male' must be in 0..1, got {male!r}")
    gender = {"male": male, "female": 1.0 - male}
    muscle = _three_way(float(settings["muscle"]), "muscle")
    weight = _three_way(float(settings["weight"]), "weight")
    height = _three_way(float(settings["height"]), "height")
    ethnic = {str(k): float(v) for k, v in settings["ethnic"].items()}
    if ethnic and not math.isclose(sum(ethnic.values()), 1.0, abs_tol=1e-9):
        raise ValueError("macro setting 'ethnic' shares must sum to one")
    mix: dict[str, float] = {}
    for sex, g in gender.items():
        for group, share in sorted(ethnic.items()):
            mix[f"{MACRO_DIR}/{group}-{sex}-{AGE}.target"] = share * g
        for (m_name, m), (w_name, w) in (
            (a, b) for a in muscle.items() for b in weight.items()
        ):
            stem = f"{sex}-{AGE}-{m_name}-{w_name}"
            mix[f"{MACRO_DIR}/universal-{stem}.target"] = g * m * w
            for h_name in ("minheight", "maxheight"):
                path = f"{MACRO_DIR}/height/{stem}-{h_name}.target"
                mix[path] = g * m * w * height[h_name]
    return {path: value for path, value in mix.items() if value > 0.0}


def morphed_rig(
    rig: MakeHumanRig, makehuman_root: Path, mix: Mapping[str, float]
) -> tuple[MakeHumanRig, list[dict[str, Any]]]:
    """``rig`` with the targets of ``mix`` applied, and what was read.

    The second value lists, per target, its path under ``makehuman_root``, its
    SHA-256 and its weight. Raises ``ValueError`` for a missing target file.
    """
    targets: list[tuple[MorphTarget, float]] = []
    used: list[dict[str, Any]] = []
    for relative, weight in sorted(mix.items()):
        path = makehuman_root.joinpath(*TARGETS_DIR, relative)
        if not path.is_file():
            raise ValueError(f"missing MakeHuman target: {relative}")
        targets.append((read_target(path), float(weight)))
        rel = "/".join((*TARGETS_DIR, relative))
        used.append({"path": rel, "sha256": sha256_file(path), "weight": float(weight)})
    vertices = apply_targets(rig.vertices, targets)
    return dataclasses.replace(rig, vertices=vertices), used

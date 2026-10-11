"""Closure-consistent fitted swings as grip-kinetics input (issue #11739, OSV-7).

The OSV-10 club-face fixtures (``tests/fixtures/club_face/swing_q_<club>.npz``)
hold every second sample of the 1 kHz same-input reference of the shared
ground-support fit (IK marker RMS about 33 mm), replayed identically by every
engine.  Their columns follow the ``coordinate_order`` recorded with the
address poses (``address_poses.json``).  Engines consume them through
:func:`map_coordinates`, which maps columns BY NAME onto the engine's own
coordinate order and refuses to guess: a missing, duplicated or unused
coordinate is an error, never a silent zero.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The fixtures keep every second sample of the 1 kHz reference.
FIXTURE_DT_S = 0.002
FIXTURE_CLUBS = ("driver", "iron7")


@dataclass(frozen=True)
class CoordinateSwing:
    """A uniformly sampled coordinate trajectory in an engine's column order."""

    names: list[str]
    time_s: np.ndarray  # (n,)
    q: np.ndarray  # (n, len(names)), float64
    sha256: str  # of the source file


def map_coordinates(
    source_names: Sequence[str],
    q: np.ndarray,
    target_names: Sequence[str],
    allow_unused: bool = False,
) -> np.ndarray:
    """Reorder the columns of ``q`` from ``source_names`` to ``target_names``.

    Preconditions: names are unique, ``q`` is finite with shape
    ``(n, len(source_names))``, and every target name exists in the source.
    Source columns that the target does not use are an error unless
    ``allow_unused``.  Postcondition: ``out[:, j] == q[:, source.index(t_j)]``.

    Raises:
        ValueError: on any violated precondition.
    """
    src, tgt = list(source_names), list(target_names)
    for label, names in (("source", src), ("target", tgt)):
        if len(set(names)) != len(names):
            raise ValueError(f"{label} coordinate names contain duplicates")
    arr = np.asarray(q, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(src):
        raise ValueError(
            f"q must have shape (n, {len(src)}), got {getattr(arr, 'shape', None)}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("q must be finite")
    missing = [name for name in tgt if name not in src]
    if missing:
        raise ValueError(f"target coordinates missing from the source: {missing}")
    unused = [name for name in src if name not in tgt]
    if unused and not allow_unused:
        raise ValueError(f"source coordinates unused by the target: {unused}")
    index = {name: k for k, name in enumerate(src)}
    return arr[:, [index[name] for name in tgt]]


def load_coordinate_swing(
    npz_path: Path,
    poses_path: Path,
    club: str,
    target_names: Sequence[str],
    dt_s: float = FIXTURE_DT_S,
) -> CoordinateSwing:
    """Load one fixture swing and map it onto ``target_names``.

    Raises:
        ValueError: for an unknown club, a non-positive ``dt_s`` or a
            coordinate set that does not map (see :func:`map_coordinates`).
    """
    if club not in FIXTURE_CLUBS:
        raise ValueError(f"club must be one of {FIXTURE_CLUBS}, got {club!r}")
    if not dt_s > 0.0:
        raise ValueError("dt_s must be positive")
    raw = Path(npz_path).read_bytes()
    with np.load(npz_path, allow_pickle=False) as data:
        q = np.asarray(data["q"], dtype=float)
    poses = json.loads(Path(poses_path).read_text(encoding="utf-8"))["poses"]
    source = poses[club]["coordinate_order"]
    mapped = map_coordinates(source, q, target_names)
    return CoordinateSwing(
        names=list(target_names),
        time_s=np.arange(q.shape[0]) * float(dt_s),
        q=mapped,
        sha256=hashlib.sha256(raw).hexdigest(),
    )

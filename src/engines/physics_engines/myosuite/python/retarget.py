"""Pure-numpy coordinate retargeting for MyoSuite kinematic replay (MS-52, #10345)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition

Array: TypeAlias = NDArray[np.float64]

_DEFAULT_MAP = Path(__file__).with_name("coordinate_map_anthro.json")


@dataclass(frozen=True)
class RetargetMap:
    """Document coordinate order projected into a MyoSuite joint vector."""

    source_names: tuple[str, ...]
    target_names: tuple[str, ...]
    source_to_target: dict[str, tuple[int, float]]
    chain_order: tuple[str, ...]
    neutral_target: Array
    omitted_target: tuple[str, ...]
    marker_sites: dict[str, dict[str, Any]] | None = None

    @property
    def n_source(self) -> int:
        return len(self.source_names)

    @property
    def n_target(self) -> int:
        return len(self.target_names)

    def project_to_source(self, q_target: Array) -> Array:
        """Best-effort inverse for mapped coordinates (identity on 1:1 entries)."""
        q = np.asarray(q_target, dtype=np.float64).reshape(-1)
        out = (
            self.neutral_target[: self.n_source].copy()
            if self.n_source
            else np.zeros(0)
        )
        if self.n_source != len(self.source_names):
            out = np.zeros(self.n_source, dtype=np.float64)
        out = np.zeros(self.n_source, dtype=np.float64)
        for src_name, (tgt_idx, sign) in self.source_to_target.items():
            src_idx = self.source_names.index(src_name)
            out[src_idx] = float(q[tgt_idx]) / sign if sign != 0 else 0.0
        return out


@precondition(lambda doc: isinstance(doc, Mapping), "coordinate map document required")
def load_retarget_map(doc: Mapping[str, Any]) -> RetargetMap:
    """Load a retarget map from the MS-51 coordinate map JSON document."""
    source_names = tuple(str(x) for x in doc["source_coordinates"])
    target_names = tuple(str(x) for x in doc["target_coordinates"])
    name_to_source = {name: idx for idx, name in enumerate(source_names)}
    source_to_target: dict[str, tuple[int, float]] = {}
    for entry in doc.get("mappings", ()):
        src = str(entry["source"])
        tgt = str(entry["target"])
        sign = float(entry.get("sign", 1.0))
        if src not in name_to_source:
            raise ValueError(f"Unknown source coordinate {src!r}")
        if tgt not in target_names:
            raise ValueError(f"Unknown target coordinate {tgt!r}")
        source_to_target[src] = (target_names.index(tgt), sign)
    neutral = doc.get("neutral_target", {})
    neutral_target = np.array(
        [float(neutral.get(name, 0.0)) for name in target_names], dtype=np.float64
    )
    mapped_targets = {target_names[t_idx] for t_idx, _ in source_to_target.values()}
    omitted_target = tuple(name for name in target_names if name not in mapped_targets)
    chain = tuple(str(x) for x in doc.get("chain_order", target_names))
    markers = doc.get("marker_sites")
    return RetargetMap(
        source_names=source_names,
        target_names=target_names,
        source_to_target=source_to_target,
        chain_order=chain,
        neutral_target=neutral_target,
        omitted_target=omitted_target,
        marker_sites=dict(markers) if isinstance(markers, Mapping) else None,
    )


def default_retarget_map() -> RetargetMap:
    """Load the bundled anthro coordinate map shipped with the engine adapter."""
    return load_retarget_map(json.loads(_DEFAULT_MAP.read_text()))


@precondition(
    lambda q_source, rmap: q_source.shape[-1] == rmap.n_source, "source width"
)
@postcondition(
    lambda result: result.ndim == 1 and bool(np.all(np.isfinite(result))),
    "finite 1-D target vector",
)
def retarget_frame(q_source: Array, rmap: RetargetMap) -> Array:
    """Map one source coordinate vector into the MyoSuite joint order."""
    q = np.asarray(q_source, dtype=np.float64).reshape(-1)
    if q.shape[0] != rmap.n_source:
        raise ValueError(
            f"Expected {rmap.n_source} source coordinates, got {q.shape[0]}"
        )
    out = rmap.neutral_target.copy()
    for src_name, (tgt_idx, sign) in rmap.source_to_target.items():
        src_idx = rmap.source_names.index(src_name)
        out[tgt_idx] = sign * float(q[src_idx])
    return interpolate_unmapped(out, rmap)


def interpolate_unmapped(q_target: Array, rmap: RetargetMap) -> Array:
    """Fill unmapped target coordinates from the nearest mapped chain ancestor."""
    out = np.asarray(q_target, dtype=np.float64).copy()
    mapped = {
        name: idx
        for name, idx in ((n, rmap.target_names.index(n)) for n in rmap.target_names)
        if any(t_idx == idx for t_idx, _ in rmap.source_to_target.values())
    }
    chain_index = {name: idx for idx, name in enumerate(rmap.chain_order)}
    for name in rmap.target_names:
        idx = rmap.target_names.index(name)
        if name in mapped:
            continue
        ancestor_value = float(rmap.neutral_target[idx])
        for candidate in sorted(
            (n for n in mapped if chain_index.get(n, -1) < chain_index.get(name, 999)),
            key=lambda n: chain_index.get(n, -1),
            reverse=True,
        ):
            ancestor_value = float(out[mapped[candidate]])
            break
        out[idx] = ancestor_value
    return out


@precondition(lambda q_traj, rmap: q_traj.ndim == 2, "trajectory must be 2-D")
def retarget_trajectory(q_traj: Array, rmap: RetargetMap) -> Array:
    """Retarget a (frames, n_source) trajectory."""
    q = np.asarray(q_traj, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != rmap.n_source:
        raise ValueError(
            f"Expected trajectory shape (frames, {rmap.n_source}), got {q.shape}"
        )
    return np.stack([retarget_frame(row, rmap) for row in q], axis=0)


def source_coordinate_index(
    coordinate_order: Sequence[str], rmap: RetargetMap
) -> Array:
    """Build gather indices aligning a candidate vector to the map source order."""
    lookup = {name: idx for idx, name in enumerate(coordinate_order)}
    missing = [name for name in rmap.source_names if name not in lookup]
    if missing:
        raise ValueError(f"Candidate missing mapped coordinates: {missing[:5]}")
    return np.array([lookup[name] for name in rmap.source_names], dtype=np.int64)

"""Explicit reduced-model to upper/full-body topology mappings (CO-05 #10609).

Continuation from a reduced seed is allowed only through a named mapping.
Pelvis teleport, unlimited root actuation, and pasted club animation are
refused by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.tour_baselines.registry import (
    get_golf_model,
    init_default_registry,
)

__all__ = [
    "TopologyMapping",
    "get_topology_mapping",
    "map_reduced_seed_to_body",
]


@dataclass(frozen=True)
class TopologyMapping:
    """Named DOF map from a reduced source model into a body target."""

    source_model_id: str
    target_model_id: str
    source_nq: int
    target_nq: int
    joint_map: tuple[tuple[int, int], ...]
    pelvis_indices: tuple[int, ...]
    allows_pelvis_teleport: bool = False
    allows_unlimited_root_actuation: bool = False
    allows_pasted_club_animation: bool = False

    def __post_init__(self) -> None:
        if not self.source_model_id or not self.target_model_id:
            raise ValueError("source_model_id and target_model_id required")
        if self.source_nq < 1 or self.target_nq < 1:
            raise ValueError("source_nq and target_nq must be >= 1")
        for src, dst in self.joint_map:
            if src < 0 or src >= self.source_nq:
                raise ValueError(f"joint_map source index out of range: {src}")
            if dst < 0 or dst >= self.target_nq:
                raise ValueError(f"joint_map target index out of range: {dst}")
        for idx in self.pelvis_indices:
            if idx < 0 or idx >= self.target_nq:
                raise ValueError(f"pelvis index out of range: {idx}")
        if self.allows_pelvis_teleport:
            raise ValueError("pelvis teleport is refused for club-only candidates")
        if self.allows_unlimited_root_actuation:
            raise ValueError("unlimited root actuation is refused")
        if self.allows_pasted_club_animation:
            raise ValueError("pasted club animation is refused")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_model_id": self.source_model_id,
            "target_model_id": self.target_model_id,
            "source_nq": self.source_nq,
            "target_nq": int(self.target_nq),
            "joint_map": [list(pair) for pair in self.joint_map],
            "pelvis_indices": list(self.pelvis_indices),
            "allows_pelvis_teleport": self.allows_pelvis_teleport,
            "allows_unlimited_root_actuation": self.allows_unlimited_root_actuation,
            "allows_pasted_club_animation": self.allows_pasted_club_animation,
        }


def _default_joint_map(source_nq: int, target_nq: int) -> tuple[tuple[int, int], ...]:
    """Map leading reduced joints into non-pelvis body DOFs."""
    # Reserve the first three target DOFs as pelvis/root (held at prior).
    pelvis_count = min(3, target_nq)
    free_start = pelvis_count
    n_map = min(source_nq, max(0, target_nq - free_start))
    return tuple((i, free_start + i) for i in range(n_map))


def get_topology_mapping(
    *,
    source_model_id: str,
    target_model_id: str,
) -> TopologyMapping:
    """Return the explicit topology map for a reduced→body continuation."""
    if not source_model_id or not target_model_id:
        raise ValueError("source_model_id and target_model_id required")
    init_default_registry()
    source = get_golf_model(source_model_id)
    target = get_golf_model(target_model_id)
    source_nq = int(source.dof)
    target_nq = int(target.dof)
    pelvis = tuple(range(min(3, target_nq)))
    return TopologyMapping(
        source_model_id=source_model_id,
        target_model_id=target_model_id,
        source_nq=source_nq,
        target_nq=target_nq,
        joint_map=_default_joint_map(source_nq, target_nq),
        pelvis_indices=pelvis,
        allows_pelvis_teleport=False,
        allows_unlimited_root_actuation=False,
        allows_pasted_club_animation=False,
    )


@precondition(
    lambda q_reduced, mapping, target_nq: (
        isinstance(q_reduced, np.ndarray) and target_nq is not None
    ),
    "q_reduced and target_nq required",
)
@postcondition(
    lambda result: isinstance(result, np.ndarray) and result.ndim == 1,
    "must return 1-D ndarray",
)
def map_reduced_seed_to_body(
    *,
    q_reduced: NDArray[np.floating],
    mapping: TopologyMapping | None,
    target_nq: int,
) -> NDArray[np.float64]:
    """Lift a reduced seed into body coordinates through an explicit mapping.

    Raises
    ------
    ValueError
        When ``mapping`` is missing, dimensions disagree, or values are
        nonfinite. Pelvis/root DOFs stay at the zero prior (no teleport).
    """
    if mapping is None:
        raise ValueError(
            "topology mapping required for reduced-seed continuation; "
            "implicit lift refused"
        )
    q_src = np.asarray(q_reduced, dtype=np.float64)
    if q_src.ndim != 1:
        raise ValueError("q_reduced must be 1-D")
    if not np.all(np.isfinite(q_src)):
        raise ValueError("q_reduced must be finite")
    if int(target_nq) != int(mapping.target_nq):
        raise ValueError(
            f"target_nq={target_nq} incompatible with mapping "
            f"target_nq={mapping.target_nq}"
        )
    # Shorter seeds pad; longer seeds keep only the leading mapped DOFs.
    if q_src.size < mapping.source_nq:
        padded = np.zeros(mapping.source_nq, dtype=np.float64)
        padded[: q_src.size] = q_src
        q_src = padded
    elif q_src.size > mapping.source_nq:
        q_src = q_src[: mapping.source_nq].copy()

    q_body = np.zeros(int(mapping.target_nq), dtype=np.float64)
    for src, dst in mapping.joint_map:
        q_body[dst] = q_src[src]
    # Pelvis/root remain at prior zero — never free teleport.
    for idx in mapping.pelvis_indices:
        q_body[idx] = 0.0
    return q_body

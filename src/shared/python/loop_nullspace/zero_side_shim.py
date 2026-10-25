"""Legacy allocator zero-side consistency checks (NSX-24, #12567).

The modern three-case diagnostic is :func:`~.decomposition.zero_side_solve` on a
:class:`~.contracts.ClosedLoopSnapshot`.  Legacy shims that only receive
``tau_rnea`` and Jacobians use :func:`check_zero_side_consistency` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class ZeroSideConsistency(str, Enum):
    """Three-case classification for locking actuated coordinates to zero torque."""

    CONSISTENT_UNIQUE = "consistent_unique"
    CONSISTENT_NONUNIQUE = "consistent_nonunique"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True)
class ZeroSideCheckResult:
    """Algebraic consistency of zeroing selected actuated coordinates."""

    status: ZeroSideConsistency
    residual_norm: float
    residual_scaled: float
    nullity: int


def check_zero_side_consistency(
    tau_rnea: Array,
    actuated_indices: Sequence[int] | NDArray[np.integer[Any]],
    zero_joint_indices: Sequence[int] | NDArray[np.integer[Any]],
    j_ground: Array | None = None,
    j_grip: Array | None = None,
    *,
    atol: float = 1e-4,
) -> ZeroSideCheckResult:
    """Classify whether zeroing ``zero_joint_indices`` is consistent with ``tau_rnea``.

    Forms the balance ``S_free^T tau_free + J_g^T f_g + J_c^T lambda = b`` with
    the zeroed coordinates removed from the free torque block.  Raises
    ``ValueError`` when ``atol`` is not positive or index sets are empty.
    """
    if atol <= 0.0 or not np.isfinite(atol):
        raise ValueError(f"atol must be positive and finite, got {atol}")
    b = np.asarray(tau_rnea, dtype=np.float64).ravel()
    if b.ndim != 1:
        raise ValueError("tau_rnea must be a 1-D vector")
    nv = b.size
    act_set: set[int] = set()
    for raw_idx in actuated_indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= nv:
            raise ValueError(f"actuated index {idx} out of range for nv={nv}")
        act_set.add(idx)

    zero_set = {int(i) for i in zero_joint_indices}
    if not zero_set:
        raise ValueError("zero_joint_indices must be non-empty")
    for idx in sorted(zero_set):
        if idx < 0 or idx >= nv:
            raise ValueError(f"zero joint index {idx} out of range for nv={nv}")
        if idx not in act_set:
            raise ValueError(f"zero joint index {idx} must belong to actuated_indices")
    zero_rows = np.asarray(sorted(zero_set), dtype=np.intp)
    b_zero = b[zero_rows]

    blocks: list[Array] = []
    if j_ground is not None and j_ground.size > 0:
        jg = np.asarray(j_ground, dtype=np.float64)
        blocks.append(jg[:, zero_rows].T)
    if j_grip is not None and j_grip.size > 0:
        jc = np.asarray(j_grip, dtype=np.float64)
        blocks.append(jc[:, zero_rows].T)
    a_mat = (
        np.hstack(blocks) if blocks else np.zeros((zero_rows.size, 0), dtype=np.float64)
    )

    if a_mat.shape[1] == 0:
        res_norm = float(np.linalg.norm(b_zero, ord=np.inf))
        rank = 0
    else:
        sol, _, rank_val, _ = np.linalg.lstsq(a_mat, b_zero, rcond=None)
        res_norm = float(np.linalg.norm(a_mat @ sol - b_zero, ord=np.inf))
        rank = int(rank_val)

    scale = max(1.0, float(np.linalg.norm(b_zero, ord=np.inf)))
    res_scaled = res_norm / scale
    nullity = max(0, int(a_mat.shape[1]) - rank)
    if res_norm > atol:
        status = ZeroSideConsistency.INCONSISTENT
    elif nullity == 0:
        status = ZeroSideConsistency.CONSISTENT_UNIQUE
    else:
        status = ZeroSideConsistency.CONSISTENT_NONUNIQUE
    return ZeroSideCheckResult(
        status=status,
        residual_norm=res_norm,
        residual_scaled=res_scaled,
        nullity=nullity,
    )

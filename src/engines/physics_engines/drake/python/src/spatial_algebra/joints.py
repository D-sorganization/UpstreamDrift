"""Drake-facing joint helpers (shared implementation)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.shared.python.spatial_algebra.joints import (
    S_PX,
    S_PY,
    S_PZ,
    S_RX,
    S_RY,
    S_RZ,
)
from src.shared.python.spatial_algebra.joints import jcalc as _shared_jcalc


def jcalc(
    jtype: str, q: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return transform and motion subspace for the joint type."""
    xj_transform, s_subspace, _ = _shared_jcalc(jtype, q)
    return xj_transform, s_subspace


__all__ = ["S_PX", "S_PY", "S_PZ", "S_RX", "S_RY", "S_RZ", "jcalc"]

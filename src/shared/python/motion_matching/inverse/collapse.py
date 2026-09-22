"""Mode-collapse diagnostics and retained cVAE plateau evidence (NM-06).

The historical inverse cVAE exhibited a hard reconstruction plateau on the
compact dataset (val_recon stuck at the mean-prediction baseline). That
evidence is retained here as a named constant — not re-invented as success.
Mixture / multi-proposal ablations must prove genuine diversity via these
diagnostics before claiming multimodal feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CVAE_PLATEAU_EVIDENCE",
    "ModeCollapseDiagnostic",
    "diagnose_mode_collapse",
]

# Retained plateau evidence from the inverse cVAE investigation
# (see inverse/regressor.py module docstring and issue #4076 follow-ups).
CVAE_PLATEAU_EVIDENCE: Mapping[str, bool | str] = {
    "val_recon_plateau": True,
    "mean_prediction_baseline": True,
    "note": (
        "cVAE val_recon plateaued at the mean-prediction baseline on the "
        "compact dataset; deterministic regressor became the production path. "
        "Mixture proposals require collapse diagnostics — plateau is not success."
    ),
}


@dataclass(frozen=True, slots=True)
class ModeCollapseDiagnostic:
    """Summary of pairwise diversity across proposal modes/samples."""

    is_collapsed: bool
    effective_mode_count: int
    min_pairwise_l2: float
    mean_pairwise_l2: float
    n_samples: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "is_collapsed": self.is_collapsed,
            "effective_mode_count": self.effective_mode_count,
            "min_pairwise_l2": self.min_pairwise_l2,
            "mean_pairwise_l2": self.mean_pairwise_l2,
            "n_samples": self.n_samples,
        }


def diagnose_mode_collapse(
    proposals: NDArray[np.floating],
    *,
    min_pairwise_l2: float = 0.05,
) -> ModeCollapseDiagnostic:
    """Flag collapse when pairwise L2 distances fall below ``min_pairwise_l2``.

    Design by Contract:
    - ``proposals`` finite, shape ``(n_modes, control_dim)`` with ``n_modes >= 1``.
    - ``min_pairwise_l2`` finite and > 0.
    """
    arr = np.asarray(proposals, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError(
            f"proposals must have shape (n_modes, control_dim); got {arr.shape}"
        )
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("proposals values must be finite")
    if not np.isfinite(min_pairwise_l2) or min_pairwise_l2 <= 0.0:
        raise ValueError("min_pairwise_l2 must be a positive finite float")

    n = int(arr.shape[0])
    if n == 1:
        return ModeCollapseDiagnostic(
            is_collapsed=True,
            effective_mode_count=1,
            min_pairwise_l2=0.0,
            mean_pairwise_l2=0.0,
            n_samples=1,
        )

    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(float(np.linalg.norm(arr[i] - arr[j])))
    dist_arr = np.asarray(distances, dtype=np.float64)
    min_d = float(dist_arr.min())
    mean_d = float(dist_arr.mean())
    # Count modes that are at least min_pairwise_l2 from all earlier kept modes.
    kept: list[int] = [0]
    for i in range(1, n):
        if all(float(np.linalg.norm(arr[i] - arr[k])) >= min_pairwise_l2 for k in kept):
            kept.append(i)
    effective = len(kept)
    return ModeCollapseDiagnostic(
        is_collapsed=effective <= 1 or min_d < min_pairwise_l2,
        effective_mode_count=effective,
        min_pairwise_l2=min_d,
        mean_pairwise_l2=mean_d,
        n_samples=n,
    )

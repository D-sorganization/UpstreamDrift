"""Fitter implementations sub-package.

Concrete :class:`~body_part_viz.contracts.ShapeFitter` strategies covering
the realistic mocap-to-shape attachments. Each fitter ships in its own
module so adding a new strategy does not touch the existing ones.
"""

from __future__ import annotations

from .between_two import BetweenTwoMarkersFitter
from .cluster_kabsch import ClusterKabschFitter
from .procrustes_anisotropic import ProcrustesAnisotropicFitter

__all__ = [
    "BetweenTwoMarkersFitter",
    "ClusterKabschFitter",
    "ProcrustesAnisotropicFitter",
]

"""Anthropometric estimators.

This sub-package hosts pluggable estimators that turn published
anthropometric ratios (de Leva, Dempster, Zatsiorsky-Seluyanov)
plus per-subject scalars (height, mass, segment length) into a
fully validated :class:`anthropometrics.SegmentProperties`, and
estimator implementations producing :class:`~anthropometrics.SegmentProperties`
or related primitives from various input modalities (mocap, regression
tables, manual measurements).

Public estimators
-----------------
* :func:`from_inertia_calc.inertia_from_cylinder`
* :func:`from_inertia_calc.inertia_from_ellipsoid`
* :func:`from_inertia_calc.inertia_from_gyration_radii`
* :func:`from_inertia_calc.build_segment_properties_with_inertia`
* :func:`from_mocap.estimate_segment_lengths_from_markers`
"""

from __future__ import annotations

from .from_inertia_calc import (
    build_segment_properties_with_inertia,
    inertia_from_cylinder,
    inertia_from_ellipsoid,
    inertia_from_gyration_radii,
)
from .from_mocap import SegmentDef, estimate_segment_lengths_from_markers

__all__ = [
    "SegmentDef",
    "build_segment_properties_with_inertia",
    "estimate_segment_lengths_from_markers",
    "inertia_from_cylinder",
    "inertia_from_ellipsoid",
    "inertia_from_gyration_radii",
]

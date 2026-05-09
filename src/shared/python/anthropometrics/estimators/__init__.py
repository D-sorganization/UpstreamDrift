"""Anthropometric inertia estimators.

This sub-package hosts pluggable estimators that turn published
anthropometric ratios (de Leva, Dempster, Zatsiorsky-Seluyanov)
plus per-subject scalars (height, mass, segment length) into a
fully validated :class:`anthropometrics.SegmentProperties`.

Public estimators
-----------------
* :func:`from_inertia_calc.inertia_from_cylinder`
* :func:`from_inertia_calc.inertia_from_ellipsoid`
* :func:`from_inertia_calc.inertia_from_gyration_radii`
* :func:`from_inertia_calc.build_segment_properties_with_inertia`
"""

from __future__ import annotations

from .from_inertia_calc import (
    build_segment_properties_with_inertia,
    inertia_from_cylinder,
    inertia_from_ellipsoid,
    inertia_from_gyration_radii,
)

__all__ = [
    "build_segment_properties_with_inertia",
    "inertia_from_cylinder",
    "inertia_from_ellipsoid",
    "inertia_from_gyration_radii",
]

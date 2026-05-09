"""Anthropometric inertia estimators and regression-based estimators.

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

Regression-based :class:`~anthropometrics.Estimator` implementations
--------------------------------------------------------------------
* :class:`from_de_leva.DeLevaEstimator` — wraps the
  ``humanoid_character_builder.core.anthropometry`` ratio table
  (de Leva 1996). The ratios are NOT duplicated — the wrapper
  reads the same single source of truth used by the rest of the
  codebase.
* :class:`from_dempster.DempsterEstimator` — loads
  :file:`ratios/dempster_1955.json`.
* :class:`from_zatsiorsky.ZatsiorskyEstimator` — loads
  :file:`ratios/zatsiorsky_seluyanov_1985.json`.

Each estimator produces a fully-validated
:class:`~anthropometrics.SubjectAnthropometrics` from raw subject
height + mass.
"""

from __future__ import annotations

from .from_de_leva import DeLevaEstimator
from .from_dempster import DempsterEstimator
from .from_inertia_calc import (
    build_segment_properties_with_inertia,
    inertia_from_cylinder,
    inertia_from_ellipsoid,
    inertia_from_gyration_radii,
)
from .from_mocap import SegmentDef, estimate_segment_lengths_from_markers
from .from_zatsiorsky import ZatsiorskyEstimator

__all__ = [
    "DeLevaEstimator",
    "DempsterEstimator",
    "SegmentDef",
    "ZatsiorskyEstimator",
    "build_segment_properties_with_inertia",
    "estimate_segment_lengths_from_markers",
    "inertia_from_cylinder",
    "inertia_from_ellipsoid",
    "inertia_from_gyration_radii",
]

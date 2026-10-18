"""Pelvis skin width: subject hip breadth or a stature prior, plus containment.

HRG-C4 (#13187). Only the femoral heads belong inside the pelvis solid (the
thigh blends in through the hip ball), so the width is

1. the subject's measured hip breadth, when the spec provides one; else
2. ``HIP_BREADTH_STATURE_FRACTION * stature`` (the ANSUR II male mean, a population prior, not a measurement); else
3. the spec's own hint width (``spec_hint``);

raised only when the femoral heads need more room: each hip joint centre
(HJC) keeps ``FEMORAL_HEAD_RADIUS_M + HIP_CONTAINMENT_MARGIN_M`` of lateral
clearance. A raise is reported as ``containment_dominated``, never clipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FEMORAL_HEAD_RADIUS_M = 0.025
HIP_CONTAINMENT_MARGIN_M = 0.010
# Standing hip breadth / stature, per-subject ratio, ANSUR II (2012 U.S. Army
# anthropometric survey; Gordon et al. 2015, NATICK/TR-15/007; public CSV from
# https://www.openlab.psu.edu/ansur2/). Column ``hipbreadth`` over ``stature``.
# Male, n = 4082 (sha256 0547aea0170e5293de519389981135e75803f02e6d40c77ace740decc0caac64):
#   mean 0.1970, sd 0.0129, median 0.1964, p5 0.1770, p95 0.2200.
# The spec records no subject sex; the AN-1 driver subject is male (de Leva
# 1996 male table), so the male mean is used. A female subject needs 0.2174
# (n = 1986, sd 0.0153, p5 0.1928, p95 0.2430, sha256 ed7e800a...1757a), or
# ``subject.hip_breadth_m``.
HIP_BREADTH_STATURE_FRACTION = 0.1970
_MAX_LEVEL = 0.999


@dataclass(frozen=True)
class PelvisWidth:
    """Resolved pelvis skin width (full mediolateral width at the widest level).

    ``target_m`` is the measured/prior width, ``containment_m`` the width the
    HJCs require (section correction included) and ``width_m`` the larger.
    ``source`` is ``measured``, ``stature_prior`` or ``spec_hint``;
    ``status`` is ``target`` (a measured value), ``prior`` or
    ``containment_dominated``.
    """

    target_m: float
    containment_m: float
    width_m: float
    source: str
    status: str


def resolve_pelvis_width(
    hjc_lateral_m: np.ndarray,
    hjc_level: np.ndarray,
    *,
    hip_breadth_m: float | None = None,
    stature_m: float | None = None,
    hint_width_m: float | None = None,
) -> PelvisWidth:
    """Pelvis width for HJCs at lateral offsets from the midline.

    Preconditions: equal-length 1-D arrays; ``hjc_level`` in [0, 1) is each
    HJC's height above the widest level over the ellipsoid half height (the
    ``1/sqrt(1-level^2)`` section correction); any given size is positive and
    finite. Postcondition: ``width_m >= containment_m`` and
    ``width_m >= target_m``.
    """
    lateral = np.abs(np.asarray(hjc_lateral_m, dtype=float))
    level = np.asarray(hjc_level, dtype=float)
    if lateral.ndim != 1 or lateral.shape != level.shape or lateral.size == 0:
        raise ValueError("hjc_lateral_m and hjc_level must be equal-length 1-D arrays")
    if np.any(level < 0.0) or np.any(level >= 1.0):
        raise ValueError("hjc_level must lie in [0, 1)")
    for name, value in (
        ("hip_breadth_m", hip_breadth_m),
        ("stature_m", stature_m),
        ("hint_width_m", hint_width_m),
    ):
        if value is not None and not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    clearance = FEMORAL_HEAD_RADIUS_M + HIP_CONTAINMENT_MARGIN_M
    need = (lateral + clearance) / np.sqrt(1.0 - np.minimum(level, _MAX_LEVEL) ** 2)
    containment = 2.0 * float(need.max())
    if hip_breadth_m is not None:
        target, source, kind = hip_breadth_m, "measured", "target"
    elif stature_m is not None:
        target = HIP_BREADTH_STATURE_FRACTION * stature_m
        source, kind = "stature_prior", "prior"
    elif hint_width_m is not None:
        target, source, kind = hint_width_m, "spec_hint", "prior"
    else:
        raise ValueError("need hip_breadth_m, stature_m or hint_width_m")
    if containment > target:
        return PelvisWidth(
            target, containment, containment, source, "containment_dominated"
        )
    return PelvisWidth(target, containment, target, source, kind)

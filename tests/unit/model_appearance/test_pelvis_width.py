"""Pelvis width from a hip-breadth prior with femoral-head containment (HRG-C4).

Issue #13187 (epic #13160). C1 sized the pelvis to enclose each whole thigh
radius, giving a 0.394 m skin width. Only the femoral heads belong inside the
pelvis: the width is now the subject's measured hip breadth, else a
stature-scaled prior, raised only when containment needs more (reported as
``containment_dominated``).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.body_meshes import pelvis_width_report
from src.shared.python.model_appearance.pelvis_width import (
    FEMORAL_HEAD_RADIUS_M,
    HIP_BREADTH_STATURE_FRACTION,
    HIP_CONTAINMENT_MARGIN_M,
    resolve_pelvis_width,
)
from tests.unit.model_appearance.test_pelvis_loft import (
    ANTHRO,
    BEST_FIT_HIPS,
    SYMMETRIC_HIPS,
    _hjcs,
    _pelvis_body,
    _skin,
    _with_hips,
    body_sections,
)

pytestmark = pytest.mark.unit

STATURE = 1.71
# 220 mm (fitted, pre-C3) and 178 mm (corrected, post-C3) HJC spacings.
HIPS_220 = {"hip_r": (0.0, -0.110, -0.08), "hip_l": (0.0, 0.110, -0.08)}
HIPS_178 = {"hip_r": (0.0, -0.089, -0.08), "hip_l": (0.0, 0.089, -0.08)}
HIPS_WIDE = {"hip_r": (0.0, -0.160, -0.08), "hip_l": (0.0, 0.160, -0.08)}
TARGET = HIP_BREADTH_STATURE_FRACTION * STATURE


def _width(spec: dict[str, Any]) -> float:
    body = _pelvis_body(spec)
    verts = _skin(spec, body).mesh_in_body_frame.vertices
    ml = body_sections(spec)[body].ml_axis
    return float(np.ptp(verts @ ml))


def _lateral_clearance(spec: dict[str, Any]) -> float:
    body = _pelvis_body(spec)
    verts = _skin(spec, body).mesh_in_body_frame.vertices
    ml = body_sections(spec)[body].ml_axis
    planes = ConvexHull(verts).equations
    out = []
    for p in _hjcs(spec, body):
        for sign in (1.0, -1.0):
            slope = planes[:, :3] @ (sign * ml)
            dist = -(planes[:, :3] @ p + planes[:, 3])
            out.append(float((dist[slope > 1e-9] / slope[slope > 1e-9]).min()))
    return min(out)


@pytest.mark.parametrize("hips", [HIPS_220, HIPS_178], ids=["220mm", "178mm"])
def test_skin_width_equals_the_prior_target_within_1mm(hips: dict) -> None:
    spec = _with_hips(hips)
    assert abs(_width(spec) - TARGET) < 1e-3
    report = pelvis_width_report(spec)[_pelvis_body(spec)]
    assert report.status == "prior"
    assert report.source == "stature_prior"
    assert abs(report.width_m - TARGET) < 1e-9


def test_width_is_independent_of_the_hjc_spacing() -> None:
    assert abs(_width(_with_hips(HIPS_220)) - _width(_with_hips(HIPS_178))) < 1e-3


@pytest.mark.parametrize("hips", [HIPS_220, HIPS_178], ids=["220mm", "178mm"])
def test_hjcs_have_35mm_lateral_clearance(hips: dict) -> None:
    need = FEMORAL_HEAD_RADIUS_M + HIP_CONTAINMENT_MARGIN_M
    assert need == pytest.approx(0.035)
    assert _lateral_clearance(_with_hips(hips)) >= need


def test_containment_dominates_and_is_reported_when_it_needs_more_width() -> None:
    spec = _with_hips(HIPS_WIDE)
    body = _pelvis_body(spec)
    report = pelvis_width_report(spec)[body]
    assert report.status == "containment_dominated"
    assert report.width_m > report.target_m
    assert abs(_width(spec) - report.width_m) < 1e-3
    assert _lateral_clearance(spec) >= 0.035 - 1e-3


def test_measured_hip_breadth_overrides_the_prior() -> None:
    spec = _with_hips(HIPS_178)
    spec["subject"]["hip_breadth_m"] = 0.31
    report = pelvis_width_report(spec)[_pelvis_body(spec)]
    assert report.source == "measured"
    assert abs(_width(spec) - 0.31) < 1e-3


def test_no_stature_or_breadth_reports_the_hint_fallback() -> None:
    spec = _with_hips(HIPS_178)
    del spec["subject"]["stature_m"]
    report = pelvis_width_report(spec)[_pelvis_body(spec)]
    assert report.source == "spec_hint"
    assert report.width_m > 0


def test_lower_pole_is_referenced_from_the_mean_hjc_height() -> None:
    spec = _with_hips(BEST_FIT_HIPS)
    body = _pelvis_body(spec)
    verts = _skin(spec, body).mesh_in_body_frame.vertices
    up = body_sections(spec)[body].trunk_up_axis()
    z = _hjcs(spec, body) @ up
    expected = float(z.mean()) - library.PELVIS_CROTCH_DROP_M
    assert float((verts @ up).min()) == pytest.approx(expected, abs=1e-3)
    assert float((verts @ up).min()) <= float(z.min()) - 0.025


def test_resolver_preconditions() -> None:
    lat, lvl = np.array([0.09, 0.09]), np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        resolve_pelvis_width(lat, lvl, hip_breadth_m=-0.1)
    with pytest.raises(ValueError):
        resolve_pelvis_width(lat, np.array([0.0]), stature_m=1.7)
    with pytest.raises(ValueError):
        resolve_pelvis_width(lat, np.array([0.0, 1.0]), stature_m=1.7)
    r = resolve_pelvis_width(lat, lvl, stature_m=1.7)
    assert r.width_m >= 2 * (0.09 + 0.035)


def test_json_roundtrip_of_the_spec_is_unchanged() -> None:
    assert (
        json.loads(ANTHRO.read_text(encoding="utf-8"))["subject"]["stature_m"] == 1.71
    )
    assert SYMMETRIC_HIPS  # fixture is shared with the C1 gates


# ANSUR II male standing hip breadth / stature, per-subject ratio, n = 4082
# (public CSV sha256 0547aea0...caac64): mean 0.1970, sd 0.0129, p5 0.1770,
# p95 0.2200. Female (n = 1986) mean is 0.2174; a female subject needs that.
ANSUR_II_MALE_MEAN = 0.1970
ANSUR_II_MALE_SD = 0.0129


def test_hip_breadth_prior_matches_the_ansur_ii_male_mean() -> None:
    assert pytest.approx(ANSUR_II_MALE_MEAN, abs=5e-4) == HIP_BREADTH_STATURE_FRACTION
    assert abs(HIP_BREADTH_STATURE_FRACTION - ANSUR_II_MALE_MEAN) < ANSUR_II_MALE_SD

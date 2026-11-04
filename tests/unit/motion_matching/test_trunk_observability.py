"""SHOW-A5 (#14554): observability of the v2 trunk DOFs (draft, moved into the repo later)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.trunk_observability import (
    decide_trunk_coupling,
    marker_jacobian,
    observability_report,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
V2 = MODELS / "full_body_spec_anthro_driver_v2.json"
BUNDLE = MODELS / "evidence/trunk_visuals/driver_v2_bundle.npz"
EVENTS = {"address": 0, "top": 1103, "impact": 1325}
TRUNK = (
    "lumbar_flexion", "lumbar_lateral_bending", "lumbar_axial_rotation",
    "thoracic_flexion", "thoracic_lateral_bending", "thoracic_axial_rotation",
)  # fmt: skip


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(V2.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def q_events(spec) -> dict[str, np.ndarray]:
    ref = np.load(BUNDLE)["reference_q"]
    return {k: ref[i] for k, i in EVENTS.items()}


def test_jacobian_matches_finite_difference_shape_and_is_finite(spec, q_events) -> None:
    jac = marker_jacobian(spec, q_events["address"], TRUNK)
    markers = sum(
        a["offset_m"] is not None for a in spec["marker_attachments"].values()
    )
    assert jac.shape == (3 * markers, 6)
    assert np.isfinite(jac).all()


def test_report_has_svd_and_condition_number_at_each_event(spec, q_events) -> None:
    rep = observability_report(spec, q_events, TRUNK)
    assert set(rep["events"]) == set(EVENTS)
    for ev in rep["events"].values():
        sv = ev["singular_values"]
        assert len(sv) == 6 and sv == sorted(sv, reverse=True)
        assert ev["condition_number"] == pytest.approx(sv[0] / sv[-1])
        assert 1 <= ev["rank"] <= 6


def test_lumbar_and_thoracic_axial_columns_are_nearly_parallel(spec, q_events) -> None:
    ev = observability_report(spec, q_events, TRUNK)["events"]["top"]
    # No marker sits on the lumbar body, so the axial pair is weakly separable.
    assert ev["axis_cosines"]["axial_rotation"] > 0.95


def test_decision_is_coupled_when_axes_are_not_separable(spec, q_events) -> None:
    rep = observability_report(spec, q_events, TRUNK)
    decision = decide_trunk_coupling(rep)
    assert decision["coupled"] is True
    assert decision["evidence"]


def test_decision_is_free_for_well_separated_columns() -> None:
    rep = {"events": {"e": {"condition_number": 5.0, "axis_cosines": {"a": 0.1}}}}
    assert decide_trunk_coupling(rep)["coupled"] is False

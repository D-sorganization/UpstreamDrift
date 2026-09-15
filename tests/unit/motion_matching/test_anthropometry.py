"""Tests for de Leva segment scaling and posture metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching import anthropometry as anth
from src.shared.python.motion_matching import posture_metrics as post

pytestmark = pytest.mark.unit


def test_reference_subject_reproduces_the_table() -> None:
    thigh = anth.segment_parameters(
        anth.REFERENCE_STATURE_M, anth.REFERENCE_MASS_KG, "thigh"
    )
    assert thigh.length_m == pytest.approx(0.4222)
    assert thigh.mass_kg == pytest.approx(0.1416 * 73.0)
    assert thigh.com_from_proximal_m == pytest.approx(0.4095 * 0.4222)
    sag, tra, lon = thigh.inertia_principal_kg_m2
    assert sag == pytest.approx(thigh.mass_kg * (0.329 * 0.4222) ** 2)
    assert lon < tra <= sag
    total = sum(
        row.mass_fraction
        for name, row in anth.DE_LEVA_MALE.items()
        if name in ("head", "trunk")
    ) + 2 * sum(
        anth.DE_LEVA_MALE[n].mass_fraction
        for n in ("upper_arm", "forearm", "hand", "thigh", "shank", "foot")
    )
    assert total == pytest.approx(1.0, abs=0.01)  # the table sums to body mass
    parts = sum(
        anth.DE_LEVA_MALE[n].mass_fraction
        for n in ("upper_trunk", "middle_trunk", "lower_trunk")
    )
    assert parts == pytest.approx(anth.DE_LEVA_MALE["trunk"].mass_fraction, abs=0.001)


def test_scaling_and_stature_estimate() -> None:
    tall = anth.segment_parameters(1.90, 90.0, "shank")
    assert tall.length_m == pytest.approx(0.4340 * 1.90 / 1.741)
    assert tall.mass_kg == pytest.approx(0.0433 * 90.0)
    stature = anth.stature_from_segment_lengths({"shank": 0.4340, "forearm": 0.2689})
    assert stature == pytest.approx(1.741)
    assert anth.mass_check(108.0, 1.74, 78.0) == pytest.approx(108.0 / 78.0)
    assert set(anth.whole_body(1.8, 80.0)) == set(anth.DE_LEVA_MALE)
    with pytest.raises(ValueError):
        anth.segment_parameters(1.8, 80.0, "tail")
    with pytest.raises(ValueError):
        anth.segment_parameters(0.0, 80.0, "thigh")
    with pytest.raises(ValueError):
        anth.stature_from_segment_lengths({})


def test_segment_tilt_and_spine_bend_split_by_plane() -> None:
    up, fwd, right = np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    pelvis = np.array([np.sin(np.radians(20)), 0.0, np.cos(np.radians(20))])
    trunk = np.array([np.sin(np.radians(45)), 0.0, np.cos(np.radians(45))])
    tilt = post.segment_tilt(trunk, up, fwd, right)
    assert tilt.total_deg == pytest.approx(45.0)
    assert tilt.forward_deg == pytest.approx(
        45.0
    ) and tilt.lateral_deg == pytest.approx(0.0)
    bend = post.spine_bend(pelvis, trunk, up, fwd, right)
    assert bend.total_deg == pytest.approx(25.0)
    assert bend.forward_deg == pytest.approx(
        25.0
    ) and bend.lateral_deg == pytest.approx(0.0)
    sideways = np.array([0.0, np.sin(np.radians(10)), np.cos(np.radians(10))])
    assert post.spine_bend(up, sideways, up, fwd, right).lateral_deg == pytest.approx(
        10.0
    )
    with pytest.raises(ValueError):
        post.segment_tilt(trunk, up, up, right)
    normal = post.plane_normal(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]]), up
    )
    np.testing.assert_allclose(normal, up, atol=1e-12)
    with pytest.raises(ValueError):
        post.plane_normal(np.zeros((2, 3)), up)

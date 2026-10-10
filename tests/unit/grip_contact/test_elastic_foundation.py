"""Elastic-foundation parameters matched to the shared pad law (#11739)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.elastic_foundation import (
    elastic_foundation_parameters,
    winkler_patch_factor_m,
)
from src.shared.python.grip_contact.pad_contact import build_pad_model

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


@pytest.fixture(scope="module")
def pads():
    interface = GripInterface.from_spec(json.loads(SPEC.read_text(encoding="utf-8")))
    return build_pad_model(interface, 1104.0)


def test_patch_factor_matches_the_closed_form() -> None:
    a = 1 / 0.008 + 1 / 0.0127
    b = 1 / 0.008
    assert winkler_patch_factor_m(0.008, 0.0127) == pytest.approx(
        math.pi / math.sqrt(a * b)
    )
    with pytest.raises(ValueError):
        winkler_patch_factor_m(0.0, 0.0127)


def test_squeeze_and_tangent_stiffness_match_the_shared_law(pads) -> None:
    ef = elastic_foundation_parameters(pads)
    n = pads.layout.pad_count
    d0 = ef.preload_penetration_m
    assert ef.analytic_force_n(d0) == pytest.approx(ef.force_per_pad_n, rel=1e-12)
    assert n * ef.analytic_force_n(d0) == pytest.approx(1104.0, rel=1e-9)
    h = 1e-9
    tangent = (ef.analytic_force_n(d0 + h) - ef.analytic_force_n(d0 - h)) / (2 * h)
    assert tangent == pytest.approx(pads.law.stiffness_n_m, rel=1e-6)
    assert d0 == pytest.approx(2.0 * pads.layout.preload_penetration_m, rel=1e-12)


def test_friction_and_dissipation_are_the_shared_values(pads) -> None:
    ef = elastic_foundation_parameters(pads)
    law = pads.law
    assert (ef.static_friction, ef.dynamic_friction) == (
        law.static_friction,
        law.dynamic_friction,
    )
    assert ef.dissipation_s_m == law.dissipation_s_m
    assert ef.transition_velocity_m_s == law.transition_velocity_m_s
    assert ef.layout.preload_penetration_m == ef.preload_penetration_m
    assert ef.layout.pad_count == pads.layout.pad_count


def test_calibrated_stiffness_is_validated(pads) -> None:
    ef = elastic_foundation_parameters(pads)
    assert ef.with_stiffness(2.0 * ef.stiffness_n_m3).stiffness_n_m3 == pytest.approx(
        2.0 * ef.stiffness_n_m3
    )
    with pytest.raises(ValueError):
        ef.with_stiffness(-1.0)
    assert ef.analytic_force_n(-1e-3) == 0.0

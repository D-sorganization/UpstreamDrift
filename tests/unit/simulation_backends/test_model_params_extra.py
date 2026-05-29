"""Supplementary unit tests for :mod:`simulation_backends.model_params`.

These cover the validation/error branches and the small helper properties left
uncovered by the primary parameter tests:

* :meth:`GolfModelParams.from_yaml` rejecting empty text and non-mapping YAML;
* the ``plane_inclination_deg`` finite-value validator;
* :attr:`UpperSegmentParams.effective_inertia_about_com` for both the
  uniform-rod default (``inertia_about_com_kg_m2 is None``) and an explicit
  value;
* :attr:`GolfModelParams.projected_gravity` with gravity disabled and with the
  swing-plane projection turned off;
* the ``num_joints`` / ``state_dim`` shape properties;
* a ``to_yaml`` round-trip of a *non-default* (mass-perturbed) model.

All RNG is seeded; no optional dependencies are touched.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.simulation_backends import GolfModelParams
from src.shared.python.simulation_backends.model_params import (
    LowerSegmentParams,
    UpperSegmentParams,
)

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(0)


def _segments() -> tuple[UpperSegmentParams, LowerSegmentParams]:
    """Return a valid upper/lower segment pair for fresh-construction tests."""
    upper = UpperSegmentParams(
        length_m=0.7,
        mass_kg=8.0,
        center_of_mass_ratio=0.5,
        inertia_about_com_kg_m2=None,
    )
    lower = LowerSegmentParams(
        length_m=1.1,
        shaft_mass_kg=0.3,
        clubhead_mass_kg=0.2,
        shaft_com_ratio=0.6,
    )
    return upper, lower


# --------------------------------------------------------------------------- #
# from_yaml error branches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_from_yaml_rejects_empty_text(text: str) -> None:
    """Empty / whitespace YAML is a precondition violation."""
    with pytest.raises(ValueError, match="non-empty"):
        GolfModelParams.from_yaml(text)


@pytest.mark.parametrize("text", ["- a\n- b", "42", "just a scalar string"])
def test_from_yaml_rejects_non_mapping(text: str) -> None:
    """YAML that does not decode to a mapping is rejected."""
    with pytest.raises(ValueError, match="mapping"):
        GolfModelParams.from_yaml(text)


# --------------------------------------------------------------------------- #
# plane_inclination_deg finite validator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_plane_inclination_rejected(bad: float) -> None:
    """A NaN/inf inclination is rejected at construction.

    ``model_copy`` deliberately skips validation, so the model must be built
    through the validating constructor. Pydantic raises a ``ValidationError``
    (a ``ValueError`` subclass) for the non-finite, out-of-``[-90, 90]`` value.
    """
    upper, lower = _segments()
    with pytest.raises(ValueError):
        GolfModelParams(upper=upper, lower=lower, plane_inclination_deg=bad)


# --------------------------------------------------------------------------- #
# UpperSegmentParams.effective_inertia_about_com
# --------------------------------------------------------------------------- #
def test_effective_inertia_uniform_rod_when_none() -> None:
    """With ``inertia_about_com_kg_m2 is None`` a uniform-rod value is used."""
    seg = UpperSegmentParams(
        length_m=0.8,
        mass_kg=5.0,
        center_of_mass_ratio=0.5,
        inertia_about_com_kg_m2=None,
    )
    # Uniform rod about its COM: (1/12) m L^2.
    expected = (1.0 / 12.0) * seg.mass_kg * seg.length_m**2
    assert seg.effective_inertia_about_com == pytest.approx(expected)


def test_effective_inertia_uses_explicit_value_when_given() -> None:
    """An explicit inertia is returned verbatim (no uniform-rod fallback)."""
    seg = UpperSegmentParams(
        length_m=0.8,
        mass_kg=5.0,
        center_of_mass_ratio=0.5,
        inertia_about_com_kg_m2=0.123,
    )
    assert seg.effective_inertia_about_com == pytest.approx(0.123)
    # The explicit value differs from the uniform-rod default, proving the
    # branch is taken.
    uniform = (1.0 / 12.0) * seg.mass_kg * seg.length_m**2
    assert seg.effective_inertia_about_com != pytest.approx(uniform)


# --------------------------------------------------------------------------- #
# projected_gravity branches
# --------------------------------------------------------------------------- #
def test_projected_gravity_zero_when_gravity_disabled() -> None:
    """Disabling gravity zeroes the projected gravity regardless of tilt."""
    params = GolfModelParams.default().model_copy(
        update={"gravity_enabled": False, "plane_inclination_deg": 30.0}
    )
    assert params.projected_gravity == 0.0


def test_projected_gravity_full_when_not_constrained_to_plane() -> None:
    """Without plane projection the full gravity magnitude is returned."""
    params = GolfModelParams.default().model_copy(
        update={"constrained_to_plane": False, "plane_inclination_deg": 30.0}
    )
    assert params.projected_gravity == pytest.approx(params.gravity_m_s2)


def test_projected_gravity_uses_cosine_when_constrained() -> None:
    """The constrained, gravity-enabled path applies the cosine projection."""
    params = GolfModelParams.default().model_copy(
        update={
            "gravity_enabled": True,
            "constrained_to_plane": True,
            "plane_inclination_deg": 30.0,
        }
    )
    expected = params.gravity_m_s2 * math.cos(math.radians(30.0))
    assert params.projected_gravity == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# shape properties
# --------------------------------------------------------------------------- #
def test_num_joints_and_state_dim() -> None:
    """The planar double pendulum exposes 2 joints and a 4-dim state."""
    params = GolfModelParams.default()
    assert params.num_joints == 2
    assert params.state_dim == 4


# --------------------------------------------------------------------------- #
# to_yaml round-trip of a NON-default model
# --------------------------------------------------------------------------- #
def test_to_yaml_round_trips_non_default_model() -> None:
    """A mass-perturbed model survives ``to_yaml`` / ``from_yaml`` unchanged."""
    base = GolfModelParams.default()
    perturbed = base.model_copy(
        update={
            "upper": base.upper.model_copy(update={"mass_kg": 9.0}),
            "lower": base.lower.model_copy(update={"clubhead_mass_kg": 0.42}),
            "damping_wrist": base.damping_wrist + 0.05,
        }
    )
    # Sanity: the perturbed model is genuinely different from the baseline.
    assert perturbed != base

    restored = GolfModelParams.from_yaml(perturbed.to_yaml())
    assert restored == perturbed
    assert restored.upper.mass_kg == pytest.approx(9.0)
    assert restored.lower.clubhead_mass_kg == pytest.approx(0.42)

"""Tests for humanoid anthropometry helpers."""

import pytest
from humanoid_character_builder.core.anthropometry import estimate_segment_masses

pytestmark = pytest.mark.unit


def test_estimate_segment_masses_normalized_total() -> None:
    total_mass = 75.0
    masses = estimate_segment_masses(total_mass)
    # de Leva Table 4 closes to its printed precision (female 99.99 %); the
    # masses are not rescaled to hide that (#13010).
    assert sum(masses.values()) == pytest.approx(total_mass, rel=2e-4)
    assert all(mass >= 0.0 for mass in masses.values())

"""Tests for humanoid anthropometry helpers."""

from humanoid_character_builder.core.anthropometry import estimate_segment_masses


def test_estimate_segment_masses_normalized_total() -> None:
    total_mass = 75.0
    masses = estimate_segment_masses(total_mass)
    assert abs(sum(masses.values()) - total_mass) < 1e-6
    assert all(mass >= 0.0 for mass in masses.values())

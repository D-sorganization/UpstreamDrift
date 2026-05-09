"""Shared deep-equality assertions for adapter round-trip tests."""

from __future__ import annotations

import numpy as np

from anthropometrics import SubjectAnthropometrics


def assert_subjects_equal(
    a: SubjectAnthropometrics,
    b: SubjectAnthropometrics,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Assert two subjects are numerically and metadata-identical."""
    assert a.subject_id == b.subject_id
    assert a.height_m == b.height_m
    assert a.mass_kg == b.mass_kg
    assert a.sex == b.sex
    assert a.source_method == b.source_method
    assert a.age_years == b.age_years
    assert len(a.segments) == len(b.segments)
    for (name_a, p_a), (name_b, p_b) in zip(a.segments, b.segments, strict=True):
        assert name_a == name_b
        assert p_a.name == p_b.name
        assert p_a.body_part_id == p_b.body_part_id
        assert p_a.length_m == p_b.length_m
        assert p_a.proximal_marker == p_b.proximal_marker
        assert p_a.distal_marker == p_b.distal_marker
        assert p_a.mass_kg == p_b.mass_kg
        assert p_a.source_method == p_b.source_method
        assert p_a.source_subject_height_m == p_b.source_subject_height_m
        assert p_a.source_subject_mass_kg == p_b.source_subject_mass_kg
        np.testing.assert_allclose(p_a.com_xyz_m, p_b.com_xyz_m, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            p_a.inertia_tensor, p_b.inertia_tensor, rtol=rtol, atol=atol
        )

"""Contracts for anthropometric and joint-limit headline authority regeneration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    ScaledAuthorityConfig,
    build_scaled_authority,
    validate_scaled_authority,
)

pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def nominal_authority():
    return build_scaled_authority(ScaledAuthorityConfig(case_indices=(0,)))


def test_scaled_authority_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="case_indices"):
        ScaledAuthorityConfig(case_indices=())
    with pytest.raises(ValueError, match="case_indices"):
        ScaledAuthorityConfig(case_indices=(18,))
    with pytest.raises(ValueError, match="height_scale"):
        ScaledAuthorityConfig(height_scale=0.0)
    with pytest.raises(ValueError, match="joint_limit_scale"):
        ScaledAuthorityConfig(joint_limit_scale=0.49)


def test_nominal_scaled_authority_reproduces_committed_selected_case(
    nominal_authority,
) -> None:
    authority = nominal_authority
    assert authority.solution_q.shape == (18, 13, 20)
    assert authority.feasible.shape == (18, 13)
    assert authority.selected_case_indices.tolist() == [0]
    assert authority.selected_failure_class.shape == (1, 13)
    assert np.all(authority.selected_failure_class == "feasible")
    assert np.all(authority.feasible[0])
    assert authority.maximum_nominal_state_error_rad <= 1.0e-8
    assert authority.source_sha256
    validate_scaled_authority(authority, ScaledAuthorityConfig(case_indices=(0,)))


def test_scaled_authority_detects_configuration_or_digest_mismatch(
    nominal_authority,
) -> None:
    config = ScaledAuthorityConfig(case_indices=(0,))
    authority = nominal_authority

    with pytest.raises(RuntimeError, match="configuration"):
        validate_scaled_authority(authority, replace(config, body_mass_scale=1.01))
    with pytest.raises(RuntimeError, match="digest"):
        validate_scaled_authority(replace(authority, authority_sha256="0" * 64), config)

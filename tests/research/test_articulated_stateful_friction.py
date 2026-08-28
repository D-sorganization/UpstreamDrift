"""Stateful tangential-compliance countermodel contracts for #9153."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_stateful_friction import (
    StatefulFrictionConfig,
    TangentialRegime,
    TangentialState,
    advance_stateful_friction,
)


def _config() -> StatefulFrictionConfig:
    return StatefulFrictionConfig(
        tangential_stiffness_n_m=1000.0,
        friction_coefficient=0.5,
    )


def test_elastic_stick_accumulates_state_and_closes_incremental_energy() -> None:
    result = advance_stateful_friction(
        TangentialState.zero(),
        tangential_displacement_increment_m=np.array([0.001, 0.0, 0.0]),
        normal_load_n=10.0,
        active=True,
        config=_config(),
    )

    assert result.regime is TangentialRegime.STICK
    np.testing.assert_allclose(result.force_on_club_n, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.state.elastic_displacement_m, [0.001, 0.0, 0.0])
    assert result.frictional_dissipation_j == pytest.approx(0.0)
    assert result.constitutive_work_j == pytest.approx(
        result.elastic_energy_change_j + result.frictional_dissipation_j
    )


def test_radial_return_caps_force_and_retains_positive_slip_dissipation() -> None:
    result = advance_stateful_friction(
        TangentialState.zero(),
        tangential_displacement_increment_m=np.array([0.01, 0.0, 0.0]),
        normal_load_n=10.0,
        active=True,
        config=_config(),
    )

    assert result.regime is TangentialRegime.SLIP
    assert np.linalg.norm(result.force_on_club_n) == pytest.approx(5.0)
    assert np.linalg.norm(result.state.elastic_displacement_m) == pytest.approx(0.005)
    assert result.plastic_slip_increment_m == pytest.approx(0.005)
    assert result.frictional_dissipation_j == pytest.approx(0.025)
    assert result.frictional_dissipation_j > 0.0
    assert result.constitutive_work_j == pytest.approx(
        result.elastic_energy_change_j + result.frictional_dissipation_j
    )


def test_opening_releases_stored_energy_as_explicit_dissipation() -> None:
    state = TangentialState(np.array([0.002, 0.0, 0.0]))
    result = advance_stateful_friction(
        state,
        tangential_displacement_increment_m=np.zeros(3),
        normal_load_n=0.0,
        active=False,
        config=_config(),
    )

    assert result.regime is TangentialRegime.OPEN
    np.testing.assert_allclose(result.state.elastic_displacement_m, 0.0)
    assert result.release_dissipation_j == pytest.approx(0.002)
    assert result.elastic_energy_change_j == pytest.approx(-0.002)
    assert result.constitutive_work_j == pytest.approx(0.0)


def test_return_map_is_rotation_equivariant() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    increment = np.array([0.01, 0.002, 0.0])
    direct = advance_stateful_friction(
        TangentialState.zero(),
        tangential_displacement_increment_m=increment,
        normal_load_n=10.0,
        active=True,
        config=_config(),
    )
    rotated = advance_stateful_friction(
        TangentialState.zero(),
        tangential_displacement_increment_m=rotation @ increment,
        normal_load_n=10.0,
        active=True,
        config=_config(),
    )

    np.testing.assert_allclose(
        rotated.force_on_club_n, rotation @ direct.force_on_club_n
    )
    np.testing.assert_allclose(
        rotated.state.elastic_displacement_m,
        rotation @ direct.state.elastic_displacement_m,
    )
    assert rotated.frictional_dissipation_j == pytest.approx(
        direct.frictional_dissipation_j
    )


def test_closed_elastic_cycle_returns_stored_energy_without_dissipation() -> None:
    loaded = advance_stateful_friction(
        TangentialState.zero(),
        tangential_displacement_increment_m=np.array([0.001, 0.0, 0.0]),
        normal_load_n=10.0,
        active=True,
        config=_config(),
    )
    unloaded = advance_stateful_friction(
        loaded.state,
        tangential_displacement_increment_m=np.array([-0.001, 0.0, 0.0]),
        normal_load_n=10.0,
        active=True,
        config=_config(),
    )

    assert loaded.constitutive_work_j + unloaded.constitutive_work_j == pytest.approx(
        0.0
    )
    assert loaded.frictional_dissipation_j == pytest.approx(0.0)
    assert unloaded.frictional_dissipation_j == pytest.approx(0.0)
    np.testing.assert_allclose(unloaded.state.elastic_displacement_m, 0.0)


def test_contract_rejects_nonphysical_load_and_state() -> None:
    with pytest.raises(ValueError, match="normal_load_n"):
        advance_stateful_friction(
            TangentialState.zero(),
            tangential_displacement_increment_m=np.zeros(3),
            normal_load_n=-1.0,
            active=True,
            config=_config(),
        )
    with pytest.raises(ValueError, match="elastic_displacement_m"):
        TangentialState(np.array([np.nan, 0.0, 0.0]))

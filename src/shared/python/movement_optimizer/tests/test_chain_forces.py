# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for chain force estimates (models/chain_forces.py)."""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.models import (
    ChainForceField,
    ChainForceHistory,
    chain_force_field,
    chain_force_history,
    link_accelerations,
)
from movement_optimizer.models.chain_dynamics import (
    ChainConfig,
    ChainRollout,
    ChainState,
    initial_catenary_angles,
    simulate_chain,
)

_SEGMENTS = 4
_STEPS = 6
_DT = 0.02


def _make_rollout() -> tuple[ChainConfig, ChainRollout]:
    config = ChainConfig(segment_count=_SEGMENTS)
    start = ChainState(
        initial_catenary_angles(_SEGMENTS, 0.2),
        np.zeros(_SEGMENTS, dtype=np.float64),
    )
    return config, simulate_chain(config, start, _STEPS, _DT)


def _single_state_rollout() -> tuple[ChainConfig, ChainRollout]:
    config = ChainConfig(segment_count=_SEGMENTS)
    start = ChainState.stationary(config, 0.1).validated(config)
    return config, ChainRollout(
        states=(start,),
        positions=start.node_positions(config)[np.newaxis, ...],
        energy_j=np.zeros(1, dtype=np.float64),
        tip_speed_m_s=np.zeros(1, dtype=np.float64),
    )


def test_link_accelerations_shape() -> None:
    config, rollout = _make_rollout()
    accel = link_accelerations(config, rollout, _DT)
    assert accel.shape == (_STEPS + 1, _SEGMENTS, 2)
    assert np.all(np.isfinite(accel))


def test_link_accelerations_single_state_is_zero() -> None:
    config, rollout = _single_state_rollout()
    accel = link_accelerations(config, rollout, _DT)
    assert accel.shape == (1, _SEGMENTS, 2)
    assert np.allclose(accel, 0.0)


def test_link_accelerations_rejects_nonpositive_dt() -> None:
    config, rollout = _make_rollout()
    with pytest.raises(ValueError, match="dt_s"):
        link_accelerations(config, rollout, 0.0)


def test_link_accelerations_rejects_empty_rollout() -> None:
    config = ChainConfig(segment_count=_SEGMENTS)
    empty = ChainRollout(
        states=(),
        positions=np.zeros((0, _SEGMENTS + 1, 2), dtype=np.float64),
        energy_j=np.zeros(0, dtype=np.float64),
        tip_speed_m_s=np.zeros(0, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="at least one state"):
        link_accelerations(config, empty, _DT)


def test_chain_force_field_shapes_and_gravity() -> None:
    config, rollout = _make_rollout()
    field = chain_force_field(config, rollout, _DT, frame_index=2)
    assert isinstance(field, ChainForceField)
    for array in (field.midpoints_m, field.gravity_n, field.tension_n, field.net_force_n):
        assert array.shape == (_SEGMENTS, 2)
        assert np.all(np.isfinite(array))
    expected = config.link_mass_kg * config.gravity_m_s2
    assert np.allclose(field.gravity_n[:, 0], 0.0)
    assert np.allclose(field.gravity_n[:, 1], expected)


def test_chain_force_field_net_force_matches_mass_times_accel() -> None:
    config, rollout = _make_rollout()
    accel = link_accelerations(config, rollout, _DT)[3]
    field = chain_force_field(config, rollout, _DT, frame_index=3)
    assert np.allclose(field.net_force_n, config.link_mass_kg * accel)


def test_chain_force_field_top_link_tension_exceeds_bottom() -> None:
    config, rollout = _make_rollout()
    field = chain_force_field(config, rollout, _DT, frame_index=0)
    top = float(np.linalg.norm(field.tension_n[0]))
    bottom = float(np.linalg.norm(field.tension_n[-1]))
    assert top >= bottom


@pytest.mark.parametrize("bad_index", [-1, _STEPS + 1, 999])
def test_chain_force_field_rejects_out_of_range_index(bad_index: int) -> None:
    config, rollout = _make_rollout()
    with pytest.raises(ValueError, match="frame_index"):
        chain_force_field(config, rollout, _DT, bad_index)


def test_chain_force_field_rejects_nonpositive_dt() -> None:
    config, rollout = _make_rollout()
    with pytest.raises(ValueError, match="dt_s"):
        chain_force_field(config, rollout, -1.0, 0)


def test_chain_force_history_shapes() -> None:
    config, rollout = _make_rollout()
    history = chain_force_history(config, rollout, _DT)
    assert isinstance(history, ChainForceHistory)
    frames = _STEPS + 1
    assert history.time_s.shape == (frames,)
    assert history.link_tension_n.shape == (frames, _SEGMENTS)
    assert history.max_tension_n.shape == (frames,)
    assert history.curvature_rad.shape == (frames, _SEGMENTS - 1)
    assert history.max_curvature_rad.shape == (frames,)
    assert np.all(np.isfinite(history.link_tension_n))


def test_chain_force_history_max_tension_matches_per_link_max() -> None:
    config, rollout = _make_rollout()
    history = chain_force_history(config, rollout, _DT)
    assert np.allclose(history.max_tension_n, np.max(history.link_tension_n, axis=1))


def test_chain_force_history_rejects_nonpositive_dt() -> None:
    config, rollout = _make_rollout()
    with pytest.raises(ValueError, match="dt_s"):
        chain_force_history(config, rollout, 0.0)


def test_chain_force_history_rejects_empty_rollout() -> None:
    config = ChainConfig(segment_count=_SEGMENTS)
    empty = ChainRollout(
        states=(),
        positions=np.zeros((0, _SEGMENTS + 1, 2), dtype=np.float64),
        energy_j=np.zeros(0, dtype=np.float64),
        tip_speed_m_s=np.zeros(0, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="at least one state"):
        chain_force_history(config, empty, _DT)

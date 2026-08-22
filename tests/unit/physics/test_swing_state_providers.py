"""Tests for the swing-state provider seam (issue #8819).

The provider contract kills false engine attribution: a provider may only
return a ``SwingState`` stamped with its own ``provider_id``, and an
unavailable provider refuses to produce a state at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics.swing_ball_flight_pipeline import SwingState
from src.shared.python.physics.swing_state_providers import (
    REASON_NOT_IMPLEMENTED,
    REASON_NOT_INSTALLED,
    ManualSwingStateProvider,
    MuJoCoSwingStateProvider,
    SwingStateConfig,
    SwingStateProvider,
    UnimplementedEngineProvider,
    _BaseSwingStateProvider,
    available_swing_state_providers,
)

pytestmark = pytest.mark.unit

_MUJOCO_INSTALLED = MuJoCoSwingStateProvider().is_available()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_lists_all_historic_engine_choices_in_gui_order(self):
        ids = [p.provider_id for p in available_swing_state_providers()]
        assert ids == ["mujoco", "drake", "pinocchio", "manual"]

    def test_all_entries_satisfy_protocol(self):
        for provider in available_swing_state_providers():
            assert isinstance(provider, SwingStateProvider)

    def test_provider_availability(self):
        availability = {
            p.provider_id: p.is_available() for p in available_swing_state_providers()
        }
        assert availability == {
            "mujoco": _MUJOCO_INSTALLED,  # implemented (#8975); env-gated
            "drake": False,
            "pinocchio": False,
            "manual": True,
        }

    def test_unavailable_providers_carry_honest_reason(self):
        for provider in available_swing_state_providers():
            if not provider.is_available():
                assert provider.availability_reason() in (
                    REASON_NOT_IMPLEMENTED,
                    REASON_NOT_INSTALLED,
                )


# ---------------------------------------------------------------------------
# ManualSwingStateProvider
# ---------------------------------------------------------------------------


class TestManualProvider:
    def test_is_available(self):
        assert ManualSwingStateProvider().is_available() is True
        assert ManualSwingStateProvider().availability_reason() == ""

    def test_builds_swing_state_from_config(self):
        provider = ManualSwingStateProvider()
        config = SwingStateConfig(
            clubhead_speed_ms=50.0, loft_deg=34.0, clubhead_mass_kg=0.300
        )
        state = provider.get_swing_state(config)
        assert isinstance(state, SwingState)
        np.testing.assert_array_equal(
            state.clubhead_velocity, np.array([50.0, 0.0, 0.0])
        )
        np.testing.assert_array_equal(state.clubhead_angular_velocity, np.zeros(3))
        assert state.clubhead_loft_deg == pytest.approx(34.0)
        assert state.clubhead_mass == pytest.approx(0.300)

    def test_engine_name_equals_provider_id(self):
        state = ManualSwingStateProvider().get_swing_state(SwingStateConfig())
        assert state.engine_name == "manual"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"clubhead_speed_ms": 0.0},
            {"clubhead_speed_ms": -5.0},
            {"clubhead_speed_ms": float("nan")},
            {"loft_deg": 0.0},
            {"loft_deg": 90.0},
            {"loft_deg": float("inf")},
            {"clubhead_mass_kg": 0.0},
            {"clubhead_mass_kg": -0.2},
        ],
    )
    def test_invalid_config_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ManualSwingStateProvider().get_swing_state(SwingStateConfig(**kwargs))


# ---------------------------------------------------------------------------
# UnimplementedEngineProvider
# ---------------------------------------------------------------------------


class TestUnimplementedEngineProvider:
    def test_never_available(self):
        provider = UnimplementedEngineProvider("drake", "pydrake")
        assert provider.is_available() is False

    def test_get_swing_state_refuses_with_reason(self):
        provider = UnimplementedEngineProvider("drake", "pydrake")
        with pytest.raises(ValueError, match="not available"):
            provider.get_swing_state(SwingStateConfig())

    def test_reason_not_installed_for_missing_module(self):
        provider = UnimplementedEngineProvider(
            "ghost", "definitely_not_a_real_module_xyz"
        )
        assert provider.availability_reason() == REASON_NOT_INSTALLED

    def test_reason_not_implemented_for_installed_module(self):
        # numpy is guaranteed installed; the reason must say "not yet
        # implemented" rather than falsely claiming a missing install.
        provider = UnimplementedEngineProvider("fake", "numpy")
        assert provider.availability_reason() == REASON_NOT_IMPLEMENTED

    def test_empty_ids_rejected(self):
        with pytest.raises(ValueError):
            UnimplementedEngineProvider("", "mujoco")
        with pytest.raises(ValueError):
            UnimplementedEngineProvider("mujoco", "")


# ---------------------------------------------------------------------------
# MuJoCoSwingStateProvider (#8975)
# ---------------------------------------------------------------------------


class TestMuJoCoProvider:
    def test_unavailable_without_mujoco_package(self, monkeypatch):
        import importlib.util as ilu

        monkeypatch.setattr(ilu, "find_spec", lambda name: None)
        provider = MuJoCoSwingStateProvider()
        assert provider.is_available() is False
        assert provider.availability_reason() == REASON_NOT_INSTALLED
        with pytest.raises(ValueError, match="not available"):
            provider.get_swing_state(SwingStateConfig())

    @pytest.mark.skipif(not _MUJOCO_INSTALLED, reason="mujoco not installed")
    def test_available_with_empty_reason(self):
        provider = MuJoCoSwingStateProvider()
        assert provider.is_available() is True
        assert provider.availability_reason() == ""

    @pytest.mark.skipif(not _MUJOCO_INSTALLED, reason="mujoco not installed")
    def test_produces_real_engine_sourced_swing_state(self):
        config = SwingStateConfig(clubhead_speed_ms=45.0, loft_deg=10.5)
        state = MuJoCoSwingStateProvider().get_swing_state(config)

        assert state.engine_name == "mujoco"  # honest attribution
        speed = float(np.linalg.norm(state.clubhead_velocity))
        assert speed == pytest.approx(45.0, rel=0.10)
        # Real dynamics: the clubhead rotates through impact.
        assert float(np.linalg.norm(state.clubhead_angular_velocity)) > 0.1
        assert np.linalg.norm(state.clubhead_orientation) == pytest.approx(
            1.0, abs=1e-6
        )
        # Mass/MOI come from the MJCF model, not the config.
        assert state.clubhead_mass > 0.0
        assert state.clubhead_mass != config.clubhead_mass_kg
        assert state.clubhead_moi > 0.0

    @pytest.mark.skipif(not _MUJOCO_INSTALLED, reason="mujoco not installed")
    def test_metadata_records_model_method_and_residual(self):
        state = MuJoCoSwingStateProvider().get_swing_state(
            SwingStateConfig(clubhead_speed_ms=40.0)
        )
        metadata = state.metadata
        assert metadata["method"] == "mujoco_forward_dynamics"
        assert metadata["model_asset"].startswith("src.engines.physics_engines.mujoco.")
        assert metadata["timestep_s"] > 0.0
        assert "achieved_speed_ms" in metadata
        assert "speed_residual_rel" in metadata


# ---------------------------------------------------------------------------
# Contract: false attribution is a hard error
# ---------------------------------------------------------------------------


class _LyingProvider(_BaseSwingStateProvider):
    """Claims to be 'manual' but stamps its output 'mujoco'."""

    provider_id = "manual"

    def _build_swing_state(self, config: SwingStateConfig) -> SwingState:
        return SwingState(
            clubhead_velocity=np.array([config.clubhead_speed_ms, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([0.0, 0.0, 1.0]),
            engine_name="mujoco",  # false attribution
        )


def test_false_engine_attribution_raises_contract_error():
    with pytest.raises(ValueError, match="false engine attribution"):
        _LyingProvider().get_swing_state(SwingStateConfig())


class _NaNProvider(_BaseSwingStateProvider):
    """Produces a non-finite clubhead velocity (must violate postcondition)."""

    provider_id = "manual"

    def _build_swing_state(self, config: SwingStateConfig) -> SwingState:
        return SwingState(
            clubhead_velocity=np.array([float("nan"), 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([0.0, 0.0, 1.0]),
            engine_name="manual",
        )


def test_non_finite_swing_state_raises_contract_error():
    with pytest.raises(ValueError, match="non-finite"):
        _NaNProvider().get_swing_state(SwingStateConfig())

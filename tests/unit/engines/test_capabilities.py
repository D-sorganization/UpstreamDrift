"""Tests for src.shared.python.engine_core.capabilities (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)


class TestCapabilityLevel:
    def test_has_full(self) -> None:
        assert CapabilityLevel.FULL is not None

    def test_has_partial(self) -> None:
        assert CapabilityLevel.PARTIAL is not None

    def test_has_none(self) -> None:
        assert CapabilityLevel.NONE is not None

    def test_full_not_none(self) -> None:
        assert CapabilityLevel.FULL != CapabilityLevel.NONE

    def test_partial_not_none(self) -> None:
        assert CapabilityLevel.PARTIAL != CapabilityLevel.NONE


class TestEngineCapabilitiesDefaults:
    def test_default_name_empty(self) -> None:
        caps = EngineCapabilities()
        assert caps.engine_name == ""

    def test_default_capabilities_none(self) -> None:
        caps = EngineCapabilities()
        assert caps.mass_matrix == CapabilityLevel.NONE
        assert caps.jacobian == CapabilityLevel.NONE
        assert caps.parameter_gradients == CapabilityLevel.NONE
        assert caps.state_control_gradients == CapabilityLevel.NONE
        assert caps.forward_sim == CapabilityLevel.NONE
        assert caps.contact_step == CapabilityLevel.NONE
        assert caps.trajectory_opt == CapabilityLevel.NONE
        assert caps.video_export == CapabilityLevel.NONE

    def test_has_video_export_false_by_default(self) -> None:
        caps = EngineCapabilities()
        assert caps.has_video_export is False

    def test_has_dataset_export_false_by_default(self) -> None:
        caps = EngineCapabilities()
        assert caps.has_dataset_export is False

    def test_has_contact_forces_false_by_default(self) -> None:
        caps = EngineCapabilities()
        assert caps.has_contact_forces is False

    def test_has_measurements_false_by_default(self) -> None:
        caps = EngineCapabilities()
        assert caps.has_measurements is False

    def test_has_gradient_and_rollout_capabilities_false_by_default(self) -> None:
        caps = EngineCapabilities()
        assert caps.has_parameter_gradients is False
        assert caps.has_state_control_gradients is False
        assert caps.has_forward_sim is False
        assert caps.has_contact_step is False
        assert caps.has_trajectory_opt is False


class TestEngineCapabilitiesCustom:
    def _make_caps(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_name="TestEngine",
            mass_matrix=CapabilityLevel.FULL,
            jacobian=CapabilityLevel.PARTIAL,
            video_export=CapabilityLevel.FULL,
            dataset_export=CapabilityLevel.PARTIAL,
            contact_forces=CapabilityLevel.FULL,
            parameter_gradients=CapabilityLevel.PARTIAL,
            state_control_gradients=CapabilityLevel.FULL,
            forward_sim=CapabilityLevel.FULL,
            contact_step=CapabilityLevel.PARTIAL,
            trajectory_opt=CapabilityLevel.PARTIAL,
            measurements=CapabilityLevel.PARTIAL,
        )

    def test_engine_name_stored(self) -> None:
        caps = self._make_caps()
        assert caps.engine_name == "TestEngine"

    def test_has_video_export_true(self) -> None:
        caps = self._make_caps()
        assert caps.has_video_export is True

    def test_has_dataset_export_true_for_partial(self) -> None:
        caps = self._make_caps()
        assert caps.has_dataset_export is True

    def test_has_contact_forces_true(self) -> None:
        caps = self._make_caps()
        assert caps.has_contact_forces is True

    def test_has_measurements_true_for_partial(self) -> None:
        caps = self._make_caps()
        assert caps.has_measurements is True

    def test_has_gradient_and_rollout_capabilities_true(self) -> None:
        caps = self._make_caps()
        assert caps.has_parameter_gradients is True
        assert caps.has_state_control_gradients is True
        assert caps.has_forward_sim is True
        assert caps.has_contact_step is True
        assert caps.has_trajectory_opt is True


class TestEngineCapabilitiesToDict:
    def test_to_dict_has_engine_name(self) -> None:
        caps = EngineCapabilities(engine_name="TestEngine")
        d = caps.to_dict()
        assert d["engine_name"] == "TestEngine"

    def test_to_dict_has_required_keys(self) -> None:
        caps = EngineCapabilities()
        d = caps.to_dict()
        for key in [
            "mass_matrix",
            "jacobian",
            "parameter_gradients",
            "state_control_gradients",
            "forward_sim",
            "contact_step",
            "trajectory_opt",
            "video_export",
            "dataset_export",
        ]:
            assert key in d

    def test_to_dict_level_strings(self) -> None:
        caps = EngineCapabilities(mass_matrix=CapabilityLevel.FULL)
        d = caps.to_dict()
        assert d["mass_matrix"] == "full"

    def test_to_dict_none_level_string(self) -> None:
        caps = EngineCapabilities()
        d = caps.to_dict()
        assert d["mass_matrix"] == "none"


class TestEngineCapabilitiesFromDict:
    def test_capabilities_roundtrip(self) -> None:
        original = EngineCapabilities(
            engine_name="RoundTrip",
            mass_matrix=CapabilityLevel.FULL,
            parameter_gradients=CapabilityLevel.PARTIAL,
            state_control_gradients=CapabilityLevel.FULL,
            forward_sim=CapabilityLevel.FULL,
            contact_step=CapabilityLevel.PARTIAL,
            trajectory_opt=CapabilityLevel.PARTIAL,
            video_export=CapabilityLevel.PARTIAL,
        )
        d = original.to_dict()
        restored = EngineCapabilities.from_dict(d)
        assert restored.engine_name == "RoundTrip"
        assert restored.mass_matrix == CapabilityLevel.FULL
        assert restored.parameter_gradients == CapabilityLevel.PARTIAL
        assert restored.state_control_gradients == CapabilityLevel.FULL
        assert restored.forward_sim == CapabilityLevel.FULL
        assert restored.contact_step == CapabilityLevel.PARTIAL
        assert restored.trajectory_opt == CapabilityLevel.PARTIAL
        assert restored.video_export == CapabilityLevel.PARTIAL

    def test_from_dict_missing_keys_defaults_to_none(self) -> None:
        caps = EngineCapabilities.from_dict({})
        assert caps.mass_matrix == CapabilityLevel.NONE

    def test_from_dict_unknown_level_defaults_to_none(self) -> None:
        caps = EngineCapabilities.from_dict({"mass_matrix": "unknown"})
        assert caps.mass_matrix == CapabilityLevel.NONE

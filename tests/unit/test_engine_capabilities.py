"""Tests for src.shared.python.engine_core.capabilities (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)

# ---------------------------------------------------------------------------
# CapabilityLevel enum
# ---------------------------------------------------------------------------


class TestCapabilityLevel:
    def test_engine_capabilities_three_levels(self) -> None:
        assert len(CapabilityLevel) == 3

    def test_full_partial_none_exist(self) -> None:
        assert CapabilityLevel.FULL
        assert CapabilityLevel.PARTIAL
        assert CapabilityLevel.NONE

    def test_levels_are_distinct(self) -> None:
        assert CapabilityLevel.FULL != CapabilityLevel.PARTIAL
        assert CapabilityLevel.FULL != CapabilityLevel.NONE
        assert CapabilityLevel.PARTIAL != CapabilityLevel.NONE


# ---------------------------------------------------------------------------
# EngineCapabilities dataclass
# ---------------------------------------------------------------------------


class TestEngineCapabilitiesDefaults:
    def test_all_capabilities_default_none(self) -> None:
        caps = EngineCapabilities()
        for attr in (
            "mass_matrix",
            "jacobian",
            "contact_forces",
            "inverse_dynamics",
            "drift_acceleration",
            "video_export",
            "dataset_export",
            "force_visualization",
            "model_positioning",
            "measurements",
        ):
            assert getattr(caps, attr) == CapabilityLevel.NONE

    def test_engine_name_default_empty(self) -> None:
        caps = EngineCapabilities()
        assert caps.engine_name == ""

    def test_extra_default_empty_dict(self) -> None:
        caps = EngineCapabilities()
        assert caps.extra == {}

    def test_frozen_prevents_mutation(self) -> None:
        caps = EngineCapabilities()
        with pytest.raises((AttributeError, TypeError)):
            caps.engine_name = "MuJoCo"  # type: ignore[misc]


class TestEngineCapabilitiesHasProperties:
    def test_has_video_export_none(self) -> None:
        caps = EngineCapabilities()
        assert caps.has_video_export is False

    def test_has_video_export_full(self) -> None:
        caps = EngineCapabilities(video_export=CapabilityLevel.FULL)
        assert caps.has_video_export is True

    def test_has_video_export_partial(self) -> None:
        caps = EngineCapabilities(video_export=CapabilityLevel.PARTIAL)
        assert caps.has_video_export is True

    def test_has_dataset_export_false_by_default(self) -> None:
        assert EngineCapabilities().has_dataset_export is False

    def test_has_dataset_export_true_when_full(self) -> None:
        caps = EngineCapabilities(dataset_export=CapabilityLevel.FULL)
        assert caps.has_dataset_export is True

    def test_has_force_visualization_false_by_default(self) -> None:
        assert EngineCapabilities().has_force_visualization is False

    def test_has_contact_forces_false_by_default(self) -> None:
        assert EngineCapabilities().has_contact_forces is False

    def test_has_contact_forces_partial(self) -> None:
        caps = EngineCapabilities(contact_forces=CapabilityLevel.PARTIAL)
        assert caps.has_contact_forces is True

    def test_has_measurements_false_by_default(self) -> None:
        assert EngineCapabilities().has_measurements is False


class TestEngineCapabilitiesToDict:
    def test_engine_capabilities_returns_dict(self) -> None:
        caps = EngineCapabilities(engine_name="TestEngine")
        result = caps.to_dict()
        assert isinstance(result, dict)

    def test_engine_name_in_dict(self) -> None:
        caps = EngineCapabilities(engine_name="MuJoCo")
        assert caps.to_dict()["engine_name"] == "MuJoCo"

    def test_none_level_serialized_as_none_string(self) -> None:
        caps = EngineCapabilities()
        assert caps.to_dict()["mass_matrix"] == "none"

    def test_full_level_serialized_as_full_string(self) -> None:
        caps = EngineCapabilities(jacobian=CapabilityLevel.FULL)
        assert caps.to_dict()["jacobian"] == "full"

    def test_partial_level_serialized_as_partial_string(self) -> None:
        caps = EngineCapabilities(contact_forces=CapabilityLevel.PARTIAL)
        assert caps.to_dict()["contact_forces"] == "partial"

    def test_all_fields_present(self) -> None:
        caps = EngineCapabilities()
        d = caps.to_dict()
        for key in (
            "engine_name",
            "mass_matrix",
            "jacobian",
            "contact_forces",
            "inverse_dynamics",
            "drift_acceleration",
            "video_export",
            "dataset_export",
            "force_visualization",
            "model_positioning",
            "measurements",
        ):
            assert key in d


class TestEngineCapabilitiesFromDict:
    def test_engine_capabilities_roundtrip(self) -> None:
        original = EngineCapabilities(
            engine_name="Pinocchio",
            jacobian=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.PARTIAL,
        )
        restored = EngineCapabilities.from_dict(original.to_dict())
        assert restored.engine_name == original.engine_name
        assert restored.jacobian == original.jacobian
        assert restored.contact_forces == original.contact_forces

    def test_unknown_level_falls_back_to_none(self) -> None:
        caps = EngineCapabilities.from_dict({"mass_matrix": "unknown_value"})
        assert caps.mass_matrix == CapabilityLevel.NONE

    def test_empty_dict_uses_defaults(self) -> None:
        caps = EngineCapabilities.from_dict({})
        assert caps.engine_name == ""
        assert caps.mass_matrix == CapabilityLevel.NONE

    def test_case_insensitive_level_parsing(self) -> None:
        caps = EngineCapabilities.from_dict({"jacobian": "FULL"})
        assert caps.jacobian == CapabilityLevel.FULL

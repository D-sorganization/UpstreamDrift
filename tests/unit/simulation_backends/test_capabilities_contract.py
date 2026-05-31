"""Backend capability adapter contract tests."""

from __future__ import annotations

from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.simulation_backends.capabilities import (
    backend_capability_level,
    backend_to_engine_capabilities,
)
from src.shared.python.simulation_backends.protocol import BackendCapabilities


def test_backend_capabilities_answer_canonical_queries() -> None:
    caps = BackendCapabilities(
        name="mjwarp",
        device="cuda",
        supports_batched=True,
        is_differentiable=False,
        provides_dynamics=False,
    )

    assert caps.level_for(Capability.BATCHED_ROLLOUT) == CapabilityLevel.FULL
    assert caps.level_for(Capability.DIFFERENTIABLE_ROLLOUT) == CapabilityLevel.NONE
    assert caps.level_for(Capability.MASS_MATRIX) == CapabilityLevel.NONE
    assert caps.level_for(Capability.FORWARD_SIM) == CapabilityLevel.FULL


def test_backend_capabilities_keep_legacy_flag_aliases_queryable() -> None:
    caps = BackendCapabilities(
        name="ode",
        supports_batched=False,
        is_differentiable=True,
        provides_dynamics=True,
    )

    assert caps.supports("supports_batched") is False
    assert caps.supports("is_differentiable") is True
    assert caps.supports("provides_dynamics") is True
    assert backend_capability_level(caps, "provides_dynamics") == CapabilityLevel.FULL


def test_backend_to_engine_capabilities_is_narrow_adapter() -> None:
    backend_caps = BackendCapabilities(
        name="ode",
        supports_batched=False,
        is_differentiable=False,
        provides_dynamics=True,
    )

    engine_caps = backend_to_engine_capabilities(backend_caps)

    assert engine_caps.engine_name == "ode"
    assert engine_caps.level_for(Capability.MASS_MATRIX) == CapabilityLevel.FULL
    assert engine_caps.level_for(Capability.FORWARD_SIM) == CapabilityLevel.FULL
    assert engine_caps.level_for(Capability.BATCHED_ROLLOUT) == CapabilityLevel.NONE
    assert engine_caps.extra == {
        "backend_device": "cpu",
        "supports_batched": False,
        "is_differentiable": False,
        "provides_dynamics": True,
    }

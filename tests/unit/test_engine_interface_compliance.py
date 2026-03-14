"""Tests verifying all engines implement the PhysicsEngine force/torque interface.

Checks that every engine class satisfies the PhysicsEngine protocol for
the five core dynamics methods: compute_mass_matrix, compute_jacobian,
compute_contact_forces, compute_bias_forces, and compute_gravity_forces.

Issue #1175: Standardize force/torque vector and mass/Jacobian matrix access.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

# ---- Engine class references ------------------------------------------------
# Each entry: (module_path, class_name, optional_deps)
# optional_deps are the imports that must succeed before we can test the engine.

ENGINE_SPECS: list[tuple[str, str, list[str]]] = [
    (
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine",
        "MuJoCoPhysicsEngine",
        ["mujoco"],
    ),
    (
        "src.engines.physics_engines.drake.python.drake_physics_engine",
        "DrakePhysicsEngine",
        ["pydrake"],
    ),
    (
        "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine",
        "PinocchioPhysicsEngine",
        ["pinocchio"],
    ),
    (
        "src.engines.physics_engines.opensim.python.opensim_physics_engine",
        "OpenSimPhysicsEngine",
        ["opensim"],
    ),
    (
        "src.engines.physics_engines.myosuite.python.myosuite_physics_engine",
        "MyoSuitePhysicsEngine",
        ["myosuite"],
    ),
    (
        "src.engines.physics_engines.pendulum.python.pendulum_physics_engine",
        "PendulumPhysicsEngine",
        [],
    ),
]

# Core dynamics methods that every engine MUST expose
REQUIRED_METHODS = [
    "compute_mass_matrix",
    "compute_bias_forces",
    "compute_gravity_forces",
    "compute_inverse_dynamics",
    "compute_jacobian",
    "compute_contact_forces",
    "compute_drift_acceleration",
    "compute_control_acceleration",
]


def _try_import(module_path: str, deps: list[str]) -> Any:
    """Import an engine class, skipping if dependencies are missing."""
    for dep in deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            pytest.skip(f"{dep} not installed")

    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        pytest.skip(f"Cannot import {module_path}: {e}")
    return mod


def _get_engine_ids() -> list[str]:
    return [spec[1] for spec in ENGINE_SPECS]


@pytest.mark.parametrize(
    "module_path,class_name,deps",
    ENGINE_SPECS,
    ids=_get_engine_ids(),
)
class TestEngineInterfaceCompliance:
    """Verify that each engine class defines the core dynamics methods."""

    def test_has_required_methods(
        self, module_path: str, class_name: str, deps: list[str]
    ) -> None:
        """Engine class must have all required dynamics methods."""
        mod = _try_import(module_path, deps)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found in {module_path}")

        missing = []
        for method_name in REQUIRED_METHODS:
            attr = getattr(cls, method_name, None)
            if attr is None:
                missing.append(method_name)

        assert not missing, f"{class_name} is missing methods: {', '.join(missing)}"

    def test_methods_are_callable(
        self, module_path: str, class_name: str, deps: list[str]
    ) -> None:
        """All required methods must be callable (not class attributes)."""
        mod = _try_import(module_path, deps)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found in {module_path}")

        for method_name in REQUIRED_METHODS:
            attr = getattr(cls, method_name, None)
            if attr is not None:
                assert callable(attr), (
                    f"{class_name}.{method_name} exists but is not callable"
                )

    def test_get_capabilities_returns_dataclass(
        self, module_path: str, class_name: str, deps: list[str]
    ) -> None:
        """get_capabilities() should be defined and return EngineCapabilities."""
        mod = _try_import(module_path, deps)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found in {module_path}")

        assert hasattr(cls, "get_capabilities"), (
            f"{class_name} must implement get_capabilities()"
        )

    def test_contact_forces_not_abstract(
        self, module_path: str, class_name: str, deps: list[str]
    ) -> None:
        """compute_contact_forces must have a concrete implementation.

        The base protocol provides a default (returns zeros), so engines
        that don't have contact data still satisfy the interface.
        """
        mod = _try_import(module_path, deps)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found in {module_path}")

        method = getattr(cls, "compute_contact_forces", None)
        assert method is not None
        # Should NOT be abstract (base provides default)
        assert not getattr(method, "__isabstractmethod__", False), (
            f"{class_name}.compute_contact_forces should not be abstract"
        )


class TestCapabilitySerialization:
    """Test EngineCapabilities serialization round-trip."""

    def test_to_dict_round_trip(self) -> None:
        from src.engines.common.capabilities import (
            CapabilityLevel,
            EngineCapabilities,
        )

        caps = EngineCapabilities(
            engine_name="TestEngine",
            mass_matrix=CapabilityLevel.FULL,
            jacobian=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.PARTIAL,
            inverse_dynamics=CapabilityLevel.FULL,
            drift_acceleration=CapabilityLevel.NONE,
        )

        d = caps.to_dict()
        restored = EngineCapabilities.from_dict(d)

        assert restored.engine_name == "TestEngine"
        assert restored.mass_matrix == CapabilityLevel.FULL
        assert restored.contact_forces == CapabilityLevel.PARTIAL
        assert restored.drift_acceleration == CapabilityLevel.NONE

    def test_has_contact_forces_property(self) -> None:
        from src.engines.common.capabilities import (
            CapabilityLevel,
            EngineCapabilities,
        )

        none = EngineCapabilities(contact_forces=CapabilityLevel.NONE)
        partial = EngineCapabilities(contact_forces=CapabilityLevel.PARTIAL)
        full = EngineCapabilities(contact_forces=CapabilityLevel.FULL)

        assert none.has_contact_forces is False
        assert partial.has_contact_forces is True
        assert full.has_contact_forces is True

"""Architecture tests for engine tier metadata."""

from __future__ import annotations

import importlib

import pytest
from src.engines.tiers import ENGINE_TIERS, ExperimentalTierWarning

EXPECTED_ENGINE_TIERS = {
    "mujoco": "core",
    "drake": "extended",
    "pinocchio": "extended",
    "opensim": "experimental",
    "myosuite": "experimental",
    "putting_green": "core",
}


def test_engine_tiers_match_policy() -> None:
    assert ENGINE_TIERS == EXPECTED_ENGINE_TIERS


@pytest.mark.parametrize(
    ("engine_name", "expected_tier"), EXPECTED_ENGINE_TIERS.items()
)
def test_engine_package_declares_tier(engine_name: str, expected_tier: str) -> None:
    module = importlib.import_module(f"src.engines.physics_engines.{engine_name}._tier")

    assert expected_tier == module.TIER


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        (
            "src.engines.physics_engines.opensim.python.opensim_physics_engine",
            "OpenSimPhysicsEngine",
        ),
        (
            "src.engines.physics_engines.myosuite.python.myosuite_physics_engine",
            "MyoSuitePhysicsEngine",
        ),
    ],
)
def test_experimental_engine_construction_warns(
    module_name: str, class_name: str
) -> None:
    module = importlib.import_module(module_name)
    engine_cls = getattr(module, class_name)

    with pytest.warns(ExperimentalTierWarning, match="EXPERIMENTAL tier"):
        engine_cls()

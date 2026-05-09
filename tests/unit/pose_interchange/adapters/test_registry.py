"""Tests for :data:`ADAPTER_REGISTRY`."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.adapters import ADAPTER_REGISTRY
from src.shared.python.pose_interchange.protocol import PoseConventionAdapter

pytestmark = pytest.mark.unit

EXPECTED_ENGINES = {"drake", "mujoco", "pinocchio", "opensim", "simscape"}


def test_registry_has_all_engines() -> None:
    assert set(ADAPTER_REGISTRY.keys()) == EXPECTED_ENGINES


def test_registry_values_are_adapter_classes() -> None:
    for engine_name, adapter_cls in ADAPTER_REGISTRY.items():
        assert isinstance(adapter_cls, type)
        instance = adapter_cls()
        assert isinstance(instance, PoseConventionAdapter)
        assert instance.engine_name == engine_name


def test_registry_keys_match_class_engine_names() -> None:
    for engine_name, adapter_cls in ADAPTER_REGISTRY.items():
        assert adapter_cls.engine_name == engine_name

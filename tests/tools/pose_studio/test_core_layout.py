"""Additional invariant checks for the pure-data core layout.

These complement ``tests.unit.tools.pose_studio.test_core`` by asserting
contracts that are easy to break when adding new joints:

* :data:`SUPPORTED_ENGINES` items are unique and are intersected with
  both registries.
* :class:`EngineStatus` round-trips through its string value (it is a
  :class:`str` subclass).
* :data:`JOINT_REGION_LAYOUT` region names are non-empty and unique.
"""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.adapters import ADAPTER_REGISTRY
from src.shared.python.pose_interchange.services import (
    KINEMATICS_SERVICE_REGISTRY,
)
from src.tools.pose_studio.core import (
    JOINT_REGION_LAYOUT,
    SUPPORTED_ENGINES,
    EngineStatus,
    joint_region_partitions_reference_fields,
)

pytestmark = pytest.mark.unit


def test_supported_engines_unique() -> None:
    assert len(set(SUPPORTED_ENGINES)) == len(SUPPORTED_ENGINES)


def test_supported_engines_subset_of_both_registries() -> None:
    """Every advertised engine must have BOTH an adapter and a service."""
    for engine in SUPPORTED_ENGINES:
        assert engine in ADAPTER_REGISTRY
        assert engine in KINEMATICS_SERVICE_REGISTRY


def test_engine_status_str_roundtrip() -> None:
    """EngineStatus is a (str, Enum) so str-comparison must work both ways."""
    assert EngineStatus.MOCK == "mock"
    assert EngineStatus("live") is EngineStatus.LIVE
    assert EngineStatus("error") is EngineStatus.ERROR


def test_joint_region_names_unique_and_non_empty() -> None:
    names = list(JOINT_REGION_LAYOUT)
    assert len(set(names)) == len(names), "duplicate region name"
    for name, joints in JOINT_REGION_LAYOUT.items():
        assert name.strip(), "region name must be non-empty"
        assert joints, f"region {name!r} must have at least one joint"


def test_partition_helper_function() -> None:
    # Direct invocation (also covered transitively in test_core but kept here
    # so this file's invariants are self-contained).
    assert joint_region_partitions_reference_fields() is True

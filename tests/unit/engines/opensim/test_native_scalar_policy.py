"""Portable scalar executor policy admission."""

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_scalar_replay import (
    NativeScalarReplayPolicy,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("accuracy", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_scalar_accuracy_rejects_invalid_values(accuracy: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        NativeScalarReplayPolicy(accuracy)


def test_scalar_policy_rejects_undeclared_observer() -> None:
    with pytest.raises(TypeError):
        NativeScalarReplayPolicy(1e-8, constrained_cold_start=object())


@pytest.mark.parametrize("paths", [["/forceset/contact"], ("relative",), ("/a", "/a")])
def test_scalar_contact_paths_require_immutable_unique_native_paths(
    paths: tuple,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        NativeScalarReplayPolicy(1e-8, contact_force_paths=paths)

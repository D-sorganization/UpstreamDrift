"""Source-policy tests for zero-DOF structural WeldJoint admission."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
    _audit_component_policy,
)

pytestmark = pytest.mark.unit


class _Frame:
    def __init__(self, path: str, concrete_class: str = "Body") -> None:
        self.path = path
        self.concrete_class = concrete_class

    def getConcreteClassName(self) -> str:
        return self.concrete_class

    def getAbsolutePathString(self) -> str:
        return self.path


class _Weld:
    def __init__(
        self,
        *,
        coordinates: int = 0,
        parent: str | None = "/bodyset/parent",
        child: str | None = "/bodyset/child",
        native: bool = True,
        parent_type: str = "Body",
        child_type: str = "Body",
    ) -> None:
        self.coordinates = coordinates
        self.parent = None if parent is None else _Frame(parent, parent_type)
        self.child = None if child is None else _Frame(child, child_type)
        self.native = native

    def getConcreteClassName(self) -> str:
        return "WeldJoint"

    def getAbsolutePathString(self) -> str:
        return "/jointset/structural_weld"

    def numCoordinates(self) -> int:
        return self.coordinates

    def getParentFrame(self) -> _Frame | None:
        return self.parent

    def getChildFrame(self) -> _Frame | None:
        return self.child


def _downcast(expected: type | None = None):
    if expected is None:
        return staticmethod(lambda _component: None)
    return staticmethod(
        lambda component: (
            component
            if isinstance(component, expected) and getattr(component, "native", True)
            else None
        )
    )


def _opensim_api() -> SimpleNamespace:
    return SimpleNamespace(
        WeldJoint=SimpleNamespace(safeDownCast=_downcast(_Weld)),
        PhysicalFrame=SimpleNamespace(safeDownCast=_downcast(_Frame)),
        Controller=SimpleNamespace(safeDownCast=_downcast()),
        PositionMotion=SimpleNamespace(safeDownCast=_downcast()),
        Force=SimpleNamespace(safeDownCast=_downcast()),
        Muscle=SimpleNamespace(safeDownCast=_downcast()),
    )


def _audit(component: object) -> None:
    _audit_component_policy((component,), frozenset(), False, None, _opensim_api())


def test_native_zero_coordinate_weld_with_two_valid_frames_is_admitted() -> None:
    _audit(_Weld())


def test_structural_admission_does_not_widen_strict_base_component_set() -> None:
    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_muscle_bundle,
    )

    assert "WeldJoint" not in native_muscle_bundle._COMPONENTS


@pytest.mark.parametrize(
    ("component", "message"),
    [
        (_Weld(native=False), "native concrete class"),
        (_Weld(coordinates=1), "zero coordinates"),
        (_Weld(parent=None), "parent frame"),
        (_Weld(child=None), "child frame"),
        (_Weld(parent="", child="/bodyset/child"), "parent frame"),
        (_Weld(parent_type="PluginFrame"), "parent frame type"),
        (_Weld(child_type="PhysicalOffsetFramePlugin"), "child frame type"),
        (
            _Weld(parent="/bodyset/same", child="/bodyset/same"),
            "distinct frames",
        ),
    ],
)
def test_weld_joint_mutations_fail_closed(component: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _audit(component)


@pytest.mark.parametrize("concrete_class", ["WeldConstraint", "PluginWeldJoint"])
def test_only_exact_weld_joint_class_uses_structural_admission(
    concrete_class: str,
) -> None:
    component = SimpleNamespace(
        getConcreteClassName=lambda: concrete_class,
        getAbsolutePathString=lambda: "/jointset/not_a_weld",
    )
    with pytest.raises(
        ValueError, match=f"unreviewed constrained muscle component: {concrete_class}"
    ):
        _audit(component)


@pytest.mark.parametrize(
    "concrete_class",
    [
        "Controller",
        "CoordinateLimitForce",
    ],
)
def test_weld_admission_does_not_waive_controller_or_force_guards(
    concrete_class: str,
) -> None:
    class _Other:
        def getConcreteClassName(self) -> str:
            return concrete_class

        def getAbsolutePathString(self) -> str:
            return "/forceset/unreviewed"

    api = _opensim_api()
    api.Controller = SimpleNamespace(safeDownCast=_downcast(_Other))
    api.Force = SimpleNamespace(safeDownCast=_downcast(_Other))
    with pytest.raises(ValueError):
        _audit_component_policy((_Weld(), _Other()), frozenset(), False, None, api)

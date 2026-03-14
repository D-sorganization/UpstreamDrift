from unittest.mock import Mock

from src.api.models.requests import ForceOverlayRequest
from src.api.routes.force_overlays import (
    _is_filtered_out,
    _magnitude_to_color,
    _resolve_body_name,
    _resolve_joint_names,
    _should_include_force_type,
)


def test_magnitude_to_color():
    assert _magnitude_to_color(0, 10) == [0.0, 1.0, 1.0, 1.0]
    assert _magnitude_to_color(5, 10) == [1.0, 1.0, 0.0, 1.0]
    assert _magnitude_to_color(10, 10) == [1.0, 0.0, 0.0, 1.0]  # based on t logic


def test_resolve_joint_names():
    engine = Mock()
    engine.joint_names = ["j1", "j2"]
    assert _resolve_joint_names(engine, 2) == ["j1", "j2"]

    engine = Mock(spec=[])
    assert _resolve_joint_names(engine, 2) == ["joint_0", "joint_1"]


def test_should_include_force_type():
    req = ForceOverlayRequest(
        enabled=True,
        force_types=["applied"],
        color_by_magnitude=True,
        scale_factor=0.01,
    )
    assert _should_include_force_type(req, "applied")
    assert not _should_include_force_type(req, "contact")

    req = ForceOverlayRequest(
        enabled=True, force_types=["all"], color_by_magnitude=True, scale_factor=0.01
    )
    assert _should_include_force_type(req, "contact")


def test_resolve_body_name():
    names = ["j1", "j2"]
    assert _resolve_body_name(names, 0) == "j1"
    assert _resolve_body_name(names, 2) == "joint_2"


def test_is_filtered_out():
    req = ForceOverlayRequest(
        enabled=True,
        force_types=["all"],
        body_filter=["j1"],
        scale_factor=0.01,
        color_by_magnitude=True,
    )
    assert not _is_filtered_out(req, "j1")
    assert _is_filtered_out(req, "j2")

    req = ForceOverlayRequest(
        enabled=True,
        force_types=["all"],
        body_filter=None,
        scale_factor=0.01,
        color_by_magnitude=True,
    )
    assert not _is_filtered_out(req, "j1")

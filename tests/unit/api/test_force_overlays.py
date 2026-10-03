"""Unit tests for internal helper functions in force overlays route (#1199, #11307)."""

from __future__ import annotations

from unittest.mock import Mock
import pytest

from src.api.models.requests import ForceOverlayRequest
from src.api.routes.force_overlays import (
    _build_overlay_response,
    _get_sim_time,
)

pytestmark = pytest.mark.unit


def test_get_sim_time_none_engine() -> None:
    manager = Mock()
    manager.get_active_engine.return_value = None
    assert _get_sim_time(manager) == 0.0


def test_get_sim_time_dict_state() -> None:
    engine = Mock()
    engine.get_state.return_value = {"time": 1.25}
    manager = Mock()
    manager.get_active_engine.return_value = engine
    assert _get_sim_time(manager) == 1.25


def test_get_sim_time_time_attribute() -> None:
    engine = Mock(spec=["time"])
    engine.time = 2.5
    manager = Mock()
    manager.get_active_engine.return_value = engine
    assert _get_sim_time(manager) == 2.5


def test_get_sim_time_exception_handling() -> None:
    manager = Mock()
    manager.get_active_engine.side_effect = RuntimeError("Engine unavailable")
    assert _get_sim_time(manager) == 0.0


def test_build_overlay_response_disabled() -> None:
    manager = Mock()
    manager.get_active_engine.return_value = None
    config = ForceOverlayRequest(
        enabled=False,
        force_types=["applied"],
        color_by_magnitude=True,
        scale_factor=0.01,
    )
    resp = _build_overlay_response(manager, config)
    assert resp.glyphs is None
    assert resp.frame is None
    assert resp.unavailable_reason == "Force overlay disabled in request"
    assert resp.vectors == []
    assert resp.total_force_magnitude == 0.0


def test_build_overlay_response_no_engine() -> None:
    manager = Mock()
    manager.get_active_engine.side_effect = RuntimeError("No active engine")
    config = ForceOverlayRequest(
        enabled=True,
        force_types=["applied"],
        color_by_magnitude=True,
        scale_factor=0.01,
    )
    resp = _build_overlay_response(manager, config)
    assert resp.glyphs is None
    assert resp.frame is None
    assert (
        resp.unavailable_reason == "No force/torque frame available from active engine"
    )

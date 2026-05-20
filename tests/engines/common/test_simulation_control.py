"""Coverage for simulation_control.py — controller lifecycle + dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.common.simulation_control import (
    ForceOverlay,
    MeasurementResult,
    SimulationController,
    SimulationMode,
)
from src.shared.python.core.contracts import PreconditionError


class _StubController(SimulationController):
    """Minimal concrete controller for testing the base class."""

    def __init__(self) -> None:
        super().__init__()
        self.step_count = 0

    def _do_step(self) -> None:
        self.step_count += 1

    def translate_body(self, body_name: str, delta: np.ndarray) -> bool:
        return True

    def rotate_body(self, body_name: str, axis: np.ndarray, angle: float) -> bool:
        return True

    def measure_distance(self, body_a: str, body_b: str) -> MeasurementResult:
        return MeasurementResult(
            type="distance", value=1.5, unit="m", point_a=body_a, point_b=body_b
        )

    def measure_angle(self, body_a: str, body_b: str, body_c: str) -> MeasurementResult:
        return MeasurementResult(
            type="angle",
            value=0.5,
            unit="rad",
            point_a=body_a,
            point_b=body_b,
        )


# ---------------------------------------------------------------------------
# MeasurementResult
# ---------------------------------------------------------------------------


class TestMeasurementResult:
    def test_to_dict_without_vector(self) -> None:
        r = MeasurementResult(type="distance", value=1.0, unit="m", point_a="a")
        d = r.to_dict()
        assert d == {
            "type": "distance",
            "value": 1.0,
            "unit": "m",
            "point_a": "a",
            "point_b": "",
        }

    def test_to_dict_with_vector(self) -> None:
        r = MeasurementResult(
            type="vec",
            value=1.0,
            unit="m",
            point_a="a",
            point_b="b",
            vector=np.array([1.0, 2.0, 3.0]),
        )
        d = r.to_dict()
        assert d["vector"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# ForceOverlay
# ---------------------------------------------------------------------------


class TestForceOverlay:
    def test_defaults(self) -> None:
        o = ForceOverlay(body_name="club")
        np.testing.assert_array_equal(o.force, np.zeros(3))
        np.testing.assert_array_equal(o.torque, np.zeros(3))
        assert o.scale == 1.0
        assert o.color == (255, 100, 100)

    def test_to_dict_round_trip(self) -> None:
        o = ForceOverlay(
            body_name="club",
            force=np.array([1.0, 0, 0]),
            torque=np.array([0, 1.0, 0]),
            scale=2.0,
            color=(0, 255, 0),
            label="grip",
        )
        d = o.to_dict()
        assert d["body_name"] == "club"
        assert d["force"] == [1.0, 0.0, 0.0]
        assert d["torque"] == [0.0, 1.0, 0.0]
        assert d["color"] == [0, 255, 0]
        assert d["label"] == "grip"
        assert d["scale"] == 2.0


# ---------------------------------------------------------------------------
# SimulationController
# ---------------------------------------------------------------------------


class TestSimulationController:
    def test_initial_mode_is_idle(self) -> None:
        c = _StubController()
        assert c.mode == SimulationMode.IDLE
        assert not c.is_running
        assert not c.is_paused

    def test_start_pause_resume_stop_cycle(self) -> None:
        c = _StubController()
        assert c.start() is True
        assert c.is_running
        assert c.pause() is True
        assert c.is_paused
        assert c.start() is True  # resume
        assert c.is_running
        assert c.stop() is True
        assert c.mode == SimulationMode.IDLE

    def test_single_step_runs_do_step_and_pauses(self) -> None:
        c = _StubController()
        assert c.single_step() is True
        assert c.step_count == 1
        assert c.mode == SimulationMode.PAUSED

    def test_overlays_add_and_clear(self) -> None:
        c = _StubController()
        c.add_force_overlay(ForceOverlay(body_name="a"))
        c.add_force_overlay(ForceOverlay(body_name="b"))
        assert len(c.overlays) == 2
        # overlays is a copy
        c.overlays.clear()
        assert len(c.overlays) == 2
        c.clear_overlays()
        assert c.overlays == []

    def test_measurements_serialization(self) -> None:
        c = _StubController()
        c._measurements.append(c.measure_distance("a", "b"))
        c._measurements.append(c.measure_angle("a", "b", "c"))
        results = c.get_measurements()
        assert len(results) == 2
        assert results[0]["type"] == "distance"
        c.clear_measurements()
        assert c.get_measurements() == []

    def test_start_from_running_violates_precondition(self) -> None:
        c = _StubController()
        c.start()
        with pytest.raises(PreconditionError):
            c.start()

    def test_pause_from_idle_violates_precondition(self) -> None:
        c = _StubController()
        with pytest.raises(PreconditionError):
            c.pause()

    def test_stop_from_idle_violates_precondition(self) -> None:
        c = _StubController()
        with pytest.raises(PreconditionError):
            c.stop()

"""TDD tests for simulation WebSocket route fixes (issue #2481).

Bugs covered:
1. Engine name normalisation: EngineType(engine_type.upper()) fails for all valid
   lowercase engine names because enum values are lowercase strings.
2. set_state signature mismatch: engine.set_state expects (q, v) as two ndarray
   arguments, but the route passed the raw dict from the JSON config.
3. get_state serialisation: engine.get_state() returns (np.ndarray, np.ndarray)
   which cannot be JSON-serialised; frames must convert arrays to lists.

Unit tests use helper functions extracted from simulation_ws and do NOT require
httpx. Integration tests (TestClient) are skipped when httpx is not installed.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.api.routes.simulation_ws import (
    _apply_initial_state,
    _compute_real_time_sleep_delay,
    _engine_state_to_dict,
    _engine_type_from_str,
    _get_simulation_speed_factor,
    _handle_client_commands,
    _wait_for_resume_or_stop,
)
from src.shared.python.engine_core.engine_registry import EngineType

try:
    _HAS_TESTCLIENT = True
except RuntimeError:
    _HAS_TESTCLIENT = False

requires_testclient = pytest.mark.skipif(
    not _HAS_TESTCLIENT, reason="httpx not installed"
)

# ---------------------------------------------------------------------------
# 1. Engine name normalisation — pure unit tests
# ---------------------------------------------------------------------------


class TestEngineTypeNormalisation:
    """_engine_type_from_str must handle any case variant of valid names."""

    def test_lowercase_accepted(self) -> None:
        """'mujoco' must map to EngineType.MUJOCO."""
        assert _engine_type_from_str("mujoco") == EngineType.MUJOCO

    def test_uppercase_accepted(self) -> None:
        """'MUJOCO' must also map to EngineType.MUJOCO."""
        assert _engine_type_from_str("MUJOCO") == EngineType.MUJOCO

    def test_mixed_case_accepted(self) -> None:
        """'MuJoCo' must map to EngineType.MUJOCO."""
        assert _engine_type_from_str("MuJoCo") == EngineType.MUJOCO

    def test_all_valid_names_round_trip(self) -> None:
        """Every EngineType value must round-trip through _engine_type_from_str."""
        for et in EngineType:
            assert _engine_type_from_str(et.value) == et
            assert _engine_type_from_str(et.value.upper()) == et

    def test_invalid_name_raises_value_error(self) -> None:
        """Unknown engine name must raise ValueError."""
        with pytest.raises(ValueError):
            _engine_type_from_str("nosuchengine")


# ---------------------------------------------------------------------------
# 2. set_state signature — pure unit tests
# ---------------------------------------------------------------------------


class TestApplyInitialState:
    """_apply_initial_state must call engine.set_state(q, v) correctly."""

    def _make_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.set_state.return_value = None
        return engine

    def test_set_state_called_with_two_ndarray_args(self) -> None:
        """set_state receives (q, v) as two separate numpy arrays."""
        engine = self._make_engine()
        _apply_initial_state(engine, {"q": [1.0, 2.0], "v": [3.0, 4.0]})

        engine.set_state.assert_called_once()
        args = engine.set_state.call_args.args
        assert len(args) == 2, f"Expected 2 args (q, v), got {len(args)}"
        q, v = args
        assert isinstance(q, np.ndarray)
        assert isinstance(v, np.ndarray)
        assert list(q) == pytest.approx([1.0, 2.0])
        assert list(v) == pytest.approx([3.0, 4.0])

    def test_empty_initial_state_dict_uses_empty_arrays(self) -> None:
        """Missing q/v keys default to empty arrays."""
        engine = self._make_engine()
        _apply_initial_state(engine, {})
        args = engine.set_state.call_args.args
        assert len(args) == 2
        q, v = args
        assert len(q) == 0
        assert len(v) == 0

    def test_engine_without_set_state_is_skipped(self) -> None:
        """Engines lacking set_state must not raise."""
        engine = MagicMock(spec=[])  # no set_state attr
        _apply_initial_state(engine, {"q": [0.0], "v": [0.0]})  # must not raise


# ---------------------------------------------------------------------------
# 3. get_state JSON serialisation — pure unit tests
# ---------------------------------------------------------------------------


class TestEngineStateToDict:
    """_engine_state_to_dict must produce a JSON-serialisable dict."""

    def _make_engine(
        self,
        q: list[float] | None = None,
        v: list[float] | None = None,
    ) -> MagicMock:
        engine = MagicMock()
        engine.get_state.return_value = (
            np.array(q or [0.1, 0.2]),
            np.array(v or [0.3, 0.4]),
        )
        return engine

    def test_returns_dict_with_q_and_v(self) -> None:
        engine = self._make_engine([0.1, 0.2], [0.3, 0.4])
        result = _engine_state_to_dict(engine)
        assert "q" in result
        assert "v" in result
        assert result["q"] == pytest.approx([0.1, 0.2])
        assert result["v"] == pytest.approx([0.3, 0.4])

    def test_result_is_json_serialisable(self) -> None:
        engine = self._make_engine([0.5, 1.0, 1.5], [2.0, 2.5, 3.0])
        result = _engine_state_to_dict(engine)
        try:
            json.dumps(result)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"State dict is not JSON-serialisable: {exc}")

    def test_q_and_v_are_plain_lists(self) -> None:
        """q and v must be plain Python lists, not numpy arrays."""
        engine = self._make_engine([0.1], [0.2])
        result = _engine_state_to_dict(engine)
        assert isinstance(result["q"], list)
        assert isinstance(result["v"], list)

    def test_engine_without_get_state_returns_empty(self) -> None:
        """Engines lacking get_state return an empty dict."""
        engine = MagicMock(spec=[])  # no get_state attr
        result = _engine_state_to_dict(engine)
        assert result == {}

    def test_numpy_float32_serialises(self) -> None:
        """numpy float32 values inside arrays must be JSON-serialisable."""
        engine = MagicMock()
        engine.get_state.return_value = (
            np.array([0.1], dtype=np.float32),
            np.array([0.2], dtype=np.float32),
        )
        result = _engine_state_to_dict(engine)
        json.dumps(result)  # must not raise


class _Stats:
    def __init__(self, speed_factor: float) -> None:
        self.speed_factor = speed_factor


class _SimulationService:
    def __init__(self, speed_factor: float) -> None:
        self.stats = _Stats(speed_factor)


class _AppState:
    def __init__(self, speed_factor: float | None = None) -> None:
        if speed_factor is not None:
            self.simulation_service = _SimulationService(speed_factor)


class _App:
    def __init__(self, speed_factor: float | None = None) -> None:
        self.state = _AppState(speed_factor)


class _WebSocket:
    def __init__(self, speed_factor: float | None = None) -> None:
        self.app = _App(speed_factor)


class TestSimulationSpeedFactor:
    """Simulation WebSocket loop must read speed from shared service state."""

    def test_uses_service_speed_factor_when_available(self) -> None:
        websocket: Any = _WebSocket(speed_factor=2.5)
        assert _get_simulation_speed_factor(websocket, {}) == pytest.approx(2.5)

    def test_falls_back_to_config_when_service_missing(self) -> None:
        websocket: Any = _WebSocket()
        assert _get_simulation_speed_factor(
            websocket, {"speed_factor": 1.5}
        ) == pytest.approx(1.5)

    def test_invalid_speed_falls_back_to_default(self) -> None:
        websocket: Any = _WebSocket(speed_factor=0.0)
        assert _get_simulation_speed_factor(websocket, {}) == pytest.approx(1.0)


class TestRealTimeSleepDelay:
    """Real-time pacing must scale inverse to requested simulation speed."""

    def test_returns_remaining_delay_at_default_speed(self) -> None:
        assert _compute_real_time_sleep_delay(0.002, 1.0, 0.0005) == pytest.approx(
            0.0015
        )

    def test_scales_delay_for_faster_speed(self) -> None:
        assert _compute_real_time_sleep_delay(0.002, 2.0, 0.0) == pytest.approx(0.001)

    def test_never_returns_negative_delay(self) -> None:
        assert _compute_real_time_sleep_delay(0.002, 2.0, 0.01) == pytest.approx(0.0)


class TestClientCommandHandling:
    """_handle_client_commands and _wait_for_resume_or_stop must update config/stats on set_speed."""

    @pytest.mark.anyio
    async def test_handle_client_commands_set_speed(self) -> None:
        websocket: Any = _WebSocket(speed_factor=1.0)

        async def mock_receive_json() -> dict[str, Any]:
            return {"action": "set_speed", "speed_factor": 4.5}

        websocket.receive_json = mock_receive_json

        config = {"speed_factor": 1.0}
        command = await _handle_client_commands(websocket, config)

        assert command == "continue"
        assert config["speed_factor"] == pytest.approx(4.5)
        assert (
            websocket.app.state.simulation_service.stats.speed_factor
            == pytest.approx(4.5)
        )

    @pytest.mark.anyio
    async def test_wait_for_resume_or_stop_set_speed(self) -> None:
        websocket: Any = _WebSocket(speed_factor=1.0)

        messages: list[dict[str, Any]] = [
            {"action": "set_speed", "speed_factor": 3.0},
            {"action": "resume"},
        ]

        async def mock_receive_json() -> dict[str, Any]:
            return messages.pop(0)

        websocket.receive_json = mock_receive_json

        async def mock_send_json(data: Any) -> None:
            pass

        websocket.send_json = mock_send_json

        config = {"speed_factor": 1.0}
        stopped = await _wait_for_resume_or_stop(websocket, config)

        assert not stopped
        assert config["speed_factor"] == pytest.approx(3.0)
        assert (
            websocket.app.state.simulation_service.stats.speed_factor
            == pytest.approx(3.0)
        )

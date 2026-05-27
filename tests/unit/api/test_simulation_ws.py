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
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import src.api.routes.simulation_ws as simulation_ws_module
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.routes import simulation_ws
from src.api.routes.simulation_ws import (
    _apply_initial_state,
    _compute_real_time_sleep_delay,
    _engine_state_to_dict,
    _engine_type_from_str,
    _get_simulation_speed_factor,
    _handle_client_commands,
    _is_numeric_sequence,
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


class _FailingEngine:
    def step(self, _timestep: float) -> None:
        raise RuntimeError("sensitive simulation failure")


class _EngineManager:
    def switch_engine(self, _engine_type: EngineType) -> bool:
        return True

    def get_active_physics_engine(self) -> object:
        return _FailingEngine()


class TestSimulationStreamErrors:
    """Unexpected simulation failures should preserve tracebacks server-side."""

    def test_unexpected_error_hides_internal_details(self) -> None:
        app = FastAPI()
        app.state.engine_manager = _EngineManager()
        app.include_router(simulation_ws.router)

        async def fake_resolve_ws_user(_websocket: Any) -> object:
            return object()

        with (
            patch(
                "src.api.routes.simulation_ws.resolve_ws_user",
                side_effect=fake_resolve_ws_user,
            ),
            patch("src.api.routes.simulation_ws.logger.exception") as logger_exception,
            TestClient(app).websocket_connect("/ws/simulate/mujoco") as websocket,
        ):
            websocket.send_json(
                {"action": "start", "config": {"duration": 0.01, "timestep": 0.01}}
            )

            running = websocket.receive_json()
            assert running == {"status": "running", "duration": 0.01}

            error = websocket.receive_json()
            assert error == {"error": "Internal server error"}

        logger_exception.assert_called_once()


class _RouteAppState:
    def __init__(self) -> None:
        self.engine_manager = object()


class _RouteApp:
    def __init__(self) -> None:
        self.state = _RouteAppState()


class _RouteWebSocket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)
        self.app = _RouteApp()
        self.accepted = False
        self.closed = False
        self.sent: list[dict[str, Any]] = []

    async def accept(self) -> None:
        self.accepted = True

    async def receive_json(self) -> dict[str, Any]:
        return self._messages.pop(0)

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_simulation_stream_sanitizes_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected runtime failures must be logged with traceback and sanitized."""

    websocket = _RouteWebSocket([{"action": "start", "config": {}}])

    async def fake_load_simulation_engine(*_args: Any, **_kwargs: Any) -> object:
        return object()

    async def fake_run_simulation_loop(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("top secret backend detail")

    monkeypatch.setattr(
        simulation_ws_module,
        "_load_simulation_engine",
        fake_load_simulation_engine,
    )
    monkeypatch.setattr(
        simulation_ws_module,
        "_run_simulation_loop",
        fake_run_simulation_loop,
    )
    monkeypatch.setattr(
        simulation_ws_module,
        "resolve_ws_user",
        AsyncMock(return_value=object()),
    )

    with caplog.at_level("ERROR"):
        await simulation_ws_module.simulation_stream(websocket, "mujoco")

    assert websocket.accepted is True
    assert websocket.closed is True
    assert websocket.sent == [{"error": "Internal server error"}]
    assert "top secret backend detail" not in json.dumps(websocket.sent)
    assert any(
        record.message == "Simulation WebSocket failed for engine=mujoco"
        and record.exc_info is not None
        for record in caplog.records
    )


@pytest.mark.anyio
async def test_simulation_stream_rejects_invalid_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid duration must fail at the WS boundary before engine load."""

    websocket = _RouteWebSocket([{"action": "start", "config": {"duration": 0}}])
    load_engine = AsyncMock()

    async def fake_resolve_ws_user(_websocket: Any) -> object:
        return object()

    monkeypatch.setattr(simulation_ws_module, "resolve_ws_user", fake_resolve_ws_user)
    monkeypatch.setattr(simulation_ws_module, "_load_simulation_engine", load_engine)

    await simulation_ws_module.simulation_stream(websocket, "mujoco")

    assert websocket.accepted is True
    assert websocket.closed is True
    # The WS numeric-field guard fires before Pydantic and emits a specific message.
    assert websocket.sent == [{"error": "duration must be a positive finite number"}]
    load_engine.assert_not_called()


@pytest.mark.anyio
async def test_simulation_stream_rejects_malformed_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed q/v vectors must be rejected before set_state runs."""

    websocket = _RouteWebSocket(
        [{"action": "start", "config": {"initial_state": {"q": "bad-state"}}}]
    )
    load_engine = AsyncMock()

    async def fake_resolve_ws_user(_websocket: Any) -> object:
        return object()

    monkeypatch.setattr(simulation_ws_module, "resolve_ws_user", fake_resolve_ws_user)
    monkeypatch.setattr(simulation_ws_module, "_load_simulation_engine", load_engine)

    await simulation_ws_module.simulation_stream(websocket, "mujoco")

    assert websocket.accepted is True
    assert websocket.closed is True
    assert websocket.sent == [{"error": "Invalid simulation config"}]
    load_engine.assert_not_called()


# ---------------------------------------------------------------------------
# 5. _is_numeric_sequence — pure unit tests (issue #5918)
# ---------------------------------------------------------------------------


class TestIsNumericSequence:
    """_is_numeric_sequence guards np.array() against non-numeric inputs."""

    def test_valid_float_list(self) -> None:
        """A list of floats must be accepted."""
        assert _is_numeric_sequence([1.0, 2.5, 3.14]) is True

    def test_valid_int_list(self) -> None:
        """A list of ints must be accepted."""
        assert _is_numeric_sequence([0, 1, 2]) is True

    def test_empty_list_is_valid(self) -> None:
        """An empty list is a valid (zero-length) numeric sequence."""
        assert _is_numeric_sequence([]) is True

    def test_string_is_rejected(self) -> None:
        """A bare string must be rejected."""
        assert _is_numeric_sequence("not-a-list") is False

    def test_none_is_rejected(self) -> None:
        """None must be rejected (would produce NaN array)."""
        assert _is_numeric_sequence(None) is False

    def test_list_with_string_element_is_rejected(self) -> None:
        """A list containing a non-numeric element must be rejected."""
        assert _is_numeric_sequence([1.0, "bad"]) is False

    def test_bool_elements_are_rejected(self) -> None:
        """Lists of booleans must be rejected (bool is a subclass of int)."""
        assert _is_numeric_sequence([True, False]) is False

    def test_tuple_of_numbers_is_valid(self) -> None:
        """A tuple of numbers is also accepted."""
        assert _is_numeric_sequence((1.0, 2.0)) is True

    def test_nested_list_is_rejected(self) -> None:
        """A nested list must be rejected."""
        assert _is_numeric_sequence([[1.0], [2.0]]) is False


# ---------------------------------------------------------------------------
# 6. _apply_initial_state validation — issue #5918
# ---------------------------------------------------------------------------


class TestApplyInitialStateValidation:
    """_apply_initial_state must guard set_state() against non-numeric q/v."""

    def _make_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.set_state.return_value = None
        return engine

    def test_string_q_is_ignored_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-numeric q must produce a warning and skip set_state."""
        engine = self._make_engine()
        with caplog.at_level("WARNING"):
            _apply_initial_state(engine, {"q": "bad-state", "v": []})
        engine.set_state.assert_not_called()
        assert any("initial_state.q" in r.message for r in caplog.records)

    def test_none_q_is_ignored_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """None q must produce a warning and skip set_state."""
        engine = self._make_engine()
        with caplog.at_level("WARNING"):
            _apply_initial_state(engine, {"q": None, "v": []})
        engine.set_state.assert_not_called()
        assert any("initial_state.q" in r.message for r in caplog.records)

    def test_string_v_is_ignored_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-numeric v must produce a warning and skip set_state."""
        engine = self._make_engine()
        with caplog.at_level("WARNING"):
            _apply_initial_state(engine, {"q": [1.0], "v": "bad-v"})
        engine.set_state.assert_not_called()
        assert any("initial_state.v" in r.message for r in caplog.records)

    def test_valid_q_and_v_calls_set_state(self) -> None:
        """Valid numeric q and v must still invoke set_state."""
        engine = self._make_engine()
        _apply_initial_state(engine, {"q": [0.1, 0.2], "v": [0.3, 0.4]})
        engine.set_state.assert_called_once()

    def test_missing_q_and_v_defaults_to_empty_arrays(self) -> None:
        """Missing q/v keys must default to empty arrays (existing contract)."""
        engine = self._make_engine()
        _apply_initial_state(engine, {})
        engine.set_state.assert_called_once()
        args = engine.set_state.call_args.args
        assert len(args[0]) == 0
        assert len(args[1]) == 0


# ---------------------------------------------------------------------------
# 7. _handle_client_commands JSON parse error isolation — issue #5918
# ---------------------------------------------------------------------------


class TestHandleClientCommandsJsonError:
    """_handle_client_commands must handle invalid JSON from the client."""

    @pytest.mark.anyio
    async def test_invalid_json_sends_error_and_continues(self) -> None:
        """A json.JSONDecodeError must trigger an error response, not a crash."""
        websocket: Any = _WebSocket(speed_factor=1.0)

        async def mock_receive_json() -> dict[str, Any]:
            raise json.JSONDecodeError("Expecting value", "", 0)

        sent_messages: list[dict[str, Any]] = []

        async def mock_send_json(data: dict[str, Any]) -> None:
            sent_messages.append(data)

        websocket.receive_json = mock_receive_json
        websocket.send_json = mock_send_json

        config: dict[str, Any] = {}
        result = await _handle_client_commands(websocket, config)

        assert result == "continue"
        assert sent_messages == [
            {"error": "invalid_json", "message": "Message must be valid JSON"}
        ]

    @pytest.mark.anyio
    async def test_timeout_still_returns_continue(self) -> None:
        """TimeoutError (no message yet) must still return 'continue' cleanly."""
        websocket: Any = _WebSocket(speed_factor=1.0)

        async def mock_receive_json() -> dict[str, Any]:
            raise TimeoutError

        websocket.receive_json = mock_receive_json

        config: dict[str, Any] = {}
        result = await _handle_client_commands(websocket, config)
        assert result == "continue"


# ---------------------------------------------------------------------------
# _validate_ws_numeric_fields — unit tests for issue #5918
# ---------------------------------------------------------------------------


class TestValidateWsNumericFields:
    """_validate_ws_numeric_fields must reject non-finite and out-of-bounds values."""

    def _call(self, config: dict[str, Any]) -> str | None:
        return simulation_ws_module._validate_ws_numeric_fields(config)

    # speed_factor checks
    def test_nan_speed_factor_rejected(self) -> None:
        assert self._call({"speed_factor": float("nan")}) is not None

    def test_inf_speed_factor_rejected(self) -> None:
        assert self._call({"speed_factor": float("inf")}) is not None

    def test_neg_inf_speed_factor_rejected(self) -> None:
        assert self._call({"speed_factor": float("-inf")}) is not None

    def test_string_nan_speed_factor_rejected(self) -> None:
        assert self._call({"speed_factor": "NaN"}) is not None

    def test_valid_speed_factor_accepted(self) -> None:
        assert self._call({"speed_factor": 2.0}) is None

    # duration checks
    def test_nan_duration_rejected(self) -> None:
        assert self._call({"duration": float("nan")}) is not None

    def test_inf_duration_rejected(self) -> None:
        assert self._call({"duration": float("inf")}) is not None

    def test_zero_duration_rejected(self) -> None:
        assert self._call({"duration": 0.0}) is not None

    def test_negative_duration_rejected(self) -> None:
        assert self._call({"duration": -1.0}) is not None

    def test_duration_at_cap_rejected(self) -> None:
        """duration >= 3600s must be rejected."""
        assert self._call({"duration": 3600.0}) is not None

    def test_duration_just_below_cap_accepted(self) -> None:
        assert self._call({"duration": 3599.9}) is None

    def test_valid_duration_accepted(self) -> None:
        assert self._call({"duration": 10.0}) is None

    # timestep checks
    def test_nan_timestep_rejected(self) -> None:
        assert self._call({"timestep": float("nan")}) is not None

    def test_inf_timestep_rejected(self) -> None:
        assert self._call({"timestep": float("inf")}) is not None

    def test_zero_timestep_rejected(self) -> None:
        assert self._call({"timestep": 0.0}) is not None

    def test_negative_timestep_rejected(self) -> None:
        assert self._call({"timestep": -0.001}) is not None

    def test_timestep_at_max_rejected(self) -> None:
        """timestep >= 1.0s must be rejected."""
        assert self._call({"timestep": 1.0}) is not None

    def test_timestep_too_small_rejected(self) -> None:
        """timestep < 1e-6 must be rejected."""
        assert self._call({"timestep": 1e-7}) is not None

    def test_valid_timestep_accepted(self) -> None:
        assert self._call({"timestep": 0.002}) is None

    def test_empty_config_accepted(self) -> None:
        """No numeric fields supplied should not raise."""
        assert self._call({}) is None


# ---------------------------------------------------------------------------
# WS handler integration: non-finite speed_factor/timestep/duration error frames
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_simulation_stream_rejects_nan_speed_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-finite speed_factor must send an error frame and not load an engine."""
    websocket = _RouteWebSocket(
        [{"action": "start", "config": {"speed_factor": float("nan")}}]
    )
    load_engine = AsyncMock()

    monkeypatch.setattr(
        simulation_ws_module, "resolve_ws_user", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(simulation_ws_module, "_load_simulation_engine", load_engine)

    await simulation_ws_module.simulation_stream(websocket, "mujoco")

    assert websocket.accepted is True
    assert websocket.closed is True
    assert len(websocket.sent) == 1
    assert "error" in websocket.sent[0]
    load_engine.assert_not_called()


@pytest.mark.anyio
async def test_simulation_stream_rejects_inf_speed_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Infinite speed_factor must send an error frame and not load an engine."""
    websocket = _RouteWebSocket(
        [{"action": "start", "config": {"speed_factor": float("inf")}}]
    )
    load_engine = AsyncMock()

    monkeypatch.setattr(
        simulation_ws_module, "resolve_ws_user", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(simulation_ws_module, "_load_simulation_engine", load_engine)

    await simulation_ws_module.simulation_stream(websocket, "mujoco")

    assert websocket.accepted is True
    assert websocket.closed is True
    assert len(websocket.sent) == 1
    assert "error" in websocket.sent[0]
    load_engine.assert_not_called()


@pytest.mark.anyio
async def test_simulation_stream_rejects_duration_over_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """duration >= 3600s must send a specific error frame and not load an engine."""
    websocket = _RouteWebSocket([{"action": "start", "config": {"duration": 3600.0}}])
    load_engine = AsyncMock()

    monkeypatch.setattr(
        simulation_ws_module, "resolve_ws_user", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(simulation_ws_module, "_load_simulation_engine", load_engine)

    await simulation_ws_module.simulation_stream(websocket, "mujoco")

    assert websocket.accepted is True
    assert websocket.closed is True
    assert len(websocket.sent) == 1
    assert "error" in websocket.sent[0]
    load_engine.assert_not_called()


@pytest.mark.anyio
async def test_simulation_stream_rejects_timestep_over_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """timestep >= 1.0s must send a specific error frame and not load an engine."""
    websocket = _RouteWebSocket(
        [{"action": "start", "config": {"duration": 1.0, "timestep": 1.5}}]
    )
    load_engine = AsyncMock()

    monkeypatch.setattr(
        simulation_ws_module, "resolve_ws_user", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(simulation_ws_module, "_load_simulation_engine", load_engine)

    await simulation_ws_module.simulation_stream(websocket, "mujoco")

    assert websocket.accepted is True
    assert websocket.closed is True
    assert len(websocket.sent) == 1
    assert "error" in websocket.sent[0]
    load_engine.assert_not_called()


@pytest.mark.anyio
async def test_simulation_stream_rejects_timestep_too_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """timestep < 1e-6s must send a specific error frame and not load an engine."""
    websocket = _RouteWebSocket(
        [{"action": "start", "config": {"duration": 1.0, "timestep": 1e-9}}]
    )
    load_engine = AsyncMock()

    monkeypatch.setattr(
        simulation_ws_module, "resolve_ws_user", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(simulation_ws_module, "_load_simulation_engine", load_engine)

    await simulation_ws_module.simulation_stream(websocket, "mujoco")

    assert websocket.accepted is True
    assert websocket.closed is True
    assert len(websocket.sent) == 1
    assert "error" in websocket.sent[0]
    load_engine.assert_not_called()

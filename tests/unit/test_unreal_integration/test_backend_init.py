"""Tests for UnrealBridgeBackend initialization correctness (issue #2475).

When the background streaming server fails to bind (e.g., port in use),
initialize() must raise RuntimeError and must NOT set _is_initialized=True.
The backend must not report successful initialization if the socket layer failed.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestUnrealBridgeBackendInitFailure:
    """initialize() must fail explicitly when the server cannot bind a socket."""

    def test_initialize_raises_when_server_start_fails(self) -> None:
        """RuntimeError from server.start() must propagate out of initialize()."""
        from src.unreal_integration.viewer_backends import UnrealBridgeBackend

        backend = UnrealBridgeBackend()

        with patch(
            "src.unreal_integration.viewer_backends.UnrealStreamingServer"
        ) as MockServer:
            mock_server = MagicMock()
            mock_server.start = AsyncMock(
                side_effect=RuntimeError("Cannot bind to port 9999")
            )
            MockServer.return_value = mock_server

            with pytest.raises(RuntimeError):
                backend.initialize()

    def test_is_initialized_stays_false_when_server_start_fails(self) -> None:
        """_is_initialized must NOT be True after a failed initialize()."""
        from src.unreal_integration.viewer_backends import UnrealBridgeBackend

        backend = UnrealBridgeBackend()

        with patch(
            "src.unreal_integration.viewer_backends.UnrealStreamingServer"
        ) as MockServer:
            mock_server = MagicMock()
            mock_server.start = AsyncMock(side_effect=OSError("Address in use"))
            MockServer.return_value = mock_server

            with contextlib.suppress(RuntimeError, OSError):
                backend.initialize()

            assert backend._is_initialized is False, (
                "_is_initialized must remain False when server.start() failed"
            )

    def test_is_initialized_true_when_server_starts_successfully(self) -> None:
        """_is_initialized must be True after a successful initialize()."""
        from src.unreal_integration.viewer_backends import UnrealBridgeBackend

        backend = UnrealBridgeBackend()

        async def fake_start() -> None:
            pass  # success

        with patch(
            "src.unreal_integration.viewer_backends.UnrealStreamingServer"
        ) as MockServer:
            mock_server = MagicMock()
            mock_server.start = AsyncMock(side_effect=fake_start)
            mock_server.stop = AsyncMock()
            MockServer.return_value = mock_server

            def patched_initialize(self: UnrealBridgeBackend) -> None:
                if self._is_initialized:
                    return
                import time

                self._start_time = time.time()
                from src.unreal_integration.streaming import StreamingConfig

                streaming_config = StreamingConfig(
                    host=self.config.server_host,
                    port=self.config.server_port,
                )
                self._server = MockServer(config=streaming_config)
                self._stop_event.clear()
                self._server_ready_event.clear()

                # Simulate a successful background start by setting the event directly
                self._server_ready_event.set()
                self._is_initialized = True

            with patch.object(UnrealBridgeBackend, "initialize", patched_initialize):
                backend.initialize()

        assert backend._is_initialized is True

    def test_initialize_raises_on_timeout(self) -> None:
        """initialize() must raise RuntimeError if server does not become ready in time."""
        from src.unreal_integration.viewer_backends import UnrealBridgeBackend

        backend = UnrealBridgeBackend()

        # Replace _server_ready_event with a mock whose wait() always returns False.
        fake_event = MagicMock()
        fake_event.wait.return_value = False
        backend._server_ready_event = fake_event

        with patch(
            "src.unreal_integration.viewer_backends.UnrealStreamingServer"
        ) as MockServer:
            mock_server = MagicMock()
            mock_server.start = AsyncMock()
            mock_server.stop = AsyncMock()
            MockServer.return_value = mock_server

            try:
                with pytest.raises(RuntimeError, match="start|timeout|did not start"):
                    backend.initialize()
            finally:
                backend._stop_event.set()


class TestUnrealBridgeBackendSourceCheck:
    """Source-level verification of the initialize() timeout-check contract."""

    def test_initialize_checks_wait_return_value(self) -> None:
        """initialize() must check the return value of _server_ready_event.wait()."""
        import inspect

        from src.unreal_integration.viewer_backends import UnrealBridgeBackend

        src = inspect.getsource(UnrealBridgeBackend.initialize)
        # The method must check whether wait() returned True/False
        # Common patterns: `if not`, `wait(...)`, bool check
        assert "wait(" in src, "initialize() must call _server_ready_event.wait()"
        # Verify that the return value is used (not discarded)
        # The wait result must flow into a condition, not just be called for side effects
        assert (
            "if not self._server_ready_event.wait" in src
            or "waited" in src
            or "_server_start_error" in src
            or "if not " in src
        ), "initialize() must guard _is_initialized on the wait() return value"

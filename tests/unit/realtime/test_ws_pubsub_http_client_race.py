"""Tests for WSPubSub._http_client initialization race (#6980).

Two threads calling publish() simultaneously could both see _http_client is
None and construct two httpx.Client instances, leaking a connection pool.
The fix adds a class-level threading.Lock that guards client initialization.
"""

from __future__ import annotations

import contextlib
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.realtime import ws_pubsub as ws_mod


@pytest.mark.unit
class TestWSPubSubHttpClientRace:
    """Verify that _http_client is initialized at most once under concurrent load."""

    def setup_method(self) -> None:
        """Reset class-level state between tests."""
        ws_mod.WSPubSub._http_client = None

    def teardown_method(self) -> None:
        """Restore class-level state after each test."""
        client = ws_mod.WSPubSub._http_client
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()
        ws_mod.WSPubSub._http_client = None

    def test_concurrent_publishes_create_exactly_one_client(self) -> None:
        """Two threads racing to publish must not create more than one httpx.Client."""
        creation_count = 0

        class _CountingClient:
            """Tracks how many instances were created."""

            def __init__(self, **kwargs: object) -> None:
                nonlocal creation_count
                creation_count += 1
                self._post_response = MagicMock()
                self._post_response.status_code = 200

            def post(self, *args: object, **kwargs: object) -> MagicMock:
                return self._post_response

            def close(self) -> None:
                pass

        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def _publish(pubsub: ws_mod.WSPubSub) -> None:
            barrier.wait()  # synchronize both threads to maximise race window
            try:
                pubsub.publish("test/channel", {"x": 1})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with (
            patch.object(ws_mod.WSPubSub, "_ensure_backend_resolved"),
            patch.object(ws_mod.WSPubSub, "backend", "python", create=True),
        ):
            pubsub = object.__new__(ws_mod.WSPubSub)
            pubsub.host = "127.0.0.1"
            pubsub.port = 9999
            pubsub.backend = "python"
            pubsub._rust_server = None
            pubsub._backend_resolved = True

            # Patch the publish URL to something unused (no real server needed because
            # we mock the client class before the check-then-create path).
            with patch("httpx.Client", side_effect=_CountingClient):
                t1 = threading.Thread(target=_publish, args=(pubsub,))
                t2 = threading.Thread(target=_publish, args=(pubsub,))
                t1.start()
                t2.start()
                t1.join(timeout=5)
                t2.join(timeout=5)

        assert not errors, f"publish raised: {errors}"
        assert creation_count == 1, (
            f"Expected exactly 1 httpx.Client created, got {creation_count}; "
            "check-then-act race is not protected by a lock"
        )

    def test_http_client_lock_attribute_exists(self) -> None:
        """WSPubSub must expose _http_client_lock as a threading.Lock."""
        assert hasattr(ws_mod.WSPubSub, "_http_client_lock"), (
            "WSPubSub must have a class-level _http_client_lock"
        )
        assert isinstance(ws_mod.WSPubSub._http_client_lock, type(threading.Lock())), (
            "_http_client_lock must be a threading.Lock instance"
        )

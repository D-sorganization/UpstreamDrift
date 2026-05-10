"""Unit tests for the realtime facade in ``src.shared.python.realtime``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.shared.python.realtime as facade_module
from src.shared.python.realtime import (
    Subscription,
    publish,
    subscribe,
    validate_channel,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_backends() -> None:
    """Reset the module-level backend singletons between tests."""
    facade_module._file_backend = None
    facade_module._ws_backend = None
    yield
    facade_module._file_backend = None
    facade_module._ws_backend = None


@pytest.fixture()
def patched_file_backend(monkeypatch: pytest.MonkeyPatch):
    backend = MagicMock()
    backend.publish = MagicMock()
    backend.subscribe = MagicMock(
        return_value=Subscription(
            channel="target/active",
            callback=lambda _p: None,
            _unsubscribe=lambda: None,
        )
    )
    monkeypatch.setattr(facade_module, "_get_file_backend", lambda: backend)
    return backend


@pytest.fixture()
def patched_ws_backend(monkeypatch: pytest.MonkeyPatch):
    backend = MagicMock()
    backend.publish = MagicMock()
    backend.subscribe = MagicMock(
        return_value=Subscription(
            channel="pose/canonical",
            callback=lambda _p: None,
            _unsubscribe=lambda: None,
        )
    )
    monkeypatch.setattr(facade_module, "_get_ws_backend", lambda: backend)
    return backend


def test_facade_exports() -> None:
    # Ensure the symbols are re-exported at the package surface.
    assert callable(publish)
    assert callable(subscribe)
    assert callable(validate_channel)
    assert Subscription is not None


def test_publish_auto_routes_high_freq_to_ws(
    patched_file_backend, patched_ws_backend
) -> None:
    publish("pose/canonical", {"v": 1})
    patched_ws_backend.publish.assert_called_once_with("pose/canonical", {"v": 1})
    patched_file_backend.publish.assert_not_called()


def test_publish_auto_routes_low_freq_to_file(
    patched_file_backend, patched_ws_backend
) -> None:
    publish("target/active", {"v": 1})
    patched_file_backend.publish.assert_called_once_with("target/active", {"v": 1})
    patched_ws_backend.publish.assert_not_called()


def test_publish_explicit_transport_overrides_auto(
    patched_file_backend, patched_ws_backend
) -> None:
    publish("pose/canonical", {"v": 1}, transport="file")
    patched_file_backend.publish.assert_called_once()
    patched_ws_backend.publish.assert_not_called()


def test_publish_invalid_transport_raises(
    patched_file_backend, patched_ws_backend
) -> None:
    with pytest.raises(ValueError):
        publish("pose/canonical", {"v": 1}, transport="invalid")  # type: ignore[arg-type]


def test_subscribe_auto_routes_high_freq_to_ws(
    patched_file_backend, patched_ws_backend
) -> None:
    cb = lambda _p: None  # noqa: E731
    sub = subscribe("engine/mujoco/state", cb)
    patched_ws_backend.subscribe.assert_called_once_with("engine/mujoco/state", cb)
    patched_file_backend.subscribe.assert_not_called()
    assert isinstance(sub, Subscription)


def test_subscribe_auto_routes_low_freq_to_file(
    patched_file_backend, patched_ws_backend
) -> None:
    cb = lambda _p: None  # noqa: E731
    sub = subscribe("session/marker", cb)
    patched_file_backend.subscribe.assert_called_once_with("session/marker", cb)
    patched_ws_backend.subscribe.assert_not_called()
    assert isinstance(sub, Subscription)


def test_subscribe_invalid_channel_raises(
    patched_file_backend, patched_ws_backend
) -> None:
    with pytest.raises(ValueError):
        subscribe("BAD/Name", lambda _p: None)


def test_unknown_channel_routes_to_file_by_default(
    patched_file_backend, patched_ws_backend
) -> None:
    publish("custom/unregistered", {"v": 1})
    patched_file_backend.publish.assert_called_once()
    patched_ws_backend.publish.assert_not_called()


def test_real_file_backend_round_trip(tmp_path: Path) -> None:
    """End-to-end: facade with real file backend (no patches)."""
    import os

    os.environ["UPSTREAM_DRIFT_REALTIME_ROOT"] = str(tmp_path)
    try:
        facade_module._file_backend = None
        publish("target/active", {"answer": 42}, transport="file")
        expected = tmp_path / "target__active.json"
        assert expected.exists()
        import json

        assert json.loads(expected.read_text(encoding="utf-8")) == {"answer": 42}
    finally:
        os.environ.pop("UPSTREAM_DRIFT_REALTIME_ROOT", None)
        facade_module._file_backend = None

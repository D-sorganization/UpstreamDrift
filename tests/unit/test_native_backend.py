"""Tests for pendulum_simulator.native_backend (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.pendulum_simulator.native_backend import (
    get_native_backend_info,
    golfer_native_available,
)


class TestNativeBackend:
    def test_get_info_returns_dict(self) -> None:
        info = get_native_backend_info()
        assert isinstance(info, dict)

    def test_golfer_native_available_returns_bool(self) -> None:
        avail = golfer_native_available()
        assert isinstance(avail, bool)

    def test_info_not_empty(self) -> None:
        info = get_native_backend_info()
        assert len(info) > 0

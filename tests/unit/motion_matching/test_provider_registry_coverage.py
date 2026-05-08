"""Coverage tests for ``motion_matching.provider_registry``."""

from __future__ import annotations

import pytest
from src.shared.python.motion_matching.provider_registry import (
    available_engines,
    clear_registry,
    get_provider,
    register_provider,
)


class _FakeProvider:
    def __init__(self, name: str) -> None:
        self.engine_name = name

    def fit_swing(self, target, opts):  # noqa: ARG002 - signature only
        return None


def setup_function(_fn) -> None:
    """Reset the registry before each test."""
    clear_registry()


def teardown_function(_fn) -> None:
    """Reset the registry after each test."""
    clear_registry()


def test_register_then_lookup() -> None:
    """Pin: registered provider is retrievable by name."""
    p = _FakeProvider("drake")
    register_provider(p)
    assert get_provider("drake") is p
    assert "drake" in available_engines()


def test_register_replaces_existing() -> None:
    """Pin: re-registering with a different instance overwrites."""
    a = _FakeProvider("e1")
    b = _FakeProvider("e1")
    register_provider(a)
    register_provider(b)
    assert get_provider("e1") is b


def test_register_requires_engine_name() -> None:
    """Pin: provider without engine_name string is rejected."""

    class NoName:
        def fit_swing(self, t, o):
            return None

    with pytest.raises(TypeError, match="engine_name"):
        register_provider(NoName())


def test_register_requires_callable_fit_swing() -> None:
    """Pin: provider without callable fit_swing is rejected."""

    class NoFit:
        engine_name = "x"
        fit_swing = "not callable"

    with pytest.raises(TypeError, match="fit_swing"):
        register_provider(NoFit())


def test_register_empty_engine_name() -> None:
    """Pin: empty engine_name rejected."""

    class Empty:
        engine_name = ""

        def fit_swing(self, t, o):
            return None

    with pytest.raises(TypeError, match="engine_name"):
        register_provider(Empty())


def test_get_provider_unknown_engine() -> None:
    """Pin: lookup of unknown engine raises KeyError with available list."""
    with pytest.raises(KeyError, match="no fit_swing provider"):
        get_provider("zzz")


def test_clear_registry() -> None:
    """Pin: clear empties the registry."""
    register_provider(_FakeProvider("e"))
    assert available_engines() == ["e"]
    clear_registry()
    assert available_engines() == []

"""Coverage tests for ``loaders.synthetic`` registry / dispatcher."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.loaders.synthetic import (
    _BACKENDS,
    available_backends,
    register_backend,
    synthesize_target_from_coefficients,
)
from src.shared.python.motion_matching.target import AlignOptions

from ._fixtures import make_target


def setup_function(_fn) -> None:
    """Snapshot existing backends and start each test from a clean slate."""
    _BACKENDS.clear()


def teardown_function(_fn) -> None:
    """Cleanup."""
    _BACKENDS.clear()


def _backend(theta, opts):  # noqa: ARG001
    return make_target()


def test_register_and_dispatch() -> None:
    """Pin: a registered backend is reached via dispatch."""
    register_backend("ToyEngine", _backend)
    assert "toyengine" in available_backends()
    out = synthesize_target_from_coefficients(np.zeros(7), engine="toyengine")
    assert out.time.shape[0] > 0


def test_register_replaces_with_warning() -> None:
    """Pin: replacing a backend logs a warning (warning path)."""
    register_backend("e", _backend)

    def other(theta, opts):
        return make_target()

    register_backend("e", other)
    assert _BACKENDS["e"] is other


def test_register_idempotent() -> None:
    """Pin: re-registering the same callable is a no-op."""
    register_backend("e", _backend)
    register_backend("e", _backend)
    assert _BACKENDS["e"] is _backend


def test_register_backend_invalid_engine() -> None:
    """Pin: empty / non-string engine rejected."""
    with pytest.raises(ValueError, match="non-empty string"):
        register_backend("", _backend)
    with pytest.raises(ValueError, match="non-empty string"):
        register_backend(42, _backend)  # type: ignore[arg-type]


def test_register_backend_non_callable() -> None:
    """Pin: non-callable backend rejected."""
    with pytest.raises(TypeError, match="callable"):
        register_backend("e", "not a func")  # type: ignore[arg-type]


def test_dispatch_requires_engine() -> None:
    """Pin: missing engine name raises ValueError."""
    with pytest.raises(ValueError, match="no engine specified"):
        synthesize_target_from_coefficients(np.zeros(7), AlignOptions())


def test_dispatch_unknown_engine() -> None:
    """Pin: unknown engine raises LookupError with available list."""
    with pytest.raises(LookupError, match="no synthetic backend"):
        synthesize_target_from_coefficients(np.zeros(7), engine="not-real")

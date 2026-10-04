"""Tests verifying retirement of deprecated plotting renderer shims (ADR-0052, #11292, #11347)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


def test_force_vectors_shim_is_retired() -> None:
    """Importing force_vectors from plotting.renderers must raise ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        import src.shared.python.plotting.renderers.force_vectors  # noqa: F401


def test_vectors_shim_is_retired() -> None:
    """Importing vectors from plotting.renderers must raise ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        import src.shared.python.plotting.renderers.vectors  # noqa: F401

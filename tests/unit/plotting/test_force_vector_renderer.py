"""Tests verifying retirement of deprecated ForceVectorRenderer (ADR-0052, #11292, #11347)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


def test_force_vector_renderer_is_retired() -> None:
    """ForceVectorRenderer is retired; users must use matplotlib_glyphs."""
    with pytest.raises(ModuleNotFoundError):
        from src.shared.python.plotting.renderers.force_vectors import (  # noqa: F401
            ForceVectorRenderer,
        )

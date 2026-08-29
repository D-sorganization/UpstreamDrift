"""Fixtures for the launcher_embed coverage suite.

Forces the offscreen Qt platform and clears the embeddable-tool
registry between tests so fixture tools do not leak across cases.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _clear_embed_registry() -> Iterator[None]:
    """Give each test an empty registry, then restore the incoming state."""
    from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY

    snapshot = dict(EMBEDDABLE_TOOL_REGISTRY)
    EMBEDDABLE_TOOL_REGISTRY.clear()
    try:
        yield
    finally:
        EMBEDDABLE_TOOL_REGISTRY.clear()
        EMBEDDABLE_TOOL_REGISTRY.update(snapshot)

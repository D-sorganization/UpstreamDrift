"""Shared fixtures and setup for AI adapter unit tests.

Each adapter module (``gemini_adapter``, ``openai_adapter``, ``anthropic_adapter``,
``ollama_adapter``) tries to ``import`` its optional third-party dependency at
module-import time (wrapped in a ``try/except``). When CI runs without those
packages installed, the import would capture ``HAS_*`` as ``False`` unless we
install stubs in ``sys.modules`` before the adapters are first imported.

Rather than asking each test file to perform a module-level
``sys.modules[...] = MagicMock()`` assignment (banned by CLAUDE.md), we centralise
the stub installation in ``pytest_configure`` so it runs once, before any
adapter module is imported. The stubs are removed in ``pytest_unconfigure``
so the test session exits cleanly.

The helper dicts come from ``tests._mocks.physics_stubs`` so individual
per-test fixtures can still use ``monkeypatch.setitem`` for narrower scopes.
"""

from __future__ import annotations

import sys

import pytest

from tests._mocks.physics_stubs import (
    anthropic_stubs,
    google_genai_stubs,
    httpx_stubs,
    openai_stubs,
)

# Track the exact sys.modules keys we install so teardown only removes our
# own entries.
_installed_keys: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    """Install optional-dependency stubs before adapter modules are imported."""
    combined: dict = {}
    combined.update(google_genai_stubs())
    combined.update(openai_stubs())
    combined.update(anthropic_stubs())
    combined.update(httpx_stubs())

    for key, value in combined.items():
        if key not in sys.modules:
            sys.modules[key] = value
            _installed_keys.append(key)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove stubs installed by :func:`pytest_configure`."""
    while _installed_keys:
        key = _installed_keys.pop()
        sys.modules.pop(key, None)

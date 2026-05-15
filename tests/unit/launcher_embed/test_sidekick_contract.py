"""Contract tests for the Sidekick embeddable-tool adapter.

Verifies the adapter satisfies the
:class:`~src.shared.python.launcher_embed.EmbeddableTool` protocol,
exposes the capabilities documented for the chat panel, and registers
itself with the embeddable-tool registry once the launcher bootstrap
imports it. See issue #5460.

The adapter and the launcher tile id are both ``sidekick``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from src.shared.python.launcher_embed import (
    EMBEDDABLE_TOOL_REGISTRY,
    EmbedCapabilities,
    EmbeddableTool,
)
from src.tools.sidekick._embed_adapter import _SidekickEmbedAdapter

pytestmark = [pytest.mark.unit]


def _drop_sidekick_from_modules() -> None:
    """Evict the sidekick package from ``sys.modules``.

    Required because the adapter's self-registration runs as a
    side-effect of importing :mod:`src.tools.sidekick.__init__`.
    Once imported (e.g. by another test or at module collection time)
    Python caches the module; a subsequent ``__import__`` from the
    bootstrap will not re-execute the registration. Tests that assert
    on the registration side-effect therefore need a clean slate.
    """
    for name in [
        "src.tools.sidekick",
        "src.tools.sidekick._embed_adapter",
    ]:
        sys.modules.pop(name, None)


@pytest.fixture
def _registry_snapshot() -> Iterator[None]:
    """Snapshot the registry around tests that mutate it via bootstrap."""
    snapshot = dict(EMBEDDABLE_TOOL_REGISTRY)
    try:
        yield
    finally:
        EMBEDDABLE_TOOL_REGISTRY.clear()
        EMBEDDABLE_TOOL_REGISTRY.update(snapshot)


# --- Protocol conformance -----------------------------------------------


def test_adapter_satisfies_embeddable_tool_protocol() -> None:
    adapter = _SidekickEmbedAdapter()
    assert isinstance(adapter, EmbeddableTool)


def test_adapter_tool_id_is_sidekick() -> None:
    assert _SidekickEmbedAdapter().tool_id == "sidekick"


# --- Capabilities values -------------------------------------------------


def test_embed_capabilities_match_spec() -> None:
    caps = _SidekickEmbedAdapter().embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    # The chat panel is dock-shaped (vertical messages + compose row).
    assert caps.prefers_dock is True
    assert caps.min_size == (360, 480)
    assert caps.requires_separate_qapplication is False


def test_is_dirty_default_is_false() -> None:
    # Chat history is auto-persisted; nothing for the host to prompt
    # the user about on close.
    assert _SidekickEmbedAdapter().is_dirty() is False


def test_cleanup_is_idempotent() -> None:
    adapter = _SidekickEmbedAdapter()
    # Cleanup is allowed before any widget has been handed out, and
    # repeated calls are a quiet no-op.
    adapter.cleanup()
    adapter.cleanup()


def test_cleanup_releases_widgets_and_calls_delete_later() -> None:
    """cleanup() forwards to deleteLater on each handed-out widget."""
    adapter = _SidekickEmbedAdapter()

    class _FakeWidget:
        def __init__(self) -> None:
            self.delete_calls = 0

        def deleteLater(self) -> None:
            self.delete_calls += 1

    fake_a = _FakeWidget()
    fake_b = _FakeWidget()
    adapter._widgets.extend([fake_a, fake_b])

    adapter.cleanup()

    assert fake_a.delete_calls == 1
    assert fake_b.delete_calls == 1
    # Subsequent calls are no-ops because the widget list was drained.
    adapter.cleanup()
    assert fake_a.delete_calls == 1
    assert fake_b.delete_calls == 1


# --- Bootstrap side-effect ----------------------------------------------


def test_bootstrap_registers_sidekick(_registry_snapshot) -> None:  # noqa: ANN001
    """Running the launcher bootstrap registers the sidekick adapter.

    The bootstrap module is listed in
    :data:`src.launchers.embedded_tool_bootstrap.bootstrap_embeddable_tools`'s
    ``adapter_modules`` list; importing it triggers the adapter's
    self-registration. We assert against the registry directly (rather
    than the ``bootstrap_embeddable_tools()`` return value) because
    other adapters in the list may legitimately fail to import in
    headless / minimal-dependency environments where their heavyweight
    optional dependencies (NumPy, PyQt6, etc.) are absent. Sidekick
    itself only imports the contract module at adapter-module import
    time — PyQt6 is deferred to :meth:`create_main_widget`.
    """
    from src.launchers import embedded_tool_bootstrap

    # Force a clean bootstrap so this test does not depend on whether
    # another test in the session already triggered it.
    embedded_tool_bootstrap.reset_bootstrap_state()
    EMBEDDABLE_TOOL_REGISTRY.clear()
    _drop_sidekick_from_modules()

    embedded_tool_bootstrap.bootstrap_embeddable_tools()

    assert "sidekick" in EMBEDDABLE_TOOL_REGISTRY
    tool = EMBEDDABLE_TOOL_REGISTRY["sidekick"]
    assert tool.tool_id == "sidekick"
    assert tool.embed_capabilities().supports_embedded is True


def test_package_import_registers_sidekick(_registry_snapshot) -> None:  # noqa: ANN001
    """Importing :mod:`src.tools.sidekick` registers the adapter directly."""
    EMBEDDABLE_TOOL_REGISTRY.pop("sidekick", None)
    _drop_sidekick_from_modules()

    import src.tools.sidekick  # noqa: F401

    assert "sidekick" in EMBEDDABLE_TOOL_REGISTRY
    tool = EMBEDDABLE_TOOL_REGISTRY["sidekick"]
    assert tool.tool_id == "sidekick"

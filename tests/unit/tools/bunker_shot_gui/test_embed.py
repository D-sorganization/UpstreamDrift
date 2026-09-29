"""The single embed adapter for the BunkerShot3D workbench (issue #8618).

``bunker_shot_gui`` used to define **two** adapters for one ``tool_id`` -- one
in ``_embed_adapter`` and one inside ``gui`` -- and the registry rejects a
duplicate id, so whichever registered second was silently dropped and the
launcher's view of the tool depended on import order. There is now exactly
one, and these tests pin that.

Importing the adapter must not import Qt; only ``create_main_widget`` may,
which is why the widget-building test is the one guarded by ``importorskip``.
"""

from __future__ import annotations

import pytest

from src.shared.python.launcher_embed import (
    EMBEDDABLE_TOOL_REGISTRY,
    EmbedCapabilities,
    EmbeddableTool,
)
from src.tools.bunker_shot_gui import BunkerShotGuiAdapter

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture()
def adapter() -> BunkerShotGuiAdapter:
    """Return a fresh adapter."""
    return BunkerShotGuiAdapter()


def test_tool_id_is_stable(adapter: BunkerShotGuiAdapter) -> None:
    assert BunkerShotGuiAdapter.tool_id == "bunker_shot_gui"
    assert adapter.tool_id == "bunker_shot_gui"


def test_adapter_satisfies_the_embeddable_protocol(
    adapter: BunkerShotGuiAdapter,
) -> None:
    assert isinstance(adapter, EmbeddableTool)


def test_importing_the_package_registers_exactly_one_adapter() -> None:
    registered = EMBEDDABLE_TOOL_REGISTRY.get("bunker_shot_gui")
    assert isinstance(registered, BunkerShotGuiAdapter)


def test_workbench_declares_itself_embeddable(adapter: BunkerShotGuiAdapter) -> None:
    capabilities = adapter.embed_capabilities()
    assert isinstance(capabilities, EmbedCapabilities)
    assert capabilities.supports_embedded is True
    assert capabilities.requires_separate_qapplication is False


def test_minimum_size_fits_two_design_columns(
    adapter: BunkerShotGuiAdapter,
) -> None:
    width, height = adapter.embed_capabilities().min_size
    assert width >= 1000
    assert height >= 700


def test_capabilities_are_stable_across_calls(
    adapter: BunkerShotGuiAdapter,
) -> None:
    assert adapter.embed_capabilities() == adapter.embed_capabilities()


def test_cleanup_before_a_widget_exists_is_safe(
    adapter: BunkerShotGuiAdapter,
) -> None:
    adapter.cleanup()
    assert adapter._widget is None


def test_cleanup_is_idempotent(adapter: BunkerShotGuiAdapter) -> None:
    adapter.cleanup()
    adapter.cleanup()
    assert adapter._widget is None


def test_the_workbench_holds_no_unsaved_state(
    adapter: BunkerShotGuiAdapter,
) -> None:
    assert adapter.is_dirty() is False


def test_the_adapter_defers_its_gui_import() -> None:
    """Only ``create_main_widget`` may reach for Qt.

    The widget-building half of the contract is exercised in
    ``tests/tools/bunker_shot_gui``, which owns the session QApplication.
    This directory stays Qt-free on purpose.
    """
    import inspect

    from src.tools.bunker_shot_gui import _embed_adapter

    module_source = inspect.getsource(_embed_adapter)
    assert "import PyQt" not in module_source
    assert "from PyQt" not in module_source
    assert "from .gui import" in inspect.getsource(
        BunkerShotGuiAdapter.create_main_widget
    )

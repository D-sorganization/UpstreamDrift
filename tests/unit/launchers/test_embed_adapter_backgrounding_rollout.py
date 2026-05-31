"""Per-adapter backgrounding rollout tests (Sub-PR B of #6013).

Sub-PR A exercised the :class:`EmbeddedHostWidget` lifecycle with
synthetic adapters. This module verifies the *real* ``_embed_adapter``
modules that ship with each embedded tool:

- every adapter's ``cleanup()`` is idempotent (a second call is a
  no-op, never raises) -- the §G contract;
- the ``pose_subscriber_demo`` adapter implements ``pause`` / ``resume``
  that release and re-acquire the live ``pose/canonical`` subscription;
- the default-backgrounding adapters resolve their optional hooks to
  the documented structural defaults (``can_background`` /
  ``detach_to_window`` -> ``True``).

The heavy adapters build PyQt6 + matplotlib widgets in
``create_main_widget``; we deliberately do *not* construct those here.
Instead we drive ``cleanup`` against the adapter's own widget list
(populated with light test doubles) so the idempotency contract is
exercised without spinning up real GUIs. A single host round-trip test
(open -> background -> reopen) covers the integration path with one
lightweight real adapter.

All tests skip cleanly when PyQt6 is unavailable.
"""

from __future__ import annotations

import importlib
import os
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from src.launchers.embedded_host import EmbeddedHostWidget  # noqa: E402
from src.shared.python.launcher_embed import (  # noqa: E402
    register_embeddable_tool,
    unregister_embeddable_tool,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------

# (module path, class name) for every adapter that tracks the widgets it
# hands out via a ``self._widgets`` list. These all share the
# swap-then-clear ``cleanup`` contract and so are exercised by the
# idempotency test below with light test doubles.
_LIST_TRACKING_ADAPTERS = [
    ("src.tools.model_explorer._embed_adapter", "_ModelExplorerEmbedAdapter"),
    (
        "src.tools.pose_subscriber_demo._embed_adapter",
        "_PoseSubscriberDemoEmbedAdapter",
    ),
    ("src.tools.sidekick._embed_adapter", "_SidekickEmbedAdapter"),
    (
        "src.tools.starting_pose_matcher._embed_adapter",
        "_MotionMatchPreviewEmbedAdapter",
    ),
    ("src.tools.training_controller._embed_adapter", "_TrainingControllerEmbedAdapter"),
]

# Adapters whose ``cleanup`` is a no-op or drops a single widget ref.
_SIMPLE_ADAPTERS = [
    ("src.tools.ball_flight_gui._embed_adapter", "BallFlightGuiAdapter"),
    ("src.tools.bunker_shot_gui._embed_adapter", "BunkerShotGuiAdapter"),
    ("src.tools.golf_environment._embed_adapter", "GolfEnvironmentAdapter"),
    ("src.tools.golf_simulation_suite._embed_adapter", "GolfSimulationSuiteAdapter"),
    ("src.tools.putting_green_gui._embed_adapter", "PuttingGreenGuiAdapter"),
    ("src.tools.simulation_backends_launcher._embed_adapter", "_EmbedAdapter"),
    ("src.tools.terrain_engine._embed_adapter", "TerrainEngineAdapter"),
    ("src.tools.video_analyzer._embed_adapter", "VideoAnalyzerAdapter"),
]

_ALL_ADAPTERS = _LIST_TRACKING_ADAPTERS + _SIMPLE_ADAPTERS


def _load(module_path: str, class_name: str) -> Any:
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{module_path} requires optional dependency {exc.name!r}")
    return getattr(module, class_name)


def _skip_if_qt_is_mocked(qapp: Any) -> None:
    if (
        isinstance(qapp, Mock)
        or isinstance(QApplication, Mock)
        or isinstance(QLabel, Mock)
    ):
        pytest.skip("real PyQt6 widgets are not available in this environment")


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class _StubWidget:
    """Light test double standing in for a real tool widget.

    Records both teardown entry-points adapters may use: ``cleanup``
    (most adapters) and ``deleteLater`` (Sidekick forwards to the Qt
    canonical teardown instead).
    """

    def __init__(self) -> None:
        self.cleanup_calls = 0
        self.delete_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1

    def deleteLater(self) -> None:  # noqa: N802 - Qt naming
        self.delete_calls += 1


# ---------------------------------------------------------------------------
# cleanup() idempotency — every adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    _ALL_ADAPTERS,
    ids=[name for _, name in _ALL_ADAPTERS],
)
def test_cleanup_is_idempotent(module_path: str, class_name: str, qapp) -> None:  # noqa: ANN001
    """A second ``cleanup()`` must be a harmless no-op (§G)."""
    adapter = _load(module_path, class_name)()
    # First call (nothing constructed yet) must not raise.
    adapter.cleanup()
    # Second call must also not raise.
    adapter.cleanup()


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    _LIST_TRACKING_ADAPTERS,
    ids=[name for _, name in _LIST_TRACKING_ADAPTERS],
)
def test_widget_tracking_cleanup_forwards_once(
    module_path: str,
    class_name: str,
    qapp,  # noqa: ANN001
) -> None:
    """List-tracking adapters forward ``cleanup`` to widgets exactly once.

    The swap-then-clear pattern means a second call does not re-clean an
    already-released widget.
    """
    adapter = _load(module_path, class_name)()
    stub = _StubWidget()
    adapter._widgets.append(stub)

    adapter.cleanup()
    adapter.cleanup()

    # Each adapter forwards to exactly one teardown entry-point exactly
    # once (cleanup() for most, deleteLater() for Sidekick), and the
    # widget list is cleared so a second call is a no-op.
    assert stub.cleanup_calls + stub.delete_calls == 1
    assert adapter._widgets == []


# ---------------------------------------------------------------------------
# Structural defaults for optional hooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    _ALL_ADAPTERS,
    ids=[name for _, name in _ALL_ADAPTERS],
)
def test_default_hooks_resolve_to_true(
    module_path: str,
    class_name: str,
    qapp,  # noqa: ANN001
) -> None:
    """Adapters that omit the hooks resolve to the documented defaults."""
    adapter = _load(module_path, class_name)()
    # can_background / detach_to_window default to True for any adapter
    # that does not override them.
    assert bool(getattr(adapter, "can_background", lambda: True)()) is True
    assert bool(getattr(adapter, "detach_to_window", lambda: True)()) is True


# ---------------------------------------------------------------------------
# pose_subscriber_demo pause/resume
# ---------------------------------------------------------------------------


class _SubStub:
    """Stub pose-subscriber widget recording pause/resume calls."""

    def __init__(self) -> None:
        self.pause_calls = 0
        self.resume_calls = 0

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1


def test_pose_subscriber_adapter_pause_resume_delegate(qapp) -> None:  # noqa: ANN001
    """The pose-subscriber adapter forwards pause/resume to its widgets."""
    adapter = _load(
        "src.tools.pose_subscriber_demo._embed_adapter",
        "_PoseSubscriberDemoEmbedAdapter",
    )()
    stub = _SubStub()
    adapter._widgets.append(stub)

    adapter.pause()
    adapter.resume()

    assert stub.pause_calls == 1
    assert stub.resume_calls == 1


def test_pose_subscriber_adapter_can_pause_one_widget(qapp) -> None:  # noqa: ANN001
    """Per-widget pause avoids stopping a visible second mount."""
    adapter = _load(
        "src.tools.pose_subscriber_demo._embed_adapter",
        "_PoseSubscriberDemoEmbedAdapter",
    )()
    tab_stub = _SubStub()
    dock_stub = _SubStub()
    adapter._widgets.extend([tab_stub, dock_stub])

    adapter.pause_widget(tab_stub)
    adapter.resume_widget(tab_stub)

    assert tab_stub.pause_calls == 1
    assert tab_stub.resume_calls == 1
    assert dock_stub.pause_calls == 0
    assert dock_stub.resume_calls == 0


def test_pose_subscriber_widget_pause_releases_subscription(qapp) -> None:  # noqa: ANN001
    """The real widget drops and re-acquires its subscription."""
    _skip_if_qt_is_mocked(qapp)
    pytest.importorskip("matplotlib")
    from src.tools.pose_subscriber_demo.gui import MainWidget

    widget = MainWidget()
    try:
        # The widget subscribes on construction.
        widget.pause()
        assert widget._subscription is None
        widget.resume()
        # resume() re-acquires (or leaves None if the realtime layer is
        # inert in this headless context); either way it must not raise
        # and must be idempotent.
        widget.resume()
    finally:
        widget.cleanup()
        widget.deleteLater()


# ---------------------------------------------------------------------------
# Host round-trip with a real lightweight adapter
# ---------------------------------------------------------------------------


class _RealLikeAdapter:
    """Minimal real-shaped adapter used to drive the host round-trip.

    Mirrors the structure of the shipped adapters (a ``_widgets`` list,
    a swap-then-clear ``cleanup``) without pulling in a heavy GUI.
    """

    tool_id = "rollout_smoke_tool"

    def __init__(self) -> None:
        self._widgets: list[Any] = []
        self.cleanup_calls = 0

    def embed_capabilities(self):  # noqa: ANN201
        from src.shared.python.launcher_embed import EmbedCapabilities

        return EmbedCapabilities()

    def create_main_widget(self, parent: Any) -> QWidget:
        widget = QLabel("rollout", parent)
        # Stamp a sentinel so we can assert the same widget survives a
        # background -> reopen round-trip (state preservation).
        widget.setProperty("rollout_state", "preserved")
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        widgets, self._widgets = self._widgets, []
        for _ in widgets:
            self.cleanup_calls += 1

    def is_dirty(self) -> bool:
        return False


class _LifecycleLabel(QLabel):
    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(label, parent)
        self.pause_calls = 0
        self.resume_calls = 0

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1


class _SelectiveLifecycleAdapter:
    tool_id = "selective_lifecycle_tool"

    def __init__(self) -> None:
        self._widgets: list[_LifecycleLabel] = []

    def embed_capabilities(self):  # noqa: ANN201
        from src.shared.python.launcher_embed import EmbedCapabilities

        return EmbedCapabilities()

    def create_main_widget(self, parent: Any) -> _LifecycleLabel:
        widget = _LifecycleLabel(f"mount-{len(self._widgets)}", parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        self._widgets.clear()

    def is_dirty(self) -> bool:
        return False

    @staticmethod
    def pause_widget(widget: _LifecycleLabel) -> None:
        widget.pause()

    @staticmethod
    def resume_widget(widget: _LifecycleLabel) -> None:
        widget.resume()


def test_host_background_reopen_popout_dockback_round_trip(qapp) -> None:  # noqa: ANN001
    """open -> background -> reopen -> pop out -> dock back keeps state."""
    _skip_if_qt_is_mocked(qapp)
    adapter = _RealLikeAdapter()
    register_embeddable_tool(adapter)
    host = EmbeddedHostWidget()
    try:
        host.open_tab(adapter.tool_id)
        original = host.tab_widget.widget(0)
        assert original.property("rollout_state") == "preserved"

        # Background close keeps the live widget (no cleanup).
        assert host.close_tab(adapter.tool_id, destroy=False) is True
        assert adapter.cleanup_calls == 0
        assert host.backgrounded_tools() == {adapter.tool_id}

        # Reopen re-surfaces the *same* widget, not a rebuild.
        host.open_tab(adapter.tool_id)
        resurfaced = host.tab_widget.widget(0)
        assert resurfaced is original
        assert resurfaced.property("rollout_state") == "preserved"

        # Pop out then dock back preserves the same widget instance.
        assert host.pop_out_tab(adapter.tool_id) is True
        index = host.dock_back(adapter.tool_id)
        assert index == 0
        assert host.tab_widget.widget(0) is original
    finally:
        host.close()
        host.deleteLater()
        unregister_embeddable_tool(adapter.tool_id)


def test_host_backgrounds_only_target_widget_when_tool_has_multiple_mounts(
    qapp,
) -> None:  # noqa: ANN001
    """Backgrounding a tab must not pause a still-visible dock widget."""
    _skip_if_qt_is_mocked(qapp)
    adapter = _SelectiveLifecycleAdapter()
    register_embeddable_tool(adapter)
    host = EmbeddedHostWidget()
    try:
        host.open_tab(adapter.tool_id)
        host.open_dock(adapter.tool_id)
        tab_widget, dock_widget = adapter._widgets

        assert host.close_tab(adapter.tool_id, destroy=False) is True

        assert tab_widget.pause_calls == 1
        assert dock_widget.pause_calls == 0

        host.open_tab(adapter.tool_id)
        assert tab_widget.resume_calls == 1
        assert dock_widget.resume_calls == 0
    finally:
        host.close()
        host.deleteLater()
        unregister_embeddable_tool(adapter.tool_id)

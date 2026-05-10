# Embedding a Tool in the Launcher

For tool authors. This page explains how to make a `src/tools/<your_tool>/`
package mountable as a tab or dock in the launcher's
[embedded view](../user_guide/launcher/embedded_view.md), using the
`EmbeddableTool` Protocol.

> **Background.** Design rationale lives in
> [ADR-0013](../adr/0013-launcher-composability.md). For the
> end-user experience you're enabling, see
> [`docs/user_guide/launcher/embedded_view.md`](../user_guide/launcher/embedded_view.md).

---

## A. The contract surface

The full contract lives in
[`src/shared/python/launcher_embed/contract.py`](../../src/shared/python/launcher_embed/contract.py).
Two pieces:

### `EmbedCapabilities` — declare how you want to be embedded

```python
from src.shared.python.launcher_embed import EmbedCapabilities

caps = EmbedCapabilities(
    supports_embedded=True,         # default; False for legacy tools
    prefers_dock=False,              # True for sidebars / status panels
    min_size=(640, 480),             # rule of thumb: your design size
    requires_separate_qapplication=False,  # True only for GL / pygame
)
```

Frozen dataclass, validated in `__post_init__`. `min_size` must be a
2-tuple of strictly positive ints; passing `(True, True)` (which
Python would otherwise treat as `(1, 1)` because `bool` is a subclass
of `int`) is explicitly rejected.

### `EmbeddableTool` — the runtime-checkable Protocol

```python
class EmbeddableTool(Protocol):
    tool_id: str
    def embed_capabilities(self) -> EmbedCapabilities: ...
    def create_main_widget(self, parent: Any) -> Any: ...
    def cleanup(self) -> None: ...
    def is_dirty(self) -> bool: ...
```

The Protocol is `@runtime_checkable`, so you can `isinstance(x,
EmbeddableTool)` to validate at registration time. Note that
`parent` and the widget return type are spelled `typing.Any` —
that's deliberate, so the contract module doesn't need to import
PyQt6 (and headless CI / docs builders can introspect it without
the GUI extras installed).

Once you've implemented the Protocol, register your tool:

```python
from src.shared.python.launcher_embed import register_embeddable_tool

register_embeddable_tool(MyEmbeddableTool())
```

The launcher discovers registered tools via the registry; the
right-click menu's **Launch in Tab** / **Launch in Dock** items
become live for any model whose `tool_id` matches a registered
embeddable.

---

## B. The standard refactor — `QMainWindow` → `MainWidget` factory

Most existing tools are written as `QMainWindow` subclasses with
their UI built in `__init__` and their resources owned by the
window itself. To embed cleanly you split that into:

1. A `MainWidget` (a plain `QWidget` subclass) that holds **all**
   the UI and runtime state.
2. A thin `QMainWindow` wrapper for standalone launches that just
   sets `MainWidget` as its central widget.
3. A `_embed_adapter.py` module that implements `EmbeddableTool`
   and registers itself.

Pose Studio is the canonical worked example. The adapter looks like:

```python
# src/tools/pose_studio/_embed_adapter.py
"""EmbeddableTool adapter for Pose Studio.

Wraps PoseStudioMainWidget so the launcher can mount it as a tab or
dock without owning a QMainWindow.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    register_embeddable_tool,
)


class PoseStudioEmbedAdapter:
    """EmbeddableTool implementation for Pose Studio."""

    tool_id = "pose_studio"

    def __init__(self) -> None:
        self._widget: Any = None  # PoseStudioMainWidget | None

    def embed_capabilities(self) -> EmbedCapabilities:
        # Pose Studio is a primary workspace tool — wants a tab, not a dock.
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(960, 720),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import keeps the contract module Qt-free.
        from src.tools.pose_studio.gui import PoseStudioMainWidget

        if self._widget is None:
            self._widget = PoseStudioMainWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        widget = self._widget
        self._widget = None
        if widget is None:
            return
        # Stop timers, close engine sessions, save dirty state, etc.
        if hasattr(widget, "shutdown"):
            widget.shutdown()
        widget.deleteLater()

    def is_dirty(self) -> bool:
        widget = self._widget
        if widget is None or not hasattr(widget, "is_dirty"):
            return False
        return bool(widget.is_dirty())


register_embeddable_tool(PoseStudioEmbedAdapter())
```

Key shape notes:

- The adapter is **separate from the widget**. It owns the registry
  identity (`tool_id`) and the lifecycle hooks; the widget owns the
  UI. This split means the widget can stay PyQt-heavy while the
  adapter stays import-cheap.
- `create_main_widget` is **idempotent**: opening the same tool in a
  tab twice surfaces the existing tab rather than building a second
  widget. The host enforces this via its registry, but your adapter
  should still cache so the host's `restore_state()` path works
  cleanly.
- `cleanup()` is **idempotent**. The host calls it on close, on
  parent shutdown, and during `closeEvent`. Drop the widget
  reference first, then tear down resources.
- The PyQt6 import is **lazy** — inside `create_main_widget`, not
  at module top-level. This lets the launcher introspect adapters
  on a headless system without `DLL load failed` blowups.

For the legacy `QMainWindow` standalone path, leave your existing
`__main__.py` alone — it can continue to construct a `QMainWindow`
that embeds the same `MainWidget`. Both code paths share the widget
class; only the chrome differs.

---

## C. Capability declaration cheat-sheet

| Flag                             | Set to `True` when…                                                                                                                                               | Set to `False` when…                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `supports_embedded`              | Your tool can run as a child widget (no top-level window required).                                                                                               | The tool needs camera / audio capture, uses pygame, or otherwise hard-codes its window setup. |
| `prefers_dock`                   | The tool is a small status panel, parameter sidebar, or live-preview widget that pairs with a primary workspace.                                                  | The tool is a primary workspace (Pose Studio, Cross-Engine Dashboard).                        |
| `requires_separate_qapplication` | The tool manages its own GL context that conflicts with the launcher's `QApplication` (rare — almost always a sign the tool should be `supports_embedded=False`). | Default. The vast majority of tools share the launcher's `QApplication`.                      |

Notes:

- **Camera / audio capture.** If your tool opens a webcam or
  microphone via Qt's multimedia APIs, set `supports_embedded=False`.
  Multiple `QMediaCaptureSession` instances in the same process
  fight over device handles on Windows; running as a separate
  subprocess sidesteps the issue entirely.
- **pygame.** If your tool uses pygame for rendering or input,
  `supports_embedded=False`. pygame creates its own window and
  event loop and does not coexist with a Qt event loop.
- **Tools that need their own `QApplication`.** Almost always
  legacy. If you're writing a new tool and you think you need this,
  you almost certainly don't — share the launcher's `QApplication`
  and use `QOpenGLWidget` for any GL surface.
- **`min_size` rule of thumb.** Set to your tool's "smallest layout
  that still works." For Pose Studio that's `(960, 720)` because the
  joint tree, 3D viewer, and inspector each need ~300 px wide.
  For a status panel, `(280, 180)` is plenty. The host respects this
  as a minimum when sizing the tab pane or floating dock.

---

## D. Cleanup contract — what to release

Your `cleanup()` runs when:

- The user closes the tab or dock (and confirmed any dirty-state
  prompt).
- The launcher window's `closeEvent` fires.
- A test fixture tears down the host.

Release these things, in this order:

1. **Stop timers.** `QTimer` instances keep running after the widget
   hides. Call `timer.stop()`. Forgotten timers fire on a destroyed
   widget and crash on Linux.
2. **Disconnect signals you connected to objects with longer
   lifetimes.** If you `register_callback`-ed onto a singleton or
   a `realtime` channel, call its unregister/unsubscribe partner.
3. **Save dirty state.** If `is_dirty()` would have returned `True`
   and the user clicked through the prompt, you should still write
   any auto-save artifacts (crash recovery files, etc.) before
   destruction.
4. **Release file handles, network sockets, engine sessions.** Call
   `engine.close()` on any physics-engine instance you opened. Drop
   open file handles. Disconnect from FastAPI WebSocket channels.
5. **Drop the widget reference.** Set `self._widget = None` and call
   `widget.deleteLater()` to let Qt schedule destruction.

`cleanup()` must be **idempotent**: calling it twice is safe.
Defensive guards (`if widget is None: return`) at the top are the
norm, not the exception.

---

## E. Testing template

Every embed adapter should ship with a minimal smoke test that
exercises capability declaration and the create/cleanup loop. The
template:

```python
# tests/unit/tools/pose_studio/test_embed_adapter.py
"""Smoke test for the Pose Studio EmbeddableTool adapter."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

# pytest-qt's qtbot fixture is the canonical way to drive widget
# lifecycle in tests. ``qtbot`` requires PyQt6 — skip if the
# environment doesn't have the GUI extras.
PyQt6 = pytest.importorskip("PyQt6")


def test_adapter_capabilities_are_valid() -> None:
    from src.shared.python.launcher_embed import EmbedCapabilities
    from src.tools.pose_studio._embed_adapter import PoseStudioEmbedAdapter

    adapter = PoseStudioEmbedAdapter()
    caps = adapter.embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.min_size == (960, 720)


def test_adapter_create_and_cleanup_round_trip(qtbot) -> None:  # noqa: ANN001
    from src.tools.pose_studio._embed_adapter import PoseStudioEmbedAdapter

    adapter = PoseStudioEmbedAdapter()

    widget = adapter.create_main_widget(parent=None)
    qtbot.addWidget(widget)
    assert widget is not None

    # Idempotent: a second call returns the same widget instance.
    assert adapter.create_main_widget(parent=None) is widget

    # cleanup() must be safe to call twice.
    adapter.cleanup()
    adapter.cleanup()


def test_adapter_implements_protocol() -> None:
    from src.shared.python.launcher_embed import EmbeddableTool
    from src.tools.pose_studio._embed_adapter import PoseStudioEmbedAdapter

    assert isinstance(PoseStudioEmbedAdapter(), EmbeddableTool)


def test_adapter_is_dirty_default() -> None:
    from src.tools.pose_studio._embed_adapter import PoseStudioEmbedAdapter

    adapter = PoseStudioEmbedAdapter()
    # Before create_main_widget, no widget exists, so dirty must be False.
    assert adapter.is_dirty() is False
```

Mark the test `headless_safe` if your `MainWidget` builds without a
display server (matplotlib QtAgg works under `QT_QPA_PLATFORM=offscreen`;
GL widgets sometimes don't). Mark it `requires_gl` otherwise — CI
will skip on non-GL runners.

---

## F. Wiring it into the launcher

Once the adapter is registered, two more lines wire your tool into
the launcher UI:

1. **Tile entry** — `src/config/models.yaml`:

   ```yaml
   - id: "pose_studio"
     name: "Pose Studio"
     description: "Interactive cross-engine pose editor"
     type: "special_app"
     path: "src/tools/pose_studio/__main__.py"
     launcher:
       category: "tool"
       logo: "assets/pose_studio.png"
       status: "ready"
       default_launch: "tab" # auto-resolves to TAB on right-click → Launch
   ```

   `default_launch` accepts `window`, `tab`, `dock`, or `external`.
   The dispatcher rules are documented in
   `resolve_launch_mode()` and the
   [embedded view user guide](../user_guide/launcher/embedded_view.md#b-right-click-menu--launch-in-new-window--tab--dock).

2. **Adapter import** — somewhere in your tool's `__init__.py`:

   ```python
   from . import _embed_adapter  # noqa: F401  -- registers on import
   ```

   The launcher discovers adapters at startup by importing tool
   packages; the registration side-effect in `_embed_adapter` does
   the rest.

---

## See also

- [ADR-0013](../adr/0013-launcher-composability.md) — design
  rationale.
- [`realtime_ipc.md`](realtime_ipc.md) — pub-sub IPC for tools that
  need to publish or subscribe to live state across the embedded
  view.
- [`docs/user_guide/launcher/embedded_view.md`](../user_guide/launcher/embedded_view.md)
  — the user-facing experience your adapter enables.
- The Pose Studio package
  ([`src/tools/pose_studio/`](../../src/tools/pose_studio/)) —
  canonical worked example.

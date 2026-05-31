# Embedded View — Tabs, Docks, and the In-Launcher Tool Host

Until recently, every tile in the UpstreamDriftLauncher opened its tool in a
separate top-level window. With the launcher composability work
(EPIC [#4993](https://github.com/D-sorganization/UpstreamDrift/issues/4993))
you can now host tools as tabs and dock panels **inside** the
launcher itself — keep Pose Studio open in a tab, dock a canonical-pose
subscriber to the side, and drive both with one keyboard.

> **Background.** Design rationale lives in
> [ADR-0013](../../adr/0013-launcher-composability.md). For
> tool-author docs see
> [`embedding_a_tool.md`](../../development/embedding_a_tool.md) and
> [`realtime_ipc.md`](../../development/realtime_ipc.md).

> **Screenshots.** Image references in this doc point at
> `docs/assets/launcher/`. Screenshots will be added in a follow-up
> capture PR — the placeholder paths below are the contract for that
> follow-up to satisfy.

---

## A. What the embedded view is

The embedded view is a region of the launcher window that hosts
running tools as **tabs** (the central area) and **dock widgets**
(left, right, top, bottom). It is implemented by
`EmbeddedHostWidget` in
[`src/launchers/embedded_host.py`](../../../src/launchers/embedded_host.py).

You should use it when:

- You want to compare two tools side-by-side without alt-tabbing.
- A tool publishes data another tool consumes (e.g. Pose Studio →
  `pose_subscriber_demo`) and you want to watch both update live.
- A small tool (status panel, preview, parameter editor) belongs as
  a sidebar to a larger workspace tool.

Canonical-core entries follow the same path. `canonical_core_estimation`
and `canonical_core_comparison` are registered as embeddable PyQt6 tools and
also expose React routes in the shared launcher manifest, so both desktop and
React shells show the same Biomechanics workspace entries.

You should still launch in a **new window** when:

- You want the tool fullscreen.
- The tool needs its own `QApplication` (rare — see
  [`embedding_a_tool.md`](../../development/embedding_a_tool.md)
  for which tools).
- You're running a CLI / batch / external program (those route
  through `LaunchMode.EXTERNAL`).

![Launcher with embedded view enabled](../../assets/launcher/embedded_view_overview.png)

---

## B. Right-click menu — Launch in New Window / Tab / Dock

Right-clicking any tile in the launcher grid opens a context menu
with up to four launch options. Which ones are enabled depends on
the tool's declared capabilities:

| Menu item                | When enabled                                             | What it does                                                             |
| ------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Launch (default)**     | Always.                                                  | Uses the tool's declared `default_launch` from `src/config/models.yaml`. |
| **Launch in New Window** | Always.                                                  | Falls back to the legacy subprocess path. Top-level `QMainWindow`.       |
| **Launch in Tab**        | Tool implements `EmbeddableTool`.                        | Adds a tab to the embedded host's central tab area.                      |
| **Launch in Dock**       | Tool implements `EmbeddableTool` and didn't refuse dock. | Adds a `QDockWidget` to the embedded host (right side by default).       |

The mapping from menu choice to dispatcher behaviour lives in
`resolve_launch_mode()` in
[`src/launchers/launch_routing.py`](../../../src/launchers/launch_routing.py).
If you ask for **Launch in Tab** on a tool that doesn't support
embedding (legacy dashboards, tools that need their own
`QApplication`) the launcher logs a warning and silently falls back
to **Launch in New Window** — you'll see your tool open as a
top-level window and a one-line entry in the launcher diagnostic
log.

![Right-click context menu on a launcher tile](../../assets/launcher/embedded_view_context_menu.png)

---

## C. Closing a tab or dock — and the dirty-state prompt

Click the **×** on a tab or the dock's title-bar close button to
close it. If the tool reports unsaved state (its `is_dirty()` returns
`True`), you'll see a confirmation dialog:

> **Unsaved changes.** The tool `'pose_studio'` has unsaved changes.
> Close anyway?

Choose **Cancel** to keep the tool open (you can then save manually
inside the tool). Choose **Yes** to close anyway — `cleanup()` runs
on the tool, the widget is destroyed, and any in-memory state is
lost.

Tools that don't track dirty state never trigger this prompt. The
`is_dirty()` contract is documented in
[`embedding_a_tool.md`](../../development/embedding_a_tool.md).

Closing the launcher window itself runs `cleanup()` on every active
tab and dock — you don't need to close them individually first.

---

## D. Focus mode — double-click a tab

Double-click anywhere on the tab bar (including empty space to the
right of the last tab) to enter **focus mode**. The tab bar hides
and the active tool fills the entire embedded host area. Double-click
again to exit.

Focus mode is per-host, not per-tool — it's a view setting, so
opening a new tab while in focus mode keeps the bar hidden. Use it
when you want to use a tool fullscreen-style without giving up the
ability to flip back to your dock layout.

Dock widgets stay visible in focus mode; only the tab **bar** hides.
If you want a true minimal-chrome view, drag the docks closed first.

---

## E. Keyboard shortcuts

| Shortcut         | Action                                          |
| ---------------- | ----------------------------------------------- |
| `Ctrl+Shift+T`   | Open the currently-selected tile as a **tab**.  |
| `Ctrl+Shift+D`   | Open the currently-selected tile as a **dock**. |
| `Ctrl+W`         | Close the active tab (with dirty-state prompt). |
| `Ctrl+Tab`       | Cycle to the next tab.                          |
| `Ctrl+Shift+Tab` | Cycle to the previous tab.                      |

`Ctrl+Shift+T` and `Ctrl+Shift+D` mirror the right-click menu — they
go through the same `resolve_launch_mode()` fallback, so asking for
a tab on a non-embeddable tool still works (it opens a new window
and logs the fallback).

The standard close/cycle bindings (`Ctrl+W`, `Ctrl+Tab`) are wired
to the tab widget's built-in slots and behave the way they do
everywhere else on your platform.

---

## F. View → Show Embedded Host

The launcher's **View** menu has a checkbox: **Show Embedded Host**.

- **Checked** (default once any tool is opened in a tab or dock) —
  the embedded view region is visible. The tile grid stays above it
  so you can launch additional tools without leaving the launcher.
- **Unchecked** — the embedded view collapses to zero height. Active
  tabs and docks keep running (they just become invisible); toggling
  the menu back on restores them. Use this when you want the
  pure-tile-grid look.

If you uncheck the box while a dirty tool has unsaved changes, the
tool stays alive (no `cleanup()`, no dirty-state prompt) — hiding
the host is not the same as closing tools.

---

## G. Layout persistence

`EmbeddedHostWidget.state_snapshot()` returns a small JSON-shaped
dict (`{"tabs": [...], "docks": {...}, "active_tab": int}`) that the
launcher can save between sessions. As of this writing the launcher
captures the snapshot but does **not** yet persist it to disk —
that's tracked as a follow-up to
[#4993](https://github.com/D-sorganization/UpstreamDrift/issues/4993).
For now, expect to re-open your tabs each time the launcher starts;
your tool-internal state (Pose Studio's open project, etc.) is
preserved by the tool itself, not the host.

---

## See also

- [ADR-0013](../../adr/0013-launcher-composability.md) — design
  rationale for the contract, host, and IPC layer.
- [`embedding_a_tool.md`](../../development/embedding_a_tool.md) —
  for tool authors adding embedding support.
- [`realtime_ipc.md`](../../development/realtime_ipc.md) — pub-sub
  IPC between embedded tools.
- The Pose Studio user guide
  ([`docs/user_guide/pose_studio/quickstart.md`](../pose_studio/quickstart.md))
  — the first tool that adopted the embedded view.

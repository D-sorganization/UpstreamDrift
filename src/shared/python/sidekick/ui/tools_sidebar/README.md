# Sidekick Unified Tools Sidebar

This package provides the shared Sidekick sidebar toolkit for host
applications that need project-scoped utilities beside their primary workflow.

## Import contract

Pure-Python state and workspace registry APIs are safe in headless contexts:

```python
from upstream_drift_tools.ui.tools_sidebar import SidebarState, WorkspaceRegistry
```

Qt widgets are lazy imports. `SidekickSidebar` is an alias for the stable
`UnifiedToolsSidebar` class:

```python
from upstream_drift_tools.ui.tools_sidebar import SidekickSidebar
```

## Host integration

```python
sidebar = UnifiedToolsSidebar(project_root=project_root)
dock = sidebar.install_as_dock(main_window, area="right", state_path=state_path)
sidebar.file_open_requested.connect(open_file)
sidebar.context_updated.connect(update_terminal_context)
```

Call `sidebar.save_state(state_path)` during host shutdown to persist dock area,
floating state, minimized state, dock size, active tab, tab order, hidden tabs,
and popped-out tab ids.

Named Sidekick profiles are stored below a host-provided storage root:

```python
store = SidekickStateProfileStore(storage_root)
store.save_profile("Run 01", sidebar.snapshot_state())
result = store.load_profile("Run 01")
if result.ok and result.state is not None:
    sidebar.apply_state(result.state)
```

Profile files live under `<storage_root>/profiles/<name>.json`. Profile names
must be non-empty and path-safe. Loading a missing or malformed profile returns a
result without mutating the current sidebar; existing `SidebarState.load_json`
behavior remains compatible for host shutdown/startup state files.

Clear operations are intentionally guarded at the service boundary. Show
`CLEAR_SIDEKICK_DATA_WARNING` in UI code, then pass
`CLEAR_SIDEKICK_DATA_CONFIRMATION` to `clear_data(...)` only after the user
confirms.

## Flexible workflows

Sidekick supports modern tab workflow controls:

- dock on either side with `set_dock_area("left" | "right")`
- collapse/expand without losing state via `set_minimized(...)`
- reorder tabs with `move_tab(...)` or drag the movable tab bar
- hide/show configured tabs with `set_tab_visible(...)`
- pop a tab into its own window with `pop_out_tab(...)`
- redock popped-out tabs with `redock_tab(...)`
- duplicate tabs that opt in through `SidebarTabDefinition.duplicate_enabled`
- rename tabs with `rename_tab(...)` and restore defaults with
  `reset_tab_display_name(...)`

Sidekick keeps stable tab ids separate from user-facing display names. Host
applications should persist and route by `tab_id`; custom names are presentation
metadata stored under `SidebarState.tab_display_names` and resolved by the
sidebar before updating `QTabWidget` labels or pop-out window titles.

Hosts can pass `tab_definitions=[...]` to `UnifiedToolsSidebar`,
`create_tools_sidebar`, or `install_tools_sidebar` to choose which utilities are
enabled for a specific application while keeping the default import contract.

Custom tab definitions should ship non-empty `help_metadata` with at least
`title` and `summary` keys so host apps can expose a consistent Help entry from
the shared tab context menu. Optional `tips`, `examples`, and `source` fields
are rendered automatically by the built-in help dialog when provided.

The project file explorer is read-only in this first slice. It emits
`file_open_requested` for files under the scoped project root and intentionally
does not expose destructive file actions yet.

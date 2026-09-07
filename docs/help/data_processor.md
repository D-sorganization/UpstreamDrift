---
title: Data Processor
tile_id: data_processor
status: complete
---

# Data Processor

## Purpose

Data Processor is the heavier signal-processing and time-series surface:
you load tabular measurement data, filter and condition the signals,
compute statistics, and export the result. **This tile launches a widget
that lives in the Tools repository; UpstreamDrift performs no analysis of
its own here.** The UpstreamDrift code for this tile is one import
shim plus a window wrapper.

## Inputs

- Whatever `DataProcessorWidget` from the Tools repo accepts. That widget
  is not part of this repository, so its accepted file formats and column
  expectations are documented there, not here.
- A resolvable Tools repository. `resolve_tools_repo` checks, in order:
  the `TOOLS_REPO_PATH` environment variable, the pinned
  `vendor/ud-tools` gitlink, then a sibling `Tools` checkout. Without one
  the tile cannot load at all.
- When reached through the Sidekick tools sidebar instead of the
  launcher: a variable name and a comma-separated column selection (for
  example `temperature` or `temperature, pressure`) for the workspace
  export.

## Outputs

- A `QMainWindow` titled "Data Processor" hosting the Tools widget, with
  a status bar reading "Data Processor - loaded from Tools repository".
- On failure, an `_UnavailableToolWindow` placeholder carrying the import
  error and remediation text; its `is_tool_available` attribute is
  `False`.
- Via the Sidekick tab (`data_processor_tab.py`), selected result columns
  are exported into the shared Sidekick workspace under a validated
  variable name, defaulting to `data_processor_result`.
- Any file the Tools widget itself writes. Formats are defined by that
  widget, not here.

## Method

[`external_tools_adapter.py`](../../src/launchers/external_tools_adapter.py)
holds the whole UpstreamDrift-side implementation.
`_import_data_processor` resolves the Tools repo root, prepends
`<tools_repo>/src/data_processing/data_processor/python` to `sys.path`,
imports `data_processor.pyqt_widget.DataProcessorWidget`, and hands the
instance to `_wrap_external_widget`, which wraps it in a `QMainWindow`
with a minimum size of 1000x700 px. `get_data_processor_dockable_ui` is
registered in the module's `EXTERNAL_TOOLS` dict under the key
`data_processor`. That is the entire local code path - it is a loader,
not a processor.

A second, independent entry point exists in the Sidekick tools sidebar:
[`data_processor_tab.py`](../../src/shared/python/sidekick/ui/tools_sidebar/data_processor_tab.py)
lazily imports the same heavy surface on demand, deliberately so that
Sidekick startup does not depend on the full Data Processor UI stack.

## Limitations

- The registry `path`,
  `src/data_processing/data_processor/launch_pyqt6.py`, **does not exist
  in this repository** - `src/data_processing/` is not present at all.
  It is a Tools-repo-relative path, consistent with the tile's
  `provider: "tools"` marking.
- No Tools repo means no tool. You get a placeholder window, not a
  degraded mode.
- The manifest lists a `web_route` of `/tools/data-processor`, but there
  is **no such route in `ui/src/App.tsx`**. The tile's `web.mode` is
  `native-window`, so the web launcher opens the native window rather
  than serving a page; the `web_route` string is not a reachable URL.
- This page cannot document the tool's own algorithms, filter types,
  accepted file formats, units, or calculator list. All of that is Tools
  repo territory and is intentionally not guessed at here.
- Do not edit `vendor/ud-tools/`. It is a vendored, pinned copy of the
  Tools repository; fixes belong upstream in Tools.

## See Also
- [Tools Calculator Suite](tools_calculator_hub.md) - shares this exact
  entry point
- [Data Explorer](data_explorer.md) - the other Tools-provided data tile
- [Sidekick](sidekick.md)
- [Vendored Tools repository notes](../../vendor/README.md)

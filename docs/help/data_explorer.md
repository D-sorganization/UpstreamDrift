---
title: Data Explorer
tile_id: data_explorer
status: complete
---

# Data Explorer

## Purpose

Data Explorer lets you find the dataset files UpstreamDrift has produced
or that you have imported, preview their rows, read per-column summary
statistics, filter them, and export a subset. It is a browser over
dataset files on disk. It runs no simulation and computes nothing beyond
descriptive statistics.

## Inputs

- Dataset files discovered under the project's data directories. The web
  backend accepts these suffixes: `.csv`, `.json`, `.hdf5`, `.h5`,
  `.c3d` (`_SUPPORTED_DATASET_SUFFIXES` in the route module).
- Uploaded files via `POST /tools/data-explorer/import`, restricted to
  `.csv` and `.json`.
- Column expectations: none are enforced. A CSV is read by header row,
  so any header names are accepted; statistics are computed only for
  columns whose values parse as numbers. A JSON dataset must decode to
  an array of objects.
- A filter request: a column name, one operator from `eq`, `ne`, `gt`,
  `lt`, `gte`, `lte`, `contains`, a string-encoded value, and a row
  limit between 1 and 10000.

## Outputs

| Output | Format / units |
| --- | --- |
| Dataset listing | JSON: `name`, `path`, `format`, `size_bytes` (bytes), `columns` |
| Preview | JSON rows from the head of the file |
| Statistics | JSON `stats` mapping column name to a float-or-null summary, plus `row_count` |
| Filtered rows | JSON rows matching the filter, capped at the requested limit |
| Export | `csv` (comma-separated values) or `json` (array of objects) - the only two formats `GET /tools/data-explorer/export-formats` returns |

## Method

There are two distinct implementations behind this one tile.

**Desktop (PyQt6).** UpstreamDrift does not contain the Data Explorer
widget. `_import_data_explorer` in
[`external_tools_adapter.py`](../../src/launchers/external_tools_adapter.py)
imports `data_explorer.gui.MainWidget` from the **Tools** repository,
resolved by `src/launchers/tools_repo_path.resolve_tools_repo` in this
order: the `TOOLS_REPO_PATH` environment variable, the pinned
`vendor/ud-tools` gitlink, then a sibling checkout. The UpstreamDrift
side is only the adapter that puts that widget in a `QMainWindow`. When
the Tools repo cannot be resolved the tile shows an
`_UnavailableToolWindow` placeholder telling you to initialise the
submodule or set `TOOLS_REPO_PATH`.

**Web.** `/tools/data-explorer` is UpstreamDrift's own code:
[`src/api/routes/data_explorer.py`](../../src/api/routes/data_explorer.py)
(router prefix `/tools/data-explorer`) serving `GET /datasets`,
`GET /datasets/{name}/preview`, `GET /datasets/{name}/stats`,
`GET /datasets/{dataset_id}/rows`, `POST /import`,
`POST /datasets/{name}/filter`, and `GET /export-formats`, rendered by
`ui/src/pages/DataExplorer.tsx`.

## Limitations

- The registry entry's `path`, `src/data_explorer/data_explorer_app.py`,
  **does not exist in this repository**. It is a path inside the Tools
  repo. The manifest marks the tile `provider: "tools"`; the launcher
  reaches it only through the external-tools adapter.
- With no Tools repo resolvable, the desktop tile is a placeholder and
  nothing works. Only the web page functions standalone.
- The desktop and web surfaces are separate codebases with different
  feature sets. Nothing described under "Web" above is a claim about the
  Tools widget.
- Export is `.csv` and `.json` only, even though `.hdf5`, `.h5` and
  `.c3d` can be listed and previewed. Import is narrower still: `.csv`
  and `.json`.
- One filter clause at a time. `DatasetFilterRequest` takes a single
  column, operator, and value; there is no compound or nested filtering.
- Statistics are descriptive only. There is no fitting, no regression,
  and no cross-dataset join.

## See Also
- [Data Processor](data_processor.md) - the other Tools-provided data tile
- [Dataset Generator](dataset_generator.md) - produces datasets to explore
- [Vendored Tools repository notes](../../vendor/README.md)
- [Analysis tools](analysis_tools.md)

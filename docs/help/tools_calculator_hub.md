---
title: Tools Calculator Suite
tile_id: tools_calculator_hub
status: stub
---

# Tools Calculator Suite

## Purpose

The registry describes this tile as "Engineering process calculators
from the Tools repo (50+ calculators)". It is an **alias tile**: it
carries no implementation of its own and points at the same entry point
as the Data Processor tile. Its registry `surface_reason` states this
outright - "alias surface over the Tools data processor".

## Inputs

Not determinable from this repository. The tile's declared entry point
is identical to Data Processor's, and that entry point resolves into the
Tools repository, so the calculators' inputs, units, and accepted
formats are defined there.

## Outputs

Not determinable from this repository, for the same reason.

## Method

There is no UpstreamDrift code specific to this tile. Every registry
field it declares - `path`
(`src/data_processing/data_processor/launch_pyqt6.py`), `provider`
(`tools`), `working_dir`, and `python_paths` - is byte-identical to the
[Data Processor](data_processor.md) tile's. That path resolves inside
the Tools repository via
[`external_tools_adapter.py`](../../src/launchers/external_tools_adapter.py),
whose `EXTERNAL_TOOLS` registry contains keys for `video_analyzer`,
`data_explorer`, and `data_processor` - and **no key for
`tools_calculator_hub`**.

What the two tiles differ in is metadata only:

| Field | `data_processor` | `tools_calculator_hub` |
| --- | --- | --- |
| `status` | `ready` | `external` |
| `maturity` | `ready` | `beta` |
| `surfaces` | `pyqt`, `web` | `web` |
| `web_route` | `/tools/data-processor` | `/tools/calculators` |
| `order` | 15 | 44 |
| `capabilities` | data_import, signal_processing, time_series, filtering, statistics, export, process_calculators | process_calculators, engineering, analysis |

The one substantive overlap is the `process_calculators` capability,
which both tiles claim. That is consistent with this tile being a
narrower alias exposing only the calculator portion of the same
underlying Tools widget - but no code in this repository implements that
narrowing.

## Limitations

- It is a stub tile here. UpstreamDrift ships no widget, adapter,
  route handler, or launcher entry unique to it.
- Neither `/tools/calculators` nor `/tools/data-processor` exists as a
  route in `ui/src/App.tsx`. Both tiles set `web.mode` to
  `native-window`, so those `web_route` strings are not reachable URLs.
- Nothing works without a resolvable Tools repository
  (`TOOLS_REPO_PATH`, the pinned `vendor/ud-tools` gitlink, or a sibling
  checkout).
- The "50+ calculators" figure comes from the registry description and
  is not verifiable from this repository.
- Maturity is **beta**.
- Do not edit `vendor/ud-tools/`; it is a pinned vendored copy of the
  Tools repository.

## Unclear

This page is a stub because the following could not be determined from
UpstreamDrift's code:

1. **What, if anything, distinguishes this tile from `data_processor`
   at runtime.** They share an identical entry point and no local code
   branches on the tile id. Whether the launcher opens a different view,
   a calculator-only mode, or simply the same Data Processor window a
   second time is not answerable here.
2. **The calculator list, and each calculator's inputs, outputs, units,
   and method.** These live in the Tools repository, outside this
   repo's scope.
3. **Whether `/tools/calculators` is intended to become a real web
   route.** It is declared but unrouted.

Files examined: `src/config/models.yaml` (tile entry, read only),
`src/config/launcher_manifest.json`,
`src/launchers/external_tools_adapter.py`,
`src/launchers/launcher_diagnostics.py`, `ui/src/App.tsx`,
`scripts/registry/generate_registry_artifacts.py`,
`tests/launchers/test_tile_coverage_post_5556.py`,
`docs/testing/functional-test-plan.md` (row MAN-23),
`docs/development/feature_parity_matrix.md`. A repository-wide search for
`tools_calculator_hub`, `calculator_hub`, and `tools/calculators` found
matches only in registry data, generated artifacts, docs, and test
allow-lists - never in implementation code.

## See Also
- [Data Processor](data_processor.md) - shares this tile's entry point
- [Data Explorer](data_explorer.md)
- [Feature parity matrix](../development/feature_parity_matrix.md)
- [Vendored Tools repository notes](../../vendor/README.md)

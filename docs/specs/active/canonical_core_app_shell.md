# Canonical-Core App Shell Tool Registry

Status: active
Issue: [#6805](https://github.com/D-sorganization/UpstreamDrift/issues/6805)

## Problem

Canonical-core estimation and comparison need first-class launcher entries that
are visible in both the PyQt6 desktop shell and the React/Tauri shell without
creating a parallel launcher registry.

## Scope

The canonical-core workspace entries are launcher surfaces, not engine
adapters. They route users to the CC-19 estimation and CC-27 comparison service
boundaries through the existing ADR-0013 `launcher_embed` contract.

## Non-Goals

- Implementing CC-19 estimation algorithms.
- Implementing CC-27 comparison algorithms.
- Importing engine backends from launcher shell code.

## Registered Tools

| Tool ID                     | Surface    | Category       | React route                        | PyQt launch                               |
| --------------------------- | ---------- | -------------- | ---------------------------------- | ----------------------------------------- |
| `canonical_core_estimation` | Estimation | `biomechanics` | `/tools/canonical-core/estimation` | `src.tools.canonical_core._embed_adapter` |
| `canonical_core_comparison` | Comparison | `biomechanics` | `/tools/canonical-core/comparison` | `src.tools.canonical_core._embed_adapter` |

Both tools declare `shell_surfaces=["pyqt6", "react"]` in the launcher
manifest and register PyQt6 adapters through the existing
`register_embeddable_tool()` process-wide registry. The React dashboard reads
the same metadata from `/api/launcher/manifest`.

## Design Notes

- Shell metadata lives in `src/tools/canonical_core/registry.py`.
- PyQt6 imports are lazy and limited to `create_main_widget()`.
- The shell package does not import engine backends.
- CC service implementations remain behind API/service modules and can be wired
  under the same tool IDs without changing launcher discovery.

## Acceptance Criteria

- Both canonical-core tools are registered through `launcher_embed`.
- Both tools are categorized as `biomechanics`.
- Both tools expose PyQt6 and React shell metadata.
- The manifest API serializes shell metadata for React consumers.

## Validation

- `tests/unit/launcher_embed/test_canonical_core_shell.py`
- `tests/config/launcher_manifest/test_canonical_core_shell.py`
- `ui/src/api/useLauncherManifest.test.ts`

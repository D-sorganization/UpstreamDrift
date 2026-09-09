# Reference Overlay Comparison Workspace Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: C:/Users/diete/Repositories/UpstreamDrift
- Branch: feat/9866-reference-comparison-workspace
- Baseline commit: 10caddd21 (calibrated reference scene registration #9871 merged)
- Implementation commit: SELF
- Governing issue/epic: #9866, #9863

## Objective and Status

Subepic #9866: Reference Overlay Comparison Workspace, Saved Layers & Reproducible Exports.
Dual playback workspace (`ReferenceComparisonDialog`), synchronized scrubbing, event-anchor and
offset alignment modes, layer styling (opacity, color palette, skeleton visibility), comparison
session persistence (`.comparison.json`), and reproducible video/sidecar export pipeline
(`export_comparison_video`, `build_comparison_sidecar`, `ComparisonExportWorker`) with cancellation.
Focused unit and UI qualification passes.

## Files and Decisions

- `src/motion_capture/reference/comparison.py` (169 LOC): Schema data models (`ComparisonLayer`,
  `ComparisonSession`), JSON persistence helpers, and signed sidecar builder.
- `src/tools/capture_rig/reference_export.py` (248 LOC): Isolated heavy video rendering and
  background worker (`ComparisonExportWorker`) to preserve <= 500 LOC limit.
- `src/tools/capture_rig/reference_comparison.py` (419 LOC): Qt workspace dialog with dual preview,
  alignment mode switching, synchronized scrub controls, and export actions.
- `src/tools/capture_rig/library_dialog.py`: "Compare Reference…" action integration.
- `src/tools/capture_rig/reference_library_dialog.py`: "Compare with capture…" button integration.

## Validation

- 4 unit tests in `tests/motion_capture/test_reference_comparison.py` pass cleanly.
- 3 UI tests in `tests/tools/capture_rig/test_reference_comparison_ui.py` pass cleanly.
- `ruff check` and `ruff format` pass on all touched files.
- All files strictly adhere to the <= 500 LOC budget.

## Next Steps

1. Create PR linked to #9866 and arm auto-merge.
2. Advance to next subepic in backlog.

# Spine Viewer

Read-only PyQt6 viewer for segmental-spine study receipts (SPN-15b, #12772).

- `core.py` - Qt-free data layer. Opens a receipt with `load_verified_run`
  (SHA-256 verified) and exposes `viz_data` heatmaps and angle stacks.
- `gui.py` - `SpineViewerWidget`. The scientific status banner and the
  `distribution_source` line are always visible. Status is UNQUALIFIED unless the
  receipt declares `qualified` and its run verdict is `pass`.
- `_embed_adapter.py` - lazy launcher embed adapter (tile id `spine_viewer`).

Run: `python -m src.tools.spine_viewer`. Tests: `tests/tools/spine_viewer`.
Loads follow ADR-0055: compression is +F_y; the tension view is -compression.

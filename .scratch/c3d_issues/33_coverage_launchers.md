# test(launchers): bring `src/launchers/` to ≥80% coverage

## Goal

Raise coverage of `src/launchers/` to **≥80% line, ≥70% branch**. Slightly lower bar because some launcher modules are inherently GUI-shell code that's hard to unit-test without `pytest-qt` (not yet in deps).

## Current state

38 production files, 36 test files in `tests/launchers/`. Recent additions: model_card_tile_scale, layout_view_modes, layout_persistence, runtime_mode_help, info_button. Likely gaps:

- `golf_launcher.py` — main entry slots (large file; cover the pure-logic helpers)
- `launcher_dialogs.py` — Preferences and other dialogs (post-#4499 the path opens)
- `launcher_simulation.py` — the new `_launch_c3d_viewer` candidate-search logic (already partially covered by #4595's tests; add: path traversal rejection, vendor-fallback hit, all-missing toast)
- `launcher_layout_manager.py` — the new ViewMode-aware grid + LIST mode rendering (partially covered)
- `launcher_ui_setup.py` — top-bar zoom slider, view-mode combo, shortcut handlers
- `model_card.py` — drag-and-drop hooks, mouse events
- `launcher_provider_compatibility.py` — `is_engine_runtime_available` for each engine
- `theme/*.py` and `style_constants.py` — color resolution
- `about_dialog.py` — version-string assembly with various missing dependencies (numpy missing, ezc3d missing — assert graceful)
- `help_menu.py` — KeyboardShortcuts dialog scrapes registered actions

## Process

Same approach. For modules that need a `QApplication`, use `QT_QPA_PLATFORM=offscreen` + `QApplication.instance() or QApplication([])`. For dialogs, instantiate, assert widget tree, close — don't drive the event loop end-to-end unless `pytest-qt` is added as a separate effort.

## Acceptance

- [ ] `pytest tests/launchers/ --cov=src/launchers --cov-report=term-missing --cov-branch` reports **≥80% line, ≥70% branch**.
- [ ] No production changes.
- [ ] PR body lists per-file delta.
- [ ] mypy + ruff + file-size budget clean.

## Files touched

- New / extended: `tests/launchers/test_*.py`

## Out of scope

- Adding `pytest-qt` as a dependency (separate decision).
- Browser / Tauri launcher (not in `src/launchers/`).
- E2E launcher launches (covered by `tests/integration/test_golf_launcher_integration.py`).

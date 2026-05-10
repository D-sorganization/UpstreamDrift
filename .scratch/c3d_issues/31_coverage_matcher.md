# test(starting-pose-matcher): bring `src/tools/starting_pose_matcher/` to ≥85% coverage

## Goal

Raise unit + Qt-offscreen coverage of `src/tools/starting_pose_matcher/` to **≥85% line, ≥75% branch**. Test-only — no production code changes.

## Current state

14 production files; `tests/unit/tools/starting_pose_matcher/` has 7 test files. Likely gaps:

- `core.py` — pose math + load_simscape_trajectory_csv + cost helpers
- `gui.py` (2.7k lines, partially covered) — instantiate path, slot wiring (the parts that don't require a full Qt event loop)
- `gui_source_panel.py` — `DataSourcesPanel` widget, signal emission
- `live_view_controller.py` — covered partially (3 tests on main); add edge cases (target with NaN markers, club-only, ball-only)
- `session_schema.py` — v1/v2/v3/v4 round-trip, missing-field tolerance
- `skeleton_provider.py` — JsonSkeletonProvider load/save
- `providers/drake.py`, `providers/mediapipe.py`, `providers/mujoco.py`, `providers/openpose.py`, `providers/opensim.py`, `providers/pinocchio.py`, `providers/simscape.py` — error paths (model_path missing, model XML invalid, etc.)

## Process

Same approach as the tracking issue. Use the existing per-test fixtures + `QT_QPA_PLATFORM=offscreen`.

Provider tests follow a uniform shape: build the provider with a synthetic config; assert `supports_*()` flags; assert error messages on invalid inputs; skip with `pytest.importorskip("<engine>")` when the engine wheel isn't installed.

## Acceptance

- [ ] `pytest tests/unit/tools/starting_pose_matcher/ --cov=src/tools/starting_pose_matcher --cov-report=term-missing --cov-branch` reports **≥85% line, ≥75% branch**.
- [ ] PR body lists per-file delta.
- [ ] No production changes.
- [ ] mypy + ruff + file-size budget clean.

## Files touched

- New / extended: `tests/unit/tools/starting_pose_matcher/test_*.py`

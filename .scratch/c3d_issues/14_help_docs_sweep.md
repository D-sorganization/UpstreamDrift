# docs(app): comprehensive help-text sweep across launcher and matcher GUIs

## Why

The desktop application (PyQt6 classic launcher + starting-pose / motion-target matcher + per-engine simulator panels) has accumulated UI surface faster than its in-app help text. Tooltips, "What's this?" dialogs, status-bar hints, and menu-help entries are sparse, inconsistent, and in many places outdated relative to actual behaviour. New users — and even returning users after long gaps — frequently can't tell what a control does without reading source.

This effort is a focused sweep through the app to bring the in-app help text up to professional grade for the **current state** of the app (not the way it was three refactors ago).

## Scope

### A. Launcher (`src/launchers/`)

- Every tile gets:
  - A one-sentence `tooltip` that previews what clicking does.
  - A "What's this?" body (Qt's `QWhatsThis` mechanism) with: 2–4 sentence description, a link to the relevant user-guide page in `docs/user_guide/`, and a "Recommended when…" hint.
- The menu bar: `File`, `Tools`, `View`, `Help` — every action gets a tooltip and a `QAction.statusTip()`.
- Preferences dialog: every setting row gets a tooltip pinned to the row's QLabel; ambiguous units (e.g. "ms" vs "frames") spelled out.
- Engine-discovery status bar: hover surfaces the discovery path and "available / unavailable / requires_install" reason.

### B. Starting-pose / motion-target matcher (`src/tools/starting_pose_matcher/`)

- Source-toggle panel (added by issue #4482): tooltips on each toggle, a status-bar hint when toggling switches the active target type.
- Timeline / playback (added by issue #4481): tooltips on play/pause/speed/loop/trail; `statusTip` showing current frame and current event label.
- Layer-visibility checkboxes (added by issue #4481): tooltip lists the artist colour and rough description ("Body markers — 28 anatomical points, semi-transparent dots").
- File-picker dialogs: title + `selectedFilter` defaults reflect the active source toggle.
- Skeleton-provider combo: tooltip describing each provider (Drake / MuJoCo / Pinocchio / OpenSim / Simscape / MediaPipe / OpenPose) — current capability + what it produces.

### C. Per-engine simulator panels (`src/engines/*/python/...`)

Audit each engine's PyQt6 panel for missing tooltips / outdated help. Add a `statusTip` for every long-running action so the bottom status bar tells the user what's happening.

### D. Help menu

A new top-level "Help" submenu structure (idempotent — extend if it already exists):

- "User Guide" → opens `docs/user_guide/index.md` in the system browser
- "Motion-Match Loaders" → opens `docs/user_guide/motion_matching/loading_targets.md` (added by issue #4487)
- "Keyboard Shortcuts" → opens a modal listing every QShortcut + its action name, scraped at runtime from registered actions
- "Report a Bug" → opens https://github.com/D-sorganization/UpstreamDrift/issues/new
- "About" → version, build hash, Python + Qt + ezc3d versions, link to LICENSE

### E. Docstrings

For every public class / function in `src/launchers/`, `src/tools/starting_pose_matcher/`, and `src/shared/python/motion_matching/`:

- Class / function docstring describes WHY the thing exists, not just what it returns.
- All `Args:` blocks list units, ranges, and what `None`/missing means.
- Postconditions documented for any function returning a complex object.

### F. README + index

- `src/tools/starting_pose_matcher/README.md` updated to reflect post-#4481/#4482/#4486 state.
- `docs/motion_matching/README.md` (or new) becomes the index issue #4487 builds on.

## Tone + style guide

- Tooltips: ≤ 80 chars, sentence case, no period, no ellipsis.
- "What's this?" bodies: 2–4 sentences, professional, no jokes, no emojis.
- Status tips: ≤ 60 chars, present tense ("Resampling target…").
- Generic naming policy applies (no vendor / lab / person / study names anywhere).

## Acceptance criteria

- [ ] Every menu action has both a tooltip and a `statusTip`.
- [ ] Every checkbox / radio / combobox in the matcher and the preferences dialog has a tooltip.
- [ ] A headless test (`tests/ui/test_help_coverage.py`) walks the matcher's QObject tree under `QT_QPA_PLATFORM=offscreen` and asserts coverage: zero widgets in a curated whitelist that are missing a tooltip.
- [ ] Help submenu wired with the five entries above; "Keyboard Shortcuts" modal opens and renders the action table.
- [ ] About dialog shows live versions of Python, Qt, ezc3d, Numpy, the app's `VERSION` file.
- [ ] Every public function in the listed modules has a docstring; `pydocstyle` (already configured per repo) passes.
- [ ] Mypy + ruff + file-size budget clean.
- [ ] User-visible strings use the generic naming policy.

## Out of scope

- The web (Tauri/React) UI — this issue is the PyQt6 desktop surface only.
- Translations / i18n — string externalisation is a separate effort.
- Refactoring the underlying widget tree — text-only changes here.

## Files touched

Many (sweep). Likely:

- Edit: `src/launchers/*.py`
- Edit: `src/tools/starting_pose_matcher/gui.py` (and split files added by #4481/#4482)
- Edit: `src/shared/python/ui/preferences_dialog.py`
- New: `src/launchers/help_menu.py` (Help submenu builder)
- New: `src/launchers/about_dialog.py` (or extend if exists)
- Edit: `src/tools/starting_pose_matcher/README.md`
- New: `tests/ui/test_help_coverage.py`

## Sequencing

Should land **after** the matcher animated preview (#4481) and source-toggle (#4482) — those introduce the new widgets that this sweep documents. The Preferences crash fix (separate issue) must land before the headless preferences-dialog test in this sweep can assert on widget tree coverage.

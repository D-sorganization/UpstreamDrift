# bug(launcher): clicking Preferences crashes with `AttributeError: get_available_fleet_themes`

## Reproduction

1. `python3 launch_golf_suite.py --classic`
2. From the menu, open the Preferences dropdown.
3. App crashes.

## Traceback (from `crash_traceback.txt`)

```
Traceback (most recent call last):
  File "src/launchers/launcher_dialogs.py", line 144, in _show_preferences
    dialog = PreferencesDialog(self)
  File "src/shared/python/ui/preferences_dialog.py", line 134, in __init__
    self._setup_ui()
  File "src/shared/python/ui/preferences_dialog.py", line 143, in _setup_ui
    tabs.addTab(self._create_appearance_tab(), "Appearance")
  File "src/shared/python/ui/preferences_dialog.py", line 184, in _create_appearance_tab
    fleet = ThemeManager.instance().get_available_fleet_themes()
AttributeError: 'ThemeManager' object has no attribute 'get_available_fleet_themes'.
            Did you mean: 'get_available_themes'?
```

## Root cause

`src/shared/python/ui/preferences_dialog.py:184` calls `ThemeManager.instance().get_available_fleet_themes()` but `ThemeManager` (defined at `src/shared/python/theme/theme_manager.py:37`) only exposes `get_available_themes()` (`theme_manager.py:173` and `protocols.py:28,54`). The "fleet" qualifier got introduced when the fleet-theme system was added in `vendor/ud-tools` but the renaming was incomplete.

## Fix

Two correct options. Pick the one that matches the intended behaviour of the appearance tab:

### Option A — call the real method (smallest, recommended)

```python
# src/shared/python/ui/preferences_dialog.py:184
fleet = ThemeManager.instance().get_available_themes()
```

`get_available_themes()` already returns the union of core presets and fleet themes (verify against `theme_manager.py:173`). The dedupe loop above (`if name not in theme_items`) handles the overlap with the hardcoded `["Dark", "Light", "High Contrast"]` list.

### Option B — add a `get_available_fleet_themes()` method

If "fleet themes" is supposed to be a _subset_ (only themes loaded from `vendor/ud-tools`), add a new method on `ThemeManager` that filters:

```python
# src/shared/python/theme/theme_manager.py
def get_available_fleet_themes(self) -> list[str]:
    """Return only fleet-loaded themes (excludes the core Dark/Light/HighContrast presets)."""
    return [name for name in self.get_available_themes()
            if name not in {"Dark", "Light", "High Contrast"}]
```

Update `protocols.py` to add the method to the protocol(s) too.

Option A is preferable unless we have evidence Option B's filtering is required.

## Acceptance criteria

- [ ] Preferences dialog opens without exception in classic launcher.
- [ ] Theme combo lists Dark, Light, High Contrast at minimum, plus any fleet themes.
- [ ] Live-preview on theme change still works.
- [ ] Save + restore: chosen theme persists across launcher restart.
- [ ] Headless smoke test added: instantiate `PreferencesDialog` under `QT_QPA_PLATFORM=offscreen` and assert `tabs.count() >= 1` plus `theme_combo.count() >= 3`.
- [ ] Mypy + ruff clean.

## Files touched

- Edit: `src/shared/python/ui/preferences_dialog.py` (line 184) — apply Option A.
- (If Option B): `src/shared/python/theme/theme_manager.py` and `src/shared/python/theme/protocols.py`.
- New: `tests/unit/ui/test_preferences_dialog.py` (headless smoke).

## Severity

**High** — blocks all preference changes (theme, font size, etc.) in the desktop launcher.

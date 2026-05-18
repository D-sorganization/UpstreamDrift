# feat(launcher): expose multi-source motion-target preview as a launcher tile

## Why

The PyQt6 launcher (`launch_golf_suite.py --classic`) discovers tiles via `src/config/models.yaml` and the launcher manifest loader. Today the matcher GUI is reachable as the "Starting Pose Matcher" tile. After this effort the tool is materially expanded — it loads three target types, animates them, and supports a multi-source toggle. Surface that capability in the launcher.

## What to build

Update `src/config/models.yaml` and `src/config/launcher_manifest.json` so the launcher tile name and description reflect the new scope:

```yaml
- key: motion_target_preview # new key (replaces / aliases the old starting_pose_matcher)
  display_name: "Motion-Match Preview"
  description: |
    Preview club, ball, and full-body mocap targets alongside any engine model.
    Multi-source support (xlsx / .c3d / .mat); animated timeline; layer toggles.
  module: src.tools.starting_pose_matcher.__main__
  category: motion_matching
  logo: assets/logos/motion_target_preview.svg
  tags: [c3d, mocap, club, body, preview]
```

- The OLD `starting_pose_matcher` tile entry stays as a hidden alias (sets `hidden: true`) for one release so saved layouts don't break.
- New tile logo: a minimal SVG (point-cloud + skeleton outline). Place at `assets/logos/motion_target_preview.svg`. Re-use existing palette tokens from `src/launchers/launcher_theme.py`.
- Verify the model registry validation (`src.shared.python.config.model_registry`) accepts the new entry. The current launcher startup logs error `legacy model entry 'starting_pose_matcher' ...: Launcher metadata missing required fields: ['logo']` — fix that as part of this work by populating `logo:` for the legacy entry too.

## Generic naming

Tile key, display name, description, logo filename — all source-neutral.

## Acceptance criteria

- [ ] New `motion_target_preview` tile present, valid against model_registry, visible in the classic launcher.
- [ ] Old `starting_pose_matcher` entry validates (no missing `logo`) and is hidden by default.
- [ ] Launching the new tile opens the matcher GUI with the new "Data sources" panel visible by default (i.e. the source-toggle issue must merge before this is the user-visible default).
- [ ] Tile screenshot included in PR description; theme manager shows correct colours in both Light and Dark.
- [ ] Existing launcher unit tests still pass (`tests/integration/test_golf_launcher_integration.py`, `tests/api/test_launcher_*.py`).

## Files touched

- Edit: `src/config/models.yaml`
- Edit: `src/config/launcher_manifest.json`
- Edit: `src/shared/python/config/model_registry.py` (loosen / fix the legacy-entry validator)
- New: `assets/logos/motion_target_preview.svg`
- Edit: `tests/config/test_launcher_manifest.py`

## References

- Launcher manifest loader: `src/config/launcher_manifest_loader.py`
- Model registry: `src/shared/python/config/model_registry.py`
- Existing tile pattern: search `src/config/models.yaml` for `starting_pose_matcher`.

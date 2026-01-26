# Launcher Archive

This directory contains archived versions of the Golf Modeling Suite launcher
for reference and recovery purposes.

## Files

### golf_launcher_pre_refactor_ce85e6ec.py

**Source Commit:** `ce85e6ec`
**Date Archived:** 2026-01-25

This is the full launcher before the January 24, 2026 refactoring that:
- Removed the `MODEL_IMAGES` dictionary from the main launcher
- Removed custom launch methods for OpenSim, MyoSim, OpenPose
- Decomposed the launcher into modular UI components

Key features preserved in this archive:
- Full `MODEL_IMAGES` dictionary with 12 tile image mappings
- `_custom_launch_opensim()` method
- `_custom_launch_myosim()` method
- `_custom_launch_openpose()` method
- `_custom_launch_humanoid()` method (with duplicate process checking)
- `_custom_launch_comprehensive()` method
- `_custom_launch_drake()` method
- `_custom_launch_pinocchio()` method
- `_launch_urdf_generator()` method
- `_launch_c3d_viewer()` method
- `_launch_matlab_app()` method
- `SpecialApp` class (removed in Phase 3.3)

## Recovery

If the current launcher has issues, you can restore this version:

```bash
# View the archived code
cat src/launchers/_archive/golf_launcher_pre_refactor_ce85e6ec.py

# Or restore from git directly
git show ce85e6ec:src/launchers/golf_launcher.py > src/launchers/golf_launcher.py
```

## Notes

The current launcher (as of 2026-01-25) has been updated to:
1. Restore all custom launch methods
2. Include proper MODEL_IMAGES mappings
3. Add all engines to models.yaml configuration
4. Generate placeholder tile images for the launcher grid

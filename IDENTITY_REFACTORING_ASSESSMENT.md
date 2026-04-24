# Identity Refactoring Assessment (Issues #3046-#3047)

## Completed Work

### Issue #3049: Fix 10-minute Quick Start

✅ **COMPLETED** - All example scripts now work with clean install

- Removed sys.path manipulation from examples/
- Examples run after: `pip install -e .` or `pip install .`
- Branch: `fix/3049-quickstart-clean-install`
- Commit: `f1ccab9d0`

## Remaining Work for #3046-#3047: Identity Rename

### Scope Overview

This is a **large refactoring** affecting 40+ GOLF\_\* references across the codebase. Changes span environment variables, constants, database names, entry points, class names, and file names.

### High Priority: User-Facing Environment Variables

These must be updated together and backward-compatible aliases should be provided:

| Current Name        | Proposed Name           | Location(s)                           | Impact                   |
| ------------------- | ----------------------- | ------------------------------------- | ------------------------ |
| GOLF_DEFAULT_ENGINE | HUMANOID_DEFAULT_ENGINE | launch_golf_suite.py, cli_runner.py   | High - User-facing CLI   |
| GOLF_NO_BROWSER     | HUMANOID_NO_BROWSER     | launch_golf_suite.py                  | High - User-facing CLI   |
| GOLF_PORT           | HUMANOID_PORT           | launch_golf_suite.py, local_server.py | High - Server startup    |
| GOLF_SUITE_MODE     | HUMANOID_SUITE_MODE     | local_server.py, api modules          | High - Deployment config |
| GOLF_AUTH_DISABLED  | HUMANOID_AUTH_DISABLED  | api/auth, api/local_server            | High - Security config   |
| GOLF_API_SECRET_KEY | HUMANOID_API_SECRET_KEY | api/auth/security.py, config files    | High - Security config   |
| GOLF_ADMIN_PASSWORD | HUMANOID_ADMIN_PASSWORD | api/database.py, config files         | High - Security config   |

### Medium Priority: Physics Constants

These are internal constants but appear in many model configuration files:

| Current Prefix   | Proposed Prefix | Affected Files                                           | Impact                          |
| ---------------- | --------------- | -------------------------------------------------------- | ------------------------------- |
| GOLF*BALL*\*     | BALL\_\*        | src/shared/python/core/physics_constants.py, model files | Medium - Internal but pervasive |
| GOLF*CLUB*\*     | CLUB\_\*        | src/shared/python/core/physics_constants.py              | Medium - Less frequently used   |
| GOLF*SWING*\*    | SWING\_\*       | Model XML files, python modules                          | Medium - Domain-specific        |
| GOLF_TASK_POINTS | TASK_POINTS     | jacobian_diagnostics.py                                  | Low - Internal utility          |

### Low Priority: File and Class Names

These are structural/cosmetic but improve clarity:

| Current Name           | Proposed Name              | Impact                       |
| ---------------------- | -------------------------- | ---------------------------- |
| launch_golf_suite.py   | launch_humanoid_suite.py   | Low - Backward compat needed |
| golf_modeling_suite.db | humanoid_modeling_suite.db | Low - Database migration     |
| mujoco_humanoid_golf/  | mujoco_humanoid/           | Low - Structural cleanup     |
| golf*swing*\*.xml      | swing\_\*.xml              | Low - File organization      |
| drake_golf_model.py    | drake_humanoid_model.py    | Low - Module naming          |

## Key Files to Update (By Priority)

### Tier 1: Entry Points & Configuration (CRITICAL)

1. `launch_golf_suite.py` - Main CLI entry point
2. `src/shared/python/config/environment.py` - Config loading
3. `src/api/auth/security.py` - Auth secrets
4. `src/api/database.py` - DB initialization
5. `src/api/local_server.py` - Local development server
6. `pyproject.toml` - Entry point definition
7. `src/config/interim_config.yaml` - Config template

### Tier 2: Core Constants

1. `src/shared/python/core/physics_constants.py` - 20+ constants
2. `src/shared/python/core/constants.py` - General constants
3. `src/engines/loaders.py` - Engine registry with GOLF_SWING_PENDULUM

### Tier 3: Physics Engine Modules

1. `src/engines/physics_engines/mujoco/` - Multiple XML and Python files
2. `src/engines/physics_engines/drake/` - Drake models
3. `src/engines/physics_engines/pinocchio/` - Pinocchio models

### Tier 4: Model Configuration Files

1. `src/shared/models/` - YAML and XML model definitions
2. XML swing models: `*_golf_swing_*.xml` (3+ files)

### Tier 5: Tests

1. `tests/unit/` - Test files referencing GOLF\_ constants (60+ files)
2. Mocking/fixtures: `GOLF_USE_MOCK_ENGINE`, `GOLF_SUITE_MODE=test`

## Implementation Strategy

### Phase 1: Support Both Old & New Names (Non-Breaking)

```python
# In environment.py
def get_engine():
    # Try new name first, fall back to old name
    return os.getenv("HUMANOID_DEFAULT_ENGINE") or os.getenv("GOLF_DEFAULT_ENGINE")
```

### Phase 2: Update Internal Usage

- Update all code to use new names internally
- Keep old names as deprecated fallback

### Phase 3: Update Tests

- Change test fixtures to use new env var names
- Keep some tests that verify fallback compatibility

### Phase 4: Update Documentation

- Update README with new env var names
- Add migration guide for users

## Estimated Effort

- **Tier 1 (Critical)**: 4-6 hours

  - ~20 files
  - Must maintain backward compatibility
  - Requires testing each change

- **Tier 2 (Constants)**: 2-3 hours

  - ~5 files
  - Grep-and-replace mostly
  - But need to test models

- **Tier 3 (Physics engines)**: 3-4 hours

  - ~30 files
  - Some files use GOLF_SWING_PENDULUM as enum value
  - Need careful refactoring

- **Tier 4 (Models)**: 1-2 hours

  - Mostly file renames
  - Update YAML/XML references

- **Tier 5 (Tests)**: 2-3 hours
  - Many grep-and-replace
  - Verify test coverage maintained

**Total Estimated Effort**: 12-18 hours

## Recommendation

Given the scope, I recommend:

1. **Priority Phase 1** (Next 3-4 hours): Update Tier 1 files with backward compatibility

   - This unblocks users
   - Provides clear migration path
   - Can be done incrementally

2. **Priority Phase 2** (Following PR): Update Tiers 2-3 in separate PR

   - Keeps PRs focused and reviewable
   - Less risk of merge conflicts
   - Easier to test each layer

3. **Priority Phase 3** (Subsequent PR): Update tests and models
   - Can run in parallel with user code updates
   - Less critical to project identity

## Notes

- Database name change requires migration - should provide alembic migration
- Entry point in pyproject.toml should remain `upstream-drift` (already correct)
- Package name `upstream-drift` is already domain-neutral (good!)
- Consider adding deprecation warnings for old env var names
- Some class names with "golf" in them (e.g., in XML) could be left as-is since they're internal

# DRY Violations Assessment - UpstreamDrift Repository

## Summary

**Assessment Date:** 2026-02-01  
**Total Python Files:** 774  
**Estimated DRY Violations:** 750+ (Priority to fix)

---

## HIGH-PRIORITY VIOLATIONS (Immediate Action Required)

### 1. Logging Setup Duplication (5 files)

**Files with duplicate `setup_logging` functions:**

- `src/engines/physics_engines/drake/python/src/drake_gui_app.py`
- `src/shared/python/core.py`
- `src/shared/python/logger_utils.py`
- `src/shared/python/logging_config.py`
- `src/tools/model_generation/cli/main.py`

**Impact:** Medium - Inconsistent logging configuration across modules  
**Solution:** Consolidate to single `logger_utils.py` and import everywhere

---

### 2. Path Resolution Duplication (64 occurrences)

**Pattern:** `os.path.dirname(__file__)` and `Path(__file__).parent`  
**Impact:** High - Hard-coded path resolution scattered everywhere  
**Solution:** Create centralized `PathManager` or use existing `REPO_ROOT` pattern consistently

---

### 3. PyQt6 Availability Checks (110 files)

**Pattern:** `PYQT6_AVAILABLE` checks scattered across codebase  
**Impact:** Medium - Redundant conditional imports  
**Solution:** Centralize in `src/shared/python/engine_availability.py` (already exists, needs consolidation)

---

### 4. Exception Handling Patterns (480 occurrences)

**Pattern:** `except Exception as e:` with inconsistent handling  
**Impact:** Low-Medium - Not directly DRY but indicates opportunity for helper functions  
**Solution:** Create standardized exception handlers for common patterns

---

### 5. Launcher UI Duplication (6+ files)

**Files with similar QMainWindow patterns:**

- `golf_launcher.py` (2517 lines - main launcher, well-structured)
- `golf_suite_launcher.py`
- `matlab_launcher_unified.py` (174 lines - simple, clean)
- `motion_capture_launcher.py`
- `mujoco_unified_launcher.py`
- `shot_tracer.py`

**Common Duplicated Patterns:**

- Window initialization (title, size, centering)
- Stylesheet/theme setup
- Grid layout with button cards
- File launching logic (subprocess.run/os.startfile)
- QApplication setup in main()

**Impact:** HIGH - Major code duplication  
**Solution:** Create `BaseLauncher` class in `ui_components.py`

---

### 6. Config File Loading (47 occurrences)

**Patterns:**

- `json.load()` with identical error handling
- `yaml.load()` patterns
- Config validation logic

**Solution:** Create `ConfigLoader` utility class

---

### 7. Font/Theme Setup (56 files)

**Pattern:** `setStyleSheet()` and `QFont()` calls with similar values  
**Solution:** Create centralized theme module

---

### 8. Engine Detection Logic (7 files)

**Files:**

- `src/api/routes/engines.py`
- `src/shared/python/engine_availability.py`
- `src/shared/python/tests/test_launcher_integration.py`
- `src/shared/python/test_utils.py`
- `src/shared/python/ui/adapters/canvas.py`
- `src/shared/python/ui/adapters/thread.py`
- `src/tools/model_explorer/mujoco_viewer.py`

**Impact:** Medium-High - Engine availability checked in multiple places  
**Solution:** All should use `engine_availability.py` as single source of truth

---

## REMEDIATION PRIORITY ORDER

1. **Launcher UI Duplication** - Create `BaseLauncher` abstract class
2. **Logging Setup** - Consolidate to single module
3. **Engine Detection** - Enforce single source of truth
4. **Path Resolution** - Create `PathManager` utility
5. **Config Loading** - Create `ConfigLoader` utility
6. **Theme/Font Setup** - Centralize styling

---

## IMPLEMENTATION PLAN

### Phase 1: Create Base Classes (Today)

- [ ] Create `BaseLauncher` in `src/launchers/base.py`
- [ ] Migrate `MatlabLauncher` to use `BaseLauncher` (smallest, best test case)
- [ ] Add common methods: `center_window()`, `init_ui_base()`, `launch_file()`

### Phase 2: Consolidate Utilities (Next)

- [ ] Consolidate all logging to `logger_utils.py`
- [ ] Create `PathManager` class
- [ ] Create `ConfigLoader` class

### Phase 3: Migrate Remaining Launchers

- [ ] Update `motion_capture_launcher.py`
- [ ] Update `shot_tracer.py`
- [ ] Update `mujoco_unified_launcher.py`
- [ ] Update `golf_suite_launcher.py`

---

## METRICS

| Category | Count | Priority |
|----------|-------|----------|
| Logging duplications | 5 | HIGH |
| Path resolution | 64 | MEDIUM |
| PyQt6 checks | 110 | MEDIUM |
| Exception patterns | 480 | LOW |
| Launcher UI duplication | 6 | HIGH |
| Config loading | 47 | MEDIUM |
| Theme/font setup | 56 | MEDIUM |
| Engine detection | 7 | HIGH |

**Total Estimated DRY Violations: 775+**

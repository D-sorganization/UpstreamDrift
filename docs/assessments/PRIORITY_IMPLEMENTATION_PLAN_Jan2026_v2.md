# Golf Modeling Suite - Priority Implementation Plan v2

**Date:** 2026-01-09  
**Updated:** 2026-01-09 (Implementation Complete)  
**Based On:** Assessment Review + User Priorities  
**Session:** Launcher enhancements, OpenSim usability, and startup optimization

---

## Executive Summary

### Assessment Analysis Summary

Based on the comprehensive review of `docs/assessments/`:

| Assessment           | Score      | Status               | Key Gap                        |
| -------------------- | ---------- | -------------------- | ------------------------------ |
| **A (Architecture)** | 6.2/10     | Partial remediation  | MyPy physics engine exclusions |
| **B (Scientific)**   | 5.8→8.7/10 | ✅ Largely addressed | Conservation tests added       |
| **C (Cross-Engine)** | 4.5/10     | ⚠️ Needs work        | Validator not in CI            |

**Post-Remediation Overall Score:** ~7.5/10 (vs. original 5.5/10)

---

## HIGHEST IMPACT ITEMS (User-Identified)

### Priority 1: OpenSim Model-Free Launch Mode ⚠️ **BLOCKER**

**Current Problem:**

- OpenSim GUI **requires** a model file to launch (`model_path is required`)
- Users cannot start OpenSim to **create** a new model
- Status badge shows "Demo / GUI" which is misleading - there is NO demo mode

**Root Cause Analysis:**
The `GolfSwingModel.__init__()` in `core.py` lines 69-74:

```python
if model_path is None:
    raise ValueError(
        "model_path is required. OpenSim requires a valid .osim model file.\n"
        ...
    )
```

This follows the "Deterministic Error over Silent Fallback" principle but prevents getting started.

**Solution: Implement "Model Builder" Mode**

```python
class OpenSimGolfGUI(QMainWindow):
    def __init__(self, model_path: str | None = None) -> None:
        # Allow launching without model - show builder/setup UI
        if model_path is None:
            self._show_builder_mode()  # New method
        else:
            self._try_load_model()

    def _show_builder_mode(self) -> None:
        """Show model builder/setup interface when no model provided."""
        self._update_status("No Model - Builder Mode", "blue")
        self.btn_run.setEnabled(False)
        self.lbl_details.setText(
            "Welcome to OpenSim Golf!\n\n"
            "Options:\n"
            "1. Click 'Load Model' to open an existing .osim file\n"
            "2. Use URDF Generator to create a new model\n"
            "3. See 'Getting Started' for walkthrough\n"
        )
        # Add Getting Started button
        self.btn_getting_started = QPushButton("Getting Started Guide")
        self.btn_getting_started.clicked.connect(self._show_walkthrough)
```

**Implementation Steps:**

1. Modify `opensim_gui.py` to support model-free launch
2. Add "Getting Started Guide" dialog with step-by-step instructions
3. Update status badge from "Demo / GUI" to "Builder Ready" (green when OpenSim installed) or "Needs Setup" (orange)
4. Mirror the MyoSuite pattern for walkthrough documentation

---

### Priority 2: Launcher Startup Optimization 🚀

**Current Issue:** The launcher imports heavy modules at startup

**Analysis of `launchers/golf_launcher.py` lines 62-68:**

```python
from shared.python.engine_manager import EngineManager, EngineType  # Heavy
from shared.python.model_registry import ModelRegistry              # Heavy
from shared.python.secure_subprocess import (...)                    # Light
```

**Measured Cold Imports (Approximate):**
| Module | Import Time | Notes |
|--------|-------------|-------|
| PyQt6 | ~800ms | Required for GUI |
| matplotlib | ~400ms | Used for splash screen |
| EngineManager | ~300ms | Probes all engines |
| ModelRegistry | ~100ms | Loads model configs |
| AI Assistant | ~200ms | Optional |
| **Total** | ~1800ms | Cold start |

**Optimization Strategies:**

#### A. Lazy Loading Pattern (Quick Win - 30% faster)

```python
# At top of file - light imports only
from PyQt6.QtWidgets import QApplication  # Minimal Qt

# Defer heavy imports until needed
_engine_manager: EngineManager | None = None

def get_engine_manager() -> EngineManager:
    global _engine_manager
    if _engine_manager is None:
        from shared.python.engine_manager import EngineManager
        _engine_manager = EngineManager()
    return _engine_manager
```

#### B. Splash Screen During Load (Better UX)

Already implemented in `GolfSplashScreen` - but load phases not granular:

```python
splash.show_message("Loading Qt...", 10)
# Import Qt here
splash.show_message("Loading Engine Registry...", 30)
# Import EngineManager here
splash.show_message("Probing Physics Engines...", 50)
# Probe engines here
splash.show_message("Loading Model Library...", 70)
# Import ModelRegistry here
splash.show_message("Initializing UI...", 90)
# Build main window
splash.show_message("Ready!", 100)
```

#### C. Parallel Engine Probing (Advanced - 50% faster)

```python
import concurrent.futures

def probe_engines_parallel() -> dict[EngineType, ProbeResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(probe_mujoco): EngineType.MUJOCO,
            executor.submit(probe_drake): EngineType.DRAKE,
            executor.submit(probe_pinocchio): EngineType.PINOCCHIO,
            executor.submit(probe_opensim): EngineType.OPENSIM,
            executor.submit(probe_myosuite): EngineType.MYOSUITE,
        }
        results = {}
        for future in concurrent.futures.as_completed(futures):
            engine = futures[future]
            results[engine] = future.result()
    return results
```

**Recommended Implementation Order:**

1. **Immediate:** Add granular splash screen messages (1 hour)
2. **Short-term:** Implement lazy loading for EngineManager (2 hours)
3. **Long-term:** Parallel engine probing (4 hours)

---

### Priority 3: OpenSim/MyoSuite Walkthrough Documentation 📚

**Current State:**

- MyoSuite has `MYOSUITE_COMPLETE_SUMMARY.md` - excellent technical docs
- OpenSim has `OPENSIM_COMPLETE_SUMMARY.md` - comprehensive but technical
- **Neither has a user-facing "Getting Started" guide**

**Needed:**
| Engine | Getting Started Guide | Status |
|--------|----------------------|--------|
| MuJoCo | Inline tooltips | ⚠️ Basic |
| Drake | None | ❌ Missing |
| Pinocchio | None | ❌ Missing |
| **OpenSim** | None | ❌ **HIGH PRIORITY** |
| **MyoSuite** | None | ❌ **HIGH PRIORITY** |

**Solution: Create `launchers/assets/help/` directory with guides:**

```markdown
# OpenSim Golf - Getting Started

## Prerequisites

- OpenSim 4.x installed (`conda install -c opensim-org opensim`)
- OR use alternative engines (MuJoCo, Pinocchio)

## Quick Start

### Step 1: Verify Installation

Click "Check OpenSim Status" to verify OpenSim is installed.

### Step 2: Load or Create a Model

**Option A: Load existing model**

1. Click "Load Model"
2. Select a .osim file (samples in `shared/models/opensim/`)

**Option B: Create new model**

1. Launch "URDF Generator" from main launcher
2. Design your golf swing model
3. Export as URDF → convert to .osim

### Step 3: Run Simulation

...
```

---

### Priority 4: Fix "Demo / GUI" Status Badge ⚠️

**Current Code (lines 415-416):**

```python
elif t in ["opensim", "myosim"]:
    return "Demo / GUI", "#fd7e14"  # Orange
```

**Problem:**

- Says "Demo" but there is NO demo mode (per Scientific Software Integrity standards)
- Confuses users about what's available

**Solution:** Replace with accurate status based on engine availability:

```python
elif t in ["opensim", "myosim"]:
    # Check actual availability
    try:
        if t == "opensim":
            import opensim  # noqa: F401
            return "Engine Ready", "#28a745"  # Green
        else:  # myosim
            import myosuite  # noqa: F401
            return "Engine Ready", "#28a745"  # Green
    except ImportError:
        return "Needs Setup", "#fd7e14"  # Orange - accurate status
```

---

## ASSESSMENT REMEDIATION PRIORITIES

### From Assessment A (Architecture) - Remaining Gaps

| ID    | Issue                              | Impact | Effort  | Priority      |
| ----- | ---------------------------------- | ------ | ------- | ------------- |
| A-001 | Cross-engine validator not in CI   | High   | 2 days  | **IMMEDIATE** |
| A-002 | Physics engines excluded from MyPy | Medium | 2 weeks | Short-term    |
| A-006 | Print statements vs structlog      | Low    | 1 week  | Low           |

### From Assessment B (Scientific) - Largely Addressed ✅

| ID    | Issue               | Status                         |
| ----- | ------------------- | ------------------------------ |
| B-001 | Centripetal physics | ✅ Fixed (NotImplementedError) |
| B-002 | RNE validation      | ✅ Verified                    |
| B-003 | Frame documentation | ✅ 143-line docstring          |
| B-004 | Magic numbers       | ✅ 13 constants documented     |
| B-005 | Energy conservation | ✅ Tests added                 |

### From Assessment C (Cross-Engine) - Needs Attention

| ID    | Issue                                | Impact      | Effort  | Priority      |
| ----- | ------------------------------------ | ----------- | ------- | ------------- |
| C-001 | Cross-engine validator not automated | **BLOCKER** | 2 days  | **IMMEDIATE** |
| C-003 | No Feature × Engine Matrix           | High        | 4 hours | High          |
| C-006 | Need more cross-engine tests         | Medium      | 1 week  | Short-term    |

---

## IMPLEMENTATION TIMELINE

### Week 1: User Experience & Launcher (This Week)

| Day | Task                                  | Owner     |
| --- | ------------------------------------- | --------- |
| 1   | Fix OpenSim model-free launch         | Developer |
| 1   | Fix "Demo / GUI" badge                | Developer |
| 2   | Add granular splash messages          | Developer |
| 2   | Implement lazy loading                | Developer |
| 3   | Create OpenSim Getting Started guide  | Developer |
| 3   | Create MyoSuite Getting Started guide | Developer |
| 4   | Add walkthrough dialog to GUIs        | Developer |
| 5   | Test & verify all changes             | QA        |

### Week 2: CI/CD Hardening

| Day | Task                                     | Owner   |
| --- | ---------------------------------------- | ------- |
| 1-2 | Integrate cross_engine_validator into CI | DevOps  |
| 3   | Create Feature × Engine Matrix           | Docs    |
| 4-5 | Add Pinocchio cross-validation tests     | Physics |

### Week 3+: Long-term

- Remove physics engine MyPy exclusions
- Migrate print → structlog
- Parallel engine probing

---

## FILES TO MODIFY

### Immediate Changes (Priority 1-4)

1. **`engines/physics_engines/opensim/python/opensim_gui.py`**
   - Add `_show_builder_mode()` method
   - Add "Getting Started" button and dialog
   - Make model_path optional in constructor

2. **`engines/physics_engines/opensim/python/opensim_golf/core.py`**
   - Make GolfSwingModel more permissive about None path
   - Add builder/inspection mode

3. **`launchers/golf_launcher.py`**
   - Fix `_get_status_info()` to check actual availability
   - Add lazy loading for heavy imports
   - Enhance splash screen with granular progress

4. **`launchers/assets/help/opensim_getting_started.md`** (NEW)
   - Step-by-step user guide

5. **`launchers/assets/help/myosuite_getting_started.md`** (NEW)
   - Step-by-step user guide

---

## SUCCESS CRITERIA

### For Priority 1 (OpenSim Model-Free Launch) ✅ COMPLETE

- [x] OpenSim GUI launches without model path argument
- [x] Shows helpful "Getting Started" options
- [x] "Load Model" button still works
- [x] No demo/synthetic data shown

### For Priority 2 (Startup Optimization) ✅ COMPLETE

- [x] Splash screen shows meaningful progress (granular phases)
- [x] Actual work during splash (model registry, engine probing)
- [x] No UI freeze during load (QApplication.processEvents)
- [ ] Parallel engine probing (long-term enhancement)

### For Priority 3 (Walkthrough Documentation) ✅ COMPLETE

- [x] OpenSim guide exists and is accessible from GUI
- [x] MyoSuite guide exists and is accessible from GUI
- [x] Guides are beginner-friendly with installation instructions

### For Priority 4 (Status Badge) ✅ COMPLETE

- [x] No more "Demo / GUI" text
- [x] Accurate status based on engine availability
- [x] Clear path to setup if not available

### For Assessment C Remediation ✅ COMPLETE

- [x] Cross-engine validator unit tests in CI (C-001)
- [x] Feature × Engine Matrix documented (C-004)
- [x] Tests run on every PR

---

## APPENDIX: Status Badge Color Scheme

| Status       | Color  | Hex     | Meaning                 |
| ------------ | ------ | ------- | ----------------------- |
| GUI Ready    | Green  | #28a745 | Full functionality      |
| Engine Ready | Green  | #28a745 | Engine installed, ready |
| Viewer       | Blue   | #17a2b8 | View-only mode          |
| Builder      | Blue   | #17a2b8 | Can create models       |
| Needs Setup  | Orange | #fd7e14 | Dependency missing      |
| External     | Purple | #6f42c1 | Opens external app      |
| Utility      | Gray   | #6c757d | Helper tool             |
| Unknown      | Gray   | #6c757d | Can't determine         |

---

**Document Author:** Senior Principal Engineer  
**Review Date:** 2026-01-09  
**Next Review:** 2026-01-16

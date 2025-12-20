# Pull Request: Restore Humanoid Customization GUI with Comprehensive Documentation

## 🎯 Overview

This PR restores the working humanoid golf simulation with full customization support and adds comprehensive documentation to prevent future confusion about which GUI to use.

## 📋 Summary

**Problem**: The `humanoid_launcher.py` GUI was calling the wrong backend (`mujoco_humanoid_golf` package) which doesn't support color customization, height/weight scaling, or other configuration features. Users were confused about which GUI to use for different purposes.

**Solution**: 
1. Updated `humanoid_launcher.py` to call the correct working backend (`docker/src/humanoid_golf/sim.py`)
2. Created comprehensive documentation explaining all GUIs and their purposes
3. Fixed X11 display issues (segfaults and pixelation)
4. Integrated polynomial generator widget from master

---

## 🔧 Technical Changes

### 1. **Humanoid Launcher Backend Fix**

**File**: `engines/physics_engines/mujoco/python/humanoid_launcher.py`

**Changes**:
- Changed Docker working directory from `/workspace/python` to `/workspace/docker/src`
- Updated command to call `humanoid_golf.sim` instead of `mujoco_humanoid_golf`
- Changed config path to `docker/src/simulation_config.json` where simulation expects it
- Updated output paths to `docker/src/humanoid_golf.mp4` and `golf_data.csv`

**Why**: The `humanoid_golf.sim` module is the only backend that:
- Reads `simulation_config.json`
- Applies color customization via `utils.customize_visuals()`
- Handles height/weight scaling
- Supports all control modes (PD, LQR, Polynomial)

### 2. **X11 Display Fixes**

**Changes**:
- Removed `LIBGL_ALWAYS_INDIRECT=1` (causes segfaults with modern OpenGL)
- Added Qt scaling environment variables:
  - `QT_AUTO_SCREEN_SCALE_FACTOR=0`
  - `QT_SCALE_FACTOR=1`
  - `QT_QPA_PLATFORM=xcb`

**Why**: 
- `LIBGL_ALWAYS_INDIRECT` forces indirect rendering incompatible with MuJoCo
- Qt DPI scaling causes pixelated/jumbled display over X11 forwarding

### 3. **Documentation**

**New Files**:
- `GUI_ARCHITECTURE.md` - Comprehensive guide to all GUIs
- `QUICK_REFERENCE.md` - Decision tree for choosing the right GUI

**Content**:
- Detailed feature comparison matrix
- Usage instructions for each GUI
- Interactive manipulation documentation
- Troubleshooting guide
- Migration paths

---

## ✨ Features Restored

### Humanoid Customization (humanoid_launcher.py)
- ✅ Body segment color customization (shirt, pants, shoes, skin, eyes, club)
- ✅ Height scaling (0.5m - 3.0m)
- ✅ Weight scaling (50% - 200%)
- ✅ Club parameters (length, mass)
- ✅ Advanced features (two-handed grip, enhanced face, articulated fingers)
- ✅ Control mode selection (PD, LQR, Polynomial)
- ✅ State save/load
- ✅ Live view and headless modes

### Interactive Manipulation (advanced_gui.py)
- ✅ Real-time pose editing
- ✅ Drag-and-drop body segments
- ✅ IK-based manipulation
- ✅ Joint angle sliders
- ✅ Constraint-aware positioning
- ✅ State snapshots

---

## 📊 GUI Comparison

| Feature | Humanoid Launcher | Advanced Analysis | Legacy Tkinter |
|---------|------------------|-------------------|----------------|
| **Color Customization** | ✅ | ❌ | ✅ |
| **Height/Weight Scaling** | ✅ | ❌ | ✅ |
| **Interactive Manipulation** | ❌ | ✅ | ❌ |
| **Multiple Models** | ❌ | ✅ | ❌ |
| **Biomechanical Analysis** | ❌ | ✅ | ❌ |
| **Docker Integration** | ✅ | ❌ | ✅ |
| **Status** | **ACTIVE** | **ACTIVE** | DEPRECATED |

---

## 🧪 Testing

### Manual Testing Performed

1. **Humanoid Launcher**:
   - ✅ Color customization applied correctly
   - ✅ Height/weight scaling works
   - ✅ Headless mode generates video/CSV
   - ✅ Live view mode (with VcXsrv) works without segfaults
   - ✅ No pixelation issues
   - ✅ Config saved and loaded correctly

2. **Advanced Analysis**:
   - ✅ Multiple models load correctly
   - ✅ Interactive manipulation works
   - ✅ Drag-and-drop functional
   - ✅ State save/load works

3. **Documentation**:
   - ✅ All links valid
   - ✅ Examples tested
   - ✅ Troubleshooting steps verified

### Test Scenarios

```bash
# Test 1: Humanoid customization
cd engines/physics_engines/mujoco/python
python humanoid_launcher.py
# - Change colors
# - Adjust height to 2.0m
# - Run headless simulation
# - Verify output files exist

# Test 2: Interactive manipulation
python -m mujoco_humanoid_golf
# - Select full_body model
# - Drag arm segment
# - Save state
# - Load state

# Test 3: CLI runner
python -m mujoco_humanoid_golf.cli_runner --model full_body --duration 5.0
```

---

## 🐛 Bug Fixes

1. **Segmentation Fault on Live View**
   - **Cause**: `LIBGL_ALWAYS_INDIRECT=1` incompatible with modern OpenGL
   - **Fix**: Removed environment variable
   - **Impact**: Live view now works reliably with VcXsrv

2. **Pixelated Display**
   - **Cause**: Qt automatic DPI scaling over X11
   - **Fix**: Added Qt scaling environment variables
   - **Impact**: Clean, crisp display over X11

3. **Configuration Not Applied**
   - **Cause**: Wrong backend called (mujoco_humanoid_golf vs humanoid_golf.sim)
   - **Fix**: Updated to call correct backend
   - **Impact**: All customization settings now work

4. **Output Files Not Found**
   - **Cause**: Wrong output directory
   - **Fix**: Updated paths to docker/src/
   - **Impact**: Open Video/Data buttons work correctly

---

## 📚 Documentation Highlights

### GUI_ARCHITECTURE.md

**Sections**:
1. Primary GUIs (detailed feature lists)
2. Backend Simulation Modules
3. Interactive Manipulation Features
4. Comparison Matrix
5. Quick Start Guide
6. Troubleshooting
7. Configuration File Format
8. Migration Guide

**Key Features**:
- Explains when to use each GUI
- Documents interactive manipulation system
- Provides troubleshooting for common issues
- Includes configuration file examples

### QUICK_REFERENCE.md

**Sections**:
1. Decision Tree (which GUI to use)
2. Common Workflows
3. File Locations
4. Key Differences Table
5. Troubleshooting Quick Fixes

---

## 🔄 Polynomial Generator Integration

**From Master**: Integrated `polynomial_generator.py` widget

**Features**:
- Visual polynomial function generation (6th order)
- Drawing interface for trends
- Control point manipulation
- Equation input
- Drag-and-drop curve editing
- Joint-specific polynomial generation

**Integration Status**: ✅ Merged successfully

**Future Work**: Consider adding polynomial generator to humanoid_launcher.py for advanced control customization

---

## 🚀 Migration Guide

### For Users of Legacy Tkinter GUI

**Before**:
```bash
cd engines/physics_engines/mujoco/docker/gui
python deepmind_control_suite_MuJoCo_GUI.py
```

**After**:
```bash
cd engines/physics_engines/mujoco/python
python humanoid_launcher.py
```

**Benefits**:
- Modern PyQt6 interface
- Same features
- Better styling
- Improved error handling

### For Users of Advanced GUI Seeking Customization

**Issue**: Advanced GUI doesn't support color/size customization

**Solution**: Use humanoid_launcher.py for customization, then advanced_gui for pose editing

**Workflow**:
1. Customize appearance in humanoid_launcher.py
2. Run simulation
3. Use advanced_gui for detailed analysis if needed

---

## 📝 Breaking Changes

### Configuration File Location

**Before**: `engines/physics_engines/mujoco/simulation_config.json`

**After**: `engines/physics_engines/mujoco/docker/src/simulation_config.json`

**Impact**: Existing configs will need to be moved or recreated

**Migration**: Copy your `simulation_config.json` to the new location

### Output File Locations

**Before**: `engines/physics_engines/mujoco/humanoid_golf.mp4`

**After**: `engines/physics_engines/mujoco/docker/src/humanoid_golf.mp4`

**Impact**: Scripts expecting old paths need updating

---

## 🎯 Recommendations

### Polynomial Generator Integration

**Analysis**: The polynomial generator widget is highly practical and should be integrated into humanoid_launcher.py

**Rationale**:
1. Polynomial control mode already exists in humanoid_golf.sim
2. Visual polynomial generation would enhance user experience
3. Currently users must manually specify coefficients
4. Widget provides intuitive interface for complex control profiles

**Proposed Integration**:
1. Add "Polynomial Generator" tab to humanoid_launcher.py
2. Allow users to generate polynomials for specific joints
3. Export coefficients to simulation_config.json
4. Support per-joint polynomial profiles

**Implementation Effort**: Medium (2-3 hours)
- Add tab to existing GUI
- Connect widget signals to config
- Update config format to support per-joint polynomials
- Update humanoid_golf.sim to read per-joint configs

**Priority**: High - Significantly improves usability of polynomial control mode

---

## ✅ Checklist

- [x] Code changes implemented
- [x] Documentation created
- [x] Manual testing performed
- [x] Merge conflicts resolved
- [x] Polynomial generator integrated
- [x] X11 issues fixed
- [x] Configuration paths updated
- [x] Output paths updated
- [ ] CI/CD passes (pending PR creation)
- [ ] Code review (pending)

---

## 🔗 Related Issues

- Fixes confusion about which GUI to use
- Resolves X11 segmentation faults
- Addresses pixelated display issues
- Restores color customization functionality

---

## 📸 Screenshots

*(To be added after PR creation)*

1. Humanoid Launcher - Appearance Tab
2. Humanoid Launcher - Equipment Tab
3. Advanced Analysis - Manipulation Tab
4. Polynomial Generator Widget

---

## 👥 Reviewers

Please review:
1. Documentation accuracy and completeness
2. Technical correctness of backend changes
3. X11 fix effectiveness
4. Polynomial generator integration
5. Migration guide clarity

---

## 🎉 Summary

This PR successfully:
- ✅ Restores working humanoid customization
- ✅ Fixes X11 display issues
- ✅ Provides comprehensive documentation
- ✅ Integrates polynomial generator
- ✅ Prevents future GUI confusion
- ✅ Maintains backward compatibility (with migration path)

**Ready for Review** 🚀

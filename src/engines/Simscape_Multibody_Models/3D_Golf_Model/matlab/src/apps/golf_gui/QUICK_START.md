# Quick Start Guide

## 🆕 NEW: Tabbed Application (Recommended)

### How to Launch

```matlab
% From MATLAB command window, run:
launch_tabbed_app
```

Or manually:

```matlab
addpath(genpath('matlab/Scripts/Golf_GUI/Integrated_Analysis_App'));
main_golf_analysis_app();
```

### What You Get

- **Tab 1:** Model Setup (placeholder - future)
- **Tab 2:** ZTCF Calculation (placeholder - future)
- **Tab 3:** Visualization (FULLY FUNCTIONAL ✓)

### Using Tab 3 (Visualization)

**Default data loads automatically!** Just go to Tab 3 and the visualization appears.

To load your own data:
1. **Load 3 Files...** - Select BASEQ, ZTCFQ, DELTAQ separately (recommended)
2. **Load Combined...** - Select single MAT file with all datasets
3. **Load from Tab 2** - Use ZTCF calculation results
4. **Reload Defaults** - Reload the example data

Controls are on the left: Play, Stop, Speed, Display options.

---

## 🔧 Old Standalone Version

### If you want the standalone (non-tabbed) version

```matlab
% Load your data first
load('your_data_file.mat'); % Must have BASEQ, ZTCFQ, DELTAQ

% Create datasets structure
datasets = struct('BASEQ', BASEQ, 'ZTCFQ', ZTCFQ, 'DELTAQ', DELTAQ);

% Launch
addpath('matlab/Scripts/Golf_GUI/2D GUI/visualization');
SkeletonPlotter(datasets);
```

---

## 🐛 Troubleshooting

### Stuck Figures Won't Close

```matlab
% Force close all figures
close all force;
```

### Application Won't Launch

```matlab
% Reset paths and try again
restoredefaultpath;
launch_tabbed_app;
```

### "Function not found" errors

Make sure you're in the correct directory:

```matlab
cd('C:\Users\diete\Repositories\Golf_Model\matlab\Scripts\Golf_GUI');
launch_tabbed_app;
```

---

## 📚 Documentation

- **Full Guide:** `Integrated_Analysis_App/README.md`
- **Implementation Details:** `docs/TABBED_GUI_PHASE2_COMPLETE.md`
- **Signal Plotter Guide:** `docs/INTERACTIVE_SIGNAL_PLOTTER_GUIDE.md`

---

## ✅ Key Differences

| Feature | Old (Standalone) | New (Tabbed) |
|---------|-----------------|--------------|
| Interface | Single window | 3 tabs |
| Data loading | Manual script | UI buttons |
| Session save | No | Yes ✓ |
| Config persist | No | Yes ✓ |
| Future-proof | No | Yes ✓ |

**Recommendation:** Use the new tabbed version!

# Golf GUI Versions Guide

## 📁 Current Active Versions

### 1. 🆕 **NEW: Tabbed Application** (Recommended)

**Location:** `Integrated_Analysis_App/`

**Main Entry Point:** `main_golf_analysis_app.m`

**Launch:**

```matlab
launch_tabbed_app  % Easy launcher
```

**Features:**

- ✅ 3-tab interface
- ✅ Tab 3: Visualization (FULLY FUNCTIONAL)
- ⏳ Tab 1: Model Setup (placeholder)
- ⏳ Tab 2: ZTCF Calculation (placeholder)
- ✅ Session save/load
- ✅ Configuration persistence
- ✅ Menu system

**Status:** **Phase 2 Complete** - Tab 3 ready to use!

---

### 2. 📊 **OLD: Standalone Visualization** (Legacy)

**Location:** `2D GUI/visualization/`

**Main Entry Points:**

- `SkeletonPlotter.m` - 3D skeleton visualization
- `InteractiveSignalPlotter.m` - Signal plotting window
- `test_interactive_signal_plotter.m` - Test script

**Launch:**

```matlab
% Manual launch (requires data already loaded)
datasets = struct('BASEQ', BASEQ, 'ZTCFQ', ZTCFQ, 'DELTAQ', DELTAQ);
SkeletonPlotter(datasets);
```

**Features:**

- ✅ 3D skeleton visualization
- ✅ Interactive signal plotter
- ❌ No tabs
- ❌ No session save
- ❌ Manual data loading

**Status:** **Standalone/Legacy** - Works but not recommended for new work

---

## 📂 Other Versions/Archives

### 3. 🗃️ **Simscape Multibody Data Plotters**

**Location:** `Simscape Multibody Data Plotters/`

**What it is:** Older/alternative implementations

Contains:

- `Matlab Versions/SkeletonPlotter/` - Original skeleton plotter
- `Python Version/` - Python-based GUI attempts

**Status:** **Archive** - Historical versions

---

### 4. 🎥 **Motion Capture Plotter**

**Location:** `Motion Capture Plotter/`

**What it is:** Python-based motion capture visualization

Contains:

- PyQt6-based GUI
- Motion capture data analysis
- Coordinate system analysis

**Status:** **Separate Project** - Different purpose (motion capture, not simulation)

---

## 🗂️ Directory Structure

```
matlab/Scripts/Golf_GUI/
│
├── Integrated_Analysis_App/          ⭐ NEW TABBED VERSION
│   ├── main_golf_analysis_app.m
│   ├── tab1_model_setup.m
│   ├── tab2_ztcf_calculation.m
│   ├── tab3_visualization.m
│   ├── test_tabbed_app.m
│   ├── README.md
│   └── utils/
│       ├── data_manager.m
│       └── config_manager.m
│
├── 2D GUI/                           📊 OLD STANDALONE VERSION
│   └── visualization/
│       ├── SkeletonPlotter.m
│       ├── InteractiveSignalPlotter.m
│       ├── SignalDataInspector.m
│       ├── SignalPlotConfig.m
│       └── test_interactive_signal_plotter.m
│
├── Simscape Multibody Data Plotters/ 🗃️ ARCHIVE
│   ├── Matlab Versions/
│   └── Python Version/
│
├── Motion Capture Plotter/           🎥 SEPARATE PROJECT
│   └── Motion_Capture_Plotter.py
│
├── launch_tabbed_app.m               ⭐ EASY LAUNCHER
├── QUICK_START.md                    📖 QUICK GUIDE
└── VERSION_GUIDE.md                  📋 THIS FILE
```

---

## 🎯 Which Version Should I Use?

### Use the **NEW Tabbed Version** if

- ✅ You want the latest features
- ✅ You want an integrated workflow
- ✅ You need session management
- ✅ You want future updates (Tab 1 & 2 coming)

### Use the **OLD Standalone Version** if

- 🔧 You have existing scripts that use it
- 🔧 You only need basic visualization
- 🔧 You're working with legacy code

### ⚠️ Don't Use

- ❌ Anything in `Simscape Multibody Data Plotters/` - outdated
- ❌ Anything in archive folders

---

## 🚀 Recommended Launch Method

```matlab
% Navigate to the Golf_GUI folder
cd('<owner-home>\Repositories\Golf_Model\matlab\Scripts\Golf_GUI')

% Launch the new tabbed application
launch_tabbed_app
```

This will:

1. Close any stuck figures
2. Set up paths automatically
3. Launch the tabbed GUI
4. Take you to Tab 3 where you can load data

---

## 🔄 Relationship Between Versions

```
┌─────────────────────────────────────────────┐
│  NEW: Integrated_Analysis_App               │
│  ┌─────────────────────────────────────┐   │
│  │ Tab 3 wraps and uses:               │   │
│  │  ↓                                  │   │
│  │  OLD: 2D GUI/visualization/         │   │
│  │     - SkeletonPlotter.m     ←──────┼───┼─ Reused!
│  │     - InteractiveSignalPlotter.m   │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Key Point:** The NEW tabbed version **uses** the OLD visualization code internally! It's a wrapper that adds:

- Tabs
- Data management
- Session persistence
- Menu system

---

## 📊 Version Count Summary

**Active Versions:** 2

1. ⭐ **NEW Tabbed** (`Integrated_Analysis_App/`)
2. 📊 **OLD Standalone** (`2D GUI/visualization/`)

**Archived/Other:** 2+ 3. 🗃️ **Archive** (`Simscape Multibody Data Plotters/`) 4. 🎥 **Motion Capture** (`Motion Capture Plotter/`)

**Total GUI implementations in this folder:** **2 active, 2+ archived**

---

## 🔧 Migration Path

If you're using the old standalone version:

1. Your data format stays the same (BASEQ, ZTCFQ, DELTAQ)
2. All visualization features are preserved
3. You gain additional features (tabs, session management)
4. Switch is seamless - just use the new launcher

**No code changes needed to your data files!**

---

## 📞 Quick Reference

| What do you want?   | Use this                          | Location                   |
| ------------------- | --------------------------------- | -------------------------- |
| Latest & greatest   | `launch_tabbed_app`               | `Integrated_Analysis_App/` |
| Quick visualization | `SkeletonPlotter(datasets)`       | `2D GUI/visualization/`    |
| Test tabbed app     | `test_tabbed_app`                 | `Integrated_Analysis_App/` |
| Test standalone     | `test_interactive_signal_plotter` | `2D GUI/visualization/`    |

---

**Last Updated:** October 28, 2025
**Current Branch:** `feature/tabbed-gui`
**Status:** Phase 2 Complete ✅

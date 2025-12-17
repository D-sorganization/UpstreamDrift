# Integrated Golf Analysis Application

## Overview

This is a comprehensive three-tab GUI application for golf swing analysis, integrating model setup, ZTCF calculation, and visualization into a unified interface.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           Golf Swing Analysis Application                   │
│                                                              │
│  ┌────────────┬────────────────┬─────────────────────────┐ │
│  │  Tab 1:    │   Tab 2:       │   Tab 3:                │ │
│  │  Model     │   ZTCF         │   Analysis &            │ │
│  │  Setup     │   Calculation  │   Visualization         │ │
│  └────────────┴────────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Launch the Application

```matlab
% Add the application to your path
addpath(genpath('matlab/Scripts/Golf_GUI/Integrated_Analysis_App'));

% Launch the application
app_handles = main_golf_analysis_app();
```

### Run Tests

```matlab
% Run the test script to verify installation
test_tabbed_app
```

## Current Implementation Status

### ✅ Phase 2: Tabbed Framework (COMPLETE)

- [x] Main application window with tab group
- [x] Three-tab structure
- [x] Data passing utilities (`data_manager`)
- [x] Configuration management (`config_manager`)
- [x] Session save/load functionality
- [x] Tab 3: Visualization (fully functional)

### ⏳ Phase 3: Tab 1 Implementation (PLANNED)

- [ ] Model parameter input UI
- [ ] Simscape simulation integration
- [ ] Live 3D animation
- [ ] Data export to Tab 2

### ⏳ Phase 4: Tab 2 Implementation (PLANNED)

- [ ] ZTCF calculation engine
- [ ] Parallel processing
- [ ] Progress monitoring
- [ ] Results export to Tab 3

## File Structure

```
Integrated_Analysis_App/
├── main_golf_analysis_app.m      # Main entry point
├── tab1_model_setup.m             # Tab 1: Model Setup (placeholder)
├── tab2_ztcf_calculation.m        # Tab 2: ZTCF (placeholder)
├── tab3_visualization.m           # Tab 3: Visualization (functional)
├── utils/
│   ├── data_manager.m             # Data passing between tabs
│   └── config_manager.m           # Configuration management
├── config/
│   └── golf_analysis_app_config.mat  # Saved configuration
├── test_tabbed_app.m              # Test script
└── README.md                      # This file
```

## Tab Descriptions

### Tab 1: Model Setup & Simulation

**Status:** Placeholder (Phase 3)

Will provide:

- Model parameter configuration
- Initial conditions setup
- Simscape Multibody simulation
- Live 3D animation preview
- Data export to Tab 2

### Tab 2: ZTCF Calculation

**Status:** Placeholder (Phase 4)

Will provide:

- ZTCF (Zero Torque Counterfactual) calculation
- Parallel processing with progress monitoring
- Generation of BASEQ, ZTCFQ, and DELTAQ datasets
- Data export to Tab 3
- Session checkpointing

### Tab 3: Analysis & Visualization

**Status:** Fully Functional ✅

Provides:

- **Auto-loads default data on startup** - visualization appears immediately!
- Load 3 separate files (BASEQ, ZTCFQ, DELTAQ)
- Load combined MAT file
- Load from Tab 2 calculations
- Embedded 3D skeleton visualization
- Playback controls (play, speed, frame slider)
- Display options (forces, torques, trail, club)
- Dataset switching (BASEQ, ZTCFQ, DELTAQ)

## Usage

### Tab 3: Visualization (Current)

**Automatic Startup:**
- Default data loads automatically when you open Tab 3
- Visualization appears immediately - no manual loading required!

**Loading Your Own Data:**

1. **Load 3 Files...** (Recommended for separate files)
   - Select BASEQ.mat
   - Select ZTCFQ.mat
   - Select DELTAQ.mat

2. **Load Combined...** (For single combined file)
   - Select MAT file containing all three datasets

3. **Load from Tab 2** (For calculated data)
   - Uses results from ZTCF Calculation tab

4. **Reload Defaults**
   - Reloads the example data from repository

**Interactive Controls:**
- **Play/Pause** - Animate the golf swing
- **Speed Slider** - Adjust playback speed (0.1x to 2x)
- **Frame Slider** - Scrub through frames manually
- **Dataset Dropdown** - Switch between BASEQ, ZTCFQ, DELTAQ views
- **Display Checkboxes** - Toggle forces, torques, trail, club visibility
- **Clear** - Remove current visualization

### Menu Options

- **File Menu:**
  - Load/Save Session: Restore or save your entire workspace
  - Exit: Close the application

- **Tools Menu:**
  - Clear All Data: Remove all data from memory
  - Reset Configuration: Restore default settings

- **Help Menu:**
  - About: Application information
  - Documentation: Open the implementation plan

## Data Management

### Data Manager

The `data_manager` class handles data passing between tabs:

```matlab
% Store ZTCF data (from Tab 2)
app_handles.data_manager.set_ztcf_data(ztcf_data);

% Retrieve ZTCF data (in Tab 3)
ztcf_data = app_handles.data_manager.get_ztcf_data();

% Check if data exists
if app_handles.data_manager.has_ztcf_data()
    % Data is available
end
```

### Configuration Manager

The `config_manager` class handles persistent settings:

```matlab
% Load configuration
config = app_handles.config_manager.load_config();

% Save configuration
app_handles.config_manager.save_config(config);

% Reset to defaults
app_handles.config_manager.reset_config();
```

## Session Management

Sessions can be saved and restored:

1. **Auto-save:** Configuration is automatically saved on exit
2. **Manual save:** File → Save Session
3. **Load session:** File → Load Session

Session files contain:

- Simulation data (if available)
- ZTCF calculation results (if available)
- Analysis state (if available)
- Window position and preferences

## Dependencies

- MATLAB R2019b or later (for `uitabgroup` support)
- Existing visualization tools:
  - `SkeletonPlotter.m`
  - `InteractiveSignalPlotter.m`
  - `SignalPlotConfig.m`
  - `SignalDataInspector.m`

## Troubleshooting

### Application Won't Launch

1. Ensure all paths are added:

   ```matlab
   addpath(genpath('matlab/Scripts/Golf_GUI/Integrated_Analysis_App'));
   ```

2. Check for missing dependencies:

   ```matlab
   which SkeletonPlotter
   which data_manager
   ```

### Tab 3 Can't Find SkeletonPlotter

The visualization path should be automatically added, but you can manually add it:

```matlab
addpath('matlab/Scripts/Golf_GUI/2D GUI/visualization');
```

### Data Not Loading

Ensure your data file contains the required fields:

- `BASEQ` (table)
- `ZTCFQ` (table)
- `DELTAQ` (table)

## Development

### Adding Features to Tab 1 or Tab 2

1. Edit the respective tab file (`tab1_model_setup.m` or `tab2_ztcf_calculation.m`)
2. Replace placeholder UI with functional controls
3. Implement callbacks for user interactions
4. Use `app_handles.data_manager` to pass data between tabs
5. Update the `refresh_callback` to handle data updates

### Extending the Framework

The modular design allows easy extension:

- Add new tabs by creating similar tab initialization functions
- Extend data_manager with new data types
- Add custom menu items in `create_app_menu()`
- Implement additional analysis tools as needed

## References

- Implementation Plan: `docs/TABBED_GUI_IMPLEMENTATION_PLAN.md`
- Signal Plotter Guide: `docs/INTERACTIVE_SIGNAL_PLOTTER_GUIDE.md`
- Original visualization: `matlab/Scripts/Golf_GUI/2D GUI/visualization/`

## Version History

- **v1.0 (Current):** Phase 2 complete - Tabbed framework with functional Tab 3
- **Future:** Phase 3 (Tab 1) and Phase 4 (Tab 2) implementations

## Support

For issues or questions, refer to the main project documentation or the implementation plan.

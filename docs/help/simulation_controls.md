# Simulation Controls

Learn how to run, control, and interact with physics simulations.

## Overview

The simulation controls allow you to run physics simulations, adjust parameters, and capture data for analysis.

## Starting a Simulation

### From the Launcher

1. **Select an engine** - Click a tile (MuJoCo, Drake, etc.)
2. **Configure options** - Set visualization, Docker mode if needed
3. **Launch** - Click the "Launch" button or double-click the tile

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter | Launch selected model |
| Ctrl+L | Quick launch dialog |
| Ctrl+R | Reset to initial state |

## Playback Controls

### Basic Controls

| Control | Keyboard | Description |
|---------|----------|-------------|
| Play/Pause | Space | Toggle simulation |
| Step Forward | Right Arrow | Advance one timestep |
| Step Backward | Left Arrow | Go back one timestep |
| Reset | R | Return to initial state |
| Stop | Escape | Stop and reset |

### Speed Control

Adjust playback speed to observe dynamics at different rates:

| Speed | Description |
|-------|-------------|
| 0.1x | Slow motion (detailed analysis) |
| 0.5x | Half speed |
| 1.0x | Real-time |
| 2.0x | Fast forward |
| Max | As fast as possible |

**Keyboard:** Use `+` and `-` to adjust speed.

## Simulation Parameters

### Timestep

The simulation timestep affects accuracy and speed:

- **Smaller (0.0001s):** More accurate, slower
- **Default (0.002s):** Good balance
- **Larger (0.01s):** Fast but less accurate

**Recommendation:** Use 0.001-0.002s for biomechanical simulations.

### Integrator

Different integration methods are available depending on engine:

| Integrator | Accuracy | Speed | Best For |
|------------|----------|-------|----------|
| Euler | Low | Fast | Quick tests |
| RK4 | High | Medium | General use |
| Implicit | High | Slow | Stiff systems |

### Physics Options

- **Gravity:** Enable/disable, adjust magnitude
- **Contact:** Enable/disable contact physics
- **Friction:** Adjust friction coefficients
- **Damping:** Set joint damping values

## State Control

### Initial Conditions

Set the starting state for simulation:

1. **Joint Positions (q):** Initial angles/positions
2. **Joint Velocities (v):** Initial angular/linear velocities
3. **Applied Torques (tau):** Initial control inputs

### Saving and Loading States

- **Ctrl+S:** Save current state
- **Ctrl+O:** Load saved state
- **Ctrl+Shift+S:** Save state with name

States are saved as JSON files that can be shared or version-controlled.

## Recording and Export

### Recording Data

Enable recording to capture simulation data:

1. Click "Record" before starting
2. Run simulation
3. Click "Stop Recording"
4. Export data

### Export Formats

| Format | Use Case |
|--------|----------|
| CSV | Spreadsheet analysis, MATLAB |
| JSON | Structured data, programming |
| MAT | MATLAB native format |

### Exported Data

Recordings include:
- Time series (t)
- Joint positions (q)
- Joint velocities (v)
- Applied torques (tau)
- Contact forces (if applicable)
- Energy values

### Quick Export

- **Ctrl+E:** Export current data
- **Ctrl+Shift+E:** Export with options dialog

## Control Inputs

### Manual Torque Control

Use sliders or input fields to apply torques:

1. Select joint from dropdown
2. Adjust torque value (-100 to +100 Nm typical)
3. Observe system response

### Trajectory Following

Load a pre-computed trajectory:

1. Import trajectory file (CSV or JSON)
2. Set interpolation method (linear, cubic)
3. Enable "Follow Trajectory" mode
4. Play simulation

### Control Modes

| Mode | Description |
|------|-------------|
| Manual | Direct torque control via sliders |
| Trajectory | Follow pre-defined motion |
| PD Control | Position/velocity feedback |
| Free | No control (passive dynamics) |

## Tips and Best Practices

### For Accurate Simulations

1. Start with small timestep (0.001s)
2. Use RK4 integrator
3. Verify energy conservation
4. Check against known solutions

### For Fast Iteration

1. Use larger timestep initially
2. Reduce visualization frequency
3. Enable "Fast Mode" (reduced rendering)
4. Use batch simulation for parameter sweeps

### For Data Analysis

1. Always record simulation data
2. Use consistent timestep across runs
3. Export to CSV for statistical analysis
4. Include metadata in exports

## Troubleshooting

### Simulation Explodes

**Symptoms:** Values go to infinity, NaN errors

**Solutions:**
- Reduce timestep
- Check for extreme initial conditions
- Enable joint limits
- Verify model parameters

### Slow Simulation

**Solutions:**
- Increase timestep (if accuracy allows)
- Disable unnecessary visualization
- Use simpler model
- Enable hardware acceleration (if available)

### Simulation Won't Start

**Check:**
- Engine is properly loaded
- Model file exists and is valid
- Initial state is within limits
- No dependency errors in console

---

*See also: [Full User Manual](../USER_MANUAL.md) | [Engine Selection](engine_selection.md) | [Visualization](visualization.md)*

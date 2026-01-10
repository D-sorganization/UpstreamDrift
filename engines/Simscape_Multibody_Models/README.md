# MATLAB Simscape Multibody Golf Models

MATLAB/Simulink Simscape Multibody models for golf swing biomechanical analysis, providing physics-based simulation and analysis capabilities through MATLAB's powerful modeling environment.

## Overview

The Simscape Multibody Golf Models provide comprehensive biomechanical simulation and analysis tools using MATLAB's Simscape Multibody framework. These models complement the Python-based physics engines (MuJoCo, Drake, Pinocchio) with MATLAB's extensive analysis and visualization capabilities.

**Key Features:**
- **2D and 3D Models**: Progressive complexity from simplified 2D to full 3D biomechanics
- **Interactive GUIs**: MATLAB App Designer-based visualization and analysis tools
- **Motion Capture Integration**: Load and analyze experimental motion data
- **Dataset Generation**: Batch simulation for parameter sweeps and optimization
- **Simscape Multibody**: Leverage MATLAB's proven multibody dynamics solver
- **MATLAB Ecosystem**: Integration with MATLAB's optimization, signal processing, and visualization tools

## Available Models

### 2D Golf Model
**Location**: `2D_Golf_Model/`

- **Simplified planar dynamics** for educational and rapid analysis
- **Faster computation** suitable for parameter studies
- **Key joints**: Shoulder, elbow, wrist in sagittal plane
- **Applications**: Teaching, initial concept validation, parameter sensitivity

**See**: [2D_Golf_Model/README.md](2D_Golf_Model/README.md)

### 3D Golf Model
**Location**: `3D_Golf_Model/`

- **Full 3D biomechanics** with realistic joint constraints
- **Complete body model**: Legs, pelvis, torso, arms, club
- **Motion capture support**: Import and analyze experimental data
- **Interactive GUI**: Real-time visualization and analysis
- **Applications**: Research, performance analysis, biomechanical studies

**See**: [3D_Golf_Model/README.md](3D_Golf_Model/README.md)

## Prerequisites

### Required Software
- **MATLAB** R2023a or later (R2023b+ recommended)
- **Simulink** (included with MATLAB)
- **Simscape Multibody** (Add-On Toolbox)

### Optional Toolboxes
- **Optimization Toolbox**: For parameter optimization
- **Signal Processing Toolbox**: For motion data filtering
- **Statistics and Machine Learning Toolbox**: For data analysis
- **Parallel Computing Toolbox**: For batch simulations

### Installation Verification

```matlab
% Check required toolboxes
ver('simulink')
ver('sm')  % Simscape Multibody

% Verify Simscape Multibody is installed
if license('test', 'Simscape_Multibody')
    disp('Simscape Multibody is available')
else
    error('Simscape Multibody not found. Install via Add-On Explorer.')
end
```

## Quick Start

### 1. Open MATLAB
```matlab
cd /path/to/Golf_Modeling_Suite/engines/Simscape_Multibody_Models
```

### 2. Choose Your Model

#### For 2D Model:
```matlab
cd 2D_Golf_Model/matlab
open_system('golf_swing_2d.slx')  % Open Simulink model
sim('golf_swing_2d')               % Run simulation
```

#### For 3D Model:
```matlab
cd 3D_Golf_Model/matlab
open_system('golf_swing_3d.slx')  % Open Simulink model

% Or launch the interactive GUI
run('Scripts/Golf_GUI/launch_golf_gui.m')
```

### 3. Run Analysis
```matlab
% 3D Model example - analyze a swing
cd 3D_Golf_Model/matlab/Scripts
analyze_swing('swing_data.mat')
plot_swing_results()
```

## Features Comparison

| Feature | 2D Model | 3D Model |
|---------|----------|----------|
| Degrees of Freedom | 3-5 | 10-15 |
| Computation Speed | Fast (real-time) | Moderate |
| Biomechanical Accuracy | Simplified | High |
| Motion Capture Support | Limited | Full |
| Interactive GUI | Basic | Advanced |
| Parameter Studies | Excellent | Good |
| Research Applications | Educational | Professional |

## Integration with Python Engines

The Simscape models can work alongside Python engines for comprehensive analysis:

### Workflow 1: MATLAB → Python
```matlab
% 1. Generate data in MATLAB
results = run_simscape_simulation(params);
save('simscape_results.mat', 'results');

% 2. Export to Python-readable format
writetable(results_table, 'results.csv');
```

```python
# 3. Analyze in Python with MuJoCo/Drake
import pandas as pd
results = pd.read_csv('results.csv')
validate_with_mujoco(results)
```

### Workflow 2: Python → MATLAB
```python
# 1. Generate optimal trajectory in Drake
trajectory = optimize_swing_drake(model)
trajectory.to_csv('optimal_trajectory.csv')
```

```matlab
% 2. Validate in Simscape
trajectory = readtable('optimal_trajectory.csv');
simscape_results = simulate_trajectory(trajectory);
compare_results(trajectory, simscape_results);
```

## Model Capabilities

### 2D Golf Model Capabilities
- Planar swing dynamics
- Club-ball impact modeling
- Ground reaction forces (simplified)
- Parameter sensitivity analysis
- Educational visualization
- Fast parameter sweeps (1000s of simulations)

### 3D Golf Model Capabilities
- Full 3D body kinematics and dynamics
- Realistic joint constraints and ranges
- Ground contact modeling (both feet)
- Club-ball impact with spin
- Motion capture data import (C3D, CSV)
- Interactive visualization
- Performance metrics calculation
- Dataset generation for machine learning

## Advanced Features

### Motion Capture Integration (3D Model)
```matlab
% Load motion capture data
mocap_data = load_c3d_file('swing_trial_01.c3d');

% Map markers to model
marker_mapping = create_marker_set();
joint_angles = mocap_to_joint_angles(mocap_data, marker_mapping);

% Simulate with captured motion
sim_results = simulate_from_mocap(joint_angles);
```

### Batch Simulation & Dataset Generation
```matlab
% Define parameter ranges
shoulder_range = linspace(-180, 90, 20);  % degrees
wrist_range = linspace(-90, 90, 20);

% Generate dataset
dataset = generate_swing_dataset(shoulder_range, wrist_range);
save('swing_dataset.mat', 'dataset');

% Analyze results
optimal_params = find_optimal_parameters(dataset);
```

### Optimization
```matlab
% Define objective function
objective = @(params) -club_head_speed(params);  % Maximize speed

% Set bounds
lb = [-180, -90, -45];  % Lower bounds [shoulder, elbow, wrist]
ub = [90, 90, 45];      % Upper bounds

% Optimize
options = optimoptions('fmincon', 'Display', 'iter');
optimal = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);
```

## Output and Visualization

### Available Outputs
- **Kinematics**: Joint angles, velocities, accelerations
- **Dynamics**: Joint torques, forces, power
- **Club Metrics**: Head speed, trajectory, face angle
- **Ball Metrics**: Launch angle, spin rate, distance
- **Energy**: Kinetic, potential, total
- **Performance**: Swing efficiency, power transfer

### Visualization Tools
- **3D Animation**: Real-time or playback of swing motion
- **Time Series Plots**: Joint angles, velocities, torques
- **Phase Diagrams**: State-space visualization
- **Stick Figures**: Simplified motion visualization
- **Force Vectors**: Ground reactions, contact forces

## File Formats

### Input Formats
- `.slx` - Simulink models
- `.mat` - MATLAB data files
- `.c3d` - Motion capture data
- `.csv` - Comma-separated values
- `.trc` - Trajectory files (OpenSim compatible)

### Output Formats
- `.mat` - MATLAB simulation results
- `.csv` - Exportable data for external analysis
- `.fig` - MATLAB figures
- `.avi`/`.mp4` - Animation videos
- `.xlsx` - Excel spreadsheets for reporting

## Performance Considerations

### Simulation Speed
- **2D Model**: 5-50× real-time (very fast)
- **3D Model**: 0.5-5× real-time (depends on complexity)
- **Batch Mode**: Optimized for throughput

### Memory Requirements
- **2D Model**: ~100 MB RAM
- **3D Model**: ~500 MB RAM
- **Large Datasets**: 2-4 GB RAM

### Tips for Faster Simulation
1. Use fixed-step solvers when possible
2. Reduce visualization during batch runs
3. Enable parallel computing for parameter sweeps
4. Simplify contact models for preliminary studies

## Best Practices

### When to Use Simscape Models

✅ **Use Simscape for**:
- MATLAB-native workflows
- Integration with MATLAB optimization/control tools
- Educational demonstrations
- Rapid GUI development with App Designer
- Compatibility with existing MATLAB code
- Signal processing and filtering of biomechanical data

❌ **Prefer Python Engines for**:
- Muscle-level biomechanics (use MuJoCo + MyoSuite)
- High-performance batch optimization (use Drake)
- Real-time interactive simulation (use MuJoCo)
- Integration with Python ML libraries

### Model Selection Guide

**Choose 2D Model when**:
- Teaching biomechanics concepts
- Need very fast computation
- Performing large parameter studies
- Planar assumptions are acceptable

**Choose 3D Model when**:
- Researching biomechanics
- Analyzing real motion capture data
- Need accurate 3D kinematics
- Publishing scientific results

## Documentation

### Model-Specific Documentation
- [2D Golf Model Documentation](2D_Golf_Model/README.md)
- [3D Golf Model Documentation](3D_Golf_Model/README.md)

### Related Documentation
- [Engine Selection Guide](../../docs/engine_selection_guide.md) - Compare with Python engines
- [User Guide](../../docs/user_guide/README.md) - General Golf Suite documentation
- [MATLAB Simscape Docs](https://www.mathworks.com/help/sm/) - Official Simscape documentation

## Troubleshooting

### Common Issues

**Issue**: Model won't open in older MATLAB versions
- **Solution**: Models require R2023a+. Upgrade MATLAB or use compatibility mode

**Issue**: Simscape Multibody not found
- **Solution**: Install via MATLAB Add-On Explorer (Home → Add-Ons → Get Add-Ons)

**Issue**: Simulation runs very slowly
- **Solution**: Reduce visualization frequency, use fixed-step solver, simplify contact

**Issue**: Motion capture data won't import
- **Solution**: Check file format (.c3d, .trc), verify marker names match model

## Examples

Example scripts are included in each model directory:

### 2D Model Examples
```matlab
cd 2D_Golf_Model/matlab/examples
run_basic_swing           % Simple swing simulation
run_parameter_sweep       % Vary parameters
run_optimization_example  % Find optimal parameters
```

### 3D Model Examples
```matlab
cd 3D_Golf_Model/matlab/examples
run_mocap_analysis        % Motion capture workflow
run_dataset_generation    % Generate ML dataset
run_gui_demo             % Interactive GUI demo
```

## Contributing

Contributions to Simscape models are welcome:
- Improve joint models or constraints
- Add new biomechanical features
- Enhance visualization
- Optimize performance
- Add analysis scripts

See [Contributing Guide](../../docs/development/contributing.md)

## Citation

If you use these models in research, please cite:

```bibtex
@software{golf_simscape_models,
  title = {MATLAB Simscape Multibody Golf Models},
  author = {[Authors]},
  year = {2026},
  note = {Part of Golf Modeling Suite},
  url = {https://github.com/D-sorganization/Golf_Modeling_Suite}
}
```

## License

MIT License - See [LICENSE](../../LICENSE) for details.

## Support

For Simscape-specific questions:
- Review model-specific README files
- Check MATLAB documentation: `doc simscape`
- See [Golf Modeling Suite Issues](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)
- MATLAB Central forums for Simscape questions

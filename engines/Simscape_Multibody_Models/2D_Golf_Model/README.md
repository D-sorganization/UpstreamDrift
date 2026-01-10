# 2D Golf Model - Simplified Planar Biomechanics

A MATLAB/Simulink Simscape Multibody model for 2D (planar) golf swing analysis, providing fast computation and intuitive visualization for educational and parametric studies.

## Overview

The 2D Golf Model simplifies the golf swing to planar motion, enabling rapid simulation and analysis. This model is ideal for:

- **Education**: Teaching biomechanics fundamentals
- **Parameter Studies**: Fast exploration of design space (1000s of simulations)
- **Concept Validation**: Quick testing of ideas before 3D modeling
- **Optimization**: Finding optimal parameters with minimal computation
- **Simplified Analysis**: When 3D complexity isn't necessary

**Simulation Speed**: 5-50× real-time (very fast!)

## Model Description

### Degrees of Freedom

The 2D model includes:
1. **Shoulder Flexion/Extension** - Arm elevation in sagittal plane
2. **Elbow Flexion/Extension** - Forearm rotation
3. **Wrist Flexion/Extension** - Club control
4. *Optional*: Hip angle, knee angle for more complete model

**Total**: 3-5 DOF depending on configuration

### Coordinate System
- **X-axis**: Forward (direction of ball flight)
- **Y-axis**: Upward (vertical)
- **Z-axis**: Out of plane (not modeled in 2D)

### Simplifications

Compared to 3D model:
- ❌ No out-of-plane motion
- ❌ No shoulder/wrist rotation
- ❌ Simplified ground contact
- ❌ No weight transfer
- ✅ Much faster computation
- ✅ Easier visualization
- ✅ Simpler parameter space

## Quick Start

### Prerequisites
- MATLAB R2023a+ with Simulink and Simscape Multibody
- See [parent README](../README.md) for installation details

### Opening the Model

```matlab
% Navigate to model directory
cd /path/to/Golf_Modeling_Suite/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab

% Open Simulink model
open_system('golf_swing_2d.slx')
```

### Running a Simulation

```matlab
% Set simulation parameters
set_param('golf_swing_2d', 'StopTime', '2.0')  % 2 second swing

% Run simulation
sim('golf_swing_2d')

% Plot results
plot_swing_results_2d()
```

### Using the GUI

```matlab
% Launch interactive GUI
run('Scripts/launch_2d_gui.m')

% Or use from MATLAB Apps tab (if installed)
```

## File Structure

```
2D_Golf_Model/
├── matlab/
│   ├── golf_swing_2d.slx          # Main Simulink model
│   ├── Scripts/
│   │   ├── run_simulation.m       # Basic simulation script
│   │   ├── parameter_sweep.m      # Batch simulations
│   │   ├── optimize_swing.m       # Optimization example
│   │   └── Integrated_Analysis_App/  # GUI application
│   ├── Functions/
│   │   ├── kinematics_2d.m        # Kinematic calculations
│   │   ├── dynamics_2d.m          # Dynamic calculations
│   │   └── plot_utilities_2d.m    # Plotting functions
│   └── Data/
│       └── parameters_2d.mat      # Default parameters
├── matlab_optimized/              # Performance-optimized version
├── docs/                          # Documentation
└── CI_Documentation/              # Continuous integration docs
```

## Model Parameters

### Body Segment Parameters

```matlab
% Upper arm
params.upper_arm.length = 0.3;     % m
params.upper_arm.mass = 2.0;       % kg
params.upper_arm.inertia = 0.02;   % kg⋅m²

% Forearm
params.forearm.length = 0.25;      % m
params.forearm.mass = 1.5;         % kg
params.forearm.inertia = 0.015;    % kg⋅m²

% Club
params.club.length = 1.2;          % m (driver)
params.club.mass = 0.3;            % kg
params.club.inertia = 0.04;        % kg⋅m²
```

### Joint Parameters

```matlab
% Joint limits (degrees)
params.shoulder.min = -180;
params.shoulder.max = 90;

params.elbow.min = 0;
params.elbow.max = 150;

params.wrist.min = -90;
params.wrist.max = 90;

% Joint damping
params.shoulder.damping = 0.5;     % N⋅m⋅s/rad
params.elbow.damping = 0.3;
params.wrist.damping = 0.2;
```

## Running Parameter Studies

### Single Parameter Sweep

```matlab
% Vary shoulder angle at impact
shoulder_angles = linspace(-45, 45, 50);  % degrees

results = zeros(length(shoulder_angles), 1);
for i = 1:length(shoulder_angles)
    params.shoulder_impact = shoulder_angles(i);
    result = sim_2d_golf(params);
    results(i) = result.club_head_speed;
end

% Plot results
plot(shoulder_angles, results)
xlabel('Shoulder Angle at Impact (deg)')
ylabel('Club Head Speed (m/s)')
```

### Multi-Parameter Sweep

```matlab
% Grid search over shoulder and wrist
[S, W] = meshgrid(linspace(-45, 45, 20), linspace(-30, 30, 20));

speed_map = zeros(size(S));
for i = 1:numel(S)
    params.shoulder_impact = S(i);
    params.wrist_impact = W(i);
    result = sim_2d_golf(params);
    speed_map(i) = result.club_head_speed;
end

% Visualize
surf(S, W, speed_map)
xlabel('Shoulder Angle (deg)')
ylabel('Wrist Angle (deg)')
zlabel('Club Head Speed (m/s)')
```

## Optimization

### Maximize Club Head Speed

```matlab
% Objective function
objective = @(x) -compute_club_speed_2d(x);  % Minimize negative = maximize

% Variables: [shoulder, elbow, wrist] timing offsets
x0 = [0, 0.1, 0.2];  % Initial guess (seconds)

% Bounds
lb = [-0.2, 0, 0];   % Lower bounds
ub = [0.2, 0.5, 0.8];  % Upper bounds

% Optimize
options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 100);
[x_opt, fval] = fmincon(objective, x0, [], [], [], [], lb, ub, [], options);

fprintf('Optimal club speed: %.2f m/s\n', -fval);
fprintf('Shoulder timing: %.3f s\n', x_opt(1));
fprintf('Elbow timing: %.3f s\n', x_opt(2));
fprintf('Wrist timing: %.3f s\n', x_opt(3));
```

## Analysis Capabilities

### Kinematics
- Joint angles vs. time
- Joint velocities vs. time
- Club head position and trajectory
- Club head speed (max and at impact)
- Angular velocities

### Dynamics
- Joint torques vs. time
- Joint power vs. time
- Total mechanical energy
- Club head kinetic energy
- Energy transfer efficiency

### Performance Metrics
- Maximum club head speed
- Club head speed at impact
- Impact timing
- Swing duration
- Energy efficiency (mechanical work / total energy)

## Visualization

### Animation
```matlab
% Animate the swing
animate_2d_swing(simulation_results)

% Save animation as video
animate_2d_swing(simulation_results, 'SaveAs', 'swing_2d.avi')
```

### Plotting
```matlab
% Plot kinematics
plot_joint_angles_2d(results)
plot_club_trajectory_2d(results)

% Plot dynamics
plot_joint_torques_2d(results)
plot_energy_2d(results)

% Plot performance
plot_phase_diagram_2d(results, 'shoulder')
```

## Integration with Other Models

### Compare with 3D Model
```matlab
% Run 2D model
results_2d = sim_2d_golf(params);

% Convert params to 3D and run
params_3d = convert_2d_to_3d_params(params);
results_3d = sim_3d_golf(params_3d);

% Compare
compare_2d_vs_3d(results_2d, results_3d);
```

### Validate with Python Engines
```matlab
% Export 2D trajectory
trajectory_2d = results.joint_angles;
writetable(trajectory_2d, 'trajectory_2d.csv');

% Then in Python:
# import pandas as pd
# traj = pd.read_csv('trajectory_2d.csv')
# validate_with_mujoco_2d(traj)
```

## Performance Optimization

### Optimized Version
The `matlab_optimized/` directory contains a performance-optimized version:

```matlab
cd matlab_optimized
open_system('golf_swing_2d_optimized.slx')
```

**Optimizations**:
- Fixed-step solver
- Reduced visualization overhead
- Simplified contact model
- Vectorized computations

**Speed improvement**: 2-5× faster than standard version

### Batch Simulation Tips
```matlab
% Disable visualization
set_param('golf_swing_2d', 'SimMechanicsOpenEditorOnUpdate', 'off')

% Use parallel processing
parfor i = 1:num_simulations
    results{i} = sim_2d_golf(param_sets(i));
end

% Re-enable visualization
set_param('golf_swing_2d', 'SimMechanicsOpenEditorOnUpdate', 'on')
```

## Educational Use

### Teaching Examples

The model includes educational scripts in `matlab/examples/`:

1. **Basic Pendulum** - Single DOF for teaching dynamics
2. **Double Pendulum** - Chaos and sensitivity
3. **Simple Swing** - Three DOF basic golf swing
4. **Parameter Sensitivity** - How parameters affect outcome
5. **Optimization Basics** - Introduction to optimization

### Assignments

Example student assignments:

1. **Assignment 1**: Measure club head speed vs. wrist timing
2. **Assignment 2**: Find optimal joint sequence for maximum speed
3. **Assignment 3**: Compare energy transfer for different strategies
4. **Assignment 4**: Analyze effect of club length and mass

## Limitations

The 2D model has several limitations compared to 3D:

1. **No out-of-plane motion**: Cannot model face rotation or slice/hook
2. **Simplified ground contact**: No weight transfer or lateral forces
3. **Missing DOFs**: No shoulder rotation, wrist deviation, spine rotation
4. **Planar ball flight**: Ball travels only in 2D
5. **Limited realism**: Suitable for concepts, not detailed biomechanics

**When 2D is insufficient, use the [3D Model](../3D_Golf_Model/README.md)**

## Examples

### Example 1: Basic Swing
```matlab
% Run with default parameters
cd matlab/Scripts
results = run_basic_swing_2d();

% View animation
animate_2d_swing(results);
```

### Example 2: Parameter Study
```matlab
% Study effect of club length
club_lengths = linspace(0.8, 1.4, 20);  % 0.8 to 1.4 m
speeds = parameter_study_club_length(club_lengths);

plot(club_lengths, speeds)
xlabel('Club Length (m)')
ylabel('Max Club Head Speed (m/s)')
title('Club Length vs. Speed')
```

### Example 3: Optimization
```matlab
% Find optimal timing
cd matlab/Scripts
[optimal_params, max_speed] = optimize_swing_timing_2d();

fprintf('Optimal timing found:\n');
fprintf('  Club head speed: %.2f m/s\n', max_speed);
fprintf('  Shoulder start: %.3f s\n', optimal_params.shoulder_start);
fprintf('  Elbow start: %.3f s\n', optimal_params.elbow_start);
fprintf('  Wrist start: %.3f s\n', optimal_params.wrist_start);
```

## Documentation

- [Parent Simscape README](../README.md) - Simscape models overview
- [3D Model README](../3D_Golf_Model/README.md) - Full 3D model
- [Engine Selection Guide](../../../docs/engine_selection_guide.md) - Compare engines
- [MATLAB Simscape Docs](https://www.mathworks.com/help/sm/) - Official documentation

## Citation

```bibtex
@software{golf_2d_model,
  title = {2D Golf Swing Model - MATLAB Simscape Multibody},
  author = {[Authors]},
  year = {2026},
  note = {Part of Golf Modeling Suite},
  url = {https://github.com/D-sorganization/Golf_Modeling_Suite}
}
```

## License

MIT License - See [LICENSE](../../../LICENSE)

## Support

For 2D model questions:
- Check this README
- Review example scripts in `matlab/examples/`
- See [Golf Modeling Suite Issues](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)

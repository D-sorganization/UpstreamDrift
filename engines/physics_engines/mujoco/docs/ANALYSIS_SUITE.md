# Golf Swing Biomechanical Analysis Suite

## Overview

The Golf Swing Biomechanical Analysis Suite is a comprehensive, professional-grade tool for analyzing golf swing mechanics using physics-based simulation. This enhanced version builds upon the basic pendulum models to provide advanced force/torque visualization, real-time data recording, and extensive plotting capabilities.

## Key Features

### 1. **Advanced 3D Visualization**
- Real-time MuJoCo physics simulation at 60 FPS
- Multiple camera views (side, front, top, follow)
- Force and torque vector visualization with adjustable scaling
- Contact force visualization for ground reaction forces

### 2. **Biomechanical Analysis**
- Real-time extraction of joint kinematics (position, velocity, acceleration)
- Joint kinetics (forces and torques)
- Actuator power computation
- Energy analysis (kinetic, potential, total)
- Club head tracking (position, velocity, speed)
- Ground reaction forces (for models with legs)
- Center of mass tracking

### 3. **Data Recording & Export**
- Record complete swing sequences
- Export to CSV for analysis in Excel, Python, R, etc.
- Export to JSON for custom data processing
- Time-stamped biomechanical data

### 4. **Advanced Plotting**
- **Summary Dashboard**: Overview of key metrics
- **Joint Angles**: Time series of all joint positions
- **Joint Velocities**: Angular velocities over time
- **Joint Torques**: Applied torques at each joint
- **Actuator Powers**: Mechanical power output
- **Energy Analysis**: Kinetic, potential, and total energy
- **Club Head Speed**: Speed vs time with peak detection
- **Club Head Trajectory**: 3D visualization of club path
- **Phase Diagrams**: State-space plots (angle vs velocity)
- **Torque Comparison**: Contribution analysis

### 5. **Professional GUI**
- Tabbed interface for organized workflow
- Real-time metrics display
- Intuitive controls with visual feedback
- Responsive layout with adjustable panels

## User Interface

### Tabs

#### 1. **Controls Tab**
- **Model Selection**: Choose from 5 different swing models (2-28 DOF)
- **Simulation Control**: Play, pause, reset buttons
- **Recording Control**: Start/stop data recording
- **Actuator Sliders**: Control torques at each joint
  - Grouped by body part (legs, torso, arms, etc.)
  - Real-time torque display in Nm

#### 2. **Visualization Tab**
- **Camera View**: Select viewing angle
  - Side: Classic golf swing view
  - Front: Face-on view
  - Top: Bird's eye view
  - Follow: Tracking camera

- **Force & Torque Visualization**:
  - Toggle joint torque vectors
  - Adjustable torque scale (0.01% - 1.0%)
  - Toggle constraint forces
  - Adjustable force scale
  - Toggle contact forces for ground reactions

#### 3. **Analysis Tab**
- **Real-Time Metrics**:
  - Club head speed (mph and m/s)
  - Total energy (Joules)
  - Recording duration
  - Number of frames captured

- **Data Export**:
  - Export to CSV button
  - Export to JSON button

#### 4. **Plotting Tab**
- **Plot Type Selection**: Choose from 10 plot types
- **Joint Selection**: For phase diagrams
- **Generate Plot Button**: Create visualization
- **Interactive Plots**: Zoom, pan, save

## Workflow Guide

### Basic Workflow

1. **Select Model**
   - Choose appropriate model for your analysis
   - Simple models (2-3 DOF) for basic mechanics
   - Advanced models (28 DOF) for detailed biomechanics

2. **Configure Visualization**
   - Set camera angle
   - Enable force/torque vectors if needed
   - Adjust scaling for clear visualization

3. **Set Up Swing**
   - Use actuator sliders to configure joint torques
   - Observe real-time simulation
   - Adjust as needed for desired motion

4. **Record Data**
   - Click "Start Recording"
   - Let simulation run through swing sequence
   - Click "Stop Recording" when complete

5. **Analyze Results**
   - Switch to Plotting tab
   - Generate various plots to understand mechanics
   - View real-time metrics in Analysis tab

6. **Export Data**
   - Export to CSV for detailed analysis
   - Export to JSON for custom processing

### Advanced Workflow

#### Force Analysis
1. Enable torque vector visualization
2. Adjust scale to see relative magnitudes
3. Record swing while observing force patterns
4. Generate "Joint Torques" plot
5. Generate "Actuator Powers" plot
6. Analyze torque sequencing and power generation

#### Energy Analysis
1. Record complete swing (backswing through follow-through)
2. Generate "Energy Analysis" plot
3. Observe energy transfer (PE → KE)
4. Identify energy peaks and valleys
5. Correlate with swing phases

#### Club Head Analysis
1. Use model with club head (any model)
2. Record swing sequence
3. Generate "Club Head Speed" plot
4. Note peak speed timing
5. Generate "Club Head Trajectory (3D)" plot
6. Analyze swing path and plane

#### Phase Space Analysis
1. Record swing data
2. Select joint of interest
3. Generate "Phase Diagram"
4. Analyze joint behavior in state space
5. Identify patterns and singularities

## Technical Details

### Biomechanical Data Structure

Each recorded frame contains:
- **Time**: Simulation time (seconds)
- **Joint Positions**: qpos array (rad or m)
- **Joint Velocities**: qvel array (rad/s or m/s)
- **Joint Accelerations**: Computed via finite differences
- **Joint Torques**: Applied control torques (Nm)
- **Joint Forces**: Constraint forces
- **Actuator Forces**: Force/torque at each actuator
- **Actuator Powers**: Mechanical power (W)
- **Club Head Data**: Position, velocity, speed
- **Ground Reaction Forces**: Left and right foot (if applicable)
- **Energy**: Kinetic, potential, total (J)
- **Center of Mass**: Position and velocity

### Coordinate Systems

- **MuJoCo World Frame**:
  - X: Forward (direction of ball flight)
  - Y: Left
  - Z: Up

- **Joint Angles**:
  - Defined in model XML
  - Typically anatomical conventions
  - See model documentation for specifics

### Units

All quantities use SI units:
- Length: meters (m)
- Time: seconds (s)
- Angle: radians (rad) - displayed as degrees in GUI
- Force: Newtons (N)
- Torque: Newton-meters (Nm)
- Energy: Joules (J)
- Power: Watts (W)
- Speed: meters/second (m/s) - club head also shown in mph

## Export Format

### CSV Export

Columns include:
- `time`: Timestamp for each frame
- `joint_positions_0`, `joint_positions_1`, ...: Joint angles
- `joint_velocities_0`, `joint_velocities_1`, ...: Joint angular velocities
- `joint_torques_0`, `joint_torques_1`, ...: Applied torques
- `actuator_powers_0`, `actuator_powers_1`, ...: Power at each actuator
- `club_head_speed`: Club head speed
- `club_head_position_x/y/z`: 3D position
- `kinetic_energy`, `potential_energy`, `total_energy`: Energy quantities
- And more...

### JSON Export

Nested dictionary with same fields as CSV, suitable for:
- Python analysis (pandas, numpy)
- MATLAB/Octave
- R statistical analysis
- Custom visualization tools

## Tips & Best Practices

### For Accurate Analysis

1. **Recording Duration**:
   - Include full swing sequence
   - Start before backswing begins
   - Continue through follow-through

2. **Timestep**:
   - Default: 0.001s (1ms)
   - Adequate for most analyses
   - Finer timesteps for high-frequency phenomena

3. **Torque Settings**:
   - Start with small values
   - Gradually increase to desired motion
   - Avoid unrealistically large torques

4. **Force Vector Visualization**:
   - Start with low scale values
   - Increase until vectors are visible
   - Too large = cluttered display
   - Adjust iteratively

### For Performance

1. **Recording**:
   - Stop recording when not needed
   - Large datasets (>1000 frames) may slow plotting

2. **Plotting**:
   - Generate one plot at a time
   - Close plot window before generating new one
   - Use dashboard for quick overview

3. **Model Selection**:
   - Simple models (2-3 DOF) are faster
   - Advanced model (28 DOF) requires more computation

## Headless CLI for Machine Learning Runs

Run simulations without the GUI when you need to queue dozens of experiments or integrate with optimization pipelines:

```bash
python -m python.mujoco_golf_pendulum.cli_runner \
  --model full_body \
  --duration 1.5 \
  --timestep 0.001 \
  --control-config configs/full_body_constant.json \
  --output-json output/full_body.json \
  --output-csv output/full_body.csv \
  --summary
```

Control presets are JSON files listing actuator instructions:

```json
{
  "actuators": [
    {"index": 0, "type": "constant", "value": 25.0},
    {"index": 3, "type": "sine", "amplitude": 30.0, "frequency": 1.2},
    {
      "index": 8,
      "type": "polynomial",
      "coefficients": [0, 4, 0, -1, 0, 0, 0]
    }
  ]
}
```

To sweep multiple scenarios, pass a batch configuration via `--batch-config`. Each entry can reference a different model, duration, or preset, and telemetry is exported automatically so downstream machine-learning jobs can consume consistent datasets.

## Troubleshooting

### No Plot Generated
- **Cause**: No data recorded
- **Solution**: Record data before plotting

### Plot Shows "No Data"
- **Cause**: Requested data not available for this model
- **Solution**: Use model with required features (e.g., club head)

### Simulation Unstable
- **Cause**: Excessive torques
- **Solution**: Reduce actuator values, reset simulation

### Vectors Not Visible
- **Cause**: Scale too small
- **Solution**: Increase force/torque scale in Visualization tab

### Export Fails
- **Cause**: No write permissions or disk full
- **Solution**: Check file permissions and disk space

## Future Enhancements

Potential additions to the analysis suite:
- Multi-swing comparison overlay
- Automated swing phase detection
- Optimization algorithms for target swing metrics
- Real-time plotting during recording
- Custom analysis plugins
- Integration with launch monitor data
- Machine learning swing classification

## References

For model specifications and biomechanical details, see:
- `ADVANCED_BIOMECHANICAL_MODEL.md` - 28 DOF model documentation
- `models.py` - All model definitions
- `biomechanics.py` - Analysis algorithms
- `plotting.py` - Visualization code

## Support

For questions, issues, or feature requests:
- Check existing documentation
- Review example workflows
- Submit GitHub issue with details

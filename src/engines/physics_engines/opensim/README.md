# OpenSim Integration - Biomechanical Model Validation

OpenSim integration provides validation and analysis capabilities using the gold-standard biomechanical modeling platform, enabling cross-validation of Golf Modeling Suite results against established biomechanics workflows.

## Overview

OpenSim is the most widely-used biomechanical modeling platform in research and clinical applications. This integration enables:

- **Model Validation**: Cross-validate MuJoCo/Drake/Pinocchio results against OpenSim
- **Established Workflows**: Leverage decades of biomechanics research
- **Clinical Applications**: Use clinically validated models and analysis tools
- **Educational Bridge**: Connect physics-based simulation to biomechanics education
- **Research Credibility**: Validate results against the field standard

## What is OpenSim?

OpenSim is an open-source software platform for:
- Musculoskeletal modeling
- Simulation of movement
- Analysis of biomechanical data
- Study of neuromuscular control

Developed at Stanford University and used by thousands of researchers worldwide.

## Integration Features

### Model Import/Export
- Load OpenSim models (.osim format)
- Export Golf Suite models to OpenSim format
- Convert between model representations
- Preserve anatomical and mechanical properties

### Cross-Validation
- Compare joint angles and trajectories
- Validate muscle moment arms
- Check inverse dynamics results
- Verify contact forces

### Analysis Tools
- Inverse kinematics (IK)
- Inverse dynamics (ID)
- Static optimization
- Forward dynamics
- Muscle analysis

## Installation

### Prerequisites

```bash
# Install OpenSim Python package
conda install -c opensim-org opensim

# Or via pip (if available)
pip install opensim
```

**Note**: OpenSim requires Python 3.8-3.10. Create a separate conda environment if using Python 3.11+:

```bash
# Create OpenSim environment
conda create -n opensim-env python=3.10
conda activate opensim-env
conda install -c opensim-org opensim
```

### Verification

```python
import opensim
print(f"OpenSim version: {opensim.GetVersion()}")
```

## Usage

### Loading OpenSim Models

```python
from opensim_physics_engine import OpenSimPhysicsEngine

# Load an OpenSim model
engine = OpenSimPhysicsEngine(model_file="models/golfer_model.osim")

# Get model information
num_coords = engine.get_num_coordinates()
num_muscles = engine.get_num_muscles()
muscle_names = engine.get_muscle_names()
```

### Cross-Validation with MuJoCo

```python
from opensim_physics_engine import OpenSimPhysicsEngine
from cross_engine_validator import CrossEngineValidator

# Initialize both engines
opensim_engine = OpenSimPhysicsEngine("golfer_opensim.osim")
mujoco_engine = MuJoCoPhysicsEngine("golfer_mujoco.xml")

# Create validator
validator = CrossEngineValidator()
validator.add_engine("opensim", opensim_engine)
validator.add_engine("mujoco", mujoco_engine)

# Run validation
results = validator.validate_kinematics(motion_data)
print(f"RMS difference: {results.rms_error}")
print(f"Max difference: {results.max_error}")
```

### Inverse Kinematics

```python
from opensim_golf.core import OpenSimGolfAnalyzer

# Initialize analyzer
analyzer = OpenSimGolfAnalyzer(model_file="golfer.osim")

# Load motion capture data
marker_data = analyzer.load_markers("swing_markers.trc")

# Solve inverse kinematics
ik_results = analyzer.solve_inverse_kinematics(
    marker_data=marker_data,
    time_range=(0, 2.0),
    accuracy=1e-5
)

# Get joint angles
joint_angles = ik_results.get_coordinates()
```

### Inverse Dynamics

```python
# Solve inverse dynamics
id_results = analyzer.solve_inverse_dynamics(
    motion_file="ik_results.mot",
    grfs_file="ground_reaction_forces.mot"
)

# Get joint torques
joint_torques = id_results.get_forces()
```

### Muscle Analysis

```python
from muscle_analysis import OpenSimMuscleAnalyzer

# Initialize muscle analyzer
muscle_analyzer = OpenSimMuscleAnalyzer(model_file="golfer_muscles.osim")

# Analyze muscle moment arms
moment_arms = muscle_analyzer.compute_moment_arms(
    coordinate="shoulder_flexion",
    position_range=(-90, 90)
)

# Compute muscle fiber lengths during motion
muscle_lengths = muscle_analyzer.analyze_muscle_kinematics(
    motion_file="golf_swing.mot"
)
```

## Available Models

### Included Models

The suite includes OpenSim models for golf analysis:

1. **Simple Golfer (10 DOF)**
   - Upper body focus
   - No muscles (rigid body only)
   - Fast validation

2. **Full Body Golfer (23 DOF)**
   - Complete body model
   - Ground contact
   - Weight transfer analysis

3. **Musculoskeletal Golfer (23 DOF + muscles)**
   - Based on Rajagopal et al. (2016)
   - 80+ muscle-tendon units
   - Complete biomechanical analysis

### Community Models

Compatible with standard OpenSim models:
- Rajagopal et al. (2016) full body
- Hamner et al. (2010) running model
- Arnold et al. (2010) leg model
- Custom research models

## Analysis Workflows

### Workflow 1: Model Validation

```python
# 1. Create model in Golf Suite (MuJoCo)
mujoco_model = create_golf_swing_model()

# 2. Export to OpenSim
export_to_opensim(mujoco_model, "golf_model.osim")

# 3. Validate in OpenSim
validate_model("golf_model.osim")

# 4. Compare results
comparison = compare_engines(mujoco_model, "golf_model.osim")
```

### Workflow 2: Motion Capture to Simulation

```python
# 1. Load mocap markers
markers = load_motion_capture("swing.c3d")

# 2. Run IK in OpenSim
ik_solution = solve_ik_opensim(markers)

# 3. Import to Golf Suite
golf_motion = import_opensim_motion(ik_solution)

# 4. Simulate in MuJoCo with validated motion
simulate_mujoco(golf_motion)
```

### Workflow 3: Optimization

```python
# 1. Define objective (e.g., maximize club head speed)
objective = MaximizeClubHeadSpeed()

# 2. Run optimization in Golf Suite
optimal_motion = optimize_swing(objective)

# 3. Validate biomechanical constraints in OpenSim
constraints_valid = check_opensim_constraints(optimal_motion)

# 4. Analyze muscle recruitment
muscle_pattern = analyze_muscles_opensim(optimal_motion)
```

## Integration Architecture

```
Golf Modeling Suite           OpenSim
┌─────────────────┐          ┌──────────────────┐
│  MuJoCo         │          │  Model           │
│  Simulation     │◄────────►│  Validation      │
└─────────────────┘          └──────────────────┘
         │                            │
         │                            │
┌─────────────────┐          ┌──────────────────┐
│  Motion Data    │◄────────►│  IK/ID Tools     │
│  (Shared)       │          │                  │
└─────────────────┘          └──────────────────┘
         │                            │
         │                            │
┌─────────────────┐          ┌──────────────────┐
│  Analysis       │          │  Muscle          │
│  Results        │◄────────►│  Analysis        │
└─────────────────┘          └──────────────────┘
```

## API Reference

### OpenSimPhysicsEngine

```python
class OpenSimPhysicsEngine:
    def __init__(self, model_file: str)
    def load_model(self, model_file: str) -> None
    def get_state(self) -> np.ndarray
    def set_state(self, state: np.ndarray) -> None
    def step(self, dt: float) -> None
    def get_num_coordinates(self) -> int
    def get_num_muscles(self) -> int
    def get_muscle_names(self) -> List[str]
```

### OpenSimGolfAnalyzer

```python
class OpenSimGolfAnalyzer:
    def solve_inverse_kinematics(...) -> IKResults
    def solve_inverse_dynamics(...) -> IDResults
    def solve_static_optimization(...) -> SOResults
    def compute_muscle_moment_arms(...) -> np.ndarray
    def analyze_joint_reaction_forces(...) -> JRFResults
```

## Performance Considerations

### Speed Comparison

| Operation | OpenSim | MuJoCo | Speedup |
|-----------|---------|--------|---------|
| Forward simulation | 1× | 100× | 100× faster |
| Inverse kinematics | 1× | 5× | 5× faster |
| Inverse dynamics | 1× | 50× | 50× faster |
| Muscle analysis | 1× | 10× | 10× faster |

**Recommendation**: Use OpenSim for validation and detailed biomechanical analysis, MuJoCo for high-speed simulation and optimization.

## Best Practices

### When to Use OpenSim

✅ **Use OpenSim for**:
- Model validation against established standards
- Clinical biomechanical analysis
- Detailed muscle moment arm computation
- Integration with existing OpenSim workflows
- Publishing in biomechanics journals

❌ **Don't use OpenSim for**:
- Real-time simulation
- Large-scale optimization (use MuJoCo)
- Contact-rich scenarios (use MuJoCo)
- Interactive visualization (use MuJoCo/Drake)

### Validation Workflow

1. **Develop** in MuJoCo (fast iteration)
2. **Validate** with OpenSim (ensure correctness)
3. **Publish** with both (maximum credibility)

## Examples

### Example 1: Basic Validation

```python
from opensim_physics_engine import OpenSimPhysicsEngine

# Load model
engine = OpenSimPhysicsEngine("golfer.osim")

# Set initial pose
engine.set_coordinate("shoulder_flexion", 0.5)  # radians

# Get resulting muscle lengths
muscle_lengths = engine.get_muscle_lengths()
print(f"Deltoid length: {muscle_lengths['deltoid_anterior']} m")
```

### Example 2: Cross-Engine Comparison

```python
from cross_engine_validator import validate_across_engines

results = validate_across_engines(
    engines=["mujoco", "opensim"],
    motion_file="swing_motion.csv",
    tolerance=1e-3
)

results.plot_comparison()
results.save_report("validation_report.pdf")
```

## Troubleshooting

### Common Issues

**Issue**: OpenSim won't install with Python 3.11+
- **Solution**: Use Python 3.10 in a separate conda environment

**Issue**: Model file won't load
- **Solution**: Ensure .osim file is compatible with OpenSim 4.x+

**Issue**: Muscle names don't match
- **Solution**: Use muscle name mapping file (see `docs/muscle_name_mappings.md`)

## Documentation

- **[OpenSim Integration Guide](../../../docs/OPENSIM_INTEGRATION.md)**: Complete integration documentation
- **[Cross-Engine Validation](../../../docs/engine_selection_guide.md)**: Engine comparison guide
- **[OpenSim Official Docs](https://simtk-confluence.stanford.edu/display/OpenSim/Documentation)**: Upstream OpenSim documentation
- **[OpenSim Scripting Guide](https://simtk-confluence.stanford.edu/display/OpenSim/Scripting+in+Python)**: Python API documentation

## Citation

If you use OpenSim in your research, please cite:

```bibtex
@article{delp2007opensim,
  title={OpenSim: open-source software to create and analyze dynamic simulations of movement},
  author={Delp, Scott L and Anderson, Frank C and Arnold, Allison S and Loan, Peter and Habib, Ayman and John, Chand T and Guendelman, Eran and Thelen, Darryl G},
  journal={IEEE transactions on biomedical engineering},
  volume={54},
  number={11},
  pages={1940--1950},
  year={2007},
  publisher={IEEE}
}
```

## License

OpenSim is licensed under the Apache License 2.0. See the [OpenSim repository](https://github.com/opensim-org/opensim-core) for details.

## Support

For OpenSim-specific questions:
- [OpenSim Forums](https://simtk.org/plugins/phpBB/indexPhpbb.php?group_id=91&pluginname=phpBB)
- [OpenSim GitHub Issues](https://github.com/opensim-org/opensim-core/issues)

For Golf Modeling Suite integration:
- See [Golf Modeling Suite Issues](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)

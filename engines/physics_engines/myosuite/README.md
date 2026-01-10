# MyoSuite Integration - Musculoskeletal Models

MyoSuite integration brings physiologically realistic muscle-tendon dynamics to the Golf Modeling Suite, enabling detailed biomechanical analysis of golf swings with validated musculoskeletal models.

## Overview

MyoSuite provides research-grade musculoskeletal models with Hill-type muscle dynamics, converted from validated OpenSim models. This integration enables:

- **290-Muscle Full Body Models**: Complete musculoskeletal representation
- **Realistic Muscle Dynamics**: Force-length-velocity relationships and activation dynamics
- **60-4000× Faster**: Compared to OpenSim equivalents while maintaining physiological accuracy
- **MuJoCo-Based**: Leverages MuJoCo physics engine for high-performance simulation
- **Research Applications**: Golf biomechanics, injury prevention, performance optimization

## Available Models

### MyoBody Full (52 DOF, 290 muscles)
Complete full-body musculoskeletal model with all major muscle groups:
- **Torso**: 19 torso actuators
- **Upper Limbs**: Bilateral arms with complete muscle groups
- **Lower Limbs**: Legs with full muscle representation
- **Applications**: Complete golf swing biomechanics, whole-body coordination analysis

### MyoUpperBody (19 DOF, 20 actuators)
Upper body focus with muscle-based control:
- **Torso + Head**: Core stability muscles
- **Bilateral Arms**: Shoulder, elbow, and wrist muscles
- **Applications**: Upper body golf swing analysis, arm-torso coordination

### MyoArm Simple (14 DOF, 14 actuators)
Simplified bilateral arm model for rapid analysis:
- **Bilateral Arms**: Streamlined muscle groups
- **Fast Simulation**: Reduced complexity for quick iterations
- **Applications**: Arm motion analysis, initial biomechanical testing

## Features

### Hill-Type Muscle Models
- Force-length relationships (active and passive)
- Force-velocity relationships
- Tendon compliance
- Pennation angle effects
- Maximum isometric force parameterization

### Muscle Activation Dynamics
- Realistic activation/deactivation time constants
- Excitation-to-activation mapping
- Muscle fatigue modeling (optional)

### Biomechanical Analysis
- Muscle recruitment patterns
- Co-contraction analysis
- Power generation and absorption
- Muscle-induced accelerations
- Joint reaction forces

## Installation

MyoSuite is integrated with the main Golf Modeling Suite installation:

```bash
# Install from suite root
pip install -r requirements.txt

# Or install MyoSuite separately
pip install myosuite
```

## Usage

### Using with MuJoCo Launcher

MyoSuite models are available through the MuJoCo interface:

```bash
cd engines/physics_engines/mujoco/python
python -m mujoco_golf_pendulum
```

Then select one of the MyoSuite models from the GUI dropdown.

### Direct Python API

```python
from myosuite_physics_engine import MyoSuitePhysicsEngine
from muscle_analysis import MuscleAnalyzer

# Initialize MyoSuite engine
engine = MyoSuitePhysicsEngine(model_name="myo_upper_body")

# Run simulation
for step in range(1000):
    # Apply muscle activations
    activations = compute_muscle_activations(...)
    engine.step(activations)

    # Get muscle states
    muscle_forces = engine.get_muscle_forces()
    muscle_lengths = engine.get_muscle_lengths()

# Analyze muscle recruitment
analyzer = MuscleAnalyzer(engine)
recruitment_pattern = analyzer.analyze_recruitment()
power_flow = analyzer.compute_muscle_power()
```

### Muscle Analysis

```python
from muscle_analysis import MuscleAnalyzer

# Initialize analyzer
analyzer = MuscleAnalyzer(engine)

# Analyze specific muscles or muscle groups
shoulder_muscles = analyzer.get_muscle_group("shoulder")
force_profile = analyzer.get_force_profile(shoulder_muscles)
activation_pattern = analyzer.get_activation_pattern(shoulder_muscles)

# Compute muscle-induced accelerations
mia = analyzer.compute_muscle_induced_accelerations()

# Power analysis
power_gen = analyzer.compute_power_generation()
power_abs = analyzer.compute_power_absorption()
```

## Model Details

### Muscle Groups

**Upper Body**:
- Shoulder: Deltoids, rotator cuff, trapezius
- Arm: Biceps, triceps, brachialis
- Forearm: Wrist flexors/extensors, pronators/supinators
- Torso: Pectoralis, latissimus dorsi, serratus anterior

**Lower Body** (Full model only):
- Hip: Gluteals, hip flexors, adductors
- Thigh: Quadriceps, hamstrings
- Leg: Gastrocnemius, soleus, tibialis

### Conversion from OpenSim

MyoSuite models are converted from validated OpenSim models:
- **MoBL-ARMS**: Upper extremity (arms)
- **Rajagopal et al. (2015)**: Full body model

Conversion process preserves:
- Muscle attachment points
- Optimal fiber lengths
- Maximum isometric forces
- Tendon slack lengths
- Pennation angles

## Performance

MyoSuite achieves significant speedup over OpenSim:

| Model | DOF | Muscles | Speed vs OpenSim |
|-------|-----|---------|------------------|
| MyoArm Simple | 14 | 14 | 60× faster |
| MyoUpperBody | 19 | 20 | 150× faster |
| MyoBody Full | 52 | 290 | 4000× faster |

Simulation speed: **Real-time to 100× real-time** depending on model complexity

## Research Applications

### Golf Biomechanics
- Muscle activation patterns during swing phases
- Power generation and transfer through kinetic chain
- Injury risk assessment
- Performance optimization

### Injury Prevention
- Identify high-stress muscle groups
- Analyze joint loading
- Evaluate alternative swing mechanics
- Rehabilitation planning

### Performance Analysis
- Optimal muscle recruitment strategies
- Energy efficiency analysis
- Fatigue effects on swing mechanics
- Training protocol development

## Validation

MyoSuite models are validated against:
- Original OpenSim model outputs
- Experimental EMG data
- Motion capture studies
- Biomechanics literature

See `docs/MYOSUITE_INTEGRATION.md` for detailed validation results.

## Examples

Example scripts demonstrating MyoSuite capabilities:

```bash
# Run muscle analysis example
python examples/myosuite_muscle_analysis.py

# Compute muscle-induced accelerations
python examples/myosuite_induced_accelerations.py

# Optimize muscle recruitment
python examples/myosuite_optimization.py
```

## Documentation

- **[MyoSuite Integration Guide](../../../docs/MYOSUITE_INTEGRATION.md)**: Complete integration documentation
- **[MuJoCo README](../mujoco/README.md)**: MuJoCo engine documentation
- **[MyoSuite Official Docs](https://github.com/MyoHub/myosuite)**: Upstream MyoSuite documentation
- **[Muscle Analysis API](python/muscle_analysis.py)**: Python API documentation

## Limitations

### MuJoCo Dependency
MyoSuite models **require MuJoCo** and are not compatible with Drake or Pinocchio engines. For non-muscle analysis:
- Use MuJoCo for muscle-level biomechanics
- Use Drake/Pinocchio for rigid-body dynamics only

### Model Complexity
Full body model (290 muscles) requires:
- Significant computational resources
- Careful parameter tuning
- Experience with muscle modeling

Start with simpler models (MyoArm, MyoUpperBody) before using full body model.

## Citation

If you use MyoSuite models in your research, please cite:

```bibtex
@article{caggiano2022myosuite,
  title={MyoSuite: A contact-rich simulation suite for musculoskeletal motor control},
  author={Caggiano, Vittorio and Wang, Huawei and Durandau, Guillaume and Sartori, Massimo and Kumar, Vikash},
  journal={arXiv preprint arXiv:2205.13600},
  year={2022}
}
```

## License

MyoSuite is licensed under the Apache License 2.0. See the [MyoSuite repository](https://github.com/MyoHub/myosuite) for details.

## Support

For MyoSuite-specific questions:
- [MyoSuite GitHub Issues](https://github.com/MyoHub/myosuite/issues)
- [MyoSuite Discussions](https://github.com/MyoHub/myosuite/discussions)

For Golf Modeling Suite integration:
- See [Golf Modeling Suite Issues](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)

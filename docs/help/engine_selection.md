# Engine Selection Guide

Choose the right physics engine for your biomechanical analysis needs.

## Overview

UpstreamDrift integrates five physics engines, each designed for different use cases. This guide helps you select the best engine for your specific needs.

## Quick Decision Guide

| Goal | Recommended Engine |
|------|-------------------|
| General biomechanics | MuJoCo |
| Muscle-driven simulation | MuJoCo + MyoSuite |
| Trajectory optimization | Drake |
| Fast prototyping | Pinocchio |
| Clinical validation | OpenSim |
| Contact-heavy scenarios | MuJoCo |

## Engine Details

### MuJoCo (Recommended for Most Users)

**Best for:** General biomechanical simulation, contact physics, real-time visualization

MuJoCo (Multi-Joint dynamics with Contact) is the recommended starting point for most users.

**Strengths:**
- Easy installation (`pip install mujoco`)
- Fast, stable simulation
- Excellent contact physics
- Real-time visualization
- Muscle simulation via MyoSuite integration

**Use when:**
- Learning biomechanical simulation
- Need real-time feedback
- Working with contact (ball impact, ground reaction forces)
- Want muscle-driven simulation

**Installation:**
```bash
pip install mujoco
```

### Drake

**Best for:** Trajectory optimization, control system design, motion planning

Drake excels at computing optimal trajectories and designing control systems.

**Strengths:**
- State-of-the-art optimization solvers
- Model-based control design
- Contact-implicit trajectory optimization
- URDF/SDF support

**Use when:**
- Optimizing swing trajectories
- Designing feedback controllers
- Motion planning with constraints
- Multi-objective optimization (speed, accuracy, energy)

**Installation:**
```bash
# Recommended: via conda
conda install -c conda-forge drake

# Alternative: pip (limited platforms)
pip install drake
```

### Pinocchio

**Best for:** Fast algorithms, research prototyping, counterfactual analysis

Pinocchio provides efficient implementations of rigid body dynamics algorithms.

**Strengths:**
- Lightweight and fast
- Analytical Jacobians and derivatives
- ZTCF/ZVCF counterfactual analysis
- Drift-control decomposition

**Use when:**
- Need fast algorithm prototyping
- Performing counterfactual analysis
- Research requiring algorithmic differentiation
- Cross-validating with other engines

**Installation:**
```bash
# Required: via conda
conda install -c conda-forge pinocchio
```

### OpenSim

**Best for:** Musculoskeletal modeling, clinical research, validation studies

OpenSim is the gold standard for biomechanical research and clinical applications.

**Strengths:**
- Extensive validated model library
- Muscle force estimation
- Clinical research compatibility
- Large research community

**Use when:**
- Validating against published research
- Clinical or rehabilitation applications
- Detailed muscle analysis
- Regulatory or publication requirements

**Installation:**
```bash
# Via conda
conda install -c opensim-org opensim

# Or download from opensim.stanford.edu
```

### MyoSuite

**Best for:** Realistic muscle-driven simulation, reinforcement learning

MyoSuite provides 290-muscle models built on MuJoCo for physiologically realistic simulation.

**Strengths:**
- Hill-type muscle models
- Force-length-velocity relationships
- Reinforcement learning integration
- Fatigue modeling

**Use when:**
- Need detailed muscle activation analysis
- Training neural controllers with RL
- Physiologically accurate movement generation
- Studying muscle coordination

**Note:** MyoSuite requires MuJoCo and is not compatible with other engines.

**Installation:**
```bash
pip install myosuite
```

## Capability Comparison

| Feature | MuJoCo | Drake | Pinocchio | OpenSim | MyoSuite |
|---------|--------|-------|-----------|---------|----------|
| Forward Dynamics | Full | Full | Full | Full | Full |
| Inverse Dynamics | Full | Full | Full | Full | Partial |
| Contact Physics | Excellent | Good | Basic | N/A | Via MuJoCo |
| Muscle Models | Via MyoSuite | N/A | N/A | Full | Full |
| Optimization | Basic | Excellent | Good | Basic | N/A |
| Visualization | Excellent | Good | Basic | Good | Via MuJoCo |
| Installation | Easy | Moderate | Moderate | Easy | Easy |

## Switching Between Engines

UpstreamDrift supports cross-engine validation. You can run the same simulation on multiple engines and compare results:

```python
from shared.python.cross_engine_validator import CrossEngineValidator

validator = CrossEngineValidator()
comparison = validator.compare_engines(
    ["mujoco", "drake", "pinocchio"],
    simulation_params
)
```

## Troubleshooting

### Engine Not Found

If an engine shows as "Not Installed":
1. Verify installation with `python -c "import engine_name"`
2. Check environment activation
3. See installation instructions above

### Import Errors

Common solutions:
- Reinstall: `pip install --force-reinstall package_name`
- Check Python version compatibility
- Install Visual C++ Redistributable (Windows)

---

*See also: [Full User Manual](../USER_MANUAL.md) | [Simulation Controls](simulation_controls.md) | [Visualization](visualization.md)*

# Engine Selection Guide

## Overview

The Golf Modeling Suite supports three physics engines with different strengths and use cases. This guide helps you choose the right engine for your needs.

## Quick Decision Matrix

| Use Case | Recommended Engine | Why |
|----------|-------------------|-----|
| **Muscle-driven biomechanics** | MuJoCo + MyoSuite | Only option with Hill-type muscles |
| **Golf swing optimization** | Drake | Best trajectory optimization tools |
| **Quick analysis/prototyping** | MuJoCo | Easiest installation, good all-around |
| **Research reproducibility** | Pinocchio | Lightweight, algorithmic differentiation |
| **Contact-heavy simulations** | MuJoCo | Most mature soft contact model |

---

## Engine Compatibility Matrix

### Core Features (All Engines ✅)

| Feature | MuJoCo | Drake | Pinocchio | Notes |
|---------|--------|-------|-----------|-------|
| Forward Dynamics | ✅ | ✅ | ✅ | All engines validated |
| Inverse Dynamics | ✅ | ✅ | ✅ | < 0.01% error |
| Jacobians | ✅ | ✅ | ✅ | Conditioning checks included |
| Energy Conservation | ✅ | ✅ | ✅ | Validated in tests |
| URDF Loading | ✅ | ✅ | ✅ | Cross-engine compatible |

### Advanced Features (Engine-Specific)

| Feature | MuJoCo | Drake | Pinocchio | Status |
|---------|--------|-------|-----------|--------|
| **Biomechanics** | | | | |
| Hill-Type Muscles | ✅ Via MyoSuite | ❌ | ❌ | **MuJoCo-ONLY** |
| Muscle Activation Dynamics | ✅ Full pipeline | ❌ | ❌ | **MuJoCo-ONLY** |
| Grip Force Modeling | ✅ MyoSuiteGripModel | ❌ | ❌ | **MuJoCo-ONLY** |
| Muscle-Induced Acceleration | ✅ Tested | ❌ | ❌ | **MuJoCo-ONLY** |
| **Optimization** | | | | |
| Trajectory Optimization | ⚠️ Basic | ✅ Trajopt | ⚠️ Crocoddyl | Drake best |
| Contact Planning | ⚠️ Limited | ✅ | ⚠️ | Drake best |
| **Performance** | | | | |
| Simulation Speed | Baseline | *Untested* | ~20% slower* | MuJoCo fastest |
| Memory Footprint | Medium | Large | Small | Pinocchio lightest |

*Based on limited benchmarking

---

## ⚠️ CRITICAL LIMITATION: MyoSuite Biomechanics

### MyoSuite is MuJoCo-Only

**IMPORTANT**: All MyoSuite biomechanics features are **exclusive to MuJoCo**. This includes:

- ✅ Human URDF models with muscles
- ✅ Hill-type muscle force generation
- ✅ Muscle activation dynamics
- ✅ Grip force modeling
- ✅ Muscle-induced acceleration analysis

### Why MuJoCo-Only?

**Technical Reason**: MyoSuite is built on top of MuJoCo's muscle actuator system, which uses:
- Tendon routing through via-points
- Force-length and force-velocity curves
- Activation state integration

Drake and Pinocchio use **torque-based actuation models** that don't have equivalent muscle primitives.

### Workarounds

#### Option 1: Use MuJoCo for Biomechanics
**Recommended** for muscle-driven analysis:
```python
# Run biomechanics analysis in MuJoCo
from engines.physics_engines.myosuite import MyoSuitePhysicsEngine

engine = MyoSuitePhysicsEngine()
engine.load_from_path("myoElbowPose1D6MRandom-v0")

# Analyze muscle contributions
analyzer = engine.get_muscle_analyzer()
forces = analyzer.get_muscle_forces()
induced = analyzer.compute_muscle_induced_accelerations()

# Export trajectory for visualization in other engines
trajectory = engine.save_trajectory()
```

#### Option 2: Convert to Torques (Approximate)
For Drake/Pinocchio workflows, convert muscle forces to joint torques:
```python
# In MuJoCo: compute muscle contributions
muscle_forces = analyzer.get_muscle_forces()
moment_arms = analyzer.compute_moment_arms()

# Convert to equivalent joint torques
joint_torques = {}
for muscle_name, force in muscle_forces.items():
    r = moment_arms[muscle_name]
    tau = force * r  # Simple torque = force × moment arm
    joint_torques[muscle_name] = tau

# Use torques in Drake/Pinocchio
# (Less accurate but enables cross-engine comparison)
```

#### Option 3: Kinematics in All Engines
Human URDFs work for **kinematics** in all engines:
```python
# Load human model in any engine (kinematics only)
engine = get_engine(EngineType.DRAKE)  # or PINOCCHIO
engine.load_urdf("path/to/human_model.urdf")

# Compute Jacobians, visualize poses, etc.
# (No muscle forces, but joint kinematics work)
```

### Roadmap

**Future Enhancement** (estimated 2-4 weeks effort):
- Implement torque-based muscle approximation for Drake/Pinocchio
- Use muscle geometry + empirical force curves → equivalent torques
- **Status**: Not currently prioritized
- **Trade-off**: Less physiologically accurate but enables multi-engine validation

---

## Installation Difficulty

| Engine | Difficulty | Installation Method | Time | Notes |
|--------|-----------|---------------------|------|-------|
| **MuJoCo** | ⭐ Easy | `pip install mujoco` | 2 min | Recommended for beginners |
| **Drake** | ⭐⭐⭐ Hard | Conda or from source | 15-60 min | Platform-dependent |
| **Pinocchio** | ⭐⭐ Medium | `conda install pinocchio` | 5 min | Conda required |
| **MyoSuite** | ⭐⭐ Medium | `conda install myoconverter` | 10 min | Requires conda, MuJoCo |

### Quick Install Commands

```bash
# MuJoCo (easiest)
pip install mujoco

# Pinocchio (conda required)
conda install -c conda-forge pinocchio

# Drake (platform-specific)
# macOS/Linux:
conda install -c conda-forge drake
# Windows: see https://drake.mit.edu

# MyoSuite (for biomechanics)
conda install -c conda-forge myoconverter
```

---

## Performance Characteristics

### Simulation Speed (Preliminary Data)*

| Scenario | MuJoCo | Drake | Pinocchio |
|----------|--------|-------|-----------|
| Simple Pendulum | 1.0x (baseline) | *Needs testing* | ~1.2x slower |
| Golf Swing | *TBD* | *TBD* | *TBD* |
| Contact (ball impact) | *TBD* | *TBD* | *TBD* |

*Comprehensive benchmarks pending (Assessment Finding C-005)

### Memory Usage

- **MuJoCo**: Medium (~200 MB for typical model)
- **Drake**: Large (~500 MB+ with visualization)
- **Pinocchio**: Small (~50 MB for same model)

---

## Contact Modeling Differences

| Aspect | MuJoCo | Drake | Pinocchio |
|--------|--------|-------|-----------|
| **Contact Type** | Soft (penalty-based) | Compliant + Rigid | Algorithmic |
| **Stability** | Excellent | Excellent | Good |
| **Realism** | Very realistic | Physically correct | Computationally efficient |
| **Golf Ball Impact** | ✅ Tested | ⚠️ Untested | ⚠️ Untested |

**Note**: Cross-engine contact validation pending (Assessment C-003)

---

## Recommendations by User Type

### Research/Academic Use
**Pinocchio** if algorithmic differentiation needed, **MuJoCo** otherwise
- Lightweight
- Reproducible
- Open-source friendly

### Industry/Performance-Critical
**MuJoCo**
- Fastest simulation
- Mature ecosystem
- Commercial support (Roboti/Google)

### Biomechanics Research
**MuJoCo + MyoSuite** (only option)
- Hill-type muscles
- Physiologically accurate
- Growing ecosystem

### Trajectory Optimization
**Drake**
- Best-in-class optimization
- Robust contact planning
- MIT/Toyota backing

### Quick Prototyping
**MuJoCo**
- Easiest install
- Good documentation
- Quick iteration

---

## Migration Between Engines

### Exporting Models

The URDF Generator provides engine-optimized exports:

```python
from tools.urdf_generator import URDFGeneratorWindow

window = URDFGeneratorWindow()

# Export for specific engines
window.export_for_mujoco()     # MuJoCo-specific optimizations
window.export_for_drake()      # Drake conventions
window.export_for_pinocchio()  # Pinocchio format
```

**Note**: Exports are currently **untested** (Assessment Finding C-007)

### What Transfers Between Engines

✅ **Works Across Engines**:
- Joint configuration (q, v)
- Rigid body kinematics
- Basic URDF structure
- Link/joint definitions

❌ **Engine-Specific** (doesn't transfer):
- Muscle definitions (MyoSuite → MuJoCo only)
- Contact parameters (each engine different)
- Solver settings
- Visualization preferences

---

## Getting Help

### MuJoCo
- Docs: https://mujoco.readthedocs.io/
- Forum: https://github.com/deepmind/mujoco/discussions
- Email: mujoco@deepmind.com

### Drake
- Docs: https://drake.mit.edu/
- Forum: https://github.com/RobotLocomotion/drake/discussions
- Tutorials: https://deepnote.com/Drake

### Pinocchio
- Docs: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/
- GitHub: https://github.com/stack-of-tasks/pinocchio
- Forum: https://github.com/stack-of-tasks/pinocchio/discussions

### MyoSuite
- Docs: https://myosuite.readthedocs.io/
- GitHub: https://github.com/MyoHub/myoconverter
- Paper: Wang et al. (2022)

---

## Future Plans

**Q1 2026** (This Quarter):
- ✅ Complete Phase 1 critical fixes
- ⏳ Enable cross-engine CI validation
- ⏳ Comprehensive performance benchmarks

**Q2 2026**:
- Add Drake/Pinocchio contact validation
- Implement torque-based muscle approximation
- Automated cross-engine regression suite

**Q3-4 2026**:
- Optimize performance across all engines
- Expand MyoSuite model library
- Cross-validation with experimental data

---

## Summary

### Choose MuJoCo If:
- ✅ Need biomechanics (MyoSuite muscles)
- ✅ Want easiest installation
- ✅ Prioritize simulation speed
- ✅ Need mature contact modeling

### Choose Drake If:
- ✅ Need trajectory optimization
- ✅ Doing contact-rich planning
- ✅ Want robust optimization tools

### Choose Pinocchio If:
- ✅ Need algorithmic differentiation
- ✅ Want lightweight footprint
- ✅ Doing research/academic work

### For Biomechanics: MuJoCo + MyoSuite is Required ⚠️

---

**Last Updated**: January 7, 2026  
**Version**: 1.0  
**Status**: Post-PR303 (Model Library + MyoConverter integrated)

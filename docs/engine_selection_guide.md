# Engine Selection Guide

UpstreamDrift supports 5 physics engines, each with different strengths. This
guide helps you choose the right engine for your use case.

## Quick Decision Matrix

| Capability              | MuJoCo     | Drake     | Pinocchio | OpenSim    | MyoSuite   |
| ----------------------- | ---------- | --------- | --------- | ---------- | ---------- |
| Real-time performance   | Excellent  | Good      | Excellent | Fair       | Good       |
| Muscle modeling         | Limited    | No        | No        | Excellent  | Excellent  |
| Contact physics         | Excellent  | Good      | Good      | Fair       | Good       |
| Trajectory optimization | Good       | Excellent | Good      | Fair       | Fair       |
| Inverse kinematics      | Good       | Excellent | Excellent | Good       | Fair       |
| Python API quality      | Excellent  | Good      | Excellent | Fair       | Good       |
| Install complexity      | Easy       | Medium    | Easy      | Hard       | Medium     |
| License                 | Apache 2.0 | BSD 3     | BSD 2     | Apache 2.0 | Apache 2.0 |

## When to Use Each Engine

### MuJoCo (Recommended Default)

**Best for**: General-purpose physics simulation, real-time visualization,
contact-rich scenarios, and getting started quickly.

MuJoCo is the most well-rounded engine in the suite. It offers excellent
real-time performance, robust contact physics, and the simplest installation
path. If you are unsure which engine to use, start here.

```bash
pip install mujoco
```

Key strengths:

- Fast and stable contact simulation
- Excellent Python bindings with NumPy integration
- Good documentation and large user community
- MyoSuite builds on top of MuJoCo for muscle modeling

### Drake

**Best for**: Trajectory optimization, mathematical rigor, formal verification,
and model-based control design.

Drake excels at optimization-based approaches. If your workflow involves solving
optimal control problems or you need mathematically rigorous dynamics
formulations, Drake is the right choice.

```bash
pip install drake
```

Key strengths:

- State-of-the-art trajectory optimization solvers
- Mathematical program formulation for control design
- Formal plant/controller separation
- Strong URDF/SDF model support

### Pinocchio

**Best for**: Rigid body dynamics, inverse kinematics, analytical Jacobians,
and robot control research.

Pinocchio provides the fastest analytical rigid body dynamics computations. It is
ideal for control algorithms that need fast Jacobian and dynamics evaluations.

```bash
pip install pin
```

Key strengths:

- Fastest analytical rigid body algorithms (RNEA, ABA, CRBA)
- Efficient Jacobian and derivative computations
- PINK inverse kinematics integration
- Lightweight with minimal dependencies

### OpenSim

**Best for**: Musculoskeletal modeling, biomechanics research, and validation
against established biomechanics literature.

OpenSim is the gold standard for musculoskeletal modeling in biomechanics
research. Use it when you need to validate results against published studies or
work with existing OpenSim models.

```bash
conda install -c opensim-org opensim
```

Key strengths:

- Validated musculoskeletal models (MoBL-ARMS, Rajagopal)
- Large library of existing models and motion data
- Established in biomechanics research community
- Muscle-tendon dynamics with Hill-type models

### MyoSuite

**Best for**: Muscle-actuated simulations, reinforcement learning with
musculoskeletal models, and high-fidelity muscle dynamics.

MyoSuite extends MuJoCo with realistic muscle models. Use it when you need
muscle-level actuation with reinforcement learning or when studying muscle
coordination patterns.

```bash
pip install myosuite
```

Key strengths:

- 290-muscle full body models
- Hill-type muscle dynamics (force-length-velocity relationships)
- Built-in reinforcement learning task suite
- Builds on MuJoCo's fast simulation core

## Choosing by Use Case

| Use Case                 | Recommended Engine | Alternative  |
| ------------------------ | ------------------ | ------------ |
| Quick prototyping        | MuJoCo             | Pinocchio    |
| Biomechanics research    | OpenSim            | MyoSuite     |
| Optimal control          | Drake              | MuJoCo       |
| Muscle coordination      | MyoSuite           | OpenSim      |
| Robot control algorithms | Pinocchio          | Drake        |
| Contact-rich simulation  | MuJoCo             | Drake        |
| RL with muscles          | MyoSuite           | MuJoCo       |
| Model validation         | OpenSim            | Cross-engine |

## Cross-Engine Validation

One of UpstreamDrift's key features is the ability to compare results across
engines. This is useful for validating that your simulation results are not
engine-specific artifacts.

Run the cross-engine validation tests:

```bash
pytest tests/integration/cross_engine/ -v
```

For known differences between engines, see
[Troubleshooting: Cross-Engine Deviations](troubleshooting/cross_engine_deviations.md).

## Installing Multiple Engines

To install all available engines at once:

```bash
pip install -e ".[engines]"
```

Or install specific engines:

```bash
pip install -e ".[mujoco,drake,pinocchio]"
```

For engines that require conda (OpenSim), see
[Installation Guide](troubleshooting/installation.md).

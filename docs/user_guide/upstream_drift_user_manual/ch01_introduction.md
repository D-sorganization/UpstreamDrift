## Part I: Getting Started

---

# Chapter 1: Introduction

## 1.1 Project Overview

**UpstreamDrift** is a unified, open-source platform for biomechanical golf swing
analysis and physics-based simulation. It consolidates five distinct physics engines,
multiple model complexities, advanced biomechanical analysis tools, and professional
visualization capabilities into a single, cohesive software suite.

The platform bridges the gap between educational swing models and research-grade
musculoskeletal simulations, providing a continuous spectrum of model fidelity:

| Model Tier      | DOF Range | Muscles | Use Case                     | Engine(s)                |
| --------------- | --------- | ------- | ---------------------------- | ------------------------ |
| Educational     | 1--2      | 0       | Teaching, intuition building | Pendulum, MATLAB         |
| Intermediate    | 7--14     | 0       | Engineering analysis         | MuJoCo, Drake, Pinocchio |
| Advanced        | 18--28    | 0       | Research-grade rigid body    | MuJoCo, Drake, Pinocchio |
| Musculoskeletal | 28+       | 50--290 | Clinical and sports science  | OpenSim, MyoSuite        |

At its core, UpstreamDrift models the golf swing as a multi-body dynamical system
governed by the Euler-Lagrange equations of motion:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau + J^T(q) F_{\text{ext}}$$

where:

- $M(q) \in \mathbb{R}^{n \times n}$ is the joint-space inertia matrix,
- $C(q, \dot{q}) \in \mathbb{R}^{n \times n}$ captures Coriolis and centrifugal effects,
- $g(q) \in \mathbb{R}^{n}$ is the gravitational torque vector,
- $\tau \in \mathbb{R}^{n}$ is the vector of applied joint torques (or muscle-generated torques),
- $J(q) \in \mathbb{R}^{m \times n}$ is the contact Jacobian, and
- $F_{\text{ext}} \in \mathbb{R}^{m}$ represents external contact forces (ground reaction, ball impact, grip).

Each physics engine implements this fundamental equation with different numerical
methods, contact models, and integration schemes, enabling cross-validation and
comparative analysis of results.

### Key Capabilities

UpstreamDrift provides the following core capabilities:

1. **Multi-Engine Simulation**: Run the same swing model across MuJoCo, Drake,
   Pinocchio, OpenSim, and MyoSuite. Compare kinematics, dynamics, and energy
   profiles to validate results and understand engine-specific behaviors.

2. **Musculoskeletal Modeling**: Full Hill-type muscle models with activation
   dynamics, force-length-velocity relationships, and tendon compliance.
   Models range from simplified 50-muscle upper extremity representations
   to comprehensive 290-muscle full-body configurations derived from validated
   OpenSim models (MoBL-ARMS, Rajagopal 2015).

3. **Biomechanical Analysis Suite**: Inverse kinematics, inverse dynamics,
   kinematic sequence analysis, swing plane decomposition, grip contact forces,
   ground reaction force processing, energy monitoring, manipulability analysis,
   and injury risk assessment.

4. **Ball Flight Physics**: Aerodynamic drag and lift models incorporating
   spin-dependent Magnus forces, dimple-resolved drag coefficients, wind effects,
   altitude corrections, and terrain interaction for complete shot simulation.

5. **Motion Capture Integration**: Import and retarget motion capture data from
   multiple sources (OpenPose, MediaPipe, C3D files) onto simulation models
   for data-driven analysis and validation against real swing data.

6. **Professional Visualization**: Real-time 3D rendering with multiple camera
   views, force/torque vector overlays, manipulability ellipsoids, over 10 plot
   types (energy, phase diagrams, 3D trajectories, joint angles, muscle activations),
   and data export to CSV and JSON.

7. **REST API**: A FastAPI-based server with JWT authentication, rate limiting,
   CORS support, WebSocket streaming, and comprehensive endpoints for simulation
   control, analysis, engine management, video processing, and data export.

8. **Dual GUI Options**: A feature-rich PyQt6 desktop launcher and a modern
   Tauri/React web-based desktop application, both providing access to all
   engines and analysis tools.

9. **Robotics and Control**: Impedance, admittance, hybrid force-position, and
   operational space control schemes. Contact handling, constraint analysis,
   locomotion planning, and sensing integration for robotic golf swing execution.

10. **Research Tools**: Model Predictive Control (MPC), differentiable physics
    for gradient-based optimization, deformable object simulation, multi-robot
    coordination, reinforcement learning, imitation learning, and sim-to-real
    transfer pipelines.

### Platform Statistics

As of version 2.1.0, the codebase comprises:

- Approximately **1,163 Python source files** across the `src/` directory tree
- Over **180 installable packages** and dependencies
- **15 test categories** with 1,563+ test cases
- **5 physics engines** with a unified interface layer
- **7 API route modules** with 50+ REST endpoints
- **3 launcher interfaces** (unified, golf-specific, engine-specific)

## 1.2 Mission and Design Philosophy

### Mission Statement

UpstreamDrift exists to democratize biomechanical golf swing analysis by providing
an open, extensible, and scientifically rigorous simulation platform that serves
educators, researchers, engineers, and sports scientists equally well.

### Design Principles

The architecture of UpstreamDrift is guided by several foundational principles:

**1. Multi-Engine Transparency**

No single physics engine is universally superior for all aspects of golf swing
simulation. MuJoCo excels at contact-rich dynamics and muscle integration; Drake
provides robust trajectory optimization; Pinocchio delivers the fastest rigid-body
algorithms; OpenSim and MyoSuite offer clinically validated musculoskeletal models.
UpstreamDrift treats each engine as a first-class citizen and provides a unified
interface (`PhysicsEngine` protocol) that allows users to switch engines without
modifying their analysis code.

**2. Progressive Model Fidelity**

Users should be able to start with simple 2 DOF pendulum models to build intuition,
then seamlessly scale to 28 DOF humanoid models for engineering analysis, and
finally employ 290-muscle musculoskeletal models for clinical or research
applications. The platform supports this progression without requiring users to
learn entirely new tools at each level of fidelity.

**3. Design by Contract (DbC)**

Following the principles established by Bertrand Meyer, the codebase employs formal
preconditions, postconditions, and class invariants to enforce correctness at API
boundaries. This is implemented through the `contracts.py` module, which provides
decorators (`@precondition`, `@postcondition`, `@invariant`) that can be globally
enabled or disabled for performance tuning. Contract enforcement transforms many
runtime errors into clear, immediate diagnostic messages.

**4. Pragmatic Programmer Values**

The codebase explicitly follows principles from _The Pragmatic Programmer_:
DRY (Don't Repeat Yourself) to eliminate knowledge duplication, Orthogonality to
minimize coupling between modules, and the principle of Least Surprise to ensure
that APIs and interfaces behave as users expect.

**5. Security by Default**

All API endpoints enforce JWT authentication, rate limiting, and input validation.
Password hashing uses bcrypt (replacing legacy SHA256). Security headers, CORS
restrictions, trusted host middleware, and upload size limits are applied globally.
The platform has been audited and upgraded from a security grade of D+ (68/100) to
A- (92/100) and complies with OWASP Top 10, CWE-327, CWE-532, and CWE-94 standards.

**6. Observability**

Structured logging via `structlog` provides machine-parseable, context-rich log
output. Request tracing, health check endpoints, and diagnostic tools enable
effective monitoring and debugging in both development and production environments.

## 1.3 Target Audience

UpstreamDrift is designed to serve multiple user communities:

### Biomechanics Researchers

Researchers studying the biomechanics of the golf swing will find validated
musculoskeletal models, inverse kinematics/dynamics tools, muscle activation
analysis, and injury risk assessment capabilities. The multi-engine validation
framework allows results to be cross-checked across different physics simulators,
strengthening confidence in findings.

### Sports Scientists and Coaches

Practitioners seeking to understand and optimize golf swing mechanics can use the
visualization tools, kinematic sequence analysis, and swing comparison features
to gain insights from simulation data. The motion capture integration pipeline
enables analysis of real swing data within the simulation framework.

### Robotics Engineers

Engineers developing robotic golf swing systems can leverage the control module
(impedance, admittance, hybrid force-position, operational space control), the
planning module (trajectory optimization, motion planning), and the deployment
module (real-time control, digital twin, safety monitoring, teleoperation).

### Educators and Students

The pendulum models (1--2 DOF) and parameter sweep tools provide accessible
entry points for teaching concepts in rigid-body dynamics, Lagrangian mechanics,
energy conservation, and numerical simulation. The progressive model fidelity
allows students to gradually engage with more complex systems.

### Software Developers

Developers extending the platform or building integrations can rely on the
well-structured module hierarchy, typed interfaces, Design by Contract enforcement,
comprehensive test suite, and detailed API documentation (auto-generated via
FastAPI's OpenAPI/Swagger support at `/docs` and `/redoc`).

## 1.4 Version History

### Version 2.1.0 (Current Release -- February 2026)

- **Security Overhaul**: Security grade improved from D+ (68/100) to A- (92/100).
  API key hashing upgraded from SHA256 to bcrypt. JWT token generation updated
  to use timezone-aware datetime. Plaintext password logging removed. `pip-audit`
  made blocking in CI.
- **MyoSuite Integration**: Full Hill-type muscle models with 290 muscles.
  Research-grade models converted from validated OpenSim configurations
  (MoBL-ARMS, Rajagopal).
- **OpenSim Integration**: Biomechanical model validation and analysis tools.
  Tutorials and example scripts for musculoskeletal modeling.
- **Tauri/React Desktop Application**: Modern web-based desktop UI alongside
  the classic PyQt6 launcher.
- **FastAPI REST API**: Full-featured API server with JWT auth, rate limiting,
  CORS, WebSocket support, and comprehensive endpoint coverage.
- **Comprehensive Assessment Framework**: 15 quality assessment categories (A--O)
  for continuous code quality monitoring.
- **Expanded Pre-Commit Hooks**: Trailing whitespace, YAML validation, large
  file detection added alongside ruff, black, and mypy checks.

### Version 1.0.0 (January 2026)

- **Initial Stable Release**: Five physics engines (MuJoCo, Drake, Pinocchio,
  OpenSim, MyoSuite) integrated into a unified platform.
- **1,563+ Unit Tests**: Comprehensive validation across all engines.
- **Professional PyQt6 GUI**: Interactive launcher with engine selection,
  model browsing, and visualization.
- **Multi-Engine Comparison**: Cross-engine validation and comparative analysis tools.
- **URDF Generator**: Automated URDF file creation with bundled assets.
- **Core Features**: Manipulability ellipsoid visualization, flexible shaft
  dynamics, grip contact force analysis, ground reaction force processing.

### Pre-1.0 Development

The project originated as a collection of independent MATLAB Simscape Multibody
models for 2D and 3D golf swing simulation. These were progressively augmented
with Python-based physics engines (MuJoCo first, then Drake and Pinocchio),
culminating in the unified platform released as version 1.0.0.

## 1.5 Document Conventions

Throughout this manual, the following conventions are used:

- **`monospace text`** indicates code, file names, command-line input, or
  identifiers (e.g., `launch_upstream_drift.py`, `src/api/server.py`).

- **Bold text** denotes key terms, UI elements, or emphasis.

- _Italic text_ denotes book titles, foreign terms, or mild emphasis.

- Code blocks with language annotations show executable examples:

  ```python
  # Python example
  from src.shared.python.engine_core.engine_manager import EngineManager
  manager = EngineManager()
  ```

  ```bash
  # Shell command example
  python launch_upstream_drift.py --engine mujoco
  ```

- Mathematical notation uses LaTeX syntax. Inline: $E = \frac{1}{2}mv^2$.
  Display:

  $$\tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)$$

- **Warning** blocks indicate potential pitfalls or destructive operations.

- **Note** blocks provide supplementary information or tips.

- File paths are given relative to the repository root unless otherwise stated.
  The repository root is the directory containing `pyproject.toml` and
  `launch_upstream_drift.py`.

## 1.6 How to Read This Manual

The manual is organized into eight parts. Readers are encouraged to follow a
path that matches their role and objectives:

| Reader Profile          | Recommended Path                                  |
| ----------------------- | ------------------------------------------------- |
| First-time user         | Chapters 1 -> 3 -> 4 -> 5 (or any engine chapter) |
| Biomechanics researcher | Chapters 1 -> 3 -> 4 -> 8 -> 9 -> 12 -> 14 -> 15  |
| Robotics engineer       | Chapters 1 -> 3 -> 4 -> 5 -> 6 -> 7 -> 16 -> 18   |
| API developer           | Chapters 1 -> 3 -> 4 -> 20 -> 21                  |
| GUI developer           | Chapters 1 -> 3 -> 4 -> 22                        |
| Contributor             | Chapters 1 -> 2 -> 3 -> 23 -> 24                  |
| DevOps / deployment     | Chapters 1 -> 3 -> 25 -> Appendix A               |
| Student / educator      | Chapters 1 -> 3 -> 4 -> 10 -> 12                  |

Each chapter is designed to be reasonably self-contained, with cross-references
to related chapters where deeper context is required.

---

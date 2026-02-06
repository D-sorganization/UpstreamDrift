# UpstreamDrift User Manual

**Version 2.1.0**

**A Unified Platform for Biomechanical Golf Swing Analysis and Physics-Based Simulation**

---

_Copyright (c) 2026 UpstreamDrift Contributors. Licensed under the MIT License._

_Document Revision: 1.0 | Last Updated: February 2026_

_Applicable Software Version: UpstreamDrift v2.1.0_

---

## About This Manual

This document serves as the official, comprehensive user manual for **UpstreamDrift**
(formerly Golf Modeling Suite). It is intended as the authoritative reference for
installation, configuration, operation, extension, and maintenance of the platform.
The manual assumes familiarity with Python, basic physics concepts, and command-line
tools, but strives to be accessible to newcomers to biomechanical simulation.

Throughout this manual, mathematical notation is rendered in LaTeX where appropriate.
Inline equations use `$...$` delimiters and display equations use `$$...$$` delimiters.

---

## Table of Contents

### Part I: Getting Started

- **[Chapter 1: Introduction](#chapter-1-introduction)**

  - [1.1 Project Overview](#11-project-overview)
  - [1.2 Mission and Design Philosophy](#12-mission-and-design-philosophy)
  - [1.3 Target Audience](#13-target-audience)
  - [1.4 Version History](#14-version-history)
  - [1.5 Document Conventions](#15-document-conventions)
  - [1.6 How to Read This Manual](#16-how-to-read-this-manual)

- **[Chapter 2: Architecture Overview](#chapter-2-architecture-overview)**

  - [2.1 High-Level System Architecture](#21-high-level-system-architecture)
  - [2.2 Multi-Engine Design](#22-multi-engine-design)
  - [2.3 Module Structure and Package Organization](#23-module-structure-and-package-organization)
  - [2.4 Design by Contract (DbC)](#24-design-by-contract-dbc)
  - [2.5 Dependency Graph](#25-dependency-graph)
  - [2.6 Data Flow and Communication Patterns](#26-data-flow-and-communication-patterns)
  - [2.7 Security Architecture](#27-security-architecture)

- **[Chapter 3: Installation and Setup](#chapter-3-installation-and-setup)**

  - [3.1 System Prerequisites](#31-system-prerequisites)
  - [3.2 Installation via Conda (Recommended)](#32-installation-via-conda-recommended)
  - [3.3 Installation via Pip](#33-installation-via-pip)
  - [3.4 Docker Installation](#34-docker-installation)
  - [3.5 Environment Configuration](#35-environment-configuration)
  - [3.6 Verifying the Installation](#36-verifying-the-installation)
  - [3.7 Platform-Specific Notes](#37-platform-specific-notes)
  - [3.8 Development Environment Setup](#38-development-environment-setup)
  - [3.9 Troubleshooting Installation Issues](#39-troubleshooting-installation-issues)

- **[Chapter 4: Quick Start Guide](#chapter-4-quick-start-guide)**
  - [4.1 Running Your First Simulation](#41-running-your-first-simulation)
  - [4.2 Launching the GUI (PyQt6 Classic Launcher)](#42-launching-the-gui-pyqt6-classic-launcher)
  - [4.3 Launching the Web UI (Tauri/Vue.js)](#43-launching-the-web-ui-tauivuejs)
  - [4.4 Starting the API Server](#44-starting-the-api-server)
  - [4.5 Basic Workflows](#45-basic-workflows)
  - [4.6 Exploring the Example Scripts](#46-exploring-the-example-scripts)
  - [4.7 Understanding Output and Results](#47-understanding-output-and-results)

### Part II: Physics Engines

- **[Chapter 5: MuJoCo Engine](#chapter-5-mujoco-engine)**

  - [5.1 Overview and Capabilities](#51-overview-and-capabilities-mujoco)
  - [5.2 Model Library (2 DOF to 28 DOF)](#52-model-library-2-dof-to-28-dof)
  - [5.3 Contact Dynamics and Ground Interaction](#53-contact-dynamics-and-ground-interaction)
  - [5.4 Flexible Shaft Modeling](#54-flexible-shaft-modeling)
  - [5.5 Humanoid Golf Simulation](#55-humanoid-golf-simulation)
  - [5.6 MyoSuite Integration (Musculoskeletal)](#56-myosuite-integration-musculoskeletal)
  - [5.7 Motion Capture Workflow](#57-motion-capture-workflow)
  - [5.8 MuJoCo-Specific Configuration](#58-mujoco-specific-configuration)

- **[Chapter 6: Drake Engine](#chapter-6-drake-engine)**

  - [6.1 Overview and Capabilities](#61-overview-and-capabilities-drake)
  - [6.2 Trajectory Optimization](#62-trajectory-optimization)
  - [6.3 Contact Modeling in Drake](#63-contact-modeling-in-drake)
  - [6.4 System Analysis Tools](#64-system-analysis-tools)
  - [6.5 URDF Support and Model Loading](#65-urdf-support-and-model-loading)
  - [6.6 Drake Dashboard](#66-drake-dashboard)

- **[Chapter 7: Pinocchio Engine](#chapter-7-pinocchio-engine)**

  - [7.1 Overview and Capabilities](#71-overview-and-capabilities-pinocchio)
  - [7.2 Rigid Body Algorithms](#72-rigid-body-algorithms)
  - [7.3 Analytical Inverse Kinematics](#73-analytical-inverse-kinematics)
  - [7.4 Inverse Dynamics Computation](#74-inverse-dynamics-computation)
  - [7.5 Meshcat Visualization](#75-meshcat-visualization)
  - [7.6 Pinocchio Dashboard](#76-pinocchio-dashboard)

- **[Chapter 8: OpenSim Engine](#chapter-8-opensim-engine)**

  - [8.1 Overview and Capabilities](#81-overview-and-capabilities-opensim)
  - [8.2 Musculoskeletal Model Library](#82-musculoskeletal-model-library)
  - [8.3 Muscle Dynamics and Activation](#83-muscle-dynamics-and-activation)
  - [8.4 Model Validation Against Literature](#84-model-validation-against-literature)
  - [8.5 OpenSim Tutorials](#85-opensim-tutorials)

- **[Chapter 9: MyoSuite Engine](#chapter-9-myosuite-engine)**

  - [9.1 Overview and Capabilities](#91-overview-and-capabilities-myosuite)
  - [9.2 Hill-Type Muscle Models](#92-hill-type-muscle-models)
  - [9.3 290-Muscle Full-Body Model](#93-290-muscle-full-body-model)
  - [9.4 Muscle Activation Dynamics](#94-muscle-activation-dynamics)
  - [9.5 Force-Length-Velocity Relationships](#95-force-length-velocity-relationships)
  - [9.6 Research-Grade Model Validation](#96-research-grade-model-validation)

- **[Chapter 10: Pendulum and Educational Models](#chapter-10-pendulum-and-educational-models)**

  - [10.1 Simple Pendulum (1 DOF)](#101-simple-pendulum-1-dof)
  - [10.2 Double Pendulum (2 DOF)](#102-double-pendulum-2-dof)
  - [10.3 Parametric Studies with Pendulum Models](#103-parametric-studies-with-pendulum-models)
  - [10.4 MATLAB Simscape Multibody Models](#104-matlab-simscape-multibody-models)

- **[Chapter 11: Cross-Engine Comparison and Validation](#chapter-11-cross-engine-comparison-and-validation)**
  - [11.1 Unified Engine Interface](#111-unified-engine-interface)
  - [11.2 Cross-Engine Validation Framework](#112-cross-engine-validation-framework)
  - [11.3 Benchmark Methodology](#113-benchmark-methodology)
  - [11.4 Comparative Analysis Tools](#114-comparative-analysis-tools)
  - [11.5 Engine Selection Guide](#115-engine-selection-guide)

### Part III: Analysis and Simulation

- **[Chapter 12: Biomechanical Analysis](#chapter-12-biomechanical-analysis)**

  - [12.1 Kinematic Analysis](#121-kinematic-analysis)
  - [12.2 Inverse Kinematics Solver](#122-inverse-kinematics-solver)
  - [12.3 Inverse Dynamics and Torque Computation](#123-inverse-dynamics-and-torque-computation)
  - [12.4 Kinematic Sequence Analysis](#124-kinematic-sequence-analysis)
  - [12.5 Swing Plane Analysis and Visualization](#125-swing-plane-analysis-and-visualization)
  - [12.6 Grip Contact Force Analysis](#126-grip-contact-force-analysis)
  - [12.7 Ground Reaction Force Processing](#127-ground-reaction-force-processing)
  - [12.8 Energy Monitoring and Conservation](#128-energy-monitoring-and-conservation)
  - [12.9 Manipulability Analysis](#129-manipulability-analysis)
  - [12.10 Injury Risk Assessment](#1210-injury-risk-assessment)

- **[Chapter 13: Ball Flight Physics](#chapter-13-ball-flight-physics)**

  - [13.1 Aerodynamic Models](#131-aerodynamic-models)
  - [13.2 Impact Mechanics](#132-impact-mechanics)
  - [13.3 Spin Dynamics](#133-spin-dynamics)
  - [13.4 Environmental Effects](#134-environmental-effects)
  - [13.5 Terrain Interaction](#135-terrain-interaction)
  - [13.6 Shot Tracing and Visualization](#136-shot-tracing-and-visualization)

- **[Chapter 14: Muscle and Tissue Modeling](#chapter-14-muscle-and-tissue-modeling)**

  - [14.1 Hill-Type Muscle Model Theory](#141-hill-type-muscle-model-theory)
  - [14.2 Multi-Muscle Coordination](#142-multi-muscle-coordination)
  - [14.3 Muscle Equilibrium Solver](#143-muscle-equilibrium-solver)
  - [14.4 Activation Dynamics](#144-activation-dynamics)
  - [14.5 Spinal Load Analysis](#145-spinal-load-analysis)
  - [14.6 Joint Stress Computation](#146-joint-stress-computation)

- **[Chapter 15: Motion Capture and Pose Estimation](#chapter-15-motion-capture-and-pose-estimation)**
  - [15.1 Video Pose Pipeline](#151-video-pose-pipeline)
  - [15.2 OpenPose Integration](#152-openpose-integration)
  - [15.3 MediaPipe Integration](#153-mediapipe-integration)
  - [15.4 C3D File Support](#154-c3d-file-support)
  - [15.5 Motion Retargeting](#155-motion-retargeting)
  - [15.6 Marker Mapping and Calibration](#156-marker-mapping-and-calibration)

### Part IV: Robotics and Control

- **[Chapter 16: Control Systems](#chapter-16-control-systems)**

  - [16.1 Impedance Control](#161-impedance-control)
  - [16.2 Admittance Control](#162-admittance-control)
  - [16.3 Hybrid Force-Position Control](#163-hybrid-force-position-control)
  - [16.4 Operational Space Control](#164-operational-space-control)
  - [16.5 Task-Space Control with Redundancy Resolution](#165-task-space-control-with-redundancy-resolution)
  - [16.6 Contact and Constraint Handling](#166-contact-and-constraint-handling)

- **[Chapter 17: Learning and Optimization](#chapter-17-learning-and-optimization)**

  - [17.1 Reinforcement Learning for Swing Optimization](#171-reinforcement-learning-for-swing-optimization)
  - [17.2 Imitation Learning from Motion Capture](#172-imitation-learning-from-motion-capture)
  - [17.3 Sim-to-Real Transfer](#173-sim-to-real-transfer)
  - [17.4 Motion Retargeting Algorithms](#174-motion-retargeting-algorithms)
  - [17.5 Trajectory Optimization](#175-trajectory-optimization)
  - [17.6 Parameter Sweeps and Sensitivity Analysis](#176-parameter-sweeps-and-sensitivity-analysis)

- **[Chapter 18: Deployment and Real-Time Systems](#chapter-18-deployment-and-real-time-systems)**
  - [18.1 Real-Time Controller](#181-real-time-controller)
  - [18.2 Digital Twin Framework](#182-digital-twin-framework)
  - [18.3 Safety Monitoring and Collision Detection](#183-safety-monitoring-and-collision-detection)
  - [18.4 Teleoperation Interface](#184-teleoperation-interface)

### Part V: Research Tools

- **[Chapter 19: Advanced Research Modules](#chapter-19-advanced-research-modules)**
  - [19.1 Model Predictive Control (MPC)](#191-model-predictive-control-mpc)
  - [19.2 Differentiable Physics](#192-differentiable-physics)
  - [19.3 Deformable Object Simulation](#193-deformable-object-simulation)
  - [19.4 Multi-Robot Coordination](#194-multi-robot-coordination)
  - [19.5 Locomotion Planning](#195-locomotion-planning)
  - [19.6 Sensing and State Estimation](#196-sensing-and-state-estimation)

### Part VI: API and Integration

- **[Chapter 20: REST API Reference](#chapter-20-rest-api-reference)**

  - [20.1 API Architecture and Design](#201-api-architecture-and-design)
  - [20.2 Authentication and Authorization (JWT)](#202-authentication-and-authorization-jwt)
  - [20.3 Core Endpoints](#203-core-endpoints)
  - [20.4 Simulation Endpoints](#204-simulation-endpoints)
  - [20.5 Analysis Endpoints](#205-analysis-endpoints)
  - [20.6 Engine Management Endpoints](#206-engine-management-endpoints)
  - [20.7 Video Processing Endpoints](#207-video-processing-endpoints)
  - [20.8 Export Endpoints](#208-export-endpoints)
  - [20.9 WebSocket Endpoints](#209-websocket-endpoints)
  - [20.10 Rate Limiting and Security Middleware](#2010-rate-limiting-and-security-middleware)
  - [20.11 Error Codes and Response Formats](#2011-error-codes-and-response-formats)

- **[Chapter 21: Database and Data Management](#chapter-21-database-and-data-management)**

  - [21.1 SQLite Database Schema](#211-sqlite-database-schema)
  - [21.2 Data Models and ORM](#212-data-models-and-orm)
  - [21.3 Migration and Versioning](#213-migration-and-versioning)
  - [21.4 Data Export Formats (CSV, JSON)](#214-data-export-formats-csv-json)
  - [21.5 Backup and Recovery](#215-backup-and-recovery)

- **[Chapter 22: GUI and Visualization](#chapter-22-gui-and-visualization)**
  - [22.1 PyQt6 Classic Launcher](#221-pyqt6-classic-launcher)
  - [22.2 Tauri/Vue.js Web Desktop Application](#222-tauivuejs-web-desktop-application)
  - [22.3 Engine Dashboards (MuJoCo, Drake, Pinocchio)](#223-engine-dashboards-mujoco-drake-pinocchio)
  - [22.4 Model Explorer](#224-model-explorer)
  - [22.5 Pose Editor](#225-pose-editor)
  - [22.6 Plotting and Chart Library](#226-plotting-and-chart-library)
  - [22.7 3D Visualization and Rendering](#227-3d-visualization-and-rendering)
  - [22.8 Ellipsoid Visualization (Manipulability)](#228-ellipsoid-visualization-manipulability)
  - [22.9 Theme and Appearance Customization](#229-theme-and-appearance-customization)

### Part VII: Development and Operations

- **[Chapter 23: Testing and Quality Assurance](#chapter-23-testing-and-quality-assurance)**

  - [23.1 Test Architecture and Categories](#231-test-architecture-and-categories)
  - [23.2 Unit Testing](#232-unit-testing)
  - [23.3 Integration Testing](#233-integration-testing)
  - [23.4 Cross-Engine Validation Tests](#234-cross-engine-validation-tests)
  - [23.5 Physics Validation Tests](#235-physics-validation-tests)
  - [23.6 Security Tests](#236-security-tests)
  - [23.7 Performance Benchmarks](#237-performance-benchmarks)
  - [23.8 Headless Testing and CI](#238-headless-testing-and-ci)
  - [23.9 Running Tests with pytest](#239-running-tests-with-pytest)
  - [23.10 Test Coverage and Reporting](#2310-test-coverage-and-reporting)

- **[Chapter 24: Contributing and Development Workflow](#chapter-24-contributing-and-development-workflow)**

  - [24.1 Development Philosophy](#241-development-philosophy)
  - [24.2 Code Style and Formatting (Black, Ruff)](#242-code-style-and-formatting-black-ruff)
  - [24.3 Type Checking (mypy)](#243-type-checking-mypy)
  - [24.4 Pre-Commit Hooks](#244-pre-commit-hooks)
  - [24.5 Branch Strategy and Pull Requests](#245-branch-strategy-and-pull-requests)
  - [24.6 CI/CD Pipeline](#246-cicd-pipeline)
  - [24.7 Makefile Targets Reference](#247-makefile-targets-reference)

- **[Chapter 25: Deployment and Operations](#chapter-25-deployment-and-operations)**
  - [25.1 Production Deployment](#251-production-deployment)
  - [25.2 Docker Compose and Orchestration](#252-docker-compose-and-orchestration)
  - [25.3 Environment Variables Reference](#253-environment-variables-reference)
  - [25.4 Logging and Observability (structlog)](#254-logging-and-observability-structlog)
  - [25.5 Health Checks and Monitoring](#255-health-checks-and-monitoring)
  - [25.6 Security Hardening](#256-security-hardening)
  - [25.7 Backup and Disaster Recovery](#257-backup-and-disaster-recovery)

### Part VIII: Appendices

- **[Appendix A: Configuration Reference](#appendix-a-configuration-reference)**

  - [A.1 pyproject.toml Options](#a1-pyprojecttoml-options)
  - [A.2 environment.yml Specification](#a2-environmentyml-specification)
  - [A.3 Environment Variable Catalog](#a3-environment-variable-catalog)
  - [A.4 Makefile Target Reference](#a4-makefile-target-reference)

- **[Appendix B: Physics and Mathematics Reference](#appendix-b-physics-and-mathematics-reference)**

  - [B.1 Equations of Motion](#b1-equations-of-motion)
  - [B.2 Muscle Model Equations](#b2-muscle-model-equations)
  - [B.3 Aerodynamic Equations](#b3-aerodynamic-equations)
  - [B.4 Contact Mechanics](#b4-contact-mechanics)
  - [B.5 Coordinate Systems and Conventions](#b5-coordinate-systems-and-conventions)

- **[Appendix C: URDF Model Specifications](#appendix-c-urdf-model-specifications)**

  - [C.1 URDF File Format](#c1-urdf-file-format)
  - [C.2 Model Catalog](#c2-model-catalog)
  - [C.3 Custom Model Creation](#c3-custom-model-creation)

- **[Appendix D: Troubleshooting Guide](#appendix-d-troubleshooting-guide)**

  - [D.1 Common Installation Issues](#d1-common-installation-issues)
  - [D.2 Engine-Specific Problems](#d2-engine-specific-problems)
  - [D.3 GUI and Display Issues](#d3-gui-and-display-issues)
  - [D.4 API Server Issues](#d4-api-server-issues)
  - [D.5 Performance Optimization](#d5-performance-optimization)

- **[Appendix E: Glossary](#appendix-e-glossary)**

- **[Appendix F: References and Further Reading](#appendix-f-references-and-further-reading)**

- **[Appendix G: License](#appendix-g-license)**

---

---

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
   Tauri/Vue.js web-based desktop application, both providing access to all
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
- **Tauri/Vue.js Desktop Application**: Modern web-based desktop UI alongside
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
  identifiers (e.g., `launch_golf_suite.py`, `src/api/server.py`).

- **Bold text** denotes key terms, UI elements, or emphasis.

- _Italic text_ denotes book titles, foreign terms, or mild emphasis.

- Code blocks with language annotations show executable examples:

  ```python
  # Python example
  from src.shared.python.engine_manager import EngineManager
  manager = EngineManager()
  ```

  ```bash
  # Shell command example
  python launch_golf_suite.py --engine mujoco
  ```

- Mathematical notation uses LaTeX syntax. Inline: $E = \frac{1}{2}mv^2$.
  Display:

  $$\tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)$$

- **Warning** blocks indicate potential pitfalls or destructive operations.

- **Note** blocks provide supplementary information or tips.

- File paths are given relative to the repository root unless otherwise stated.
  The repository root is the directory containing `pyproject.toml` and
  `launch_golf_suite.py`.

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

# Chapter 2: Architecture Overview

## 2.1 High-Level System Architecture

UpstreamDrift is structured as a layered, modular system. The following diagram
illustrates the primary architectural layers from user-facing interfaces down to
the physics engine backends:

```
+===========================================================================+
|                        USER INTERFACE LAYER                                |
|  +-------------------+  +---------------------+  +---------------------+  |
|  | PyQt6 Classic GUI |  | Tauri/Vue.js Web UI |  | REST API (FastAPI)  |  |
|  +-------------------+  +---------------------+  +---------------------+  |
+===========================================================================+
                                    |
+===========================================================================+
|                        LAUNCHER & ORCHESTRATION                            |
|  +--------------------+  +------------------+  +----------------------+   |
|  | unified_launcher   |  | golf_launcher    |  | engine dashboards    |   |
|  +--------------------+  +------------------+  +----------------------+   |
+===========================================================================+
                                    |
+===========================================================================+
|                        SHARED SERVICES LAYER                               |
|  +-----------+ +----------+ +---------+ +----------+ +---------+          |
|  | Analysis  | | Plotting | | Optim.  | | Signal   | | UI Lib  |          |
|  +-----------+ +----------+ +---------+ +----------+ +---------+          |
|  +-----------+ +----------+ +---------+ +----------+ +---------+          |
|  | Contracts | | Logging  | | Config  | | Security | | Injury  |          |
|  +-----------+ +----------+ +---------+ +----------+ +---------+          |
+===========================================================================+
                                    |
+===========================================================================+
|                        ENGINE ABSTRACTION LAYER                            |
|  +--------------------------------------------------------------------+  |
|  |            Unified Engine Interface (PhysicsEngine Protocol)        |  |
|  +--------------------------------------------------------------------+  |
|  |  EngineManager  |  EngineRegistry  |  CrossEngineValidator          |  |
|  +--------------------------------------------------------------------+  |
+===========================================================================+
                                    |
+===========================================================================+
|                        PHYSICS ENGINE BACKENDS                             |
|  +---------+ +-------+ +-----------+ +---------+ +----------+            |
|  | MuJoCo  | | Drake | | Pinocchio | | OpenSim | | MyoSuite |            |
|  | (3.3+)  | | (1.22)| | (2.6+)    | |         | |          |            |
|  +---------+ +-------+ +-----------+ +---------+ +----------+            |
+===========================================================================+
                                    |
+===========================================================================+
|                        DOMAIN MODULES                                      |
|  +----------+ +-----------+ +------------+ +------------+                 |
|  | Robotics | | Learning  | | Deployment | | Research   |                 |
|  | contact  | | RL        | | realtime   | | MPC        |                 |
|  | control  | | imitation | | digital    | | diff.phys  |                 |
|  | locomot. | | sim2real  | |   twin     | | deformable |                 |
|  | sensing  | | retarget  | | safety     | | multi-robot|                 |
|  | planning | |           | | teleop     | |            |                 |
|  +----------+ +-----------+ +------------+ +------------+                 |
+===========================================================================+
```

### Layer Responsibilities

**User Interface Layer**: Provides three distinct interfaces for interacting with
the platform. The PyQt6 GUI offers a native desktop experience with rich widget
support. The Tauri/Vue.js application provides a modern web-based desktop experience.
The FastAPI REST API enables programmatic access, automation, and integration
with external systems.

**Launcher and Orchestration Layer**: Manages application startup, engine selection,
model loading, and inter-process communication. The `unified_launcher.py` module
serves as the primary orchestrator for the PyQt6 interface, while
`launch_golf_suite.py` provides the CLI entry point supporting all launch modes.

**Shared Services Layer**: Houses reusable libraries consumed by all engines and
modules, including biomechanical analysis routines, plotting utilities, optimization
solvers, signal processing tools, UI components, configuration management,
Design by Contract infrastructure, structured logging, and security utilities.

**Engine Abstraction Layer**: Defines the `PhysicsEngine` protocol (a Python
structural typing interface) and provides the `EngineManager` and `EngineRegistry`
for dynamic engine discovery, loading, and lifecycle management. The
`CrossEngineValidator` enables comparison of results across engines.

**Physics Engine Backends**: The five physics engines, each wrapped in an
adapter that conforms to the `PhysicsEngine` protocol. Each engine has its own
sub-directory under `src/engines/physics_engines/` containing Python bindings,
models, configuration files, and engine-specific utilities.

**Domain Modules**: Specialized modules for robotics (contact, control, locomotion,
sensing, planning), learning (RL, imitation, sim-to-real, retargeting), deployment
(real-time control, digital twin, safety, teleoperation), and research (MPC,
differentiable physics, deformable objects, multi-robot coordination).

## 2.2 Multi-Engine Design

The central architectural innovation of UpstreamDrift is its multi-engine design.
Rather than committing to a single physics simulation library, the platform
abstracts engine-specific details behind a common protocol and provides mechanisms
for running the same model across multiple engines.

### The PhysicsEngine Protocol

All engines conform to the `PhysicsEngine` protocol defined in
`src/shared/python/base_physics_engine.py`. This protocol specifies the minimum
interface that any engine adapter must implement:

```python
class PhysicsEngine(Protocol):
    """Protocol defining the interface for all physics engine adapters."""

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load a model file (URDF, MJCF, XML, etc.)."""
        ...

    def step(self, dt: float) -> None:
        """Advance the simulation by one time step."""
        ...

    def get_state(self) -> dict[str, np.ndarray]:
        """Return the current simulation state (q, qdot, etc.)."""
        ...

    def set_control(self, torques: np.ndarray) -> None:
        """Apply joint torques or control inputs."""
        ...

    def get_dynamics(self) -> dict[str, np.ndarray]:
        """Return dynamics matrices (M, C, g, etc.)."""
        ...

    def reset(self) -> None:
        """Reset the simulation to initial conditions."""
        ...
```

Each engine adapter (e.g., `MuJoCoAdapter`, `DrakeAdapter`, `PinocchioAdapter`)
implements this protocol, translating generic calls into engine-specific API
invocations. This allows analysis code to remain engine-agnostic:

```python
from src.shared.python.engine_manager import EngineManager

manager = EngineManager()
engine = manager.get_engine("mujoco")  # or "drake", "pinocchio", etc.
engine.load_model("models/humanoid_28dof.urdf")
engine.step(dt=0.001)
state = engine.get_state()
```

### Engine Capabilities Matrix

Not all engines support all features. The following matrix summarizes the
capabilities of each engine:

| Capability               | MuJoCo       | Drake   | Pinocchio  | OpenSim | MyoSuite |
| ------------------------ | ------------ | ------- | ---------- | ------- | -------- |
| Forward dynamics         | Yes          | Yes     | Yes        | Yes     | Yes      |
| Inverse dynamics         | Yes          | Yes     | Yes        | Yes     | Limited  |
| Inverse kinematics       | Yes          | Yes     | Yes        | Yes     | No       |
| Contact dynamics         | Yes          | Yes     | Limited    | Limited | Yes      |
| Trajectory optimization  | Limited      | Yes     | Via CasADi | No      | No       |
| Muscle models            | Via MyoSuite | No      | No         | Yes     | Yes      |
| URDF loading             | Via MJCF     | Yes     | Yes        | Via XML | Via MJCF |
| Real-time visualization  | Yes          | Meshcat | Meshcat    | GUI     | MuJoCo   |
| Differentiable dynamics  | Limited      | Yes     | Via CasADi | No      | No       |
| Maximum DOF (tested)     | 28           | 28      | 28         | 37      | 28       |
| Maximum muscles (tested) | 290          | 0       | 0          | 290     | 290      |

### Engine Selection Logic

The `EngineManager` implements a selection heuristic that considers:

1. **Requested features**: If the user needs muscle dynamics, engines without
   muscle support are excluded.
2. **Model format**: URDF models are natively supported by Drake and Pinocchio;
   MJCF models are natively supported by MuJoCo.
3. **Performance requirements**: Pinocchio is the fastest for pure rigid-body
   computations; MuJoCo offers the best balance of features and speed.
4. **Availability**: Only engines that are installed and importable are offered.

The `engine_availability.py` module probes the system at startup to determine
which engines are available, using lazy imports to avoid loading heavy native
libraries until they are actually needed.

## 2.3 Module Structure and Package Organization

The repository is organized into the following top-level directories:

```
UpstreamDrift/
├── src/                          # Main source code
│   ├── api/                      # FastAPI REST API server
│   │   ├── auth/                 #   Authentication (JWT, bcrypt)
│   │   ├── middleware/           #   Security headers, upload limits
│   │   ├── models/               #   Pydantic request/response models
│   │   ├── routes/               #   Endpoint definitions
│   │   ├── services/             #   Business logic services
│   │   └── utils/                #   Tracing, error codes, path validation
│   ├── config/                   # Global configuration
│   ├── deployment/               # Deployment modules
│   │   ├── digital_twin/         #   Digital twin estimator and twin
│   │   ├── realtime/             #   Real-time controller and state
│   │   ├── safety/               #   Collision detection, safety monitor
│   │   └── teleoperation/        #   Devices, teleoperation interface
│   ├── engines/                  # Physics engine backends
│   │   ├── common/               #   Shared engine utilities
│   │   ├── pendulum_models/      #   Educational pendulum models
│   │   ├── physics_engines/      #   Engine adapters
│   │   │   ├── drake/            #     Drake engine
│   │   │   ├── mujoco/           #     MuJoCo engine
│   │   │   ├── myosuite/         #     MyoSuite engine
│   │   │   ├── opensim/          #     OpenSim engine
│   │   │   ├── pinocchio/        #     Pinocchio engine
│   │   │   ├── pendulum/         #     Pendulum engine wrapper
│   │   │   └── putting_green/    #     Putting simulation
│   │   └── Simscape_Multibody_Models/  # MATLAB models
│   ├── launchers/                # GUI launchers and dashboards
│   ├── learning/                 # Machine learning modules
│   │   ├── imitation/            #   Imitation learning
│   │   ├── retargeting/          #   Motion retargeting
│   │   ├── rl/                   #   Reinforcement learning
│   │   └── sim2real/             #   Sim-to-real transfer
│   ├── research/                 # Research modules
│   │   ├── deformable/           #   Deformable object simulation
│   │   ├── differentiable/       #   Differentiable physics
│   │   ├── mpc/                  #   Model Predictive Control
│   │   └── multi_robot/          #   Multi-robot coordination
│   ├── robotics/                 # Robotics modules
│   │   ├── contact/              #   Contact models
│   │   ├── control/              #   Control algorithms
│   │   ├── core/                 #   Core robotics utilities
│   │   ├── locomotion/           #   Locomotion planning
│   │   ├── planning/             #   Motion planning
│   │   └── sensing/              #   Sensing and estimation
│   ├── shared/                   # Shared libraries
│   │   ├── matlab/               #   MATLAB utilities
│   │   ├── models/               #   Model files (URDF, MJCF, XML)
│   │   ├── python/               #   Python shared modules (~120 files)
│   │   ├── tools/                #   Development and analysis tools
│   │   └── urdf/                 #   URDF generation utilities
│   ├── tools/                    # CLI tools and utilities
│   └── unreal_integration/       # Unreal Engine integration
├── tests/                        # Test suite
│   ├── unit/                     #   Unit tests
│   ├── integration/              #   Integration tests
│   ├── cross_engine/             #   Cross-engine comparison tests
│   ├── physics_validation/       #   Physics accuracy tests
│   ├── security/                 #   Security tests
│   ├── benchmarks/               #   Performance benchmarks
│   ├── deployment/               #   Deployment tests
│   ├── learning/                 #   Learning module tests
│   ├── research/                 #   Research module tests
│   ├── analytical/               #   Analytical solution tests
│   ├── acceptance/               #   Acceptance tests
│   ├── headless/                 #   Headless (no-GUI) tests
│   └── fixtures/                 #   Test fixtures and data
├── ui/                           # Tauri/Vue.js web desktop app
│   ├── src/                      #   Vue.js source code
│   ├── src-tauri/                #   Tauri (Rust) backend
│   └── public/                   #   Static assets
├── docs/                         # Documentation
├── examples/                     # Example scripts
├── scripts/                      # Development and maintenance scripts
├── data/                         # Data files
├── assets/                       # Branding and static assets
├── installer/                    # OS-specific installers
├── vendor/                       # Vendored third-party code
├── launch_golf_suite.py          # Main CLI entry point
├── start_api_server.py           # API server entry point
├── pyproject.toml                # Project metadata and dependencies
├── environment.yml               # Conda environment specification
├── Makefile                      # Development task automation
├── Dockerfile                    # Container image definition
├── Dockerfile.unified            # Unified container image
└── conftest.py                   # Root pytest configuration
```

### The `src/shared/python/` Module

The `shared/python/` directory is the largest single module, containing
approximately 120 Python files that provide reusable functionality consumed across
all engines and analysis modules. Key components include:

- **`base_physics_engine.py`**: The `PhysicsEngine` protocol definition.
- **`engine_manager.py`**: Dynamic engine loading and lifecycle management.
- **`engine_registry.py`**: Engine capability registration and discovery.
- **`contracts.py`**: Design by Contract decorators and utilities.
- **`logging_config.py`**: Centralized structured logging via `structlog`.
- **`analysis/`**: Biomechanical analysis routines.
- **`plotting/`**: Visualization and chart generation (10+ plot types).
- **`optimization/`**: Numerical optimization solvers.
- **`signal_toolkit/`**: Signal processing utilities for time-series data.
- **`ui/`**: Reusable UI widgets and adapters (PyQt6).
- **`injury/`**: Injury risk assessment algorithms.
- **`spatial_algebra/`**: Spatial vector algebra for rigid-body computations.
- **`aerodynamics.py`**: Ball flight aerodynamic models.
- **`impact_model.py`**: Club-ball impact mechanics.
- **`hill_muscle.py`**: Hill-type muscle model implementation.
- **`multi_muscle.py`**: Multi-muscle coordination solver.
- **`flexible_shaft.py`**: Club shaft flexibility model.
- **`grip_contact_model.py`**: Two-handed grip contact analysis.
- **`ground_reaction_forces.py`**: Ground reaction force computation.
- **`terrain.py`** / **`terrain_engine.py`**: Terrain and topography models.

## 2.4 Design by Contract (DbC)

UpstreamDrift employs Design by Contract as a central quality assurance mechanism.
The `src/shared/python/contracts.py` module provides three decorator types that
enforce formal specifications at function and class boundaries.

### Preconditions

Preconditions specify what must be true **before** a function or method executes.
They validate input parameters and object state:

```python
from src.shared.python.contracts import precondition

@precondition(lambda self: self._is_initialized, "Engine must be initialized")
@precondition(lambda self, dt: dt > 0, "Time step must be positive")
def step(self, dt: float) -> None:
    """Advance the simulation by dt seconds."""
    ...
```

If a precondition fails, a `PreconditionViolationError` is raised with a clear
diagnostic message indicating which condition was violated and in which function.

### Postconditions

Postconditions specify what must be true **after** a function completes. They
validate return values and output state:

```python
from src.shared.python.contracts import postcondition

@postcondition(lambda result: result.shape[0] > 0, "Result must be non-empty")
@postcondition(
    lambda result: np.all(np.isfinite(result)),
    "Result must contain only finite values"
)
def compute_acceleration(self) -> np.ndarray:
    """Compute joint-space acceleration vector."""
    ...
```

### Class Invariants

Class invariants specify conditions that must hold true for every instance of a
class, checked after every public method call:

```python
from src.shared.python.contracts import invariant

@invariant(lambda self: self.mass > 0, "Mass must be positive")
@invariant(lambda self: len(self.joints) == self.n_dof, "Joint count must match DOF")
class RigidBody:
    """Represents a rigid body in the simulation."""
    ...
```

### Performance Considerations

Contract checking introduces runtime overhead proportional to the complexity of
the contract predicates. For performance-critical production deployments, contracts
can be globally disabled:

```python
from src.shared.python.contracts import CONTRACTS_ENABLED
import src.shared.python.contracts as contracts_module

# Disable all contract checking
contracts_module.CONTRACTS_ENABLED = False
```

When disabled, the contract decorators become zero-cost pass-throughs. In
development and testing, contracts should always remain enabled to catch
violations early.

## 2.5 Dependency Graph

The following simplified dependency graph shows the major inter-module
relationships. Arrows indicate "depends on" relationships:

```
launch_golf_suite.py
    ├── src/api/local_server
    │     └── src/api/server (FastAPI app)
    │           ├── src/api/routes/* (endpoints)
    │           ├── src/api/services/* (business logic)
    │           ├── src/api/auth/* (JWT, bcrypt)
    │           ├── src/api/middleware/* (security, upload)
    │           └── src/shared/python/engine_manager
    ├── src/launchers/golf_launcher (PyQt6)
    │     ├── src/launchers/unified_launcher
    │     ├── src/launchers/ui_components
    │     ├── src/launchers/model_registry
    │     └── src/shared/python/engine_availability
    └── src/engines/physics_engines/*/
          └── (engine-specific modules)

src/shared/python/ (consumed by all modules above)
    ├── base_physics_engine (Protocol)
    ├── engine_manager / engine_registry
    ├── contracts (DbC)
    ├── logging_config (structlog)
    ├── analysis/ plotting/ optimization/
    ├── biomechanics: hill_muscle, multi_muscle, grip_contact_model, ...
    ├── physics: aerodynamics, impact_model, flexible_shaft, terrain, ...
    └── security_utils, env_validator, config_utils, ...

src/robotics/ (contact, control, locomotion, sensing, planning)
    └── src/shared/python/

src/learning/ (rl, imitation, sim2real, retargeting)
    ├── src/engines/physics_engines/*/
    └── src/shared/python/

src/deployment/ (realtime, digital_twin, safety, teleoperation)
    ├── src/engines/physics_engines/*/
    └── src/shared/python/

src/research/ (mpc, differentiable, deformable, multi_robot)
    ├── src/engines/physics_engines/*/
    └── src/shared/python/
```

### External Dependencies

The platform depends on the following major external libraries:

| Category             | Library         | Version    | Purpose                           |
| -------------------- | --------------- | ---------- | --------------------------------- |
| Scientific Computing | NumPy           | >= 1.26.4  | Array computation, linear algebra |
| Scientific Computing | SciPy           | >= 1.13.1  | Optimization, integration, signal |
| Scientific Computing | pandas          | >= 2.2.3   | Data manipulation and analysis    |
| Scientific Computing | SymPy           | >= 1.12    | Symbolic mathematics              |
| Physics Engine       | MuJoCo          | >= 3.3.0   | Primary physics engine            |
| Physics Engine       | Drake           | >= 1.22.0  | Trajectory optimization engine    |
| Physics Engine       | Pinocchio (pin) | >= 2.6.0   | Fast rigid-body algorithms        |
| Web Framework        | FastAPI         | >= 0.126.0 | REST API framework                |
| Web Server           | Uvicorn         | >= 0.30.0  | ASGI server                       |
| Data Validation      | Pydantic        | >= 2.5.0   | Request/response model validation |
| Database             | SQLAlchemy      | >= 2.0.0   | ORM and database abstraction      |
| Authentication       | PyJWT           | >= 2.10.1  | JWT token handling                |
| Password Hashing     | bcrypt          | >= 4.1.0   | Secure password/key hashing       |
| Rate Limiting        | SlowAPI         | >= 0.1.9   | API rate limiting                 |
| GUI                  | PyQt6           | >= 6.5.0   | Desktop GUI framework             |
| Visualization        | Matplotlib      | >= 3.8.0   | Plotting and chart generation     |
| Computer Vision      | OpenCV          | >= 4.8.0   | Image/video processing            |
| Logging              | structlog       | >= 24.1.0  | Structured logging                |
| Data Formats         | PyYAML          | >= 6.0.1   | YAML configuration parsing        |
| Data Formats         | defusedxml      | >= 0.7.1   | Secure XML parsing                |
| Data Formats         | ezc3d           | >= 1.4.0   | C3D motion capture file support   |
| Security             | cryptography    | >= 44.0.1  | Cryptographic operations          |
| Expression Eval      | simpleeval      | >= 1.0.0   | Safe expression evaluation        |
| HTTP Client          | httpx           | >= 0.27.0  | Async HTTP client for testing     |

## 2.6 Data Flow and Communication Patterns

### Simulation Data Flow

A typical simulation run follows this data flow:

1. **Model Loading**: The user selects an engine and model. The `EngineManager`
   loads the appropriate model file (URDF, MJCF, or XML) into the selected
   physics engine via the `PhysicsEngine.load_model()` method.

2. **Initialization**: Initial conditions (joint positions $q_0$, velocities
   $\dot{q}_0$) are set. Muscle activation levels, control parameters, and
   environmental conditions are configured.

3. **Simulation Loop**: For each time step $\Delta t$:

   - Control inputs (torques $\tau$ or muscle activations $a$) are computed
     by the control module or provided by the user.
   - The engine advances the state: $q_{k+1}, \dot{q}_{k+1} = f(q_k, \dot{q}_k, \tau_k, \Delta t)$.
   - Contact forces $F_{\text{ext}}$ are resolved by the engine's contact solver.
   - State data is recorded by the analysis pipeline.

4. **Post-Processing**: After simulation, the analysis module computes derived
   quantities (energies, forces, injury risk metrics) and generates plots.

5. **Export**: Results are saved to CSV, JSON, or the SQLite database
   (`golf_modeling_suite.db`).

### API Communication Pattern

When accessed via the REST API, the data flow is:

```
Client (Browser/Script)
    |
    | HTTP/WebSocket (JSON)
    v
FastAPI Server (src/api/server.py)
    |
    | JWT Authentication + Rate Limiting
    v
Route Handler (src/api/routes/*.py)
    |
    | Pydantic Validation
    v
Service Layer (src/api/services/*.py)
    |
    | Engine Abstraction
    v
EngineManager -> PhysicsEngine Adapter -> Engine Backend
    |
    | Results (numpy arrays)
    v
Response Serialization (Pydantic) -> JSON -> Client
```

## 2.7 Security Architecture

The security architecture implements defense in depth:

**Authentication**: JWT (JSON Web Tokens) with configurable expiration.
Tokens are signed using the `SECRET_KEY` environment variable and verified
on every authenticated request via the `src/api/auth/dependencies.py` module.

**Password/Key Hashing**: bcrypt with automatic salt generation. The platform
previously used SHA256 (fast hash, vulnerable to brute-force); this was upgraded
in v2.1.0 as a critical security fix.

**Input Validation**: All API inputs are validated through Pydantic models with
strict type checking and field constraints. File uploads are subject to size
limits enforced by `src/api/middleware/upload_limits.py`. File paths are validated
against directory traversal attacks via `src/api/utils/path_validation.py`.

**Security Headers**: The `src/api/middleware/security_headers.py` module adds
standard security headers (X-Content-Type-Options, X-Frame-Options,
Content-Security-Policy, etc.) to all responses.

**Rate Limiting**: The SlowAPI middleware limits request rates per IP address
to prevent abuse and denial-of-service attacks.

**CORS**: Cross-Origin Resource Sharing is restricted to explicitly configured
origins via the `get_cors_origins()` configuration function.

**Trusted Hosts**: The `TrustedHostMiddleware` ensures that requests are only
accepted from configured hostnames, preventing host header injection.

**Safe Expression Evaluation**: The `simpleeval` library replaces Python's
built-in `eval()` for any user-provided mathematical expressions, preventing
arbitrary code execution (CWE-94 mitigation).

**XML Security**: All XML parsing uses `defusedxml` to prevent XML external
entity (XXE) attacks and billion-laughs denial-of-service attacks.

---

# Chapter 3: Installation and Setup

## 3.1 System Prerequisites

Before installing UpstreamDrift, ensure your system meets the following
requirements:

### Required Software

| Software           | Minimum Version | Recommended Version | Notes                        |
| ------------------ | --------------- | ------------------- | ---------------------------- |
| Python             | 3.11            | 3.12                | Python 3.13 also supported   |
| Git                | 2.30+           | Latest              | Git LFS support required     |
| Git LFS            | 2.13+           | Latest              | For large model files        |
| pip                | 23.0+           | Latest              | Python package manager       |
| conda (optional)   | 23.0+           | Latest (Miniforge)  | Recommended for binary deps  |
| Docker (optional)  | 24.0+           | Latest              | For containerized deployment |
| Node.js (optional) | 18+             | 20 LTS              | Only for Tauri/Vue.js UI dev |
| Rust (optional)    | 1.70+           | Latest              | Only for Tauri desktop app   |

### Required System Libraries

**Linux (Ubuntu/Debian)**:

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    libeigen3-dev \
    patchelf \
    ffmpeg \
    xvfb
```

**macOS**:

```bash
xcode-select --install
brew install cmake eigen ffmpeg
```

**Windows**:

- Visual C++ Redistributable 2015-2022 (required for MuJoCo)
- CMake (for building native extensions)
- Git for Windows with Git LFS

### Optional Software

| Software           | Purpose                               |
| ------------------ | ------------------------------------- |
| MATLAB R2023a+     | Simscape Multibody models (2D/3D)     |
| Simulink           | Required by MATLAB models             |
| Simscape Multibody | Required by MATLAB models             |
| CUDA Toolkit       | GPU acceleration for learning modules |
| Xvfb               | Headless rendering on Linux servers   |

### Hardware Recommendations

| Component | Minimum      | Recommended           | Notes                                  |
| --------- | ------------ | --------------------- | -------------------------------------- |
| CPU       | 4 cores      | 8+ cores              | Multi-core for parallel simulation     |
| RAM       | 8 GB         | 16+ GB                | Musculoskeletal models need 8+ GB      |
| Storage   | 10 GB free   | 20+ GB free           | Models, data, conda env                |
| GPU       | Not required | NVIDIA (CUDA capable) | Only for RL and differentiable physics |
| Display   | 1920x1080    | 2560x1440+            | For GUI visualization                  |

## 3.2 Installation via Conda (Recommended)

Conda installation is the recommended approach because it correctly handles
binary dependencies such as MuJoCo, OpenCV, and PyQt6 that can be problematic
with pip on certain platforms.

### Step 1: Clone the Repository

```bash
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
git lfs install && git lfs pull
```

### Step 2: Create the Conda Environment

The `environment.yml` file defines the complete environment, including conda
packages (for binary dependencies) and pip packages (for Python-only libraries):

```bash
# Full installation (all engines and tools)
conda env create -f environment.yml
```

This creates a conda environment named `golf-suite` with Python 3.11 and all
required dependencies. The environment includes:

- **Core scientific stack**: NumPy, SciPy, pandas, SymPy, Matplotlib
- **Physics engines**: MuJoCo (via conda), with Drake and Pinocchio available
  as optional uncommented entries
- **GUI framework**: PyQt6 (via conda, ensuring binary compatibility)
- **Web framework**: FastAPI, Uvicorn, SQLAlchemy, PyJWT, bcrypt (via pip)
- **Development tools**: pytest, black, mypy, ruff, pre-commit
- **Data formats**: PyYAML, defusedxml, ezc3d
- **Security**: structlog, simpleeval, cryptography

### Step 3: Activate the Environment

```bash
conda activate golf-suite
```

### Step 4: Install the Package in Editable Mode

The `environment.yml` includes `-e .` in its pip section, which installs the
UpstreamDrift package in editable (development) mode. If this step did not
execute during environment creation, run it manually:

```bash
pip install -e ".[dev]"
```

### Step 5: (Optional) Enable Additional Engines

To install Drake and Pinocchio, either uncomment the relevant lines in
`environment.yml` and run `conda env update -f environment.yml --prune`, or
install them separately:

```bash
# Drake
pip install drake>=1.22.0

# Pinocchio (recommended via conda for binary compatibility)
conda install -c conda-forge pinocchio crocoddyl
```

### Updating an Existing Environment

```bash
conda env update -f environment.yml --prune
```

The `--prune` flag removes packages that are no longer listed in the
environment file, keeping the environment clean.

## 3.3 Installation via Pip

For users who prefer pip over conda, or when conda is not available:

### Basic Installation

```bash
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
git lfs install && git lfs pull

# Install with core dependencies only
pip install -e .
```

This installs the minimum required packages: NumPy, SciPy, FastAPI, Uvicorn,
Pydantic, httpx, MuJoCo, simpleeval, and structlog.

### Installation with Optional Dependencies

```bash
# Install with development tools
pip install -e ".[dev]"

# Install with all physics engines
pip install -e ".[all-engines]"

# Install with analysis extras (OpenCV, scikit-learn)
pip install -e ".[analysis]"

# Install with URDF generation tools
pip install -e ".[urdf]"

# Install with GUI test support
pip install -e ".[gui-test]"

# Install everything
pip install -e ".[all]"
```

### Light Installation (UI Development)

For frontend developers who only need to work on the UI without running
physics engines:

```bash
pip install -e .
export GOLF_USE_MOCK_ENGINE=1
```

The `GOLF_USE_MOCK_ENGINE` environment variable causes the platform to use a
mock engine that returns synthetic data, eliminating the need for heavy physics
libraries during UI development.

## 3.4 Docker Installation

Docker provides an isolated, reproducible environment that is particularly useful
for CI/CD, deployment, and users who want to avoid managing system dependencies.

### Building the Docker Image

```bash
# Standard image (MuJoCo + core tools)
docker build -t upstreamdrift:latest .

# Unified image (MuJoCo + Drake + Pinocchio + full toolchain)
docker build -f Dockerfile.unified -t upstreamdrift:unified .
```

The Dockerfile uses a multi-stage build:

1. **Builder stage** (`continuumio/miniconda3:24.11.1-0`): Installs build tools,
   compiles native extensions, and installs the full conda environment including
   MuJoCo, Drake, and Pinocchio.

2. **Runtime stage** (`continuumio/miniconda3:24.11.1-0`): Copies only the
   installed conda environment from the builder, installs minimal runtime
   dependencies (GL libraries, FFmpeg, curl), creates a non-root user (`golfer`),
   and sets up the workspace.

### Running the Container

```bash
# Interactive shell
docker run -it --rm \
    -p 8000:8000 \
    -v $(pwd):/workspace \
    upstreamdrift:latest

# Start API server
docker run -d --name upstreamdrift-api \
    -p 8000:8000 \
    -e SECRET_KEY="your-secret-key" \
    -e ENVIRONMENT="production" \
    upstreamdrift:latest \
    python start_api_server.py

# Run tests
docker run --rm \
    upstreamdrift:latest \
    pytest tests/ -v --tb=short
```

### Docker Environment

The container configures the following environment:

- `PYTHONPATH=/workspace:/workspace/shared/python:/workspace/engines`
- Working directory: `/workspace`
- User: `golfer` (UID 1000, non-root for security)
- Exposed port: `8000` (API server)
- Health check: `curl -f http://localhost:8000/api/health` every 30 seconds

### Headless Rendering

For servers without a display, the container includes Xvfb and OSMesa for
software-based OpenGL rendering:

```bash
# Run with virtual framebuffer for visualization tests
docker run --rm \
    upstreamdrift:latest \
    xvfb-run pytest tests/ -v
```

## 3.5 Environment Configuration

UpstreamDrift uses environment variables for runtime configuration, supporting
both development and production deployments.

### Core Environment Variables

| Variable               | Default                            | Description                               |
| ---------------------- | ---------------------------------- | ----------------------------------------- |
| `SECRET_KEY`           | (auto-generated)                   | JWT signing secret (set in production!)   |
| `GOLF_API_SECRET_KEY`  | (fallback for above)               | Alternative name for JWT secret           |
| `GOLF_ADMIN_PASSWORD`  | (auto-generated)                   | Admin user password                       |
| `DATABASE_URL`         | `sqlite:///golf_modeling_suite.db` | SQLAlchemy database URL                   |
| `ENVIRONMENT`          | `development`                      | `development` or `production`             |
| `API_HOST`             | `127.0.0.1`                        | API server bind address                   |
| `API_PORT`             | `8000`                             | API server bind port                      |
| `GOLF_PORT`            | `8000`                             | Alternative port configuration            |
| `GOLF_NO_BROWSER`      | (unset)                            | If set, don't auto-open browser           |
| `GOLF_USE_MOCK_ENGINE` | (unset)                            | If set, use mock engine (no real physics) |
| `GOLF_DEFAULT_ENGINE`  | (unset)                            | Default engine for web UI                 |
| `GOLF_LOG_LEVEL`       | `INFO`                             | Logging level (DEBUG, INFO, WARNING, etc) |

### Security Configuration

> **Warning**: In production, you **must** set `SECRET_KEY` to a strong,
> randomly generated value. Failure to do so will result in an auto-generated
> key that changes on each restart, invalidating all existing JWT tokens.

Generate a secure key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

Set it in your environment:

```bash
export SECRET_KEY="your-generated-secret-key-here"
export GOLF_ADMIN_PASSWORD="your-strong-admin-password"
export ENVIRONMENT="production"
```

### Configuration Precedence

Configuration values are resolved in the following order (highest priority first):

1. Command-line arguments (e.g., `--port 9000`)
2. Environment variables (e.g., `API_PORT=9000`)
3. `.env` file in the repository root (if present)
4. Default values defined in source code

## 3.6 Verifying the Installation

After installation, run the verification script to confirm that all core
dependencies are available and functional:

```bash
python scripts/verify_installation.py
```

This script checks:

1. **Core scientific libraries**: NumPy, SciPy, pandas, Matplotlib, SymPy
2. **GUI framework**: PyQt6
3. **Physics engines**: MuJoCo (required), Drake (optional), Pinocchio (optional)
4. **Web framework**: FastAPI, Uvicorn
5. **Data format libraries**: PyYAML, defusedxml
6. **Security libraries**: PyJWT, bcrypt, cryptography

Expected output for a complete installation:

```
============================================================
Golf Modeling Suite - Installation Verification
============================================================

Checking core dependencies:
----------------------------------------
  numpy (v1.26.4)
  scipy (v1.13.1)
  pandas (v2.2.3)
  matplotlib (v3.8.5)
  sympy (v1.12)
  PyQt6 (v6.6.1)
  mujoco (v3.3.0)
  fastapi (v0.126.0)
  uvicorn (v0.30.1)
  yaml (v6.0.1)
  defusedxml (v0.7.1)

Checking physics engines:
----------------------------------------
  MuJoCo: Available
  Drake:  Available
  Pinocchio: Available
  OpenSim: Not installed (optional)
  MyoSuite: Not installed (optional)

============================================================
Installation verified: All core checks passed.
============================================================
```

### Running the Test Suite

For a more thorough verification, run the test suite:

```bash
# Quick verification (unit tests only, ~2 minutes)
make test-unit

# Full verification (all tests, ~10 minutes)
make test

# Specific test category
pytest tests/physics_validation/ -v --tb=short
```

### Checking System Health

The system health checker provides a comprehensive diagnostic:

```bash
python scripts/check_system_health.py
```

## 3.7 Platform-Specific Notes

### Linux

- **OpenGL**: Ensure Mesa OpenGL libraries are installed for MuJoCo visualization:
  ```bash
  sudo apt install libgl1-mesa-glx libosmesa6-dev
  ```
- **Headless servers**: Use Xvfb for rendering on servers without a display:
  ```bash
  xvfb-run python launch_golf_suite.py
  xvfb-run pytest tests/ -v
  ```
- **WSL (Windows Subsystem for Linux)**: Refer to `docs/wsl_setup.md` for
  WSL-specific configuration including display forwarding.

### macOS

- **Apple Silicon (M1/M2/M3)**: For ARM64-native conda environments:
  ```bash
  CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml
  ```
  Some engines (particularly Drake) may not yet have ARM64 builds. In this case,
  use the x86_64 conda subdir with Rosetta 2 emulation:
  ```bash
  CONDA_SUBDIR=osx-64 conda env create -f environment.yml
  ```
- **XQuartz**: Required for X11 forwarding if using SSH-based remote development.

### Windows

- **Visual C++ Redistributable**: MuJoCo requires the Visual C++ Redistributable
  2015-2022. Download from Microsoft's website if OpenGL errors occur.
- **Graphics drivers**: Update to the latest GPU drivers if you encounter
  OpenGL-related crashes.
- **Path length**: Windows has a 260-character path limit by default. Enable
  long paths via Group Policy or registry if you encounter path-related errors.
- **WSL recommended**: For the best experience on Windows, consider using WSL2
  with Ubuntu. See `run_wsl.sh` for a convenience launcher.

## 3.8 Development Environment Setup

For contributors and developers extending the platform:

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs pytest, pytest-cov, pytest-mock, pytest-timeout, pytest-xdist,
ruff, mypy, and build tools.

### Set Up Pre-Commit Hooks

```bash
pip install pre-commit
pre-commit install
```

The pre-commit configuration runs the following checks on every commit:

- **ruff**: Fast Python linter (checks for errors, style issues, import ordering)
- **black**: Code formatter (line length 88, target Python 3.11)
- **mypy**: Static type checker (advisory mode)
- **Trailing whitespace removal**
- **YAML validation**
- **Large file detection**

### IDE Configuration

**VS Code** (recommended):

- Install the Python, Pylance, and Ruff extensions.
- The project includes `pyproject.toml` settings that VS Code and Pylance
  will automatically detect for linting and formatting.

**PyCharm**:

- Mark `src/` as a source root.
- Configure the Python interpreter to use the `golf-suite` conda environment.
- Enable ruff as an external tool for linting.

### Makefile Targets

The `Makefile` provides convenient shortcuts for common development tasks:

```bash
make help      # Show all available targets
make install   # Install dependencies (pip install -r requirements.txt && pip install -e ".[dev]")
make lint      # Run ruff check and mypy
make format    # Run black, ruff format, ruff check --fix
make test      # Run full test suite (pytest tests/ -v --tb=short)
make test-unit # Run unit tests only
make test-int  # Run integration tests only
make check     # Run lint + test
make docs      # Build Sphinx documentation
make clean     # Remove __pycache__, .pytest_cache, .mypy_cache, build artifacts
make all       # install + format + lint + test
```

## 3.9 Troubleshooting Installation Issues

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'mujoco'`

**Solution**: Ensure MuJoCo is installed. With conda: `conda install -c conda-forge mujoco`.
With pip: `pip install mujoco>=3.3.0`.

---

**Problem**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution**: Install Mesa OpenGL libraries:

```bash
sudo apt install libgl1-mesa-glx libgl1-mesa-dev
```

---

**Problem**: `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`

**Solution**: Install XCB libraries or set the platform to offscreen:

```bash
sudo apt install libxcb-xinerama0 libxcb-cursor0
# or for headless:
export QT_QPA_PLATFORM=offscreen
```

---

**Problem**: Conda environment creation fails with solver conflicts.

**Solution**: Use the libmamba solver for faster, more reliable dependency resolution:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -f environment.yml
```

---

**Problem**: `pip install -e .` fails with `build-backend` errors.

**Solution**: Ensure hatchling is installed: `pip install hatchling`. Then retry
the installation.

---

For additional troubleshooting, consult `docs/troubleshooting/installation.md`.

---

# Chapter 4: Quick Start Guide

This chapter walks through the essential workflows to get you productive with
UpstreamDrift as quickly as possible. Each section includes executable commands
and expected output.

## 4.1 Running Your First Simulation

The fastest way to run a simulation is through the example scripts in the
`examples/` directory.

### Example 1: Basic Simulation

```bash
python examples/01_basic_simulation.py
```

This script:

1. Initializes the `EngineManager` and selects the MuJoCo engine (default).
2. Loads a 7 DOF humanoid golf model.
3. Applies a sequence of joint torques representing a simplified golf swing.
4. Runs the simulation for 2 seconds at $\Delta t = 0.001$ s (2000 time steps).
5. Records joint positions, velocities, and energies.
6. Generates plots of the swing trajectory and energy profile.

### Example 2: Parameter Sweeps

```bash
python examples/02_parameter_sweeps.py
```

This script demonstrates how to sweep over model parameters (e.g., club mass,
shaft stiffness, grip friction) and observe their effect on club head velocity
at impact. The parameter sweep uses the kinetic energy at impact as the
objective function:

$$E_{\text{impact}} = \frac{1}{2} m_{\text{club}} v_{\text{club}}^2$$

where $m_{\text{club}}$ is the club head mass and $v_{\text{club}}$ is the
club head velocity at the moment of ball contact.

### Example 3: Injury Risk Tutorial

```bash
python examples/03_injury_risk_tutorial.py
```

This script demonstrates the injury risk assessment pipeline, computing joint
stress metrics, spinal load estimates, and soft tissue strain indicators across
a full swing cycle.

### Programmatic Simulation

For custom simulations, use the Python API directly:

```python
#!/usr/bin/env python3
"""Minimal simulation example."""

import numpy as np
from src.shared.python.engine_manager import EngineManager

# 1. Initialize the engine
manager = EngineManager()
engine = manager.get_engine("mujoco")

# 2. Load a model
engine.load_model("src/shared/models/urdf/humanoid_7dof.urdf")

# 3. Set initial conditions
engine.reset()

# 4. Simulation loop
dt = 0.001  # 1 ms time step
duration = 2.0  # 2 seconds
n_steps = int(duration / dt)

trajectory = []
for step in range(n_steps):
    # Apply torques (example: sinusoidal torque on shoulder joint)
    t = step * dt
    torques = np.zeros(7)
    torques[0] = 50.0 * np.sin(2 * np.pi * t / duration)  # Shoulder

    engine.set_control(torques)
    engine.step(dt)

    state = engine.get_state()
    trajectory.append(state)

# 5. Access results
print(f"Simulation complete: {n_steps} steps")
print(f"Final joint positions: {trajectory[-1]['q']}")
print(f"Final joint velocities: {trajectory[-1]['qdot']}")
```

## 4.2 Launching the GUI (PyQt6 Classic Launcher)

The classic PyQt6 launcher provides a native desktop interface with engine
selection, model browsing, visualization, and analysis tools.

### Launch Command

```bash
# Via the main entry point
python launch_golf_suite.py --classic

# Or equivalently
golf-suite --classic
```

### GUI Features

The classic launcher provides:

- **Engine Selection Panel**: Choose from available engines (MuJoCo, Drake,
  Pinocchio, etc.) with capability indicators.
- **Model Browser**: Browse and select from available models organized by
  complexity (pendulum, intermediate, advanced, musculoskeletal).
- **Simulation Controls**: Start, pause, reset, and step through simulations.
  Configure time step, duration, and integration parameters.
- **Visualization Viewport**: Real-time 3D rendering with multiple camera views,
  force/torque vector overlays, and manipulability ellipsoids.
- **Analysis Panel**: Access inverse kinematics, inverse dynamics, kinematic
  sequence analysis, energy plots, and data export tools.
- **Engine Dashboards**: Dedicated dashboards for MuJoCo, Drake, and Pinocchio
  with engine-specific controls and visualizations.

### Startup Sequence

The launcher uses an async startup system with a splash screen:

1. **Splash screen displayed immediately** (< 100 ms after launch).
2. **Background worker thread** loads heavy modules (MuJoCo, engine probes).
3. **Progress bar updates** reflect actual loading progress.
4. **Main window opens** with pre-loaded resources (no duplicate loading).

### Keyboard Shortcuts

| Shortcut | Action                   |
| -------- | ------------------------ |
| `Space`  | Start / pause simulation |
| `R`      | Reset simulation         |
| `S`      | Step forward one frame   |
| `Ctrl+E` | Export current data      |
| `Ctrl+P` | Capture screenshot       |
| `1`-`5`  | Switch camera view       |
| `Ctrl+Q` | Quit application         |

## 4.3 Launching the Web UI (Tauri/Vue.js)

The modern web-based desktop application uses Tauri (Rust backend) and Vue.js
(TypeScript frontend) for a contemporary user experience.

### Launch Command (Default)

```bash
# Default launch mode (recommended)
python launch_golf_suite.py

# Or equivalently
golf-suite
```

This starts the FastAPI backend server and opens the web UI in your default
browser at `http://localhost:8000`.

### Development Mode

For active UI development with hot-reload:

```bash
cd ui
npm install
npm run dev
```

This starts the Vite development server with hot module replacement (HMR).
The UI source code is in `ui/src/` and uses:

- **Vue.js 3** with Composition API
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Vite** for fast builds and HMR
- **Vitest** for unit testing

### Building the Desktop Application

To build a native desktop application using Tauri:

```bash
cd ui
npm install
npm run tauri build
```

This produces a standalone executable for your platform (Linux AppImage, macOS
.dmg, Windows .exe) that bundles the FastAPI backend and Vue.js frontend into
a single distributable application.

## 4.4 Starting the API Server

The REST API provides programmatic access to all simulation and analysis
capabilities.

### Launch the API Server

```bash
# Via the main entry point
python launch_golf_suite.py --api-only

# Or directly
python start_api_server.py

# Or via the golf-suite CLI
golf-suite --api-only --port 8000
```

### API Documentation

Once the server is running, interactive documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/api/health`

### Quick API Test

```bash
# Check server health
curl http://localhost:8000/api/health

# Authenticate (get JWT token)
curl -X POST http://localhost:8000/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username": "admin", "password": "your-password"}'

# List available engines (authenticated)
curl http://localhost:8000/api/engines \
    -H "Authorization: Bearer <your-jwt-token>"

# Start a simulation
curl -X POST http://localhost:8000/api/simulation/start \
    -H "Authorization: Bearer <your-jwt-token>" \
    -H "Content-Type: application/json" \
    -d '{
        "engine": "mujoco",
        "model": "humanoid_7dof",
        "duration": 2.0,
        "dt": 0.001
    }'
```

### API Server Configuration

The API server accepts the following command-line arguments and environment
variables:

| Argument       | Environment Variable | Default       | Description                |
| -------------- | -------------------- | ------------- | -------------------------- |
| `--port`       | `API_PORT`           | `8000`        | Server port                |
| (n/a)          | `API_HOST`           | `127.0.0.1`   | Server bind address        |
| (n/a)          | `ENVIRONMENT`        | `development` | `development`/`production` |
| (n/a)          | `SECRET_KEY`         | (generated)   | JWT signing key            |
| `--no-browser` | `GOLF_NO_BROWSER`    | (unset)       | Suppress browser launch    |

In development mode (`ENVIRONMENT=development`), the server enables auto-reload,
which watches for file changes and automatically restarts.

## 4.5 Basic Workflows

### Workflow 1: Compare Engines on the Same Model

This workflow runs the same simulation on multiple engines and compares results:

```python
from src.shared.python.engine_manager import EngineManager
from src.shared.python.comparative_analysis import ComparativeAnalysis

manager = EngineManager()
model_path = "src/shared/models/urdf/humanoid_7dof.urdf"

results = {}
for engine_name in ["mujoco", "drake", "pinocchio"]:
    engine = manager.get_engine(engine_name)
    engine.load_model(model_path)
    engine.reset()

    # Run simulation
    for _ in range(2000):
        engine.step(dt=0.001)

    results[engine_name] = engine.get_state()

# Compare
analyzer = ComparativeAnalysis()
comparison = analyzer.compare(results)
analyzer.plot_comparison(comparison)
```

### Workflow 2: Motion Capture to Simulation

This workflow imports motion capture data and drives a simulation model:

```python
from src.shared.python.video_pose_pipeline import (
    VideoPosePipeline,
    VideoProcessingConfig,
)

# Configure the pipeline
config = VideoProcessingConfig(
    input_path="data/swing_video.mp4",
    pose_method="mediapipe",
    output_format="json",
)

# Run pose estimation
pipeline = VideoPosePipeline(config)
pose_data = pipeline.process()

# Retarget to simulation model
from src.learning.retargeting import MotionRetargeter

retargeter = MotionRetargeter(
    source_skeleton="mediapipe",
    target_model="humanoid_28dof",
)
joint_angles = retargeter.retarget(pose_data)
```

### Workflow 3: Injury Risk Assessment

```python
from src.shared.python.injury import InjuryRiskAssessor

assessor = InjuryRiskAssessor()

# Load simulation results (from any engine)
trajectory = load_simulation_results("output/swing_data.json")

# Assess injury risk
report = assessor.assess(trajectory)

print(f"Overall risk level: {report.overall_risk}")
print(f"Peak spinal load: {report.spinal_load_peak:.1f} N")
print(f"Shoulder stress: {report.shoulder_stress:.1f} Nm")
print(f"Wrist strain index: {report.wrist_strain:.3f}")
```

### Workflow 4: Ball Flight Prediction

Given club head state at impact, predict the full ball flight trajectory:

```python
from src.shared.python.aerodynamics import BallFlightModel
from src.shared.python.impact_model import ImpactModel

# Compute ball launch conditions from impact
impact = ImpactModel()
launch = impact.compute(
    club_speed=45.0,          # m/s
    attack_angle=-3.0,        # degrees (negative = descending)
    club_path=2.0,            # degrees (positive = in-to-out)
    face_angle=1.0,           # degrees (positive = open)
    dynamic_loft=12.0,        # degrees
)

# Simulate ball flight
flight = BallFlightModel()
trajectory = flight.simulate(
    launch_speed=launch.ball_speed,     # m/s
    launch_angle=launch.launch_angle,   # degrees
    spin_rate=launch.spin_rate,         # rpm
    spin_axis=launch.spin_axis,         # degrees
    wind_speed=5.0,                     # m/s
    wind_direction=180.0,               # degrees (headwind)
    altitude=0.0,                       # meters above sea level
)

print(f"Carry distance: {trajectory.carry:.1f} m")
print(f"Total distance: {trajectory.total:.1f} m")
print(f"Max height: {trajectory.apex:.1f} m")
print(f"Flight time: {trajectory.time:.2f} s")
```

The aerodynamic model accounts for:

- Drag force: $F_D = \frac{1}{2} \rho C_D A v^2$
- Lift force (Magnus effect): $F_L = \frac{1}{2} \rho C_L A v^2$
- Gravitational force: $F_g = m g$

where $\rho$ is air density, $C_D$ and $C_L$ are drag and lift coefficients
(dependent on Reynolds number $Re = \frac{\rho v d}{\mu}$ and spin ratio
$S = \frac{\omega r}{v}$), $A$ is the cross-sectional area, and $v$ is the
ball velocity.

## 4.6 Exploring the Example Scripts

The `examples/` directory contains annotated scripts for common use cases:

| Script                       | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| `01_basic_simulation.py`     | Simple forward dynamics simulation                |
| `02_parameter_sweeps.py`     | Sweep over model parameters and visualize effects |
| `03_injury_risk_tutorial.py` | Injury risk assessment pipeline walkthrough       |
| `motion_training_demo.py`    | Motion capture import and retargeting demo        |

Each script includes detailed comments explaining the workflow, parameters,
and expected output. They are designed to be run directly:

```bash
python examples/01_basic_simulation.py
```

## 4.7 Understanding Output and Results

### Output Directory

Simulation results are stored in the `output/` directory by default. Each
simulation run creates a timestamped subdirectory containing:

```
output/
├── 2026-02-05_143022_mujoco_7dof/
│   ├── trajectory.csv          # Joint positions and velocities over time
│   ├── dynamics.csv            # Forces, torques, energies over time
│   ├── metadata.json           # Simulation parameters and configuration
│   ├── plots/                  # Generated visualization plots
│   │   ├── joint_angles.png
│   │   ├── energy_profile.png
│   │   ├── phase_diagram.png
│   │   └── 3d_trajectory.png
│   └── analysis/               # Derived analysis results
│       ├── kinematic_sequence.json
│       ├── peak_torques.csv
│       └── injury_risk_report.json
```

### Database Storage

The SQLite database (`golf_modeling_suite.db`) in the repository root stores
structured simulation results, user configurations, and analysis metadata.
It is accessed by the API server and can be queried directly:

```bash
sqlite3 golf_modeling_suite.db ".tables"
sqlite3 golf_modeling_suite.db "SELECT * FROM simulations LIMIT 5;"
```

### Data Export Formats

Results can be exported in two formats:

- **CSV**: Tabular data with headers, suitable for spreadsheet analysis and
  external tools (MATLAB, R, Excel).
- **JSON**: Hierarchical data with full metadata, suitable for programmatic
  consumption and API responses.

Export is available through the GUI (File > Export), the API (`/api/export/`
endpoints), and programmatically via the `src/shared/python/export.py` module.

---

_This concludes Part I: Getting Started. The subsequent parts of this manual
cover the physics engines in depth (Part II), analysis and simulation tools
(Part III), robotics and control (Part IV), research tools (Part V), API and
integration (Part VI), development and operations (Part VII), and appendices
(Part VIII)._

---

_UpstreamDrift v2.1.0 -- User Manual -- Part I_

_Document generated for UpstreamDrift by the documentation team._

---

# UpstreamDrift User Manual -- Part II: Physics Engine Architecture

---

# Chapter 5: Physics Engine Framework

## 5.1 Overview

The UpstreamDrift Golf Modeling Suite is built on a **multi-engine architecture** that
allows users to swap between different physics backends without changing the higher-level
simulation, analytics, or GUI code. The framework achieves this through four
cooperating components:

1. **`PhysicsEngine` Protocol** -- the unified abstract interface every backend must satisfy.
2. **`BasePhysicsEngine`** -- a convenience base class that factors out common behaviour
   and enforces Design by Contract (DbC).
3. **`EngineRegistry` / `EngineManager`** -- the discovery, registration, and runtime
   switching layer.
4. **`EngineProbes`** -- lightweight readiness checks that detect whether a particular
   backend (MuJoCo, Drake, Pinocchio, ...) is installed, configured, and functional.

All source files discussed in this chapter reside under:

```
src/shared/python/
    interfaces.py
    base_physics_engine.py
    engine_manager.py
    engine_registry.py
    engine_probes.py
    contracts.py
```

---

## 5.2 The `PhysicsEngine` Protocol

**File:** `src/shared/python/interfaces.py`

`PhysicsEngine` is defined as a `typing.Protocol` decorated with `@runtime_checkable`.
Any class that structurally conforms to this protocol can be used anywhere a
`PhysicsEngine` is expected -- no explicit inheritance is required. The protocol
also extends `Checkpointable`, enabling state serialisation and rollback.

### 5.2.1 State Machine

Every engine implementation must respect the following lifecycle:

```
UNINITIALIZED  --[load_from_path / load_from_string]--> INITIALIZED
INITIALIZED    --[reset]------------------------------>  INITIALIZED  (t = 0)
INITIALIZED    --[step(dt)]--------------------------->  INITIALIZED  (t += dt)
```

### 5.2.2 Core State Methods

| Method             | Signature                             | Description                                                                                                                          |
| ------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `get_state()`      | `() -> tuple[np.ndarray, np.ndarray]` | Returns $(q, v)$ where $q \in \mathbb{R}^{n_q}$ are generalized coordinates and $v \in \mathbb{R}^{n_v}$ are generalized velocities. |
| `set_state(q, v)`  | `(np.ndarray, np.ndarray) -> None`    | Sets state and triggers `forward()` to recompute derived quantities.                                                                 |
| `set_control(u)`   | `(np.ndarray) -> None`                | Stores a control vector $u \in \mathbb{R}^{n_u}$ for the next `step()`.                                                              |
| `get_time()`       | `() -> float`                         | Returns the current simulation time $t \geq 0$.                                                                                      |
| `get_full_state()` | `() -> dict[str, Any]`                | Batched query returning `{'q', 'v', 't', 'M'}` in one call (performance optimisation).                                               |

### 5.2.3 Dynamics Interface

All engines expose the following dynamics quantities:

**Mass Matrix.** The joint-space inertia matrix $M(q) \in \mathbb{R}^{n_v \times n_v}$ is
symmetric positive definite:

$$M(q) = M(q)^T, \quad \lambda_{\min}(M) > 0$$

```python
def compute_mass_matrix(self) -> np.ndarray:
    """Returns M(q), shape (n_v, n_v)."""
```

**Bias Forces.** The combined Coriolis, centrifugal, and gravitational force vector:

$$b(q, v) = C(q, v)\,v + g(q) \in \mathbb{R}^{n_v}$$

```python
def compute_bias_forces(self) -> np.ndarray:
    """Returns b = C(q,v)*v + g(q), shape (n_v,)."""
```

**Gravity Forces.** The configuration-dependent gravity vector alone:

$$g(q) \in \mathbb{R}^{n_v}$$

```python
def compute_gravity_forces(self) -> np.ndarray:
    """Returns g(q), shape (n_v,)."""
```

**Inverse Dynamics.** Given a desired acceleration $\ddot{q}$, compute the required
generalised forces:

$$\tau = M(q)\,\ddot{q} + C(q, v)\,v + g(q)$$

```python
def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
    """Returns tau = ID(q, v, qacc), shape (n_v,)."""
```

**Jacobian.** The spatial Jacobian for a named body frame maps joint velocities
to linear and angular body-frame velocities:

$$J = \frac{\partial x}{\partial q} \in \mathbb{R}^{6 \times n_v}$$

```python
def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
    """Returns {'linear': (3, n_v), 'angular': (3, n_v), 'spatial': (6, n_v)}."""
```

### 5.2.4 Section F -- Drift-Control Decomposition

A central design requirement of UpstreamDrift is the **superposition principle**:

$$\ddot{q}_{\text{full}} = \ddot{q}_{\text{drift}} + \ddot{q}_{\text{control}}$$

where

$$\ddot{q}_{\text{drift}} = M^{-1}\bigl(-C(q,v)\,v - g(q)\bigr)$$
$$\ddot{q}_{\text{control}} = M^{-1}\,\tau$$

```python
def compute_drift_acceleration(self) -> np.ndarray:
    """Passive acceleration with zero control inputs, shape (n_v,)."""

def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
    """Control-attributed acceleration M^{-1} tau, shape (n_v,)."""
```

This decomposition answers the fundamental biomechanics question: _How much of
the observed motion is due to active muscle effort versus passive dynamics?_

### 5.2.5 Section G -- Counterfactual Experiments

Two counterfactual methods enable causal reasoning about the swing:

**ZTCF (Zero-Torque Counterfactual, G1):** "What happens if all actuators turn off?"

$$\ddot{q}_{\text{ZTCF}} = M(q)^{-1}\bigl(-C(q,v)\,v - g(q) + J^T\lambda\bigr)$$

```python
def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray
```

**ZVCF (Zero-Velocity Counterfactual, G2):** "What happens if motion freezes?"

$$\ddot{q}_{\text{ZVCF}} = M(q)^{-1}\bigl(-g(q) + \tau + J^T\lambda\bigr)$$

```python
def compute_zvcf(self, q: np.ndarray) -> np.ndarray
```

The difference $\Delta a_{\text{velocity}} = a_{\text{full}} - a_{\text{ZVCF}}$
isolates Coriolis/centrifugal contributions.

### 5.2.6 Section B5 -- Flexible Shaft Interface (Optional)

Engines may optionally model a flexible club shaft via Euler-Bernoulli beam theory:

```python
def set_shaft_properties(
    self,
    length: float,              # Shaft length [m]
    EI_profile: np.ndarray,     # Bending stiffness [N*m^2]
    mass_profile: np.ndarray,   # Linear mass density [kg/m]
    damping_ratio: float = 0.02 # Modal damping ratio
) -> bool:
    """Returns True if shaft flexibility is supported and configured."""

def get_shaft_state(self) -> dict[str, np.ndarray] | None:
    """Returns {'deflection', 'rotation', 'velocity', 'modal_amplitudes'}."""
```

### 5.2.7 Recorder Interface

`RecorderInterface` is a separate protocol for capturing time-series data:

```python
@runtime_checkable
class RecorderInterface(Protocol):
    engine: Any

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]: ...
    def get_induced_acceleration_series(self, source_name: str | int) -> tuple[np.ndarray, np.ndarray]: ...
    def set_analysis_config(self, config: dict[str, Any]) -> None: ...
```

---

## 5.3 Design by Contract -- `contracts.py`

**File:** `src/shared/python/contracts.py`

UpstreamDrift enforces Design by Contract (DbC) principles through decorators and a
mixin class. Contracts can be globally toggled via the `CONTRACTS_ENABLED` flag for
production performance.

### 5.3.1 Exception Hierarchy

```
ContractViolationError
  |-- PreconditionError     (caller violated input requirements)
  |-- PostconditionError    (implementation violated output guarantees)
  |-- InvariantError        (object state inconsistency)
  |-- StateError            (operation in wrong lifecycle state)
```

Each exception carries structured diagnostic information:

```python
class ContractViolationError(Exception):
    def __init__(
        self,
        contract_type: str,
        message: str,
        function_name: str | None = None,
        details: dict[str, Any] | None = None,
    ): ...
```

### 5.3.2 Decorator API

**Preconditions** guard method entry:

```python
@precondition(lambda self: self._is_initialized, "Engine must be initialized")
def step(self, dt: float) -> None: ...
```

The condition callable receives the same arguments as the decorated function. If it
returns `False`, a `PreconditionError` is raised before the method body executes.

**Postconditions** guard method exit:

```python
@postcondition(check_finite, "Mass matrix must contain finite values")
def compute_mass_matrix(self) -> np.ndarray: ...
```

The condition callable receives the **return value**. If it returns `False`, a
`PostconditionError` is raised (the original return value is discarded).

**State requirements** are a specialised precondition for lifecycle checks:

```python
@require_state(lambda self: self._is_initialized, "initialized")
def get_state(self) -> EngineState: ...
```

**Invariant checking** is available via the `ContractChecker` mixin and the
`@invariant_checked` decorator:

```python
class MyEngine(ContractChecker, BasePhysicsEngine):
    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        return [
            (lambda: self._is_initialized or self.model is None,
             "Uninitialised engine must not have a model"),
        ]

    @invariant_checked
    def set_mass(self, mass: float) -> None:
        self._mass = mass  # Invariants checked after this returns
```

### 5.3.3 Utility Predicates

| Predicate                 | Signature                     | Purpose                        |
| ------------------------- | ----------------------------- | ------------------------------ |
| `check_finite`            | `(np.ndarray) -> bool`        | All values finite (no NaN/Inf) |
| `check_shape`             | `(np.ndarray, tuple) -> bool` | Shape matches expected         |
| `check_positive`          | `(value) -> bool`             | All values $> 0$               |
| `check_non_negative`      | `(value) -> bool`             | All values $\geq 0$            |
| `check_symmetric`         | `(matrix, tol) -> bool`       | $\|A - A^T\| < \text{tol}$     |
| `check_positive_definite` | `(matrix) -> bool`            | All eigenvalues $> 0$          |

### 5.3.4 Convenience Aliases

```python
requires_initialized = require_state(
    lambda self: getattr(self, "_is_initialized", False), "initialized")

requires_model_loaded = require_state(
    lambda self: getattr(self, "model", None) is not None, "model loaded")

@finite_result       # Postcondition: result has no NaN/Inf
@non_empty_result    # Postcondition: result.size > 0
```

---

## 5.4 `BasePhysicsEngine`

**File:** `src/shared/python/base_physics_engine.py`

`BasePhysicsEngine` is an abstract base class that factors out behaviour common
to all engine implementations:

```python
class BasePhysicsEngine(ContractChecker, PhysicsEngine):
    def __init__(self, allowed_dirs: list[Path] | None = None):
        self.model: Any = None
        self.data: Any = None
        self.state: EngineState | None = None
        self._is_initialized = False
```

### 5.4.1 `EngineState` Data Class

```python
class EngineState:
    def __init__(self, nq: int = 0, nv: int = 0):
        self.q: np.ndarray = np.zeros(nq)    # Positions
        self.v: np.ndarray = np.zeros(nv)    # Velocities
        self.a: np.ndarray = np.zeros(nv)    # Accelerations
        self.tau: np.ndarray = np.zeros(nv)  # Torques/forces
        self.time: float = 0.0
```

### 5.4.2 Template Method Pattern

`BasePhysicsEngine` uses the Template Method pattern. Concrete engines implement
the private `_impl` hooks while the base class handles validation:

```python
# Base class (handles preconditions, invariants, logging)
@log_errors("Failed to load model from path", reraise=True)
@invariant_checked
def load_from_path(self, path: str) -> None:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if self.allowed_dirs:
        validated_path = validate_path(path, self.allowed_dirs, strict=True)
    self._load_from_path_impl(path)    # <-- concrete engine
    self._is_initialized = True

# Concrete engine must implement:
@abstractmethod
def _load_from_path_impl(self, path: str) -> None: ...

@abstractmethod
def _load_from_string_impl(self, content: str, extension: str | None) -> None: ...
```

### 5.4.3 Invariants

The base class defines three invariants that all engines must maintain:

1. **Model consistency:** If `_is_initialized` is `True`, then `model` is not `None`.
2. **Dimension compatibility:** `len(q) == len(v)` or `len(q) == len(v) + 1`
   (quaternion representation).
3. **Non-negative time:** `state.time >= 0.0`.

### 5.4.4 Checkpointing

`BasePhysicsEngine` provides default `save_checkpoint()` / `restore_checkpoint()`
implementations that serialise the `EngineState` into a `StateCheckpoint` object.

---

## 5.5 Engine Registry

**File:** `src/shared/python/engine_registry.py`

The registry decouples engine _discovery_ from engine _instantiation_.

### 5.5.1 `EngineType` Enumeration

```python
class EngineType(Enum):
    MUJOCO    = "mujoco"
    DRAKE     = "drake"
    PINOCCHIO = "pinocchio"
    OPENSIM   = "opensim"
    MYOSIM    = "myosim"
    MATLAB_2D = "matlab_2d"
    MATLAB_3D = "matlab_3d"
    PENDULUM  = "pendulum"
```

### 5.5.2 `EngineStatus` Enumeration

```python
class EngineStatus(Enum):
    AVAILABLE   = "available"
    UNAVAILABLE = "unavailable"
    LOADING     = "loading"
    LOADED      = "loaded"
    ERROR       = "error"
```

### 5.5.3 Registration Data

```python
@dataclass
class EngineRegistration:
    engine_type: EngineType
    factory: EngineFactory        # Callable[[], PhysicsEngine]
    registration_path: Path | None = None
    requires_binary: list[str] = field(default_factory=list)
    probe_class: type | None = None
```

### 5.5.4 Global Singleton

```python
_registry = EngineRegistry()

def get_registry() -> EngineRegistry:
    return _registry
```

The `EngineRegistry` class exposes three methods:

- `register(registration)` -- add an engine.
- `get(engine_type) -> EngineRegistration | None` -- look up by type.
- `all_types() -> list[EngineType]` -- enumerate registered types.

---

## 5.6 Engine Manager

**File:** `src/shared/python/engine_manager.py`

The `EngineManager` is the user-facing facade that ties the registry, probes, and
engine factories together.

```python
class EngineManager:
    def __init__(self, suite_root: Path | None = None):
        self.current_engine: EngineType | None = None
        self.active_physics_engine: PhysicsEngine | None = None
        self.engine_status: dict[EngineType, EngineStatus] = {}
```

### 5.6.1 Initialisation Sequence

1. Compute engine paths relative to `suite_root`.
2. Instantiate all probes (MuJoCoProbe, DrakeProbe, PinocchioProbe, ...).
3. Register factories from the `LOADER_MAP` into the global `EngineRegistry`.
4. Call `_discover_engines()` to populate `engine_status`.

### 5.6.2 Key Methods

| Method                        | Description                                                   |
| ----------------------------- | ------------------------------------------------------------- |
| `get_available_engines()`     | Returns `list[EngineType]` of engines with status AVAILABLE.  |
| `switch_engine(engine_type)`  | Loads and activates a different backend. Returns `bool`.      |
| `get_active_physics_engine()` | Returns the current `PhysicsEngine` instance or `None`.       |
| `probe_all_engines()`         | Runs readiness probes for every engine and caches results.    |
| `get_diagnostic_report()`     | Returns a human-readable string report with fix instructions. |
| `cleanup()`                   | Shuts down MATLAB engines, releases references.               |

### 5.6.3 Engine Switching

```python
def switch_engine(self, engine_type: EngineType) -> bool:
    if self.engine_status[engine_type] != EngineStatus.AVAILABLE:
        return False
    self._load_engine(engine_type)
    self.current_engine = engine_type
    return True
```

Standard engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSim) are loaded through
the registry factory. Special cases (MATLAB, Pendulum) use dedicated loaders.

---

## 5.7 Engine Probes

**File:** `src/shared/python/engine_probes.py`

Each engine has a corresponding probe class that checks installation status at runtime.

### 5.7.1 `ProbeStatus` Enumeration

```python
class ProbeStatus(Enum):
    AVAILABLE            = "available"
    MISSING_BINARY       = "missing_binary"
    MISSING_ASSETS       = "missing_assets"
    VERSION_MISMATCH     = "version_mismatch"
    NOT_INSTALLED        = "not_installed"
    CONFIGURATION_ERROR  = "configuration_error"
```

### 5.7.2 `EngineProbeResult`

```python
@dataclass
class EngineProbeResult:
    engine_name: str
    status: ProbeStatus
    version: str | None
    missing_dependencies: list[str]
    diagnostic_message: str
    details: dict[str, Any] | None = None

    def is_available(self) -> bool: ...
    def get_fix_instructions(self) -> str: ...
```

### 5.7.3 Probe Implementations

| Probe            | Checks                                                                    |
| ---------------- | ------------------------------------------------------------------------- |
| `MuJoCoProbe`    | `import mujoco`, engine dir, Python modules, XML model files              |
| `DrakeProbe`     | `import pydrake`, `pydrake.multibody`, MeshCat port 7000-7010, engine dir |
| `PinocchioProbe` | `import pinocchio`, engine dir, `pinocchio_golf` module                   |
| `OpenSimProbe`   | `import opensim`, engine dir                                              |
| `MyoSimProbe`    | `import mujoco` (dependency), engine dir, `myosim_physics_engine.py`      |
| `MatlabProbe`    | `import matlab.engine`, model directory (2D or 3D), `.slx`/`.m` files     |
| `PendulumProbe`  | Engine dir, `constants.py`, `pendulum_solver.py`                          |

The `DrakeProbe` additionally tests MeshCat port availability by attempting to bind
to ports 7000 through 7010, reporting `CONFIGURATION_ERROR` if all are blocked.

---

---

# Chapter 6: MuJoCo Physics Engine

## 6.1 Overview

The MuJoCo backend is the primary physics engine for the Golf Modeling Suite. It
provides high-fidelity rigid-body simulation with contact, constraints, and
muscle-driven actuation. The implementation is spread across several modules:

```
src/engines/physics_engines/mujoco/
    python/mujoco_humanoid_golf/
        physics_engine.py       -- PhysicsEngine protocol implementation
        models.py               -- MJCF model hierarchy
        advanced_kinematics.py  -- IK, manipulability, singularity analysis
        advanced_control.py     -- Impedance, admittance, computed torque, OSC
        kinematic_forces.py     -- Coriolis, centrifugal, gravitational analysis
        inverse_dynamics.py     -- Full ID solver with force decomposition
        motion_capture.py       -- Motion capture data loading
        motion_optimization.py  -- Swing trajectory optimization
    docker/src/humanoid_golf/
        sim.py                  -- Simulation loop, controllers, visualization
```

---

## 6.2 `MuJoCoPhysicsEngine`

**File:** `physics_engine.py`

```python
class MuJoCoPhysicsEngine(PhysicsEngine):
    def __init__(self) -> None:
        self.model: mujoco.MjModel | None = None
        self.data:  mujoco.MjData  | None = None
        self.xml_path: str | None = None
```

### 6.2.1 Model Loading

Models are loaded via `mujoco.MjModel.from_xml_path()` or
`mujoco.MjModel.from_xml_string()`. Path traversal is mitigated by validating
against `ALLOWED_MODEL_DIRS`:

```python
ALLOWED_MODEL_DIRS = [
    SUITE_ROOT / "engines",
    SUITE_ROOT / "shared" / "resources",
    Path(tempfile.gettempdir()),
]
```

### 6.2.2 Core Dynamics

**Mass Matrix** -- computed via `mujoco.mj_fullM()`:

```python
@precondition(lambda self: self.is_initialized, "Engine must be initialized")
@postcondition(check_finite, "Mass matrix must contain finite values")
def compute_mass_matrix(self) -> np.ndarray:
    nv = self.model.nv
    M = np.zeros((nv, nv), dtype=np.float64)
    mujoco.mj_fullM(self.model, M, self.data.qM)
    return M
```

**Bias Forces** -- read directly from `data.qfrc_bias` (populated by
`mj_forward` / `mj_step`).

**Inverse Dynamics** -- via `mujoco.mj_inverse()`:

```python
self.data.qacc[:] = qacc
mujoco.mj_inverse(self.model, self.data)
return self.data.qfrc_inverse.copy()
```

### 6.2.3 Drift-Control Decomposition

The drift acceleration is computed by zeroing out all control inputs, running
`mj_forward`, and reading back `data.qacc`:

```python
def compute_affine_drift(self) -> np.ndarray:
    saved_ctrl = self.data.ctrl.copy()
    self.data.ctrl[:] = 0
    mujoco.mj_forward(self.model, self.data)
    drift_acc = self.data.qacc.copy()
    self.data.ctrl[:] = saved_ctrl
    mujoco.mj_forward(self.model, self.data)  # Restore
    return drift_acc
```

Control acceleration uses $\ddot{q}_{\text{control}} = M^{-1}\tau$:

```python
def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
    M = self.compute_mass_matrix()
    return np.linalg.solve(M, tau)
```

### 6.2.4 Counterfactual Experiments (ZTCF / ZVCF)

Both methods follow a save-mutate-restore pattern to prevent Observer Effect
corruption of simulation state:

```python
def compute_ztcf(self, q, v) -> np.ndarray:
    saved_qpos = self.data.qpos.copy()
    saved_qvel = self.data.qvel.copy()
    saved_ctrl = self.data.ctrl.copy()
    try:
        self.data.qpos[:] = q; self.data.qvel[:] = v
        self.data.ctrl[:] = 0               # Zero torque
        mujoco.mj_forward(self.model, self.data)
        return self.data.qacc.copy()
    finally:
        # Restore original state
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(self.model, self.data)
```

### 6.2.5 Flexible Shaft

The MuJoCo engine supports flexible club shafts using Euler-Bernoulli beam
theory. Natural frequencies are computed analytically from cantilever beam
eigenvalues:

$$\omega_n = \left(\frac{\beta_n}{L}\right)^2 \sqrt{\frac{EI}{\mu}}$$

where $\beta_n L$ takes the standard clamped-free values
$\{1.875, 4.694, 7.855, 10.996, 14.137\}$ and $\mu$ is the linear mass density.
The first three bending modes are retained.

---

## 6.3 Model Hierarchy

**File:** `models.py`

The suite provides a graduated hierarchy of MJCF models, from simple pendulums
to full-body musculoskeletal systems.

### 6.3.1 Pendulum Models

| Model                  | DOF | Description                           |
| ---------------------- | --- | ------------------------------------- |
| `CHAOTIC_PENDULUM_XML` | 1   | Single pendulum with chaotic regime   |
| `DOUBLE_PENDULUM_XML`  | 2   | Shoulder + wrist (minimal golf swing) |
| `TRIPLE_PENDULUM_XML`  | 3   | Shoulder + elbow + wrist              |

The double pendulum serves as the simplest golf-swing analogue, capturing the
essential energy-transfer mechanism from proximal to distal segments.

### 6.3.2 Multi-Segment Golf Models

| Model                                   | DOF | Bodies                      | Key Features                                                        |
| --------------------------------------- | --- | --------------------------- | ------------------------------------------------------------------- |
| `UPPER_BODY_GOLF_SWING_XML`             | 10  | Spine, bilateral arms, club | Spine torsion, bilateral shoulder/elbow/wrist                       |
| `FULL_BODY_GOLF_SWING_XML`              | 15  | Full body with legs         | Lower body (hip, knee, ankle) + upper body                          |
| `ADVANCED_BIOMECHANICAL_GOLF_SWING_XML` | 28  | Full body + scapulae        | 3-DOF spine, scapulae, forearm supination, 3-segment flexible shaft |

The advanced model includes:

- **3-DOF spine** (flexion/extension, lateral bending, axial rotation).
- **Scapular protraction/retraction** on both sides.
- **Forearm supination** for club face control.
- **3-segment flexible club shaft** with torsional spring-damper joints modelling
  shaft flex and kick.

### 6.3.3 MyoSuite Integration

For muscle-driven simulation, the suite supports MyoSuite models:

```python
MYOUPPERBODY_PATH = "myosuite/envs/myo/assets/arm/myoarm.xml"
MYOBODY_PATH = "myosuite/envs/myo/assets/body/myobody.xml"
# 52 DOF, 290 muscle actuators
```

### 6.3.4 Golf Ball Physics

Golf ball parameters follow USGA regulations and are imported from
`shared.python.constants`:

- Mass: `GOLF_BALL_MASS_KG` (0.04593 kg)
- Radius: `GOLF_BALL_RADIUS_M` (0.02135 m)

Ball bodies are defined with appropriate `condim`, `friction`, and `solref`
parameters for realistic impact dynamics.

### 6.3.5 Parametric Model Generation

Two factory functions produce custom club models:

```python
def generate_flexible_club_xml(
    n_segments: int = 3,
    shaft_length: float = 1.0,
    shaft_stiffness: float = 500.0,
    shaft_damping: float = 5.0,
) -> str:
    """Generate MJCF XML for a flexible multi-segment club."""

def generate_rigid_club_xml(
    shaft_length: float = 1.0,
    club_head_mass: float = 0.3,
) -> str:
    """Generate MJCF XML for a rigid single-segment club."""
```

---

## 6.4 Simulation and Controllers

**File:** `docker/src/humanoid_golf/sim.py`

The Docker-hosted simulation provides a complete run loop with configurable
controllers and visualization.

### 6.4.1 Controller Hierarchy

```python
class BaseController:
    def get_action(self, physics) -> np.ndarray:
        return np.zeros(physics.model.nu)

class PDController(BaseController):
    def __init__(self, actuators, target_pose, kp=60.0, kd=6.0): ...

class PolynomialController(BaseController):
    # 6th-order polynomial: u(t) = c_0 + c_1 t + ... + c_6 t^6
    def __init__(self, physics): ...

class LQRController(BaseController):
    # u = -K (x - x_target) with gain matrix K
    def __init__(self, physics, target_pose, actuators, height_scale=1.0): ...
```

**PD Controller.** For each joint in the `target_pose` dictionary:

$$\tau_j = K_p (q_j^d - q_j) - K_d \dot{q}_j$$

with default gains $K_p = 60$, $K_d = 6$.

**Polynomial Controller.** Generates time-varying torque profiles using
sixth-order polynomials. A truncated Taylor series $60t - 20t^3$ approximates
a swing-like sinusoidal torque on the right shoulder.

**LQR Controller.** Computes a linear state-feedback gain matrix $K$ such that
$u = -K(x - x^*)$. A decoupled diagonal PD fallback with $K_p = 100$,
$K_d = 10$ is used for robustness with quaternion-based root configurations.

### 6.4.2 Induced Acceleration Analysis in the Loop

Each simulation step computes induced accelerations and counterfactuals via
the `iaa_helper` module:

```python
iaa = iaa_helper.compute_induced_accelerations(physics)
cf  = iaa_helper.compute_counterfactuals(physics)
```

Data is emitted as JSON packets for real-time analytics consumption.

---

## 6.5 Advanced Kinematics

**File:** `advanced_kinematics.py`

### 6.5.1 `AdvancedKinematicsAnalyzer`

```python
class AdvancedKinematicsAnalyzer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.singularity_threshold = 30.0
        self.ik_damping = 0.01
        self.ik_max_iterations = 100
        self.ik_tolerance = 1e-4
```

### 6.5.2 Jacobian Computation

Jacobians are computed via `mujoco.mj_jacBody()`. An offset point $r$ in the
body frame adjusts the linear Jacobian:

$$J_p^{\text{offset}} = J_p^{\text{body}} - [\hat{r}]_\times \, J_r^{\text{body}}$$

where $[\hat{r}]_\times$ is the skew-symmetric matrix of the world-frame offset.

### 6.5.3 Manipulability Analysis

Yoshikawa's manipulability index is computed from the Jacobian SVD:

$$w = \sqrt{\det(J J^T)} = \prod_i \sigma_i$$

```python
@dataclass
class ManipulabilityMetrics:
    manipulability_index: float       # w = sqrt(det(J*J^T))
    condition_number: float           # sigma_max / sigma_min
    singular_values: np.ndarray       # [sigma_1, ..., sigma_m]
    is_near_singularity: bool         # True if kappa > threshold
    min_singular_value: float
    max_singular_value: float
```

Singularity is flagged when $\kappa(J) > 30$ or $\sigma_{\min} < 10^{-3}$.

### 6.5.4 Inverse Kinematics (DLS)

The IK solver uses Damped Least-Squares with nullspace projection for
redundancy resolution:

$$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} e + \alpha (I - J^+ J)(q^* - q)$$

where $e$ is the task-space error, $\lambda$ is the damping factor, $q^*$ is the
nullspace objective, and $\alpha$ is the nullspace gain. Joint limits are
enforced by clamping after each iteration.

### 6.5.5 Constraint Jacobian (Closed-Chain Analysis)

For the two-handed golf grip (closed kinematic chain), the constraint Jacobian
is:

$$J_c = J_{\text{right hand}} - J_{\text{left hand}}$$

The nullspace dimension $\dim(\ker J_c)$ determines the effective degrees of freedom
of the constrained system.

### 6.5.6 Task-Space Inertia

$$\Lambda = (J M^{-1} J^T)^{-1}$$

This matrix maps task-space forces to task-space accelerations and is essential
for operational space control.

---

## 6.6 Advanced Control

**File:** `advanced_control.py`

### 6.6.1 Control Modes

```python
class ControlMode(Enum):
    TORQUE         = "torque"
    IMPEDANCE      = "impedance"
    ADMITTANCE     = "admittance"
    HYBRID         = "hybrid"
    COMPUTED_TORQUE = "computed_torque"
    TASK_SPACE     = "task_space"
```

### 6.6.2 Impedance Control

Creates a virtual spring-damper system around the desired trajectory:

$$\tau = K_p(q_d - q) + D(\dot{q}_d - \dot{q}) + g(q)$$

```python
@dataclass
class ImpedanceParameters:
    stiffness: np.ndarray  # K [n x n] or [n]
    damping:   np.ndarray  # D [n x n] or [n]
    inertia:   np.ndarray | None = None  # M (optional)
```

Default parameters: $K_p = 100$ N/m, $D = 20$ Ns/m.

### 6.6.3 Admittance Control

The dual of impedance control, modifying position based on force error:

$$\Delta \ddot{q} = M_d^{-1}(F_d - F_{\text{measured}})$$

where $M_d$ is the desired inertia matrix.

### 6.6.4 Hybrid Force-Position Control

Combines position and force control using selection matrices:

$$\tau = S_p \tau_{\text{position}} + S_f \tau_{\text{force}}$$

```python
@dataclass
class HybridControlMask:
    force_mask: np.ndarray  # Boolean: True = force control, False = position
```

### 6.6.5 Computed Torque Control

Model-based feedforward control using inverse dynamics:

$$\tau = M(q)\ddot{q}_d + C(q, \dot{q})\dot{q} + g(q)$$

where $\ddot{q}_d = K_p(q_d - q) + K_d(\dot{q}_d - \dot{q})$.

The implementation uses `mujoco.mj_mulM()` for efficient sparse
matrix-vector multiplication.

### 6.6.6 Operational Space Control (OSC)

Full operational space controller that accounts for configuration-dependent
inertia:

$$F = \Lambda(q)(\ddot{x}_d + K_d \dot{e} + K_p e) + \mu(q, \dot{q}) + p(q)$$
$$\tau = J^T F + N^T \tau_{\text{posture}}$$

where $\Lambda = (J M^{-1} J^T)^{-1}$ is the task-space inertia and
$N = I - J^T \bar{J}^T$ is the nullspace projector. The dynamically consistent
pseudoinverse is $\bar{J} = M^{-1} J^T \Lambda$.

### 6.6.7 Trajectory Generation

The `TrajectoryGenerator` class produces smooth reference trajectories:

**Minimum-jerk trajectory:**

$$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5, \quad \tau = t/T$$

satisfying boundary conditions $s(0) = 0$, $\dot{s}(0) = 0$, $\ddot{s}(0) = 0$,
$s(1) = 1$, $\dot{s}(1) = 0$, $\ddot{s}(1) = 0$.

---

## 6.7 Kinematic Forces Analysis

**File:** `kinematic_forces.py`

### 6.7.1 `KinematicForceAnalyzer`

Analyses velocity-dependent forces that can be computed from kinematics alone,
without full inverse dynamics.

```python
class KinematicForceAnalyzer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._perturb_data = mujoco.MjData(model)  # Scratch copy for thread safety
```

### 6.7.2 Thread Safety via `MjDataContext`

All analysis methods use a dedicated `_perturb_data` copy or the `MjDataContext`
context manager to prevent Observer Effect bugs:

```python
class MjDataContext:
    """Context manager for safe MuJoCo MjData state isolation."""
    def __enter__(self) -> mujoco.MjData:
        self.qpos_backup = self.data.qpos.copy()
        # ... save all state
        return self.data

    def __exit__(self, ...):
        self.data.qpos[:] = self.qpos_backup
        # ... restore all state
        mujoco.mj_forward(self.model, self.data)
```

### 6.7.3 Coriolis Force Computation

The equations of motion are:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau$$

The Coriolis/centrifugal term $C(q, \dot{q})\dot{q}$ is computed using
MuJoCo's Recursive Newton-Euler algorithm (`mj_rne`):

```python
def compute_coriolis_forces_rne(self, qpos, qvel) -> np.ndarray:
    # With qacc=0: mj_rne returns C(q,v)*v + g(q)
    self._perturb_data.qpos[:] = qpos
    self._perturb_data.qvel[:] = qvel
    self._perturb_data.qacc[:] = 0.0
    bias = np.zeros(self.model.nv)
    mujoco.mj_rne(self.model, self._perturb_data, 0, bias)

    # Subtract gravity: with qvel=0, mj_rne returns g(q)
    self._perturb_data.qvel[:] = 0.0
    gravity = np.zeros(self.model.nv)
    mujoco.mj_rne(self.model, self._perturb_data, 0, gravity)

    return bias - gravity  # = C(q,v)*v
```

### 6.7.4 Coriolis-Centrifugal Decomposition

The total Coriolis force is decomposed into centrifugal (diagonal) and
velocity-coupling (off-diagonal) components. For each DOF $i$, the centrifugal
contribution is computed by setting all velocities except $\dot{q}_i$ to zero:

$$f_{\text{centrifugal}} = \sum_i C(q, e_i \dot{q}_i) e_i \dot{q}_i$$

$$f_{\text{coupling}} = f_{\text{total}} - f_{\text{centrifugal}}$$

Note: This decomposition is $O(n^2)$ and the combined `compute_coriolis_forces()`
at $O(n)$ is preferred for most applications.

### 6.7.5 `KinematicForceData`

```python
@dataclass
class KinematicForceData:
    time: float
    coriolis_forces: np.ndarray             # [nv]
    gravity_forces: np.ndarray              # [nv]
    centrifugal_forces: np.ndarray | None   # [nv]
    velocity_coupling_forces: np.ndarray | None
    club_head_coriolis_force: np.ndarray | None   # [3]
    club_head_centrifugal_force: np.ndarray | None
    club_head_apparent_force: np.ndarray | None
    coriolis_power: float
    centrifugal_power: float
    rotational_kinetic_energy: float
    translational_kinetic_energy: float
```

### 6.7.6 Effective Mass

The directional effective mass at a body determines the resistance to
acceleration along a given direction:

$$m_{\text{eff}} = \frac{1}{\hat{d}^T J M^{-1} J^T \hat{d}}$$

where $\hat{d}$ is the unit direction vector. Near kinematic singularities,
$m_{\text{eff}} \to \infty$. Condition number monitoring triggers warnings
when $\kappa(M) > 10^6$.

---

## 6.8 Inverse Dynamics

**File:** `inverse_dynamics.py`

### 6.8.1 `InverseDynamicsSolver`

```python
class InverseDynamicsSolver:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData): ...
```

**Core method:**

```python
def compute_required_torques(
    self, qpos, qvel, qacc, external_forces=None
) -> InverseDynamicsResult
```

**Result structure:**

```python
@dataclass
class InverseDynamicsResult:
    joint_torques: np.ndarray
    constraint_forces: np.ndarray | None
    inertial_torques: np.ndarray | None     # M(q) * qacc
    coriolis_torques: np.ndarray | None     # C(q,v) * v
    gravity_torques: np.ndarray | None      # g(q)
```

### 6.8.2 Nullspace Projection for Posture

When tracking a primary task (e.g. club head trajectory), secondary posture
objectives are projected into the task nullspace:

$$\tau = \tau_{\text{primary}} + (I - J^+ J) \, K_p (q_d - q)$$

```python
def compute_torques_with_posture(
    self, qpos, qvel, qacc_primary, qpos_desired, kp_posture, primary_body_name
) -> InverseDynamicsResult
```

### 6.8.3 Force Decomposition

```python
@dataclass
class ForceDecomposition:
    inertial: np.ndarray      # M * qacc
    coriolis: np.ndarray      # C(q,v) * v
    gravity: np.ndarray       # g(q)
    external: np.ndarray      # J^T * f_ext
    total: np.ndarray         # Sum
```

### 6.8.4 Trajectory Inverse Dynamics

```python
def solve_inverse_dynamics_trajectory(
    self, times, positions, velocities, accelerations
) -> list[InverseDynamicsResult]
```

---

## 6.9 Motion Capture

**File:** `motion_capture.py`

### 6.9.1 Data Structures

```python
@dataclass
class MotionCaptureFrame:
    time: float
    marker_positions: dict[str, np.ndarray]   # name -> [x, y, z]
    marker_velocities: dict[str, np.ndarray] | None
    body_orientations: dict[str, np.ndarray] | None
    joint_angles: np.ndarray | None

@dataclass
class MotionCaptureSequence:
    frames: list[MotionCaptureFrame]
    frame_rate: float
    marker_names: list[str]
    metadata: dict
```

### 6.9.2 Supported Formats

The `MotionCaptureSequence` class provides loaders for:

- **CSV** -- Simple tabular format with columns for time, marker x/y/z.
- **JSON** -- Structured format with metadata.
- **C3D** -- Industry-standard motion capture format (via `ezc3d` library).

---

## 6.10 Motion Optimization

**File:** `motion_optimization.py`

### 6.10.1 Objective Configuration

```python
@dataclass
class OptimizationObjectives:
    maximize_club_speed: bool = True
    minimize_energy: bool = True
    minimize_jerk: bool = True
    minimize_torque: bool = True
    weight_speed: float = 10.0
    weight_energy: float = 1.0
    weight_jerk: float = 0.5
    weight_torque: float = 0.1
    weight_accuracy: float = 5.0
```

### 6.10.2 `SwingOptimizer`

The `SwingOptimizer` class performs multi-objective trajectory optimisation,
balancing club head speed, energy efficiency, smoothness, and joint torque
minimisation. It uses the MuJoCo forward dynamics to evaluate candidate
trajectories against the physics model.

---

---

# Chapter 7: Rigid Body Dynamics Algorithms

## 7.1 Overview

The `rigid_body_dynamics/` subpackage provides pure-Python implementations of
the three foundational algorithms of multibody dynamics, following Featherstone's
formulation with 6D spatial algebra:

```
rigid_body_dynamics/
    rnea.py               -- Recursive Newton-Euler Algorithm (inverse dynamics)
    crba.py               -- Composite Rigid Body Algorithm (mass matrix)
    aba.py                -- Articulated Body Algorithm (forward dynamics)
    induced_acceleration.py -- Induced acceleration analysis
    common.py             -- Shared constants and utilities
```

All algorithms operate on a **model dictionary** with the following fields:

| Key       | Type               | Description                                       |
| --------- | ------------------ | ------------------------------------------------- |
| `NB`      | `int`              | Number of bodies                                  |
| `parent`  | `np.ndarray`       | Parent body indices ($-1$ for root)               |
| `jtype`   | `list[str]`        | Joint types (`"Rx"`, `"Ry"`, `"Rz"`, `"Px"`, ...) |
| `Xtree`   | `list[np.ndarray]` | Tree transforms ($6 \times 6$ spatial)            |
| `I`       | `list[np.ndarray]` | Spatial inertias ($6 \times 6$)                   |
| `gravity` | `np.ndarray`       | 6D spatial gravity vector                         |

---

## 7.2 Recursive Newton-Euler Algorithm (RNEA)

**File:** `rnea.py`

**Reference:** Featherstone, R. (2008). _Rigid Body Dynamics Algorithms_.
Chapter 5, Algorithm 5.1.

### 7.2.1 Problem Statement

Given joint positions $q$, velocities $\dot{q}$, and accelerations $\ddot{q}$,
compute the required joint torques:

$$\tau = \text{RNEA}(q, \dot{q}, \ddot{q}) = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q)$$

### 7.2.2 Algorithm

```python
def rnea(model, q, qd, qdd, f_ext=None) -> np.ndarray:
```

**Pass 1 -- Forward Kinematics** (root to leaves):

For each body $i$ with parent $p(i)$:

1. Compute joint transform $X_J$ and motion subspace $S_i$ from `jcalc(jtype, q[i])`.
2. Combined transform: $X_{\text{up},i} = X_J \cdot X_{\text{tree},i}$.
3. Spatial velocity:

$$v_i = X_{\text{up},i} \, v_{p(i)} + S_i \dot{q}_i$$

4. Spatial acceleration:

$$a_i = X_{\text{up},i} \, a_{p(i)} + S_i \ddot{q}_i + v_i \times^m (S_i \dot{q}_i)$$

For the root body ($p(i) = -1$), $v_i = S_i \dot{q}_i$ and the base acceleration
incorporates gravity: $a_i = X_J(-a_g) + S_i \ddot{q}_i$.

**Pass 2 -- Backward Dynamics** (leaves to root):

For each body $i$ (in reverse order):

1. Compute the net spatial force:

$$f_i = I_i a_i + v_i \times^f (I_i v_i) - f_{\text{ext},i}$$

2. Project to joint torque:

$$\tau_i = S_i^T f_i$$

3. Propagate force to parent:

$$f_{p(i)} \mathrel{+}= X_{\text{up},i}^T f_i$$

### 7.2.3 Complexity

RNEA runs in $O(n)$ time where $n$ is the number of bodies. The implementation
uses several optimizations:

- **Pre-allocated Fortran-ordered arrays** for cache-friendly column access.
- **Sparse motion subspace exploitation**: for standard revolute/prismatic joints,
  $S_i$ has a single non-zero element, reducing 6-element dot products to scalar
  reads.
- **Cached dictionary lookups** to avoid repeated hashing in tight loops.

---

## 7.3 Composite Rigid Body Algorithm (CRBA)

**File:** `crba.py`

**Reference:** Featherstone, R. (2008). _Rigid Body Dynamics Algorithms_.
Chapter 6, Algorithm 6.1.

### 7.3.1 Problem Statement

Given joint positions $q$, compute the joint-space mass matrix $H(q)$:

$$H(q) = \sum_{i=1}^{n} S_i^T \, I_i^c \, S_i$$

where $I_i^c$ is the **composite rigid body inertia** of the subtree rooted at
body $i$.

```python
def crba(model, q) -> np.ndarray:
```

### 7.3.2 Algorithm

**Forward Pass** -- compute joint transforms and motion subspaces for each body.

**Backward Pass** -- composite inertia accumulation (leaves to root):

$$I_{p(i)}^c \mathrel{+}= X_{\text{up},i}^T \, I_i^c \, X_{\text{up},i}$$

**Mass Matrix Assembly:**

For each body $i$:

1. Compute $f = I_i^c S_i$ (force due to unit acceleration at joint $i$).
2. Diagonal element: $H_{ii} = S_i^T f$.
3. Propagate $f$ up the tree. For each ancestor $j$:

$$H_{ij} = H_{ji} = S_j^T f_{\text{transformed}}$$

### 7.3.3 Complexity

CRBA runs in $O(n^2)$ time due to the off-diagonal element computation.
Optimizations include:

- **Pre-computed contiguous transposes** of $X_{\text{up}}$ for efficient `np.dot`.
- **Sparse motion subspace**: column extraction replaces full matrix-vector
  products ($36 \to 6$ operations).
- **Buffer swapping** (instead of copying) in the ancestor propagation loop.

### 7.3.4 Properties

The returned mass matrix satisfies:

- **Symmetry:** $H = H^T$ (enforced by simultaneous assignment of $H_{ij}$ and $H_{ji}$).
- **Positive definiteness:** All eigenvalues are positive (guaranteed by construction
  from positive-definite spatial inertias).

---

## 7.4 Articulated Body Algorithm (ABA)

**File:** `aba.py`

**Reference:** Featherstone, R. (2008). _Rigid Body Dynamics Algorithms_.
Chapter 7, Algorithm 7.1.

### 7.4.1 Problem Statement

Given joint positions $q$, velocities $\dot{q}$, and applied torques $\tau$,
compute the resulting joint accelerations:

$$\ddot{q} = \text{ABA}(q, \dot{q}, \tau)$$

This is equivalent to solving $M(q)\ddot{q} = \tau - C(q, \dot{q})\dot{q} - g(q)$,
but in $O(n)$ time without forming or inverting the mass matrix.

```python
def aba(model, q, qd, tau, f_ext=None) -> np.ndarray:
```

### 7.4.2 Algorithm

**Pass 1 -- Forward Kinematics** (root to leaves):

Same as RNEA Pass 1, computing velocities $v_i$ and bias accelerations $c_i$
(velocity-product terms).

Additionally, the bias force is computed:

$$p_i^A = v_i \times^f (I_i v_i) - f_{\text{ext},i}$$

**Pass 2 -- Backward Recursion** (leaves to root):

For each body $i$, compute the articulated-body inertia $I_i^A$ and bias force
$p_i^A$:

1. $U_i = I_i^A S_i$ (projected inertia along joint axis).
2. $d_i = S_i^T U_i$ (scalar joint-space inertia for 1-DOF joints).
3. $u_i = \tau_i - S_i^T p_i^A$ (net joint-space force).
4. Update parent articulated inertia:

$$I_{p(i)}^A \mathrel{+}= X_{\text{up},i}^T \left( I_i^A - \frac{U_i U_i^T}{d_i} \right) X_{\text{up},i}$$

5. Update parent bias force:

$$p_{p(i)}^A \mathrel{+}= X_{\text{up},i}^T \left( p_i^A + I_i^A c_i + \frac{u_i}{d_i} U_i \right)$$

**Pass 3 -- Forward Recursion** (root to leaves):

Compute spatial and joint accelerations:

$$\ddot{q}_i = \frac{u_i - U_i^T a_i}{d_i}$$

$$a_i = X_{\text{up},i} a_{p(i)} + c_i + S_i \ddot{q}_i$$

For root bodies, $a_i = X_{\text{up},i}(-a_g) + c_i$.

### 7.4.3 Complexity

ABA runs in $O(n)$ time with three passes. The implementation features:

- **Rank-1 update avoidance** of outer products via careful buffer management.
- **Numerical tolerance** $\epsilon = 10^{-10}$ to prevent division by zero in
  degenerate configurations.
- **Batch pre-allocation** of all work arrays at the start.

### 7.4.4 Relationship to Other Algorithms

| Algorithm | Solves                                   | Complexity | Notes            |
| --------- | ---------------------------------------- | ---------- | ---------------- |
| RNEA      | $\tau = \text{ID}(q, \dot{q}, \ddot{q})$ | $O(n)$     | Inverse dynamics |
| CRBA      | $M(q)$                                   | $O(n^2)$   | Mass matrix only |
| ABA       | $\ddot{q} = \text{FD}(q, \dot{q}, \tau)$ | $O(n)$     | Forward dynamics |

ABA is strictly more efficient than the na\"ive approach of computing $M$ via CRBA
and then solving $M\ddot{q} = \tau - b$, which requires $O(n^2)$ for CRBA plus
$O(n^2)$ or $O(n^3)$ for the solve.

---

## 7.5 Induced Acceleration Analysis

**File:** `induced_acceleration.py`

### 7.5.1 `MuJoCoInducedAccelerationAnalyzer`

This module decomposes the total joint acceleration into contributions from
individual force sources.

```python
class MuJoCoInducedAccelerationAnalyzer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData): ...
```

### 7.5.2 Mathematical Formulation

Starting from the Euler-Lagrange equation:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau + J^T f_c$$

The total acceleration decomposes as:

$$\ddot{q} = \underbrace{M^{-1}(-g)}_{\ddot{q}_g} + \underbrace{M^{-1}(-C\dot{q})}_{\ddot{q}_c} + \underbrace{M^{-1}\tau}_{\ddot{q}_\tau} + \underbrace{M^{-1}J^T f_c}_{\ddot{q}_{cn}}$$

### 7.5.3 Efficient Batch Solve

Rather than solving four separate linear systems, the implementation stacks
all right-hand sides and performs a single batch solve:

```python
rhs_stack = np.column_stack((-term_G, -term_C, tau_vec, data.qfrc_constraint))
results = np.linalg.solve(M, rhs_stack)  # One LU decomposition for all 4
```

### 7.5.4 Result Structure

```python
class InducedAccelerationResult(TypedDict):
    gravity:    np.ndarray   # M^{-1}(-g)
    velocity:   np.ndarray   # M^{-1}(-C*v)
    control:    np.ndarray   # M^{-1}(tau)
    constraint: np.ndarray   # M^{-1}(J^T f_c)
    total:      np.ndarray   # Sum of all components
```

### 7.5.5 Task-Space Decomposition

The `compute_task_space_components()` method maps joint-space induced
accelerations to Cartesian-space (world frame) accelerations for a specific body:

$$\ddot{x}_{\text{source}} = J \ddot{q}_{\text{source}}$$

The velocity-dependent term additionally includes the bias acceleration
$\dot{J}\dot{q}$:

$$\ddot{x}_{\text{velocity}} = J \ddot{q}_c + \underbrace{(\ddot{x}_{\text{total}} - J \ddot{q}_{\text{total}})}_{\dot{J}\dot{q}}$$

This decomposition reveals, for example, how much of the club head acceleration
at impact is due to gravity versus active muscle torques.

---

---

# Chapter 8: Drake Physics Engine

## 8.1 Overview

The Drake backend provides an alternative physics engine based on MIT's Drake
multibody simulation toolkit. It excels at trajectory optimization, contact
modeling via time-stepping methods, and 3D visualization through MeshCat.

```
src/engines/physics_engines/drake/
    python/
        drake_physics_engine.py    -- PhysicsEngine protocol implementation
        motion_optimization.py     -- Multi-objective trajectory optimization
        swing_plane_integration.py -- Swing plane analysis + MeshCat visualization
```

---

## 8.2 `DrakePhysicsEngine`

**File:** `drake_physics_engine.py`

```python
class DrakePhysicsEngine(PhysicsEngine):
    def __init__(self, time_step: float = DEFAULT_TIME_STEP):
        self.builder = DiagramBuilder()
        self.plant: MultibodyPlant
        self.scene_graph: Any
        result = AddMultibodyPlantSceneGraph(self.builder, time_step)
        self.plant = result[0]
        self.scene_graph = result[1]
        self.diagram: framework.Diagram | None = None
        self.context: framework.Context | None = None
        self.plant_context: framework.Context | None = None
        self._is_finalized = False
        self.simulator: analysis.Simulator | None = None
```

### 8.2.1 Drake Architecture Concepts

**MultibodyPlant** is Drake's representation of a multibody system. It
encapsulates the kinematics and dynamics of rigid and deformable bodies
connected by joints. The plant is constructed incrementally (add models, add
constraints) and then **finalized** -- after which the model topology is
immutable.

**SceneGraph** manages the geometric representation of bodies for collision
detection and visualization.

**DiagramBuilder** wires together systems (plant, scene graph, controllers,
visualizers) into a composite `Diagram`.

**Context** is Drake's mechanism for storing state. Unlike MuJoCo's single
global `MjData`, Drake uses functional-style contexts that allow multiple
independent state snapshots.

### 8.2.2 Initialization Sequence

```python
def _ensure_finalized(self) -> None:
    if not self._is_finalized:
        self.plant.Finalize()
        self.diagram = self.builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.simulator = analysis.Simulator(self.diagram, self.context)
        self.simulator.Initialize()
        self._is_finalized = True
```

### 8.2.3 Model Loading

Drake's `Parser` supports URDF, SDF, and (experimentally) MJCF formats:

```python
def load_from_path(self, path: str) -> None:
    parser = Parser(self.plant)
    parser.AddModels(path)
    self._ensure_finalized()
```

### 8.2.4 State Access

Drake uses `plant.GetPositions()` / `plant.GetVelocities()` instead of
direct array access:

```python
def get_state(self) -> tuple[np.ndarray, np.ndarray]:
    q = self.plant.GetPositions(self.plant_context)
    v = self.plant.GetVelocities(self.plant_context)
    return q, v

def set_state(self, q, v) -> None:
    self.plant.SetPositions(self.plant_context, q)
    self.plant.SetVelocities(self.plant_context, v)
```

### 8.2.5 Simulation Stepping

Drake's `Simulator.AdvanceTo()` integrates the system forward in time:

```python
def step(self, dt: float | None = None) -> None:
    current_time = self.context.get_time()
    step_size = dt if dt is not None else self.plant.time_step()
    self.simulator.AdvanceTo(current_time + step_size)
```

### 8.2.6 Dynamics Interface

**Mass Matrix:**

```python
def compute_mass_matrix(self) -> np.ndarray:
    M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
    return M
```

Drake computes $M(q)$ via the inverse dynamics identity: column $j$ of $M$ is
obtained by evaluating $\text{ID}(q, 0, e_j)$ where $e_j$ is the $j$-th
unit vector.

**Bias Forces:**

```python
def compute_bias_forces(self) -> np.ndarray:
    nv = self.plant.num_velocities()
    vdot_zero = np.zeros(nv)
    forces = self.plant.CalcInverseDynamics(
        self.plant_context, vdot_zero,
        self.plant.MakeMultibodyForces(self.plant))
    return forces
```

This evaluates $\text{ID}(q, v, 0) = C(q, v)v + g(q)$.

**Gravity Forces:**

```python
def compute_gravity_forces(self) -> np.ndarray:
    return self.plant.CalcGravityGeneralizedForces(self.plant_context)
```

**Jacobian:**

Drake's `CalcJacobianSpatialVelocity` returns a $6 \times n_v$ matrix with
angular components in rows 0--2 and linear components in rows 3--5:

```python
def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
    body = self.plant.GetBodyByName(body_name)
    J = self.plant.CalcJacobianSpatialVelocity(
        self.plant_context,
        JacobianWrtVariable.kV,
        body.body_frame(),
        np.zeros(3),
        self.plant.world_frame(),
        self.plant.world_frame())
    return {
        "linear":  J[3:, :],
        "angular": J[:3, :],
        "spatial": J,
    }
```

### 8.2.7 Drift-Control Decomposition

The Drake implementation solves $M\ddot{q} = -b$ for drift:

```python
def compute_drift_acceleration(self) -> np.ndarray:
    M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
    bias = self.compute_bias_forces()
    a_drift = -np.linalg.solve(M, bias)
    return a_drift
```

And $M\ddot{q} = \tau$ for control:

```python
def compute_control_acceleration(self, tau) -> np.ndarray:
    M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
    return np.linalg.solve(M, tau)
```

### 8.2.8 Counterfactual Experiments

Both ZTCF and ZVCF follow the same save-mutate-restore pattern as MuJoCo,
using Drake's `SetPositions`/`SetVelocities`/`GetPositions`/`GetVelocities`:

**ZTCF:** Set $\tau = 0$ at state $(q, v)$:

$$\ddot{q}_{\text{ZTCF}} = -M(q)^{-1} b(q, v)$$

**ZVCF:** Set $v = 0$ at configuration $q$:

$$\ddot{q}_{\text{ZVCF}} = M(q)^{-1}(\tau - g(q))$$

### 8.2.9 Contact Forces

The current Drake wrapper returns a placeholder zero vector for ground
reaction forces. Full contact force retrieval requires querying `ContactResults`
from a `Simulator`-managed context:

```python
def compute_contact_forces(self) -> np.ndarray:
    # Placeholder: full GRF requires ContactResults from Simulator
    return np.zeros(3)
```

---

## 8.3 Motion Optimization

**File:** `motion_optimization.py`

### 8.3.1 `DrakeMotionOptimizer`

```python
class DrakeMotionOptimizer:
    def __init__(self):
        self.objectives: list[OptimizationObjective] = []
        self.constraints: list[OptimizationConstraint] = []
```

### 8.3.2 Objective and Constraint Data Structures

```python
@dataclass
class OptimizationObjective:
    name: str
    weight: float
    target_value: float | None = None
    cost_function: Callable[[np.ndarray], float] | None = None

@dataclass
class OptimizationConstraint:
    name: str
    constraint_type: str   # 'equality', 'inequality', 'bounds'
    lower_bound: float | None = None
    upper_bound: float | None = None
    constraint_function: Callable[[np.ndarray], float] | None = None
```

### 8.3.3 SLSQP Trajectory Optimization

The optimizer formulates a nonlinear program:

$$\min_{x} \sum_i w_i \, f_i(x)$$

subject to:

$$g_j(x) \leq 0 \quad \text{(inequality constraints)}$$
$$h_k(x) = 0 \quad \text{(equality constraints)}$$

where $x$ is the flattened trajectory. The solver is SciPy's SLSQP
(Sequential Least Squares Programming):

```python
def optimize_trajectory(
    self,
    initial_trajectory: np.ndarray,    # (N, dim)
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> OptimizationResult:
    opt_result = scipy_minimize(
        total_cost, x0,
        method="SLSQP",
        constraints=scipy_constraints,
        options={"maxiter": max_iterations, "ftol": tolerance})
```

### 8.3.4 Standard Golf Objectives

The `setup_standard_golf_objectives()` method registers three default objectives:

1. **Ball speed** ($w = 1.0$): Maximize club head velocity at impact.
2. **Accuracy** ($w = 0.8$): Minimize lateral deviation from target line.
3. **Smoothness** ($w = 0.5$): Minimize second-derivative norm (jerk).

### 8.3.5 Standard Golf Constraints

The `setup_standard_golf_constraints()` method registers:

1. **Joint limits** (inequality): Joint angles must remain within anatomical range.
2. **Impact timing** (equality): Ball contact must occur at a specified time.

### 8.3.6 Result Structure

```python
@dataclass
class OptimizationResult:
    success: bool
    optimal_trajectory: np.ndarray
    optimal_cost: float
    iterations: int
    convergence_message: str
    objective_values: dict[str, float]
    constraint_violations: dict[str, float]
```

### 8.3.7 Convenience Methods

```python
def optimize_for_distance(self, initial_trajectory, target_distance=250.0) -> OptimizationResult
def optimize_for_accuracy(self, initial_trajectory, target_point) -> OptimizationResult
def export_optimization_results(self, result, output_path) -> None
```

---

## 8.4 Swing Plane Integration

**File:** `swing_plane_integration.py`

### 8.4.1 `DrakeSwingPlaneAnalyzer`

```python
class DrakeSwingPlaneAnalyzer:
    def __init__(self):
        self.analyzer = SwingPlaneAnalyzer()  # From shared module
```

### 8.4.2 Trajectory Analysis

```python
def analyze_trajectory(
    self, positions: np.ndarray, timestamps: np.ndarray | None = None
) -> SwingPlaneMetrics:
    """Fit a plane to club head positions (N, 3)."""
```

### 8.4.3 Drake Context Extraction

The analyzer can extract club head positions directly from a Drake simulation
context or log:

```python
def analyze_from_drake_context(
    self, context, plant, club_body_index, num_samples=100
) -> SwingPlaneMetrics:
    """Extract club positions from Drake plant context and analyze."""
```

### 8.4.4 Swing Plane Deviation Cost

The swing plane deviation cost penalises the sum of squared distances from each
trajectory point to the fitted swing plane:

$$C_{\text{plane}} = w \sum_{i=1}^{N} (\mathbf{n} \cdot (\mathbf{p}_i - \mathbf{p}_0))^2$$

where $\mathbf{n}$ is the plane normal, $\mathbf{p}_0$ is a point on the plane,
and $w$ is the constraint weight. This cost is registered with the
`DrakeMotionOptimizer` via:

```python
def integrate_with_optimization(
    self, trajectory_optimizer, swing_plane_constraint_weight=1.0
) -> None:
    def swing_plane_cost(trajectory):
        positions = trajectory[:, :3]
        metrics = self.analyzer.analyze(positions)
        return metrics.rmse ** 2

    trajectory_optimizer.add_objective(
        name="swing_plane_deviation",
        weight=swing_plane_constraint_weight,
        cost_function=swing_plane_cost,
        target_value=0.0)
```

### 8.4.5 MeshCat Visualization

The `visualize_with_meshcat()` method renders three visual elements into Drake's
MeshCat 3D viewer:

1. **Semi-transparent plane** -- the fitted swing plane as a triangulated quad.
2. **Trajectory line** -- club head path as a colored line strip.
3. **Deviation lines** -- (planned) lines from each point to the plane showing
   deviation magnitude.

```python
def visualize_with_meshcat(
    self,
    meshcat_visualizer,       # pydrake.geometry.Meshcat instance
    metrics: SwingPlaneMetrics,
    trajectory_positions: np.ndarray,
) -> None:
    meshcat.SetTriangleMesh(f"{prefix}/plane", mesh, Rgba(0.2, 0.5, 0.8, 0.3))
    meshcat.SetLine(f"{prefix}/trajectory", positions.T, line_width=3.0,
                    rgba=Rgba(1.0, 0.3, 0.1, 1.0))
```

### 8.4.6 Export

```python
def export_for_analysis(
    self,
    metrics: SwingPlaneMetrics,
    trajectory_positions: np.ndarray,
    output_path: str,
) -> None:
    """Export to JSON with normal vector, steepness, RMSE, trajectory."""
```

---

## 8.5 Direct Collocation (Drake Integration Notes)

While the current codebase uses SLSQP via SciPy for trajectory optimization,
Drake natively supports **direct collocation** methods for trajectory
optimization through `pydrake.trajectories` and `pydrake.solvers`.

### 8.5.1 Hermite-Simpson Collocation

In Hermite-Simpson collocation, the trajectory between knot points is
approximated by a cubic Hermite polynomial. The collocation constraint at the
midpoint ensures dynamic feasibility:

$$x_{k+\frac{1}{2}} = \frac{1}{2}(x_k + x_{k+1}) + \frac{h_k}{8}(\dot{x}_k - \dot{x}_{k+1})$$

$$\dot{x}_{k+\frac{1}{2}} = f(x_{k+\frac{1}{2}}, u_{k+\frac{1}{2}})$$

where $h_k$ is the time step between knot points $k$ and $k+1$.

### 8.5.2 Trapezoidal Collocation

The simpler trapezoidal method enforces:

$$x_{k+1} = x_k + \frac{h_k}{2}\bigl(f(x_k, u_k) + f(x_{k+1}, u_{k+1})\bigr)$$

### 8.5.3 Nonlinear Solvers

Drake supports multiple nonlinear programming solvers:

- **SNOPT** -- Sequential quadratic programming solver for large-scale NLPs.
  Preferred for trajectory optimization due to sparse Jacobian exploitation.
- **IPOPT** -- Interior point optimizer. Good for problems with many inequality
  constraints.
- **SLSQP** -- Used in the current implementation via SciPy. Suitable for
  moderate-scale problems.

The solver selection is automatic in Drake based on problem structure.

---

## 8.6 LQR Control in Drake Context

The simulation framework (`sim.py`) implements an LQR-structured controller
that can be used with either MuJoCo or Drake backends:

$$u = -K(x - x^*)$$

where $K \in \mathbb{R}^{n_u \times n_x}$ is the gain matrix and $x^*$ is
the target state. The current implementation uses a decoupled PD gain structure:

$$K_{i, q_j} = K_p, \quad K_{i, \dot{q}_j} = K_d$$

with $K_p = 100$, $K_d = 10$. The full linearisation-based LQR (via Drake's
`LinearQuadraticRegulator`) is available for models without quaternion
singularities.

---

## 8.7 Contact Modeling

Drake provides two contact models:

### 8.7.1 Point Contact (Compliant)

Forces are computed using a spring-damper model at each contact point:

$$f_n = k_n \phi + d_n \dot{\phi}$$

where $\phi$ is the penetration depth, $k_n$ is the contact stiffness, and
$d_n$ is the damping coefficient. This is analogous to MuJoCo's default
solver.

### 8.7.2 Hydroelastic Contact

Drake's hydroelastic contact model computes contact pressures over a surface
patch rather than at a single point, providing smoother and more physically
accurate contact forces. This is particularly relevant for modelling the
club-ball impact where the contact area evolves during the collision.

The pressure field is computed as:

$$p(\mathbf{x}) = E^* \, \phi(\mathbf{x})$$

where $E^*$ is the effective elastic modulus and $\phi(\mathbf{x})$ is the
local interpenetration function.

---

## 8.8 Engine Comparison: MuJoCo vs. Drake

| Feature                 | MuJoCo                               | Drake                                      |
| ----------------------- | ------------------------------------ | ------------------------------------------ |
| Primary formulation     | Maximal coordinates with constraints | Minimal coordinates (MultibodyPlant)       |
| Contact solver          | Convex optimisation (PGS, Newton)    | Time-stepping, hydroelastic                |
| Muscle support          | Native (via actuators or MyoSuite)   | Via external force elements                |
| Trajectory optimisation | Via `motion_optimization.py` + SciPy | Direct collocation (SNOPT/IPOPT/SLSQP)     |
| Visualisation           | `mujoco.viewer`, OpenGL              | MeshCat (WebGL), Drake Visualizer          |
| State management        | Single `MjData` (mutable)            | Functional `Context` (immutable snapshots) |
| MJCF model loading      | Native                               | Experimental                               |
| URDF/SDF loading        | Via conversion                       | Native                                     |
| Flexible shaft          | Modal (Euler-Bernoulli modes)        | Compliant multibody elements               |
| Counterfactuals         | Save-mutate-restore                  | Context-based snapshots                    |
| Induced acceleration    | Built-in analyser                    | Via protocol methods                       |

Both engines satisfy the `PhysicsEngine` protocol, ensuring that all GUI,
analytics, and optimization code works identically regardless of the active
backend.

---

# UpstreamDrift User Manual -- Part 3: Physics Engines and Biomechanical Analysis

---

# Chapter 9: Pinocchio Physics Engine

## 9.1 Overview

The Pinocchio engine provides high-performance rigid body dynamics based on Roy Featherstone's
spatial algebra and recursive algorithms. Unlike contact-rich simulators such as MuJoCo,
Pinocchio excels at _analytical_ computations -- Jacobians, dynamics derivatives, mass matrices,
and inverse dynamics -- all with $O(n)$ complexity in the number of joints. Within UpstreamDrift,
Pinocchio serves two primary roles:

1. **Standalone physics engine** via `PinocchioPhysicsEngine`, implementing the full
   `PhysicsEngine` protocol for forward simulation, state management, and drift-control
   decomposition.
2. **Analytical co-processor** via `PinocchioBackend` and `PinocchioWrapper`, providing
   fast Jacobians and dynamics derivatives alongside a MuJoCo-based simulation.

### Source files

| File                                                                                    | Purpose                                      |
| --------------------------------------------------------------------------------------- | -------------------------------------------- |
| `src/engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py`              | Full `PhysicsEngine` protocol implementation |
| `src/engines/physics_engines/pinocchio/python/dtack/backends/pinocchio_backend.py`      | Standalone backend for URDF-loaded models    |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py` | MuJoCo-Pinocchio bridge (`PinocchioWrapper`) |

### Dependencies

```
pip install pin            # Pinocchio (>= 2.6)
pip install numpy          # Required
pip install pink           # Optional: inverse kinematics
```

Availability is checked at import time via `src/shared/python/engine_availability.PINOCCHIO_AVAILABLE`.
When Pinocchio is absent, a `DummyPin` stub prevents `NameError` but raises `ImportError` on any
actual usage.

---

## 9.2 Core Algorithms

Pinocchio implements three foundational $O(n)$ algorithms for an articulated rigid body system
with $n$ joints, configuration vector $q \in \mathbb{R}^{n_q}$, velocity vector
$\dot{q} \in \mathbb{R}^{n_v}$, and applied torque $\tau \in \mathbb{R}^{n_v}$.

### 9.2.1 Recursive Newton-Euler Algorithm (RNEA) -- Inverse Dynamics

Given a desired acceleration $\ddot{q}$, RNEA computes the required joint torques:

$$\tau = \text{RNEA}(q, \dot{q}, \ddot{q}) = M(q)\,\ddot{q} + C(q, \dot{q})\,\dot{q} + g(q)$$

where $M(q)$ is the joint-space inertia matrix, $C(q, \dot{q})$ captures Coriolis and centrifugal
terms, and $g(q)$ is the gravity vector.

**API** (`PinocchioPhysicsEngine`):

```python
def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
    """Compute inverse dynamics tau = ID(q, v, a).

    Args:
        qacc: Desired acceleration vector (n_v,) [rad/s^2 or m/s^2]

    Returns:
        tau: Required generalized forces (n_v,) [N*m or N]
    """
```

**API** (`PinocchioBackend`):

```python
def compute_inverse_dynamics(
    self,
    q: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute inverse dynamics (RNEA).

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        a: Joint accelerations [nv]

    Returns:
        Joint torques [nv]
    """
```

**Usage**:

```python
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)

engine = PinocchioPhysicsEngine()
engine.load_from_path("/path/to/humanoid.urdf")

# Compute torques to hold current position (zero acceleration)
qacc_zero = np.zeros(engine.model.nv)
tau_gravity_compensation = engine.compute_inverse_dynamics(qacc_zero)
# tau_gravity_compensation contains the torques needed to resist gravity
```

**Special case -- Bias forces**: Setting $\ddot{q} = 0$ yields the bias force vector
$b(q, \dot{q}) = C(q, \dot{q})\,\dot{q} + g(q)$:

```python
bias = engine.compute_bias_forces()
# Equivalent to: pin.rnea(model, data, q, v, np.zeros(nv))
```

**Special case -- Gravity vector**: Setting $\ddot{q} = 0$ and $\dot{q} = 0$ yields
$g(q)$:

```python
gravity = engine.compute_gravity_forces()
# Uses: pin.computeGeneralizedGravity(model, data, q)
```

### 9.2.2 Articulated Body Algorithm (ABA) -- Forward Dynamics

Given applied torques $\tau$, ABA computes the resulting accelerations:

$$\ddot{q} = \text{ABA}(q, \dot{q}, \tau) = M(q)^{-1}\bigl(\tau - C(q,\dot{q})\,\dot{q} - g(q)\bigr)$$

This is the central algorithm for forward simulation and counterfactual analysis.

**API** (`PinocchioPhysicsEngine.step`):

```python
def step(self, dt: float | None = None) -> None:
    """Advance the simulation by one time step.

    Uses ABA for forward dynamics, then semi-implicit Euler integration:
        a = ABA(q, v, tau)
        v_next = v + a * dt
        q_next = integrate(q, v_next * dt)
    """
```

**API** (`PinocchioBackend`):

```python
def compute_forward_dynamics(
    self,
    q: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute forward dynamics (ABA).

    Returns:
        Joint accelerations [nv]
    """
```

**Usage** -- Forward simulation loop:

```python
engine = PinocchioPhysicsEngine()
engine.load_from_path("robot.urdf")

# Apply constant torque and simulate
tau = np.array([0.0, 0.0, 5.0, -2.0])  # Example torques
engine.set_control(tau)

for _ in range(1000):
    engine.step(dt=0.001)
    q, v = engine.get_state()
    print(f"t={engine.get_time():.3f}s, q={q}")
```

### 9.2.3 Composite Rigid Body Algorithm (CRBA) -- Mass Matrix

CRBA computes the joint-space inertia matrix $M(q) \in \mathbb{R}^{n_v \times n_v}$:

$$M(q) = \sum_{i=1}^{n} J_i(q)^T \, \mathcal{I}_i \, J_i(q)$$

where $J_i$ is the Jacobian of body $i$ and $\mathcal{I}_i$ is its spatial inertia.

**API**:

```python
def compute_mass_matrix(self) -> np.ndarray:
    """Compute the dense inertia matrix M(q).

    Returns:
        M: (n_v, n_v) symmetric positive definite mass matrix.

    Notes:
        Pinocchio's CRBA fills only the upper triangular part.
        The engine symmetrizes: M = triu(M) + triu(M,1).T
    """
```

**Usage**:

```python
M = engine.compute_mass_matrix()

# Verify symmetry and positive definiteness
assert np.allclose(M, M.T), "Mass matrix must be symmetric"
eigenvalues = np.linalg.eigvalsh(M)
assert np.all(eigenvalues > 0), "Mass matrix must be positive definite"

# Kinetic energy
v = engine.v
T = 0.5 * v.T @ M @ v
```

---

## 9.3 Fast Analytical Jacobians and Derivatives

One of Pinocchio's key strengths is analytical (not finite-difference) computation of Jacobians
and dynamics derivatives. This makes it ideal for trajectory optimization and sensitivity analysis.

### 9.3.1 Frame Jacobians

The geometric Jacobian maps joint velocities to the spatial velocity of a frame:

$$\begin{pmatrix} v \\ \omega \end{pmatrix} = J(q) \, \dot{q}$$

where $J(q) \in \mathbb{R}^{6 \times n_v}$ and $v, \omega$ are the linear and angular velocities
of the frame.

The Jacobian time derivative satisfies:

$$\dot{J}(q, \dot{q}) = \frac{\partial J(q)}{\partial q}\,\dot{q}$$

This is critical for operational-space control where the task-space acceleration is:

$$\ddot{x} = J(q)\,\ddot{q} + \dot{J}(q, \dot{q})\,\dot{q}$$

**API** (`PinocchioPhysicsEngine`):

```python
def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
    """Compute spatial Jacobian for a specific body.

    Args:
        body_name: Name of the body frame in the model.

    Returns:
        Dictionary with keys:
          - 'linear':  (3, n_v) position Jacobian  [m/rad or m/m]
          - 'angular': (3, n_v) rotation Jacobian   [rad/rad]
          - 'spatial': (6, n_v) combined [angular; linear]
        Returns None if body_name not found.
    """
```

**API** (`PinocchioBackend`):

```python
def compute_frame_jacobian(
    self,
    q: npt.NDArray[np.float64],
    frame_id: int | str,
    reference_frame: int = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
) -> npt.NDArray[np.float64]:
    """Compute frame Jacobian.

    Args:
        q: Joint positions [nq]
        frame_id: Frame ID (int) or frame name (str)
        reference_frame: Reference frame for Jacobian
            - pin.LOCAL: Body-fixed frame
            - pin.WORLD: World frame
            - pin.LOCAL_WORLD_ALIGNED: Body origin, world orientation (default)

    Returns:
        Jacobian matrix [6 x nv]
    """
```

**Usage** -- Club head velocity computation:

```python
engine.forward()  # Update kinematics

jac = engine.compute_jacobian("club_head")
if jac is not None:
    # Linear velocity of club head
    v_club = jac["linear"] @ engine.v   # (3,) in m/s
    club_speed = np.linalg.norm(v_club)
    print(f"Club head speed: {club_speed:.1f} m/s")

    # Angular velocity of club head
    omega_club = jac["angular"] @ engine.v  # (3,) in rad/s
```

### 9.3.2 Analytical Dynamics Derivatives

Pinocchio can compute the partial derivatives of the forward dynamics equations analytically,
which is essential for trajectory optimization and model-predictive control.

Given forward dynamics $\ddot{q} = f(q, \dot{q}, \tau)$, the derivatives are:

$$\frac{\partial \ddot{q}}{\partial q}, \quad \frac{\partial \ddot{q}}{\partial \dot{q}}, \quad \frac{\partial \ddot{q}}{\partial \tau} = M(q)^{-1}$$

**API** (`PinocchioWrapper` -- MuJoCo bridge):

```python
def compute_dynamics_derivatives(
    self,
    q: np.ndarray | None = None,
    v: np.ndarray | None = None,
    tau: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical derivatives of dynamics.

    Returns:
        Tuple of (df_dq, df_dv, df_dtau, df_du):
          - df_dq:   (n_v, n_v) partial of acceleration w.r.t. position
          - df_dv:   (n_v, n_v) partial of acceleration w.r.t. velocity
          - df_dtau: (n_v, n_v) partial of acceleration w.r.t. torque = M^{-1}
          - df_du:   (n_v, n_v) partial of acceleration w.r.t. control
    """
```

Under the hood this calls:

```python
pin.computeABADerivatives(self.pin_model, self.pin_data, q, v, tau)
# Results stored in:
#   self.pin_data.ddq_dq   -- partial(ddot{q}) / partial(q)
#   self.pin_data.ddq_dv   -- partial(ddot{q}) / partial(dot{q})
#   self.pin_data.Minv     -- M(q)^{-1} = partial(ddot{q}) / partial(tau)
```

**Usage** -- Linearization for LQR control:

```python
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.pinocchio_interface import (
    PinocchioWrapper,
)

wrapper = PinocchioWrapper(mj_model, mj_data)
wrapper.sync_mujoco_to_pinocchio()

# Get linearization matrices at current state
A_qq, A_qv, B, _ = wrapper.compute_dynamics_derivatives()

# Continuous-time state-space: dx/dt = A x + B u
# where x = [q; v], u = tau
n = wrapper.pin_model.nv
A = np.block([
    [np.zeros((n, n)), np.eye(n)],
    [A_qq,             A_qv     ]
])
# B is already M^{-1}
```

### 9.3.3 Coriolis Matrix

Pinocchio provides direct computation of the Coriolis matrix $C(q, \dot{q})$ satisfying:

$$M(q)\,\ddot{q} + C(q, \dot{q})\,\dot{q} + g(q) = \tau$$

**API** (`PinocchioWrapper`):

```python
def compute_coriolis_matrix(
    self,
    q: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Coriolis matrix [nv x nv].

    Uses: pin.computeCoriolisMatrix(model, data, q, v)
    """
```

---

## 9.4 Drift-Control Decomposition (Section F)

Pinocchio implements the Section F drift-control decomposition, which separates total
acceleration into passive and active components:

$$\ddot{q}_{full} = \underbrace{\ddot{q}_{drift}}_{\text{passive}} + \underbrace{\ddot{q}_{control}}_{\text{active}}$$

### 9.4.1 Drift Acceleration

The drift (passive) acceleration is what would occur with zero applied torques. It arises
from gravity, Coriolis, and centrifugal effects:

$$\ddot{q}_{drift} = M(q)^{-1}\bigl(-C(q, \dot{q})\,\dot{q} - g(q)\bigr) = \text{ABA}(q, \dot{q}, \mathbf{0})$$

```python
def compute_drift_acceleration(self) -> np.ndarray:
    """Compute passive (drift) acceleration with zero control inputs.

    Uses ABA with zero torque: pin.aba(model, data, q, v, zeros(nv))
    """
```

### 9.4.2 Control Acceleration

The control acceleration is the component due solely to applied torques:

$$\ddot{q}_{control} = M(q)^{-1}\,\tau$$

```python
def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
    """Compute control-attributed acceleration.

    Uses: np.linalg.solve(M, tau)
    """
```

### 9.4.3 Superposition Verification

The critical contract is: $\ddot{q}_{drift} + \ddot{q}_{control} = \ddot{q}_{full}$.

```python
# Verify superposition
a_drift = engine.compute_drift_acceleration()
a_control = engine.compute_control_acceleration(engine.tau)
a_full = pin.aba(engine.model, engine.data, engine.q, engine.v, engine.tau)

assert np.allclose(a_drift + a_control, a_full, atol=1e-10), \
    "Superposition violated!"
```

---

## 9.5 Counterfactual Experiments (Section G)

### 9.5.1 Zero-Torque Counterfactual (ZTCF)

ZTCF answers: "What would happen if all actuators turned off right now?"

$$\ddot{q}_{ZTCF} = \text{ABA}(q, \dot{q}, \mathbf{0})$$

```python
def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

    Args:
        q: Joint positions (n_q,) [rad or m]
        v: Joint velocities (n_v,) [rad/s or m/s]

    Returns:
        Acceleration under zero applied torque (n_v,)
    """
```

### 9.5.2 Zero-Velocity Counterfactual (ZVCF)

ZVCF answers: "What acceleration would occur if motion froze instantaneously?"

$$\ddot{q}_{ZVCF} = \text{ABA}(q, \mathbf{0}, \tau)$$

This isolates gravity and control from Coriolis/centrifugal effects. The velocity-dependent
contribution can then be computed as:

$$\Delta \ddot{q}_{velocity} = \ddot{q}_{full} - \ddot{q}_{ZVCF}$$

```python
def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
    """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

    Args:
        q: Joint positions (n_q,) [rad or m]

    Returns:
        Acceleration with v=0, preserving current control (n_v,)
    """
```

**Usage** -- Downswing analysis:

```python
# During downswing, separate gravity from centrifugal "whip"
q, v = engine.get_state()

a_full = pin.aba(engine.model, engine.data, q, v, engine.tau)
a_zvcf = engine.compute_zvcf(q)   # No Coriolis/centrifugal
a_ztcf = engine.compute_ztcf(q, v)  # No control torques

a_coriolis_centrifugal = a_full - a_zvcf
a_control_effect = a_full - a_ztcf

print(f"Centrifugal contribution: {np.linalg.norm(a_coriolis_centrifugal):.2f} rad/s^2")
print(f"Control contribution:     {np.linalg.norm(a_control_effect):.2f} rad/s^2")
```

---

## 9.6 URDF Model Loading and Joint Configuration

Pinocchio primarily loads models from URDF (Unified Robot Description Format) files.

### 9.6.1 Loading from URDF

```python
engine = PinocchioPhysicsEngine()
engine.load_from_path("/path/to/model.urdf")

# After loading:
print(f"Model: {engine.model_name}")
print(f"Configuration DOF (nq): {engine.model.nq}")
print(f"Velocity DOF (nv): {engine.model.nv}")
print(f"Joints: {engine.get_joint_names()}")
```

The distinction between $n_q$ (configuration) and $n_v$ (velocity) dimensions matters for
models with non-Euclidean joints. For example, a free-floating base uses quaternions
($n_q = 7$) but has 6 velocity DOF ($n_v = 6$).

### 9.6.2 Loading from XML String

```python
engine.load_from_string(urdf_content, extension="urdf")
# Uses: pin.buildModelFromXML(content)
```

### 9.6.3 Neutral Configuration

After loading, the engine initializes to the neutral (zero) configuration:

```python
q0 = pin.neutral(engine.model)  # Neutral configuration
engine.q = q0
engine.v = np.zeros(engine.model.nv)
```

### 9.6.4 Configuration Space Integration

Pinocchio handles Lie group integration correctly for non-Euclidean joints:

```python
# Correct integration on manifold (handles quaternions, SO(3), etc.)
q_next = pin.integrate(model, q, v * dt)
# NOT: q_next = q + v * dt  (wrong for quaternions!)
```

This is automatically used in `PinocchioPhysicsEngine.step()`.

---

## 9.7 MuJoCo-Pinocchio Bridge

The `PinocchioWrapper` class enables using Pinocchio's analytical algorithms alongside
MuJoCo's simulation capabilities. This is the recommended architecture for golf swing
analysis: MuJoCo handles contacts and constraints while Pinocchio provides fast Jacobians
and derivatives.

### 9.7.1 Initialization

```python
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.pinocchio_interface import (
    PinocchioWrapper,
    create_pinocchio_wrapper,
)

# From MuJoCo model
wrapper = create_pinocchio_wrapper(mj_model, mj_data)

# Or directly
wrapper = PinocchioWrapper(mj_model, mj_data, use_mjcf_parser=True)
```

The wrapper converts MuJoCo models to URDF via a temporary file, then loads the URDF in
Pinocchio.

### 9.7.2 Quaternion Convention Conversion

MuJoCo and Pinocchio use different quaternion conventions for free joints:

- MuJoCo: $(w, x, y, z)$
- Pinocchio: $(x, y, z, w)$

The wrapper handles this automatically:

```python
# Internal conversion (transparent to user)
q_pin = wrapper._mujoco_q_to_pinocchio_q(mj_data.qpos)
q_mj  = wrapper._pinocchio_q_to_mujoco_q(q_pin)
```

### 9.7.3 State Synchronization

```python
# After MuJoCo simulation step:
mujoco.mj_step(mj_model, mj_data)

# Sync to Pinocchio for analytical computation
wrapper.sync_mujoco_to_pinocchio()

# Now use Pinocchio's fast algorithms
J = wrapper.compute_end_effector_jacobian("club_head")
M = wrapper.compute_mass_matrix()
KE = wrapper.compute_kinetic_energy()
PE = wrapper.compute_potential_energy()
```

### 9.7.4 Energy Computation

```python
# Kinetic energy: T = 0.5 * v^T M v
KE = wrapper.compute_kinetic_energy()

# Potential energy via Pinocchio's built-in
PE = wrapper.compute_potential_energy()

# Total mechanical energy
E = KE + PE
```

---

## 9.8 PinocchioBackend for Standalone Usage

The `PinocchioBackend` class provides a simpler interface for standalone dynamics
computations without the full `PhysicsEngine` protocol overhead.

```python
from src.engines.physics_engines.pinocchio.python.dtack.backends.pinocchio_backend import (
    PinocchioBackend,
)

backend = PinocchioBackend("/path/to/robot.urdf")

q = np.zeros(backend.model.nq)
v = np.zeros(backend.model.nv)

# Mass matrix
M = backend.compute_mass_matrix(q)

# Bias forces (gravity + Coriolis)
# Uses pin.nle() which is more efficient than rnea with zero acceleration
b = backend.compute_bias_forces(q, v)

# Forward kinematics -- all frame placements
frames = backend.forward_kinematics(q)
for i, frame in enumerate(frames):
    print(f"Frame {i}: position = {frame.translation}")
```

---

## 9.9 PINK Inverse Kinematics Integration

Pinocchio integrates with PINK (Pinocchio INverse Kinematics) for solving IK problems
using the computed Jacobians. Given a target frame placement $X_{target} \in SE(3)$,
IK solves:

$$\Delta q = J(q)^{\dagger} \, \text{Log}(X_{current}^{-1} \, X_{target})$$

where $J^{\dagger}$ is the Moore-Penrose pseudo-inverse and Log is the $SE(3)$ logarithm map.

```python
import pink

# Define IK task
task = pink.FrameTask(
    "club_head",
    position_cost=1.0,
    orientation_cost=0.5,
)

# Set target
task.set_target(target_transform)

# Solve IK
configuration = pink.Configuration(engine.model, engine.data, engine.q)
velocity = pink.solve_ik(configuration, [task], dt=0.01)
engine.q = pin.integrate(engine.model, engine.q, velocity * 0.01)
```

---

## 9.10 Performance Comparison with MuJoCo

| Operation               | Pinocchio           | MuJoCo             | Notes                             |
| ----------------------- | ------------------- | ------------------ | --------------------------------- |
| Forward dynamics (ABA)  | $O(n)$ analytical   | $O(n)$ analytical  | Comparable speed                  |
| Inverse dynamics (RNEA) | $O(n)$ analytical   | `mj_inverse`       | Pinocchio more flexible           |
| Mass matrix (CRBA)      | $O(n^2)$ analytical | `mj_fullM`         | Comparable                        |
| Jacobians               | Analytical, $O(n)$  | Analytical         | Both fast                         |
| Dynamics derivatives    | Analytical, $O(n)$  | Finite differences | **Pinocchio far superior**        |
| Contact handling        | Limited             | Excellent          | Use MuJoCo for contacts           |
| Constraint dynamics     | Via ProxQP solver   | Built-in           | MuJoCo more mature                |
| Lie group integration   | Native              | Not applicable     | Pinocchio handles SO(3) correctly |

**Recommendation**: Use the `PinocchioWrapper` bridge for golf swing analysis. MuJoCo
simulates the swing with contacts, while Pinocchio provides analytical Jacobians and
derivatives for optimization and control design.

---

## 9.11 Complete Workflow Example

```python
import numpy as np
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)

# 1. Initialize engine
engine = PinocchioPhysicsEngine()
engine.load_from_path("models/humanoid_golfer.urdf")

# 2. Set initial configuration (address position)
q_address = np.array([...])  # Pre-computed address pose
engine.set_state(q_address, np.zeros(engine.model.nv))

# 3. Forward kinematics
engine.forward()
jac = engine.compute_jacobian("club_head")

# 4. Compute dynamics quantities
M = engine.compute_mass_matrix()
g = engine.compute_gravity_forces()
b = engine.compute_bias_forces()

# 5. Simulate with torque control
for step in range(5000):
    # PD control example
    q, v = engine.get_state()
    tau = -100.0 * (q - q_target) - 20.0 * v  # PD gains

    engine.set_control(tau)
    engine.step(dt=0.001)

    # Analyze drift vs control at each step
    a_drift = engine.compute_drift_acceleration()
    a_control = engine.compute_control_acceleration(tau)

# 6. Batched state query (performance optimization)
full_state = engine.get_full_state()
q, v, t, M = full_state["q"], full_state["v"], full_state["t"], full_state["M"]
```

---

# Chapter 10: OpenSim Physics Engine

## 10.1 Overview

The OpenSim engine provides integration with the OpenSim musculoskeletal modeling platform,
the de facto standard in biomechanics research. Within UpstreamDrift, OpenSim serves as the
**biomechanical validation engine** -- its Hill-type muscle models, established model library,
and clinical validation history make it the gold standard for verifying that simulation
results are physiologically plausible.

### Source files

| File                                                                   | Purpose                                      |
| ---------------------------------------------------------------------- | -------------------------------------------- |
| `src/engines/physics_engines/opensim/python/opensim_physics_engine.py` | Full `PhysicsEngine` protocol implementation |
| `src/engines/physics_engines/opensim/python/muscle_analysis.py`        | Muscle analysis and grip modeling            |
| `src/engines/physics_engines/opensim/python/opensim_gui.py`            | Diagnostic GUI                               |

### Dependencies

```
pip install opensim      # OpenSim Python bindings (>= 4.4)
pip install numpy
```

Availability is checked via `src/shared/python/engine_availability.OPENSIM_AVAILABLE`.

---

## 10.2 Engine Initialization and Model Loading

### 10.2.1 Loading `.osim` Models

OpenSim uses its own XML-based model format (`.osim`):

```python
from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine,
)

engine = OpenSimPhysicsEngine()
engine.load_from_path("/path/to/model.osim")

print(f"Model: {engine.model_name}")
print(f"Coordinates: {engine._model.getNumCoordinates()}")
print(f"Speeds: {engine._model.getNumSpeeds()}")
print(f"Controls: {engine._model.getNumControls()}")
```

The loading sequence:

1. `opensim.Model(path)` -- Parse the `.osim` XML file
2. `model.initSystem()` -- Build the Simbody multibody tree and initialize the `State`
3. `opensim.Manager(model)` -- Create an integration manager for time stepping

### 10.2.2 Loading from String

For dynamically generated models, loading from an XML string is supported via a temporary
file:

```python
osim_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
  <Model name="simple_pendulum">
    ...
  </Model>
</OpenSimDocument>"""

engine.load_from_string(osim_xml, extension="osim")
```

The temporary file is cleaned up automatically after loading.

### 10.2.3 State Management

OpenSim separates configuration ($Q$, generalized coordinates) from speeds ($U$,
generalized velocities):

```python
# Get state
q, v = engine.get_state()
# q: Generalized coordinates (n_q,) -- joint angles, translations
# v: Generalized speeds (n_u,) -- joint angular velocities, translational velocities

# Set state
engine.set_state(q_desired, v_desired)
# Internally calls model.realizeVelocity(state) after setting
```

---

## 10.3 Simulation and Integration

### 10.3.1 Time Stepping

OpenSim uses a variable-step integrator (Runge-Kutta-Merson by default):

```python
engine.step(dt=0.01)
# Internally:
#   manager.setInitialTime(current_time)
#   manager.setFinalTime(current_time + dt)
#   manager.integrate(current_time + dt)
```

### 10.3.2 Forward Computation

```python
engine.forward()
# Calls: model.realizeDynamics(state)
# Updates all derived quantities (forces, accelerations) without advancing time
```

### 10.3.3 Reset

```python
engine.reset()
# Re-initializes state to defaults
# Equilibrates muscles (finds static equilibrium of muscle states)
# Resets integration manager
```

The muscle equilibration step is critical: `model.equilibrateMuscles(state)` solves
for the fiber lengths that satisfy the force-length-velocity equilibrium at the current
activation levels. Without this, initial muscle transients can destabilize the simulation.

---

## 10.4 Dynamics Computations

### 10.4.1 Mass Matrix

OpenSim computes the mass matrix via the Simbody `MatterSubsystem`:

```python
M = engine.compute_mass_matrix()
# Internally:
#   matter = model.getMatterSubsystem()
#   model.realizePosition(state)
#   matter.calcM(state, m_mat)
#   # Convert opensim.Matrix to numpy
```

### 10.4.2 Bias Forces

```python
bias = engine.compute_bias_forces()
# Uses inverse dynamics with zero acceleration:
# bias = ID(q, v, 0) = C(q,v)*v + g(q)
```

### 10.4.3 Gravity Forces

OpenSim isolates gravity by temporarily zeroing velocities:

```python
g = engine.compute_gravity_forces()
# Temporarily sets v=0, computes bias (which becomes pure gravity),
# then restores original velocities
```

### 10.4.4 Inverse Dynamics

```python
qacc = np.array([...])  # Desired accelerations
tau = engine.compute_inverse_dynamics(qacc)
# Uses: opensim.InverseDynamicsSolver(model).solve(state, udot)
```

### 10.4.5 Jacobian Computation

OpenSim computes Jacobians via finite differences, since the Simbody API does not expose
direct Jacobian computation as easily as Pinocchio:

```python
jac = engine.compute_jacobian("hand_r")
# Returns dict with 'linear', 'angular', 'spatial'
```

The implementation uses a perturbation $\epsilon = \sqrt{\epsilon_{mach}} \approx 1.49 \times 10^{-8}$
(optimal for first-order finite differences balancing truncation and round-off errors,
see Nocedal & Wright, _Numerical Optimization_, Chapter 8):

$$J_{ij} \approx \frac{p(q + \epsilon\,e_j) - p(q)}{\epsilon}$$

For the angular Jacobian, the rotation difference is computed via the Rodrigues formula:

$$\theta = \arccos\left(\frac{\text{tr}(R_0^T R_1) - 1}{2}\right), \quad \hat{n} = \frac{1}{2\sin\theta}\begin{pmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{pmatrix}$$

---

## 10.5 Drift-Control Decomposition (Section F)

### 10.5.1 Drift Acceleration

```python
a_drift = engine.compute_drift_acceleration()
# Drift = -M^{-1} * bias
# (bias includes Coriolis + gravity with zero muscle forces)
```

### 10.5.2 Control Acceleration

```python
a_control = engine.compute_control_acceleration(tau)
# Control = M^{-1} * tau
```

---

## 10.6 Counterfactual Experiments (Section G)

### 10.6.1 ZTCF -- Zero-Torque Counterfactual

```python
a_ztcf = engine.compute_ztcf(q, v)
```

Implementation:

1. Save current state and controls
2. Set desired state $(q, v)$
3. Set zero control (all muscle activations = 0)
4. Realize dynamics and extract accelerations from `state.getUDot()`
5. Restore original state and controls

### 10.6.2 ZVCF -- Zero-Velocity Counterfactual

```python
a_zvcf = engine.compute_zvcf(q)
```

Implementation:

1. Save current state
2. Set state with desired $q$ and $v = 0$
3. Controls are preserved
4. Realize dynamics and extract accelerations
5. Restore original state

---

## 10.7 Muscle Model Integration (Section J)

### 10.7.1 OpenSimMuscleAnalyzer

The `OpenSimMuscleAnalyzer` class provides comprehensive muscle analysis using OpenSim's
Hill-type muscle models.

```python
analyzer = engine.get_muscle_analyzer()
# Returns OpenSimMuscleAnalyzer(model, state) or None
```

**Muscle forces** (Hill-type model output):

$$F_m = a \cdot F_{max} \cdot f_L(l_m) \cdot f_V(v_m) + F_{PE}(l_m)$$

```python
forces = analyzer.get_muscle_forces()
# Returns: {"biceps_long": 245.3, "triceps": 120.1, ...}
# Units: Newtons [N]
```

**Activation levels**:

```python
activations = analyzer.get_activation_levels()
# Returns: {"biceps_long": 0.45, "triceps": 0.12, ...}
# Range: [0, 1]

# Set activations
analyzer.set_activation_levels({"biceps_long": 0.8, "triceps": 0.2})
```

**Moment arms** -- the perpendicular distance from the muscle line of action to the
joint center of rotation:

$$r_i = \frac{\partial L_m}{\partial q_i}$$

where $L_m$ is the musculotendon length and $q_i$ is the generalized coordinate.

```python
moment_arms = analyzer.get_moment_arms()
# Returns nested dict: {"biceps_long": {"elbow_flexion": 0.035, ...}, ...}
# Units: meters [m]

# For a specific coordinate only:
moment_arms_elbow = analyzer.get_moment_arms(coordinate_name="elbow_flexion")
```

**Muscle-generated joint torques**:

$$\tau_j^{(m)} = F_m \cdot r_{mj}$$

```python
muscle_torques = analyzer.compute_muscle_joint_torques()
# Returns: {"biceps_long": np.array([0, 0, 8.6, 0, ...]), ...}
# Units: N*m
```

**Muscle-induced accelerations**:

$$\ddot{q}^{(m)} = M(q)^{-1} \, \tau^{(m)}$$

```python
induced = analyzer.compute_muscle_induced_accelerations()
# Returns: {"biceps_long": np.array([...]), ...}
# Units: rad/s^2
```

Also available directly on the engine:

```python
induced = engine.compute_muscle_induced_accelerations()
```

### 10.7.2 Comprehensive Analysis

```python
analysis = engine.analyze_muscle_contributions()
# Returns MuscleAnalysis dataclass:
#   analysis.muscle_forces       -- dict[str, float]
#   analysis.moment_arms         -- dict[str, dict[str, float]]
#   analysis.activation_levels   -- dict[str, float]
#   analysis.muscle_lengths      -- dict[str, float]
#   analysis.total_muscle_torque -- np.ndarray
```

---

## 10.8 Grip Modeling (Section J1)

The `OpenSimGripModel` class models the hand-club interface using wrapping geometry:

```python
grip_model = engine.create_grip_model()
# Returns OpenSimGripModel(model) or None
```

### 10.8.1 Cylindrical Wrapping

Muscles route around cylindrical wrapping surfaces representing the grip:

```python
grip_model.add_cylindrical_wrap(
    muscle_name="flexor_digitorum",
    grip_body_name="club_grip",
    radius=0.015,    # Shaft radius + hand thickness [m]
    length=0.25,     # Grip length [m]
    location=(0.0, 0.0, 0.0),  # In grip body frame [m]
)
```

### 10.8.2 Grip Force Analysis

```python
grip_analysis = grip_model.analyze_grip_forces(state, analyzer)
# Returns dict:
#   "total_grip_force_N": 125.0,
#   "n_grip_muscles": 8,
#   "grip_muscles": ["flexor_digitorum", ...],
#   "within_physiological_range": True  # 50-200 N per hand
```

Validation range per hand: $50 \leq F_{grip} \leq 200$ N (constants
`MIN_PHYSIOLOGICAL_GRIP_N` and `MAX_PHYSIOLOGICAL_GRIP_N`).

---

## 10.9 Musculoskeletal Model Validation Workflows

### 10.9.1 Validation Against Clinical Data

The recommended validation workflow:

1. **Load a standard OpenSim model** (e.g., `gait2392.osim`, `arm26.osim`)
2. **Apply experimental motion data** (`.mot` files for coordinates, `.sto` for muscle data)
3. **Run inverse dynamics** to compute required joint torques
4. **Compare** joint torques with published electromyography (EMG) data
5. **Verify** muscle activation patterns match known biomechanical literature

```python
# Example validation pipeline
engine = OpenSimPhysicsEngine()
engine.load_from_path("opensim-models/Arm26/arm26.osim")

# Set a known configuration from experimental data
q_experimental = np.array([0.5, 0.8, ...])  # From motion capture
v_experimental = np.array([1.2, -0.5, ...])
engine.set_state(q_experimental, v_experimental)

# Compute inverse dynamics
a_measured = np.array([...])  # From marker-based acceleration
tau = engine.compute_inverse_dynamics(a_measured)

# Compare with EMG-predicted torques
analyzer = engine.get_muscle_analyzer()
muscle_torques = analyzer.compute_muscle_joint_torques()
total_muscle_tau = sum(muscle_torques.values())

# Check consistency
residual = np.linalg.norm(tau - total_muscle_tau)
print(f"Residual torque: {residual:.2f} N*m")
```

### 10.9.2 Force-Length-Velocity Validation

The Hill-type muscle model produces characteristic curves that should match published data:

**Force-Length**: The active force-length relationship is bell-shaped, centered at the
optimal fiber length $l_{opt}$:

$$f_L(l) = e^{-\left(\frac{l - l_{opt}}{\sigma}\right)^2}$$

**Force-Velocity**: The force-velocity relationship follows the Hill equation:

$$f_V(v) = \begin{cases} \frac{v_{max} - v}{v_{max} + K\,v} & v \leq 0 \text{ (shortening)} \\ N + (N-1)\frac{v_{max} + v}{7.56\,K\,v - v_{max}} & v > 0 \text{ (lengthening)} \end{cases}$$

### 10.9.3 OpenSim File Formats

| Extension | Description            | Usage                                          |
| --------- | ---------------------- | ---------------------------------------------- |
| `.osim`   | Model definition (XML) | Musculoskeletal geometry, muscles, joints      |
| `.mot`    | Motion file            | Joint coordinates over time                    |
| `.sto`    | Storage file           | General time-series data (forces, activations) |
| `.trc`    | Marker trajectory      | 3D marker positions from motion capture        |
| `.xml`    | Setup files            | Tool configuration (IK, ID, SO settings)       |

---

## 10.10 Complete Workflow Example

```python
import numpy as np
from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine,
)

# 1. Load model
engine = OpenSimPhysicsEngine()
engine.load_from_path("path/to/golf_swing_model.osim")
engine.reset()

# 2. Set controls (muscle excitations)
n_controls = engine._model.getNumControls()
excitations = np.full(n_controls, 0.1)  # Low baseline activation
engine.set_control(excitations)

# 3. Simulate
dt = 0.01
for i in range(200):  # 2 seconds
    engine.step(dt)
    q, v = engine.get_state()
    t = engine.get_time()

    # Every 0.5s, print state
    if i % 50 == 0:
        M = engine.compute_mass_matrix()
        print(f"t={t:.2f}s, |q|={np.linalg.norm(q):.3f}, cond(M)={np.linalg.cond(M):.1f}")

# 4. Muscle analysis at final configuration
analysis = engine.analyze_muscle_contributions()
if analysis:
    print(f"Number of muscles analyzed: {len(analysis.muscle_forces)}")
    print(f"Total muscle torque norm: {np.linalg.norm(analysis.total_muscle_torque):.2f} N*m")

    # Top 5 muscles by force
    sorted_muscles = sorted(analysis.muscle_forces.items(), key=lambda x: x[1], reverse=True)
    for name, force in sorted_muscles[:5]:
        print(f"  {name}: {force:.1f} N (activation: {analysis.activation_levels[name]:.2f})")

# 5. Counterfactual analysis
a_ztcf = engine.compute_ztcf(q, v)
a_zvcf = engine.compute_zvcf(q)
print(f"ZTCF acceleration norm: {np.linalg.norm(a_ztcf):.3f} rad/s^2")
print(f"ZVCF acceleration norm: {np.linalg.norm(a_zvcf):.3f} rad/s^2")
```

---

# Chapter 11: MyoSuite Musculoskeletal Engine

## 11.1 Overview

The MyoSuite engine wraps the MyoSuite framework (Caggiano et al., 2022) to provide
MuJoCo-based musculoskeletal simulation within the UpstreamDrift `PhysicsEngine` protocol.
MyoSuite combines anatomically detailed MuJoCo muscle models with OpenAI Gym-compatible
reinforcement learning interfaces, making it ideal for:

1. **Muscle-driven simulation** with realistic Hill-type muscle dynamics
2. **Neural control policy training** via reinforcement learning (RL)
3. **Cross-validation** against OpenSim for biomechanical fidelity

### Source files

| File                                                                     | Purpose                                    |
| ------------------------------------------------------------------------ | ------------------------------------------ |
| `src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py` | `PhysicsEngine` protocol wrapper           |
| `src/engines/physics_engines/myosuite/python/muscle_analysis.py`         | Muscle analysis, grip modeling             |
| `src/shared/python/myosuite_adapter.py`                                  | RL environment adapter (`MuscleDrivenEnv`) |

### Dependencies

```
pip install myosuite     # MyoSuite (includes MuJoCo)
pip install gym          # OpenAI Gym
pip install mujoco       # MuJoCo Python bindings
pip install stable-baselines3  # Optional: for RL policy training
```

---

## 11.2 Model Configurations

MyoSuite provides pre-built musculoskeletal models of varying complexity. The models are
MuJoCo XML files with Hill-type muscle actuators:

### 11.2.1 MyoUpperBody

| Property               | Value                             |
| ---------------------- | --------------------------------- |
| Degrees of Freedom     | 19 DOF                            |
| Actuators              | 20 muscle actuators               |
| Coverage               | Bilateral arms + torso            |
| Typical Environment ID | `myoArmPose-v0`, `myoArmReach-v0` |

Suitable for: upper body swing mechanics, arm coordination analysis, shoulder-elbow coupling.

### 11.2.2 MyoBody Full

| Property           | Value                           |
| ------------------ | ------------------------------- |
| Degrees of Freedom | 52 DOF                          |
| Muscles            | 290 muscles                     |
| Coverage           | Complete body (head to feet)    |
| Typical Usage      | Full-body golf swing simulation |

This is the most anatomically complete model. Due to the high muscle count, simulation
is computationally intensive. Recommended for validation studies rather than real-time
applications.

### 11.2.3 MyoArm Simple

| Property               | Value                       |
| ---------------------- | --------------------------- |
| Degrees of Freedom     | 14 DOF                      |
| Actuators              | 14 actuators                |
| Coverage               | Simplified bilateral arms   |
| Typical Environment ID | `myoElbowPose1D6MRandom-v0` |

Suitable for: rapid prototyping, RL training, elbow/wrist analysis, grip studies.

### 11.2.4 Available MyoSuite Environments

| Category | Environment ID                 | Muscles    | Description        |
| -------- | ------------------------------ | ---------- | ------------------ |
| Elbow    | `myoElbowPose1D6MRandom-v0`    | 6          | Elbow pose control |
| Elbow    | `myoElbowPose1D6MExoRandom-v0` | 6          | With exoskeleton   |
| Hand     | `myoHandPose100Random-v0`      | 39         | Hand pose control  |
| Hand     | `myoHandReach-v0`              | 39         | Reaching task      |
| Hand     | `myoHandKey-v0`                | 39         | Key turning        |
| Arm      | `myoArmPose-v0`                | Multiple   | Arm pose control   |
| Leg      | `myoLegWalk-v0`                | Lower body | Walking            |
| Leg      | `myoLegRun-v0`                 | Lower body | Running            |

---

## 11.3 Engine Initialization

### 11.3.1 Loading Environments

Unlike other engines that load model files, MyoSuite loads Gym environment IDs:

```python
from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine,
)

engine = MyoSuitePhysicsEngine()
engine.load_from_path("myoElbowPose1D6MRandom-v0")
# Internally: gym.make(env_id) + env.reset()

print(f"Model: {engine.model_name}")        # "myoElbowPose1D6MRandom-v0"
print(f"Timestep: {engine._dt:.4f}s")       # From sim.model.opt.timestep
print(f"Initialized: {engine.is_initialized}")  # True
```

The wrapper accesses the underlying MuJoCo simulation:

```python
# Direct MuJoCo model access
mj_model = engine.model   # MuJoCo MjModel object (or None)
mj_data = engine.sim.data  # MuJoCo MjData object
```

### 11.3.2 State Management

```python
# Get state (qpos, qvel from underlying MuJoCo sim)
q, v = engine.get_state()

# Set state
engine.set_state(q_desired, v_desired)
# Calls sim.forward() to update derived quantities

# Set control (muscle activations via ctrl buffer)
u = np.array([0.5, 0.3, 0.1, ...])
engine.set_control(u)
```

### 11.3.3 Simulation Stepping

```python
engine.step(dt=0.002)
# If dt is provided, temporarily overrides sim.model.opt.timestep
# Uses sim.step() directly (bypasses Gym wrapper for low-level control)
```

---

## 11.4 Hill-Type Muscle Model

MyoSuite implements the Hill-type muscle model via MuJoCo's built-in muscle actuator dynamics.
The total muscle force is:

$$F_m = a \cdot F_{max} \cdot f_L(l_m) \cdot f_V(v_m) + F_{PE}(l_m)$$

where:

- $a \in [0, 1]$ is the muscle activation level
- $F_{max}$ is the maximum isometric force [N]
- $f_L(l_m)$ is the normalized force-length relationship
- $f_V(v_m)$ is the normalized force-velocity relationship
- $F_{PE}(l_m)$ is the passive elastic element force

### 11.4.1 Force-Length Relationship

The active force-length curve is a Gaussian centered at the optimal fiber length $l_{opt}$:

$$f_L(l) = e^{-\left(\frac{l - l_{opt}}{\sigma}\right)^2}$$

where $\sigma$ controls the width of the bell curve. At $l = l_{opt}$, the muscle can
produce maximum force ($f_L = 1$). As the muscle shortens or lengthens beyond this optimum,
force capacity decreases.

### 11.4.2 Force-Velocity Relationship

The force-velocity relationship distinguishes shortening (concentric) from lengthening
(eccentric) contractions:

$$f_V(v) = \begin{cases} \dfrac{v_{max} - v}{v_{max} + K\,v} & v \leq 0 \text{ (shortening)} \\[10pt] N + (N-1)\dfrac{v_{max} + v}{7.56\,K\,v - v_{max}} & v > 0 \text{ (lengthening)} \end{cases}$$

where:

- $v_{max}$ is the maximum shortening velocity (typically $10 \cdot l_{opt}$ per second)
- $K$ is the curvature parameter (typically 0.25)
- $N$ is the eccentric force enhancement factor (typically 1.5)
- $v < 0$ denotes shortening; $v > 0$ denotes lengthening

Key properties:

- At $v = 0$ (isometric): $f_V = 1$
- At $v = -v_{max}$ (maximum shortening): $f_V = 0$ (zero force)
- During eccentric contraction: $f_V > 1$ (force exceeds isometric maximum)

### 11.4.3 Parallel Elastic Element

The passive elastic element produces force only when the muscle is stretched beyond its
slack length $l_{slack}$:

$$F_{PE}(l) = k_{PE} \cdot \max(0,\; l - l_{slack})^2$$

where $k_{PE}$ is the passive element stiffness. This models the connective tissue
(tendons, fascia) that resists excessive stretch.

### 11.4.4 Activation Dynamics

Muscle activation follows first-order dynamics governed by the neural excitation signal $u(t)$:

$$\dot{a}(t) = \frac{u(t) - a(t)}{\tau(u, a)}$$

where the time constant $\tau$ depends on the direction of change:

$$\tau = \begin{cases} \tau_{act} & \text{if } u > a \text{ (activation, fast: 10-50 ms)} \\ \tau_{deact} & \text{if } u \leq a \text{ (deactivation, slow: 40-110 ms)} \end{cases}$$

The asymmetry between activation and deactivation time constants reflects the physiology:
calcium release (activation) is faster than calcium reuptake (deactivation).

**Usage via the RL adapter**:

```python
from src.shared.python.myosuite_adapter import MuscleDrivenEnv

env = MuscleDrivenEnv(muscle_system, task="tracking", dt=0.001)
obs = env.reset()

# Neural excitation -> activation dynamics -> muscle force
action = np.array([0.8, 0.2, 0.5])  # Excitations [0,1]
obs, reward, done, info = env.step(action)
# info["activations"] shows the filtered activation levels
```

---

## 11.5 Muscle Analysis Module

The `MyoSuiteMuscleAnalyzer` class provides detailed muscle analysis via the underlying
MuJoCo simulation.

### 11.5.1 Initialization

```python
analyzer = engine.get_muscle_analyzer()
# Returns MyoSuiteMuscleAnalyzer(sim) or None

print(f"Muscles found: {len(analyzer.muscle_names)}")
print(f"Muscle names: {analyzer.muscle_names}")
```

Muscle identification: The analyzer identifies muscle actuators by checking
`model.actuator_dyntype[i] == 2` (MuJoCo's `mjDYN_MUSCLE` constant). If the dyntype
attribute is unavailable, all actuators are treated as muscles (appropriate for MyoSuite
models).

### 11.5.2 Muscle State Queries

```python
# Current activation levels [0, 1]
activations = analyzer.get_muscle_activations()
# Sources: data.act[] if available, falls back to data.ctrl[]

# Current muscle forces [N]
forces = analyzer.get_muscle_forces()
# Sources: data.actuator_force[]

# Current muscle lengths [m]
lengths = analyzer.get_muscle_lengths()
# Sources: data.actuator_length[]

# Current contraction velocities [m/s] (negative = shortening)
velocities = analyzer.get_muscle_velocities()
# Sources: data.actuator_velocity[]
```

All queries return arrays indexed by muscle order in `analyzer.muscle_names`.

### 11.5.3 Setting Muscle Activations by Name

```python
engine.set_muscle_activations({
    "BIClong": 0.7,
    "BICshort": 0.5,
    "TRIlong": 0.1,
    "TRIlat": 0.1,
})
# Internally maps names to actuator indices, clips to [0, 1]
```

### 11.5.4 Complete Muscle State Snapshot

```python
state = engine.get_muscle_state()
# Returns MyoSuiteMuscleState dataclass:
#   state.muscle_names   -- list[str]
#   state.activations    -- np.ndarray, shape (n_muscles,)
#   state.forces         -- np.ndarray, shape (n_muscles,)
#   state.lengths        -- np.ndarray, shape (n_muscles,)
#   state.velocities     -- np.ndarray, shape (n_muscles,)
```

---

## 11.6 Moment Arm Analysis

Moment arms $r$ quantify the mechanical advantage of each muscle about each joint.
The MyoSuite analyzer computes moment arms via finite differences:

$$r_{mj} = -\frac{\partial L_m}{\partial q_j} \approx -\frac{L_m(q + \delta e_j) - L_m(q)}{\delta}$$

where $\delta = 10^{-6}$ rad (or m) is the perturbation size.

The negative sign arises because muscles typically shorten (decrease $L_m$) when the joint
flexes (increases $q_j$).

```python
moment_arms = analyzer.compute_moment_arms()
# Returns: {"BIClong": np.array([0.0, 0.035, 0.0, ...]), ...}
# Units: meters [m]
# Shape of each array: (nv,) -- one value per DOF
```

---

## 11.7 Muscle-Generated Joint Torques

Each muscle generates a torque about each joint equal to the product of its force and
moment arm:

$$\tau_j^{(m)} = F_m \cdot r_{mj}$$

```python
muscle_torques = analyzer.compute_muscle_joint_torques()
# Returns: {"BIClong": np.array([0, 8.6, 0, ...]), ...}

# Total muscle torque across all muscles
total_tau = np.zeros(engine.sim.model.nv)
for tau in muscle_torques.values():
    total_tau += tau
```

---

## 11.8 Muscle-Induced Accelerations

Each muscle's contribution to joint acceleration is:

$$\ddot{q}^{(m)} = M(q)^{-1} \, \tau^{(m)}$$

```python
induced = analyzer.compute_muscle_induced_accelerations()
# Returns: {"BIClong": np.array([...]), ...}
# Units: rad/s^2

# Verify: sum should approximate total acceleration from muscles
total_induced = sum(induced.values())
```

Also accessible directly via the engine:

```python
induced = engine.compute_muscle_induced_accelerations()
```

---

## 11.9 Muscle Co-Contraction Analysis

Co-contraction occurs when antagonist muscles are simultaneously active, increasing joint
stiffness at the expense of metabolic efficiency. The co-contraction index for an
agonist-antagonist pair is:

$$CI = \frac{2 \cdot \min(a_{agonist}, a_{antagonist})}{a_{agonist} + a_{antagonist}}$$

where $CI = 0$ means pure agonist activation and $CI = 1$ means equal co-contraction.

```python
activations = analyzer.get_muscle_activations()
muscle_names = analyzer.muscle_names

# Example: biceps vs triceps co-contraction
bic_idx = muscle_names.index("BIClong")
tri_idx = muscle_names.index("TRIlong")

a_bic = activations[bic_idx]
a_tri = activations[tri_idx]

if (a_bic + a_tri) > 0:
    ci = 2.0 * min(a_bic, a_tri) / (a_bic + a_tri)
else:
    ci = 0.0

print(f"Co-contraction index: {ci:.3f}")
```

---

## 11.10 Power Analysis

Joint power quantifies the rate of energy generation or absorption at each joint:

$$P = \tau \cdot \dot{q}$$

where $P > 0$ indicates power generation (concentric) and $P < 0$ indicates power
absorption (eccentric).

Total work over an interval $[0, T]$:

$$W = \int_0^T P \, dt = \int_0^T \tau(t) \cdot \dot{q}(t) \, dt$$

The metabolic activation cost accounts for the energy spent maintaining muscle activation
beyond mechanical work:

$$P_{metabolic} = F \cdot v + \alpha \cdot a^2 \cdot |F \cdot v|$$

where $\alpha = 0.25$ is the metabolic cost constant.

```python
activation_power = analyzer.compute_activation_power()
# Returns: {"BIClong": -1.23, "TRIlong": 0.45, ...}
# Units: Watts [W]
# Includes mechanical power + metabolic activation cost
```

---

## 11.11 Muscle Recruitment Optimization

The standard muscle recruitment problem minimizes the sum of squared activations subject
to producing the required joint torques:

$$\min_{a} \sum_{i=1}^{n_m} a_i^2$$
$$\text{subject to:} \quad \tau = R \cdot F(a)$$

where $R$ is the moment arm matrix and $F(a)$ is the vector of muscle forces as a function
of activations. Additional constraints include:

- $0 \leq a_i \leq 1$ (activation bounds)
- $F_i = a_i \cdot F_{max,i} \cdot f_L(l_i) \cdot f_V(v_i)$ (Hill model)

This is a nonlinear optimization problem because $F$ depends nonlinearly on $a$ through
the force-length and force-velocity relationships. In practice, for a given configuration
and velocity, $f_L$ and $f_V$ are constants, and the problem reduces to a quadratic program.

---

## 11.12 Comprehensive Analysis

```python
analysis = analyzer.analyze_all()
# Returns MyoSuiteMuscleAnalysis dataclass:
#   analysis.muscle_state          -- MyoSuiteMuscleState
#   analysis.moment_arms           -- dict[str, np.ndarray]
#   analysis.joint_torques         -- dict[str, np.ndarray]
#   analysis.induced_accelerations -- dict[str, np.ndarray]
#   analysis.total_muscle_torque   -- np.ndarray
#   analysis.activation_power      -- dict[str, float]
```

---

## 11.13 Grip Modeling (Section K1)

The `MyoSuiteGripModel` provides activation-driven grip force analysis:

```python
grip_model = engine.create_grip_model()
# Returns MyoSuiteGripModel(sim, analyzer) or None
```

### 11.13.1 Grip Muscle Identification

```python
grip_muscles = grip_model.get_grip_muscles()
# Filters by keywords: flexor, extensor, grip, hand, finger, thumb, etc.
```

### 11.13.2 Grip Force Computation

```python
total_grip = grip_model.compute_total_grip_force()
# Sum of forces from grip-related muscles [N]
```

### 11.13.3 Comprehensive Grip Analysis

```python
grip_info = grip_model.analyze_grip()
# Returns dict:
#   "total_grip_force_N": 350.0,
#   "mean_grip_activation": 0.65,
#   "n_grip_muscles": 12,
#   "grip_muscles": ["flexor_digitorum", ...],
#   "within_mvc_range": True,     # 200-800 N
#   "individual_forces": {...},
#   "individual_activations": {...}
```

Validation range (MVC): $200 \leq F_{grip} \leq 800$ N (constants `MIN_MVC_GRIP_N`
and `MAX_MVC_GRIP_N`).

---

## 11.14 Drift-Control Decomposition and Counterfactuals

### 11.14.1 Drift Acceleration

```python
a_drift = engine.compute_drift_acceleration()
# Sets all muscle activations to zero, computes forward dynamics,
# extracts qacc, then restores original activations
```

### 11.14.2 Control Acceleration

```python
a_control = engine.compute_control_acceleration(tau)
# M^{-1} * tau using MuJoCo mass matrix
```

### 11.14.3 ZTCF

```python
a_ztcf = engine.compute_ztcf(q, v)
# Saves state/ctrl, sets state, zeros ctrl, sim.forward(), reads qacc, restores
```

### 11.14.4 ZVCF

```python
a_zvcf = engine.compute_zvcf(q)
# Saves state, sets q with v=0, sim.forward(), reads qacc, restores
```

---

## 11.15 RL Adapter: MuscleDrivenEnv

The `MuscleDrivenEnv` class bridges the UpstreamDrift muscle models with standard RL
frameworks (Stable-Baselines3, RLlib):

```python
from src.shared.python.myosuite_adapter import MuscleDrivenEnv, train_muscle_policy

# Create environment
env = MuscleDrivenEnv(muscle_system, task="tracking", dt=0.001)

# Observation space: [q, v, a_1, ..., a_n]
obs = env.reset()

# Action space: neural excitations [0, 1] per muscle
action = np.array([0.5, 0.3, 0.1])
obs, reward, done, info = env.step(action)

# Train a policy
policy = train_muscle_policy(env, total_timesteps=100000)
# Uses SAC (Soft Actor-Critic) from stable-baselines3
```

### Reward Functions

| Task       | Reward | Description         |
| ---------- | ------ | ------------------- | --------------- | ------------------------- |
| `tracking` | $-     | q - q\_{target}     | $               | Track time-varying target |
| `reach`    | $-     | q - q\_{target}     | - 0.01 \cdot t$ | Reach target quickly      |
| `swing`    | Custom | Golf swing specific |

---

## 11.16 Cross-Validation with OpenSim

```python
from src.shared.python.cross_engine_validator import CrossEngineValidator

myosuite_engine = MyoSuitePhysicsEngine()
myosuite_engine.load_from_path("myoElbowPose1D6MRandom-v0")

opensim_engine = OpenSimPhysicsEngine()
opensim_engine.load_from_path("arm26.osim")

validator = CrossEngineValidator()
result = validator.compare_states(
    "MyoSuite", myosuite_tau,
    "OpenSim", opensim_tau,
    metric="torque"
)
print(f"Passed: {result.passed}, Severity: {result.severity}")
```

---

## 11.17 Complete Workflow Example

```python
import numpy as np
from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine,
)

# 1. Initialize
engine = MyoSuitePhysicsEngine()
engine.load_from_path("myoElbowPose1D6MRandom-v0")
engine.reset()

# 2. Get muscle information
muscle_names = engine.get_muscle_names()
print(f"Muscles: {muscle_names}")

# 3. Simulate with varying activations
trajectory_q = []
trajectory_forces = []

for i in range(1000):
    # Sinusoidal biceps activation
    activations = {
        "BIClong": 0.5 * (1 + np.sin(i * 0.01)),
        "BICshort": 0.4 * (1 + np.sin(i * 0.01)),
        "TRIlong": 0.1,
        "TRIlat": 0.1,
    }
    engine.set_muscle_activations(activations)
    engine.step(dt=0.002)

    q, v = engine.get_state()
    trajectory_q.append(q.copy())

    # Periodic muscle analysis
    if i % 100 == 0:
        state = engine.get_muscle_state()
        if state:
            trajectory_forces.append(state.forces.copy())
            print(f"t={engine.get_time():.3f}s, "
                  f"elbow={np.degrees(q[0]):.1f} deg, "
                  f"peak_force={np.max(state.forces):.1f} N")

# 4. Full analysis at final state
analysis = engine.analyze_muscle_contributions()
if analysis:
    print(f"\nFinal analysis:")
    print(f"  Total muscle torque: {np.linalg.norm(analysis.total_muscle_torque):.2f} N*m")

    for name, power in analysis.activation_power.items():
        print(f"  {name}: power={power:.2f} W")

# 5. Drift-control decomposition
a_drift = engine.compute_drift_acceleration()
print(f"\nDrift acceleration norm: {np.linalg.norm(a_drift):.3f} rad/s^2")
```

---

# Chapter 12: Biomechanical Analysis Suite

## 12.1 Overview

The Biomechanical Analysis Suite is a collection of shared analysis modules that operate
across all physics engines. These modules are implemented as Python mixins (classes designed
to be combined via multiple inheritance) and are located in
`src/shared/python/analysis/`. They compute golf-specific biomechanical metrics from
simulation data regardless of the underlying physics engine.

### Source files

| File                                              | Purpose                        |
| ------------------------------------------------- | ------------------------------ |
| `src/shared/python/analysis/swing_metrics.py`     | Golf swing kinematics/kinetics |
| `src/shared/python/analysis/energy_metrics.py`    | Energy analysis                |
| `src/shared/python/analysis/grf_metrics.py`       | Ground reaction forces         |
| `src/shared/python/analysis/angular_momentum.py`  | Angular momentum tracking      |
| `src/shared/python/analysis/phase_detection.py`   | Golf swing phase detection     |
| `src/shared/python/analysis/stability_metrics.py` | Balance and stability          |
| `src/shared/python/analysis/dataclasses.py`       | Shared data structures         |

### Architecture

All analysis modules are implemented as mixins that expect certain attributes on the host
class:

```python
class SwingAnalyzer(
    SwingMetricsMixin,
    EnergyMetricsMixin,
    GRFMetricsMixin,
    AngularMomentumMetricsMixin,
    PhaseDetectionMixin,
    StabilityMetricsMixin,
    BasicStatsMixin,
):
    """Complete analysis class combining all mixins."""

    def __init__(self, times, joint_positions, ...):
        self.times = times
        self.joint_positions = joint_positions
        self.club_head_speed = club_head_speed
        self.dt = dt
        # ... other attributes
```

---

## 12.2 Swing Metrics (`swing_metrics.py`)

The `SwingMetricsMixin` computes golf-specific kinematics including club head speed,
X-factor, tempo, and range of motion.

### 12.2.1 Club Head Speed

The club head speed is the magnitude of the linear velocity at the club head frame:

$$v_{club} = \|\dot{p}_{head}\| = \|J_{linear}(q) \cdot \dot{q}\|$$

where $\dot{p}_{head}$ is the 3D velocity of the club head and $J_{linear}$ is the
linear component of the Jacobian at the club head frame.

Peak club head speed typically occurs at or just before impact:

```python
peak = analyzer.find_club_head_speed_peak()
# Returns PeakInfo dataclass:
#   peak.value       -- peak speed [m/s]
#   peak.time        -- time of peak [s]
#   peak.index       -- array index
#   peak.prominence  -- peak prominence [m/s]
#   peak.width       -- peak width [s]
```

### 12.2.2 X-Factor

The X-Factor measures the angular separation between the shoulder plane and hip plane
during the swing:

$$X\text{-Factor}(t) = \theta_{shoulder}(t) - \theta_{hip}(t)$$

where $\theta_{shoulder}$ and $\theta_{hip}$ are the axial rotation angles (in degrees)
of the shoulder girdle and pelvis, respectively.

**API**:

```python
def compute_x_factor(
    self,
    shoulder_joint_idx: int,
    hip_joint_idx: int,
) -> np.ndarray | None:
    """Compute X-Factor (shoulder-hip rotation difference).

    Args:
        shoulder_joint_idx: Index of shoulder/torso rotation joint
        hip_joint_idx: Index of hip rotation joint

    Returns:
        X-Factor time series (degrees) or None if indices invalid
    """
```

**Usage**:

```python
x_factor = analyzer.compute_x_factor(
    shoulder_joint_idx=3,  # Torso rotation
    hip_joint_idx=0,       # Pelvis rotation
)

if x_factor is not None:
    max_x_factor = np.max(x_factor)
    print(f"Peak X-Factor: {max_x_factor:.1f} degrees")
    # Professional golfers: 45-60 degrees at top of backswing
```

### 12.2.3 X-Factor Stretch

The X-Factor velocity (stretch rate) quantifies how rapidly the X-factor increases during
the transition from backswing to downswing:

$$\dot{X}(t) = \frac{d}{dt}\bigl[\theta_{shoulder}(t) - \theta_{hip}(t)\bigr]$$

```python
def compute_x_factor_stretch(
    self,
    shoulder_joint_idx: int,
    hip_joint_idx: int,
) -> tuple[np.ndarray, float] | None:
    """Compute X-Factor velocity (stretch rate) and peak stretch.

    Returns:
        Tuple of (x_factor_velocity_array, peak_stretch_rate) or None
    """
```

**Usage**:

```python
result = analyzer.compute_x_factor_stretch(
    shoulder_joint_idx=3,
    hip_joint_idx=0,
)
if result is not None:
    x_vel, peak_stretch = result
    print(f"Peak X-Factor stretch rate: {peak_stretch:.1f} deg/s")
```

### 12.2.4 Range of Motion

```python
def compute_range_of_motion(self, joint_idx: int) -> tuple[float, float, float]:
    """Compute range of motion for a joint.

    Args:
        joint_idx: Joint index

    Returns:
        (min_angle, max_angle, rom) in degrees
    """
```

**Usage**:

```python
min_ang, max_ang, rom = analyzer.compute_range_of_motion(joint_idx=2)
print(f"Shoulder ROM: {rom:.1f} degrees ({min_ang:.1f} to {max_ang:.1f})")
```

### 12.2.5 Swing Tempo

Tempo is the ratio of backswing to downswing duration:

$$\text{Tempo} = \frac{t_{transition} - t_{address}}{t_{impact} - t_{transition}}$$

Professional golfers typically have a 3:1 tempo ratio.

```python
def compute_tempo(self) -> tuple[float, float, float] | None:
    """Compute swing tempo (backswing:downswing ratio).

    Returns:
        (backswing_duration, downswing_duration, ratio) or None
    """
```

The transition point (top of backswing) is detected as the minimum club head speed
in the first 70% of the pre-impact interval, after Savitzky-Golay smoothing.

**Usage**:

```python
tempo = analyzer.compute_tempo()
if tempo is not None:
    bs_dur, ds_dur, ratio = tempo
    print(f"Backswing: {bs_dur:.3f}s, Downswing: {ds_dur:.3f}s, Ratio: {ratio:.2f}")
```

### 12.2.6 Smash Factor

The smash factor relates ball speed to club head speed, measuring the efficiency of
energy transfer at impact:

$$SF = \frac{v_{ball}}{v_{club}}$$

Typical values: 1.44-1.50 for drivers, decreasing with shorter clubs. A higher smash
factor indicates more efficient energy transfer.

### 12.2.7 Impact Detection

```python
def detect_impact_time(self) -> float | None:
    """Detect ball impact time.

    Uses peak club head speed as proxy for impact.

    Returns:
        Impact time in seconds, or None
    """
```

### 12.2.8 Kinematic Sequence

The kinematic sequence describes the order in which body segments reach peak angular
velocity during the downswing. The optimal sequence is proximal-to-distal:

1. **Hips** -- peak first (initiate downswing)
2. **Torso** -- peak second (transfer energy from lower body)
3. **Arms** -- peak third (amplify through lever system)
4. **Club** -- peak last (maximum speed at impact)

This is stored in the `KinematicSequenceInfo` dataclass:

```python
@dataclass
class KinematicSequenceInfo:
    segment_name: str      # e.g., "hips", "torso", "arms", "club"
    peak_velocity: float   # Peak angular velocity [deg/s]
    peak_time: float       # Time of peak [s]
    peak_index: int        # Array index of peak
    order_index: int       # Actual order (0=first to peak)
```

### 12.2.9 Attack Angle, Dynamic Loft, and Face Angle

These impact parameters characterize the club-ball interaction:

- **Attack angle**: The vertical angle of the club head path at impact.
  Negative = descending blow (irons), positive = ascending (driver).
- **Dynamic loft**: The effective loft of the club face at impact, accounting for
  shaft lean and face rotation.
- **Face angle**: The horizontal orientation of the club face relative to the target
  line at impact. Open = face pointing right of target (for right-handed golfer),
  closed = left.

---

## 12.3 Energy Metrics (`energy_metrics.py`)

The `EnergyMetricsMixin` computes energy-related quantities for the multibody system.

### 12.3.1 Kinetic Energy

The total kinetic energy of the system in generalized coordinates is:

$$T = \frac{1}{2}\,\dot{q}^T\,M(q)\,\dot{q}$$

where $M(q)$ is the joint-space inertia matrix and $\dot{q}$ is the generalized velocity
vector.

### 12.3.2 Potential Energy

The gravitational potential energy is:

$$V = -\sum_{i=1}^{n} m_i\,g^T\,p_i$$

where $m_i$ is the mass of body $i$, $g$ is the gravity vector, and $p_i$ is the position
of the center of mass of body $i$.

### 12.3.3 Total Mechanical Energy

$$E = T + V$$

For a conservative system (no friction, no external forces), $E$ should be constant.
Energy drift indicates either non-conservative effects or numerical integration errors.

### 12.3.4 Energy Transfer Efficiency

The energy transfer efficiency measures how much of the total system energy is
concentrated in the club at impact:

$$\eta = \frac{E_{club}}{E_{total}} \times 100\%$$

**API**:

```python
def compute_energy_metrics(
    self,
    kinetic_energy: np.ndarray,
    potential_energy: np.ndarray,
) -> dict[str, Any]:
    """Compute energy-related metrics.

    Args:
        kinetic_energy: Kinetic energy time series [J]
        potential_energy: Potential energy time series [J]

    Returns:
        Dictionary with keys:
          - "max_kinetic_energy": float [J]
          - "max_potential_energy": float [J]
          - "max_total_energy": float [J]
          - "energy_efficiency": float [%]
          - "energy_variation": float [J] (std of total energy)
          - "energy_drift": float [J] (E_final - E_initial)
    """
```

**Usage**:

```python
# Compute energy time series from simulation data
T_series = np.array([0.5 * v.T @ M @ v for M, v in zip(mass_matrices, velocities)])
V_series = np.array([compute_PE(q) for q in configurations])

metrics = analyzer.compute_energy_metrics(T_series, V_series)
print(f"Peak kinetic energy: {metrics['max_kinetic_energy']:.1f} J")
print(f"Energy efficiency at impact: {metrics['energy_efficiency']:.1f}%")
print(f"Energy drift: {metrics['energy_drift']:.3f} J")
```

### 12.3.5 Segmental Energy Flow

The proximal-to-distal energy flow pattern is a hallmark of efficient golf swings. Energy
generated by large proximal muscles (hips, torso) transfers through the kinematic chain
to the distal segments (arms, club). This is quantified by tracking the kinetic energy
of each segment over time and observing the sequential peaking pattern.

---

## 12.4 Ground Reaction Force Metrics (`grf_metrics.py`)

The `GRFMetricsMixin` computes ground reaction force (GRF) and centre of pressure (CoP)
metrics essential for golf swing balance analysis.

### 12.4.1 Vertical GRF Analysis

The vertical ground reaction force $F_z(t)$ characterizes the golfer's interaction with
the ground. Key metrics:

- **Peak vertical force**: Maximum $F_z$ during the swing (typically 1.0-1.5 body weight
  during the downswing)
- **Loading rate**: $\dot{F}_z = dF_z/dt$, measuring how rapidly force builds

### 12.4.2 Shear Forces

The horizontal (shear) GRF components generate the rotational moments needed for
the swing:

$$F_{shear} = \sqrt{F_x^2 + F_y^2}$$

### 12.4.3 Centre of Pressure

The centre of pressure (CoP) is the point on the ground surface where the resultant
GRF acts:

$$\text{CoP} = \frac{\sum F_i \times p_i}{\sum F_i}$$

For a two-foot stance, the CoP moves between the feet during the swing, reflecting
weight transfer patterns.

**API**:

```python
def compute_grf_metrics(self) -> GRFMetrics | None:
    """Compute Ground Reaction Force and Center of Pressure metrics.

    Expects attributes:
        self.cop_position:  (N, 2) or (N, 3) array of CoP positions
        self.ground_forces: (N, 3) array of GRF vectors [N]
        self.dt:            time step [s]

    Returns:
        GRFMetrics dataclass or None if data unavailable
    """
```

The returned `GRFMetrics` dataclass contains:

```python
@dataclass
class GRFMetrics:
    cop_path_length: float       # Total CoP excursion [m]
    cop_max_velocity: float      # Maximum CoP velocity [m/s]
    cop_x_range: float           # Lateral CoP range [m]
    cop_y_range: float           # Anterior-posterior CoP range [m]
    peak_vertical_force: float | None   # Peak Fz [N]
    peak_shear_force: float | None      # Peak shear [N]
```

**Usage**:

```python
grf = analyzer.compute_grf_metrics()
if grf is not None:
    print(f"CoP path length: {grf.cop_path_length:.3f} m")
    print(f"CoP lateral range: {grf.cop_x_range:.3f} m")
    print(f"Peak vertical GRF: {grf.peak_vertical_force:.1f} N")
    print(f"Peak shear GRF: {grf.peak_shear_force:.1f} N")
```

### 12.4.4 Lateral Weight Shift

The CoP x-range (`cop_x_range`) directly quantifies the lateral weight shift during the
swing. A typical golf swing shows:

1. **Address**: CoP centered between feet
2. **Backswing**: CoP shifts toward trail foot
3. **Downswing**: CoP rapidly shifts toward lead foot
4. **Follow-through**: CoP settles over lead foot

### 12.4.5 Free Moment Analysis

The free moment (vertical torque) about the CoP is computed from the ground contact forces.
It represents the rotational resistance the ground provides to the golfer's axial rotation.

### 12.4.6 Implementation Notes

The CoP path length computation uses optimized array operations:

```python
# Manual slicing (faster than np.diff for this use case)
cop_diff = cop_position[1:] - cop_position[:-1]

# np.hypot is faster than np.linalg.norm for 2D vectors
if cop_diff.shape[1] == 2:
    cop_dist = np.hypot(cop_diff[:, 0], cop_diff[:, 1])
else:
    cop_dist = np.sqrt(np.sum(cop_diff**2, axis=1))

path_length = np.sum(cop_dist)
```

---

## 12.5 Angular Momentum (`angular_momentum.py`)

The `AngularMomentumMetricsMixin` tracks the angular momentum of the system and its
individual segments.

### 12.5.1 Segmental Angular Momentum

The angular momentum of segment $i$ about the system center of mass is:

$$\mathbf{H}_i = I_i\,\omega_i + m_i\,(r_i \times v_i)$$

where:

- $I_i$ is the inertia tensor of segment $i$ about its own center of mass
- $\omega_i$ is the angular velocity of segment $i$
- $m_i$ is the mass of segment $i$
- $r_i$ is the position vector from the system CoM to the segment CoM
- $v_i$ is the velocity of the segment CoM relative to the system CoM

The first term $I_i\,\omega_i$ is the _local_ (spin) angular momentum, and the second
term $m_i\,(r_i \times v_i)$ is the _remote_ (orbital) angular momentum.

### 12.5.2 Total Angular Momentum

The total angular momentum is the sum over all segments:

$$\mathbf{H}_{total} = \sum_{i=1}^{n} \mathbf{H}_i = \sum_{i=1}^{n} \bigl[I_i\,\omega_i + m_i\,(r_i \times v_i)\bigr]$$

### 12.5.3 Conservation Analysis

For a system with no external torques, angular momentum should be conserved:
$\dot{\mathbf{H}}_{total} = \mathbf{0}$. In a golf swing, GRF provides an external
torque, so $\mathbf{H}_{total}$ is not conserved. However, the _rate of change_ of
angular momentum equals the external moment:

$$\dot{\mathbf{H}}_{total} = \mathbf{M}_{ext} = \mathbf{r}_{CoP} \times \mathbf{F}_{GRF}$$

The variability metric quantifies how much angular momentum changes relative to its mean:

$$\text{variability} = \frac{\sigma_{|\mathbf{H}|}}{\mu_{|\mathbf{H}|}}$$

### 12.5.4 Angular Momentum Transfer

During the downswing, angular momentum transfers from proximal to distal segments
(analogous to the kinematic sequence for velocities). This transfer is what creates the
"whip" effect in the golf swing.

**API**:

```python
def compute_angular_momentum_metrics(self) -> AngularMomentumMetrics | None:
    """Compute metrics related to system angular momentum.

    Expects attributes:
        self.angular_momentum: (N, 3) array of angular momentum vectors [kg*m^2/s]
        self.times:            (N,) array of time stamps [s]

    Returns:
        AngularMomentumMetrics dataclass or None if data unavailable
    """
```

The returned dataclass:

```python
@dataclass
class AngularMomentumMetrics:
    peak_magnitude: float     # Peak |H| [kg*m^2/s]
    peak_time: float          # Time of peak [s]
    mean_magnitude: float     # Mean |H| [kg*m^2/s]
    peak_lx: float            # Peak |Hx| [kg*m^2/s]
    peak_ly: float            # Peak |Hy| [kg*m^2/s]
    peak_lz: float            # Peak |Hz| [kg*m^2/s]
    variability: float        # std(|H|) / mean(|H|) (dimensionless)
```

**Usage**:

```python
am_metrics = analyzer.compute_angular_momentum_metrics()
if am_metrics is not None:
    print(f"Peak angular momentum: {am_metrics.peak_magnitude:.2f} kg*m^2/s")
    print(f"  at time: {am_metrics.peak_time:.3f} s")
    print(f"Components: Lx={am_metrics.peak_lx:.2f}, "
          f"Ly={am_metrics.peak_ly:.2f}, Lz={am_metrics.peak_lz:.2f}")
    print(f"Variability: {am_metrics.variability:.3f}")
```

---

## 12.6 Phase Detection (`phase_detection.py`)

The `PhaseDetectionMixin` automatically segments a golf swing into canonical phases using
heuristic algorithms based on club head speed.

### 12.6.1 Golf Swing Phases

A complete golf swing is divided into the following phases:

| Phase              | Description                          | Key Event                      |
| ------------------ | ------------------------------------ | ------------------------------ |
| **Address**        | Setup position before movement       | Start of recording             |
| **Takeaway**       | Initial club movement away from ball | First significant speed        |
| **Backswing**      | Club moves to top of backswing       | Increasing shoulder turn       |
| **Transition**     | Direction change at top              | Speed minimum before downswing |
| **Downswing**      | Acceleration toward impact           | Rapid speed increase           |
| **Impact**         | Club contacts ball                   | Peak club head speed           |
| **Follow-through** | Deceleration after impact            | Speed decreasing               |
| **Finish**         | Final pose                           | Speed below threshold          |

### 12.6.2 Phase Detection Algorithm

The detection algorithm proceeds as follows:

1. **Smooth** the club head speed signal using a Savitzky-Golay filter (window = 11,
   polynomial order = 3) to remove noise while preserving sharp transitions.

2. **Find impact** as the time of peak smoothed speed:
   $$t_{impact} = \arg\max_t \tilde{v}_{club}(t)$$

3. **Find transition** (top of backswing) as the speed minimum in the first 70% of
   the pre-impact interval, avoiding the initial 5 frames of noise:
   $$t_{transition} = \arg\min_{t \in [t_5, 0.7 \cdot t_{impact}]} \tilde{v}_{club}(t)$$

4. **Find takeaway** as the first frame where speed exceeds 10% of the transition speed.

5. **Find finish** as the first post-impact frame where speed drops below 30% of peak.

6. **Assign phase boundaries** based on proportional divisions between key events.

**API**:

```python
def detect_swing_phases(self) -> list[SwingPhase]:
    """Automatically detect swing phases.

    Expects attributes:
        self.club_head_speed: (N,) array of speeds [m/s]
        self.times:           (N,) array of timestamps [s]
        self.duration:        total duration [s]

    Returns:
        List of SwingPhase objects
    """
```

Each `SwingPhase` contains:

```python
@dataclass
class SwingPhase:
    name: str           # Phase name
    start_time: float   # Phase start [s]
    end_time: float     # Phase end [s]
    start_index: int    # Array index of phase start
    end_index: int      # Array index of phase end
    duration: float     # Phase duration [s]
```

**Usage**:

```python
phases = analyzer.detect_swing_phases()
for phase in phases:
    print(f"{phase.name:15s}  {phase.start_time:.3f}s - {phase.end_time:.3f}s  "
          f"({phase.duration:.3f}s)")
```

Example output:

```
Address          0.000s - 0.120s  (0.120s)
Takeaway         0.120s - 0.280s  (0.160s)
Backswing        0.280s - 0.650s  (0.370s)
Transition       0.650s - 0.720s  (0.070s)
Downswing        0.720s - 0.950s  (0.230s)
Impact           0.948s - 0.952s  (0.004s)
Follow-through   0.950s - 1.150s  (0.200s)
Finish           1.150s - 1.500s  (0.350s)
```

### 12.6.3 Phase-Specific Statistics

After detecting phases, per-phase statistics can be computed for any signal:

```python
def compute_phase_statistics(
    self,
    phases: list[SwingPhase],
    data: np.ndarray,
) -> dict[str, SummaryStatistics]:
    """Compute statistics for each phase.

    Args:
        phases: List of detected swing phases
        data: 1D data array (same length as self.times)

    Returns:
        Dictionary mapping phase name to SummaryStatistics
    """
```

**Usage**:

```python
phases = analyzer.detect_swing_phases()

# Speed statistics per phase
speed_stats = analyzer.compute_phase_statistics(phases, analyzer.club_head_speed)
for phase_name, stats in speed_stats.items():
    print(f"{phase_name}: mean={stats.mean:.2f}, max={stats.max:.2f}, "
          f"std={stats.std:.2f} m/s")
```

### 12.6.4 Event Markers

Key discrete events within the swing:

- **Top of backswing**: Maximum X-factor / minimum club head speed before downswing
- **Impact**: Peak club head speed
- **Finish**: Speed drops below threshold

```python
impact_time = analyzer.detect_impact_time()
if impact_time is not None:
    print(f"Impact at t = {impact_time:.3f}s")
```

---

## 12.7 Stability Metrics (`stability_metrics.py`)

The `StabilityMetricsMixin` computes balance and postural stability metrics essential
for golf swing analysis.

### 12.7.1 Base of Support

The base of support (BoS) is the convex hull of all ground contact points. For a golfer,
this is approximately the rectangle defined by the outer edges of both shoes. Stability
requires the CoM projection to remain within the BoS.

### 12.7.2 CoM-CoP Distance

The horizontal distance between the centre of mass (CoM) and centre of pressure (CoP)
is a primary stability measure:

$$d_{stability} = \|CoM_{xy} - CoP_{xy}\|$$

where the subscript $xy$ denotes the horizontal (ground plane) components.

### 12.7.3 Time-to-Boundary

The time-to-boundary (TTB) estimates how long until the CoM would reach the edge of the
base of support at the current velocity:

$$TTB = \frac{d_{boundary}}{v_{CoM}}$$

where $d_{boundary}$ is the distance from the CoM projection to the nearest BoS boundary
and $v_{CoM}$ is the horizontal CoM velocity. Smaller TTB indicates less stability margin.

### 12.7.4 Dynamic Balance Index

The dynamic balance index combines CoM-CoP distance with CoM velocity to give a single
measure of balance control during dynamic movement.

### 12.7.5 Inclination Angle

The inclination angle $\theta$ measures the lean of the CoM-CoP vector from vertical:

$$\theta = \arccos\left(\frac{v_z}{\|v\|}\right)$$

where $v = CoM - CoP$ is the 3D vector from CoP to CoM, and $v_z$ is its vertical
component.

**API**:

```python
def compute_stability_metrics(self) -> StabilityMetrics | None:
    """Compute postural stability metrics.

    Expects attributes:
        self.cop_position: (N, 2) or (N, 3) array of CoP positions [m]
        self.com_position: (N, 3) array of CoM positions [m]

    Returns:
        StabilityMetrics dataclass or None if data unavailable
    """
```

The returned dataclass:

```python
@dataclass
class StabilityMetrics:
    min_com_cop_distance: float     # Minimum horizontal CoM-CoP distance [m]
    max_com_cop_distance: float     # Maximum horizontal CoM-CoP distance [m]
    mean_com_cop_distance: float    # Mean horizontal CoM-CoP distance [m]
    peak_inclination_angle: float   # Maximum lean angle [degrees]
    mean_inclination_angle: float   # Mean lean angle [degrees]
```

**Usage**:

```python
stability = analyzer.compute_stability_metrics()
if stability is not None:
    print(f"CoM-CoP distance: {stability.mean_com_cop_distance:.3f} m "
          f"(min={stability.min_com_cop_distance:.3f}, "
          f"max={stability.max_com_cop_distance:.3f})")
    print(f"Peak inclination angle: {stability.peak_inclination_angle:.1f} degrees")
    print(f"Mean inclination angle: {stability.mean_inclination_angle:.1f} degrees")
```

### 12.7.6 Implementation Notes

The implementation assumes a Z-up coordinate convention (consistent with MuJoCo). The
CoP is treated as a 2D point on the ground plane ($z=0$); if provided as 3D, only the
first two columns are used for horizontal distance computation.

Optimized operations used:

```python
# np.hypot is faster than np.linalg.norm for 2D vectors
diff = cop_xy - com_xy
dist = np.hypot(diff[:, 0], diff[:, 1])
```

---

## 12.8 Shared Dataclasses

All analysis modules use dataclasses defined in `src/shared/python/analysis/dataclasses.py`
for structured output. The complete set:

| Dataclass                | Module           | Key Fields                                                                            |
| ------------------------ | ---------------- | ------------------------------------------------------------------------------------- |
| `PeakInfo`               | Basic stats      | `value`, `time`, `index`, `prominence`, `width`                                       |
| `SummaryStatistics`      | Basic stats      | `mean`, `median`, `std`, `min`, `max`, `range`, `rms`                                 |
| `SwingPhase`             | Phase detection  | `name`, `start_time`, `end_time`, `duration`                                          |
| `KinematicSequenceInfo`  | Swing metrics    | `segment_name`, `peak_velocity`, `peak_time`, `order_index`                           |
| `GRFMetrics`             | GRF analysis     | `cop_path_length`, `cop_max_velocity`, `peak_vertical_force`                          |
| `AngularMomentumMetrics` | Angular momentum | `peak_magnitude`, `peak_time`, `peak_lx/ly/lz`, `variability`                         |
| `StabilityMetrics`       | Stability        | `min/max/mean_com_cop_distance`, `peak/mean_inclination_angle`                        |
| `CoordinationMetrics`    | Coordination     | `in_phase_pct`, `anti_phase_pct`, `mean_coupling_angle`                               |
| `JointPowerMetrics`      | Power analysis   | `peak_generation`, `peak_absorption`, `net_work`                                      |
| `ImpulseMetrics`         | Impulse          | `net_impulse`, `positive_impulse`, `negative_impulse`                                 |
| `JerkMetrics`            | Smoothness       | `peak_jerk`, `rms_jerk`, `dimensionless_jerk`                                         |
| `JointStiffnessMetrics`  | Stiffness        | `stiffness`, `r_squared`, `hysteresis_area`                                           |
| `SwingProfileMetrics`    | Profile          | `speed_score`, `sequence_score`, `stability_score`, `efficiency_score`, `power_score` |
| `PCAResult`              | PCA              | `components`, `explained_variance`, `projected_data`                                  |
| `RQAMetrics`             | Recurrence       | `recurrence_rate`, `determinism`, `laminarity`                                        |

---

## 12.9 Muscle Synergy Analysis

The `MuscleSynergyAnalyzer` (in `src/shared/python/muscle_analysis.py`) extracts coordinated
muscle activation patterns using Non-negative Matrix Factorization (NMF).

### 12.9.1 Theory

Muscle synergies represent the hypothesis that the CNS simplifies control by activating
groups of muscles together. The activation matrix $V$ (muscles $\times$ time) is
factorized as:

$$V \approx W \cdot H$$

where:

- $W \in \mathbb{R}^{n_{muscles} \times n_{synergies}}$ -- muscle weights (which muscles
  belong to each synergy)
- $H \in \mathbb{R}^{n_{synergies} \times n_{samples}}$ -- temporal activation profiles
  (when each synergy is active)

### 12.9.2 Variance Accounted For (VAF)

The quality of the factorization is measured by:

$$\text{VAF} = 1 - \frac{\sum (V - WH)^2}{\sum V^2}$$

A VAF $\geq 0.90$ (90%) is typically considered a good reconstruction.

### 12.9.3 API

```python
from src.shared.python.muscle_analysis import MuscleSynergyAnalyzer

# activation_data: (n_samples, n_muscles), non-negative
analyzer = MuscleSynergyAnalyzer(activation_data, muscle_names=["bic", "tri", ...])

# Extract specific number of synergies
result = analyzer.extract_synergies(n_synergies=3)
print(f"VAF: {result.vaf:.2%}")
print(f"Weights shape: {result.weights.shape}")      # (n_muscles, 3)
print(f"Activations shape: {result.activations.shape}")  # (3, n_samples)

# Find optimal number automatically
optimal = analyzer.find_optimal_synergies(max_synergies=10, vaf_threshold=0.90)
print(f"Optimal synergies: {optimal.n_synergies} (VAF={optimal.vaf:.2%})")
```

---

## 12.10 Complete Analysis Workflow

```python
import numpy as np

# Assume simulation data has been collected:
#   times:            (N,) timestamps
#   joint_positions:  (N, n_joints) joint angles
#   club_head_speed:  (N,) club head speed
#   cop_position:     (N, 2) centre of pressure
#   com_position:     (N, 3) centre of mass
#   ground_forces:    (N, 3) ground reaction forces
#   angular_momentum: (N, 3) total angular momentum
#   kinetic_energy:   (N,) kinetic energy
#   potential_energy: (N,) potential energy

# 1. Phase detection
phases = analyzer.detect_swing_phases()
print("=== Swing Phases ===")
for phase in phases:
    print(f"  {phase.name:15s}  {phase.start_time:.3f}s - {phase.end_time:.3f}s")

# 2. Swing metrics
impact_time = analyzer.detect_impact_time()
tempo = analyzer.compute_tempo()
x_factor = analyzer.compute_x_factor(shoulder_idx, hip_idx)

print(f"\n=== Swing Metrics ===")
print(f"  Impact time: {impact_time:.3f}s")
if tempo:
    print(f"  Tempo ratio: {tempo[2]:.2f}")
if x_factor is not None:
    print(f"  Peak X-Factor: {np.max(x_factor):.1f} deg")

# 3. Energy analysis
energy = analyzer.compute_energy_metrics(kinetic_energy, potential_energy)
print(f"\n=== Energy ===")
print(f"  Peak KE: {energy['max_kinetic_energy']:.1f} J")
print(f"  Efficiency: {energy['energy_efficiency']:.1f}%")
print(f"  Energy drift: {energy['energy_drift']:.3f} J")

# 4. GRF analysis
grf = analyzer.compute_grf_metrics()
if grf:
    print(f"\n=== Ground Reaction Forces ===")
    print(f"  CoP path length: {grf.cop_path_length:.3f} m")
    print(f"  Peak vertical GRF: {grf.peak_vertical_force:.1f} N")
    print(f"  CoP lateral range: {grf.cop_x_range:.3f} m")

# 5. Angular momentum
am = analyzer.compute_angular_momentum_metrics()
if am:
    print(f"\n=== Angular Momentum ===")
    print(f"  Peak: {am.peak_magnitude:.2f} kg*m^2/s at {am.peak_time:.3f}s")
    print(f"  Variability: {am.variability:.3f}")

# 6. Stability
stability = analyzer.compute_stability_metrics()
if stability:
    print(f"\n=== Stability ===")
    print(f"  Mean CoM-CoP distance: {stability.mean_com_cop_distance:.3f} m")
    print(f"  Peak inclination: {stability.peak_inclination_angle:.1f} deg")

# 7. Phase-specific analysis
speed_by_phase = analyzer.compute_phase_statistics(phases, club_head_speed)
print(f"\n=== Speed by Phase ===")
for phase_name, stats in speed_by_phase.items():
    print(f"  {phase_name:15s}  mean={stats.mean:.1f}  max={stats.max:.1f} m/s")
```

---

## 12.11 Summary of Equations

For reference, the key biomechanical equations used across the analysis suite:

| Quantity                   | Equation                                                       | Units            |
| -------------------------- | -------------------------------------------------------------- | ---------------- |
| Club head speed            | $v_{club} = \|\dot{p}_{head}\|$                                | m/s              |
| X-Factor                   | $X = \theta_{shoulder} - \theta_{hip}$                         | degrees          |
| Smash factor               | $SF = v_{ball} / v_{club}$                                     | dimensionless    |
| Kinetic energy             | $T = \frac{1}{2}\,\dot{q}^T\,M(q)\,\dot{q}$                    | J                |
| Potential energy           | $V = -\sum m_i\,g^T\,p_i$                                      | J                |
| Energy efficiency          | $\eta = E_{club}/E_{total} \times 100\%$                       | %                |
| Centre of pressure         | $\text{CoP} = \frac{\sum F_i \times p_i}{\sum F_i}$            | m                |
| Segmental angular momentum | $\mathbf{H}_i = I_i\,\omega_i + m_i\,(r_i \times v_i)$         | kg$\cdot$m$^2$/s |
| Total angular momentum     | $\mathbf{H}_{total} = \sum \mathbf{H}_i$                       | kg$\cdot$m$^2$/s |
| CoM-CoP stability distance | $d = \|CoM_{xy} - CoP_{xy}\|$                                  | m                |
| Time-to-boundary           | $TTB = d_{boundary} / v_{CoM}$                                 | s                |
| Hill muscle force          | $F_m = a \cdot F_{max} \cdot f_L \cdot f_V + F_{PE}$           | N                |
| Joint power                | $P = \tau \cdot \dot{q}$                                       | W                |
| Total work                 | $W = \int_0^T P\,dt$                                           | J                |
| Activation dynamics        | $\dot{a} = (u - a) / \tau$                                     | 1/s              |
| Muscle recruitment         | $\min \sum a_i^2$ s.t. $\tau = R \cdot F(a)$                   | --               |
| Jacobian                   | $J(q) = \frac{\partial \phi(q)}{\partial q}$                   | varies           |
| Jacobian derivative        | $\dot{J}(q, \dot{q}) = \frac{\partial J}{\partial q}\,\dot{q}$ | varies           |
| Drift acceleration         | $\ddot{q}_{drift} = M^{-1}(-C\dot{q} - g)$                     | rad/s$^2$        |
| Control acceleration       | $\ddot{q}_{control} = M^{-1}\tau$                              | rad/s$^2$        |
| Superposition              | $\ddot{q}_{full} = \ddot{q}_{drift} + \ddot{q}_{control}$      | rad/s$^2$        |

---

# UpstreamDrift User Manual -- Part IV: Robotics Framework

---

# Chapter 13: Contact Dynamics

The contact dynamics module (`src/robotics/contact/`) provides the foundation for
multi-body contact modeling, friction analysis, and grasp quality assessment. All
components adhere to the framework's Design by Contract methodology: immutable
data structures enforce invariants at construction time, preconditions guard every
public entry point, and postconditions verify results before they leave the module
boundary. This chapter covers the four major subsystems -- core contact types,
friction cone analysis, contact management, and grasp analysis -- in full
mathematical and programmatic detail.

---

## 13.1 Core Types and Protocols

### 13.1.1 ContactType Enumeration

The `ContactType` enum (defined in `src/robotics/core/types.py`) classifies the
geometry of contact between two bodies. The available members are:

| Member  | Description                                                                                                 |
| ------- | ----------------------------------------------------------------------------------------------------------- |
| `POINT` | Single-point contact. The simplest model, adequate for finger tips and convex surface interactions.         |
| `LINE`  | Edge or line contact. Occurs when a straight edge contacts a surface.                                       |
| `PATCH` | Surface or patch contact. Models extended contact areas such as the sole of a foot resting flat on a floor. |
| `SOFT`  | Soft or deformable contact. Used when one or both bodies exhibit compliance at the contact interface.       |

Every `ContactType` value is produced by `auto()`, making the underlying integer
representation an implementation detail. Code should compare against the symbolic
members rather than raw integers:

```python
from src.robotics.core.types import ContactType

if contact.contact_type == ContactType.PATCH:
    # Use extended contact model
    ...
```

### 13.1.2 ContactState Frozen Dataclass

`ContactState` is a frozen (immutable) dataclass that captures the complete
instantaneous state of a single contact. Immutability ensures thread safety and
prevents accidental mutation during multi-step computations. The class definition
lives in `src/robotics/core/types.py`.

**Field Reference**

| Field                  | Type                    | Default             | Description                                                     |
| ---------------------- | ----------------------- | ------------------- | --------------------------------------------------------------- |
| `contact_id`           | `int`                   | (required)          | Unique identifier assigned by the ContactManager.               |
| `body_a`               | `str`                   | (required)          | Name of the first body in the contact pair.                     |
| `body_b`               | `str`                   | (required)          | Name of the second body in the contact pair.                    |
| `position`             | `NDArray[float64]` (3,) | (required)          | Contact point in the world frame, measured in meters.           |
| `normal`               | `NDArray[float64]` (3,) | (required)          | Unit contact normal pointing from body B toward body A.         |
| `penetration`          | `float`                 | `0.0`               | Penetration depth in meters. Must be non-negative.              |
| `normal_force`         | `float`                 | `0.0`               | Scalar normal force magnitude in newtons. Must be non-negative. |
| `friction_force`       | `NDArray[float64]` (3,) | `np.zeros(3)`       | Friction (tangential) force vector in newtons.                  |
| `friction_coefficient` | `float`                 | `0.5`               | Coulomb friction coefficient $\mu$. Must be non-negative.       |
| `contact_type`         | `ContactType`           | `ContactType.POINT` | Geometric classification of the contact.                        |
| `is_active`            | `bool`                  | `True`              | Whether the contact is currently active.                        |

**Construction-Time Invariants**

The `__post_init__` method enforces the following invariants immediately after
construction:

1. `position.shape == (3,)` -- the contact point must be a 3-vector.
2. `normal.shape == (3,)` -- the contact normal must be a 3-vector.
3. `friction_force.shape == (3,)` -- the friction force must be a 3-vector.
4. `penetration >= 0` -- negative penetration is physically meaningless.
5. `normal_force >= 0` -- tensile normal forces violate the unilateral contact
   assumption.
6. `friction_coefficient >= 0` -- negative friction is non-physical.
7. The normal vector is automatically normalized: if $\|\mathbf{n}\| > 10^{-10}$
   then the stored value becomes $\hat{\mathbf{n}} = \mathbf{n} / \|\mathbf{n}\|$.

All NumPy array fields are coerced to `float64` via `np.asarray` during
construction, regardless of the dtype passed in by the caller. Because the
dataclass is frozen, these assignments use `object.__setattr__` to bypass the
write protection that `frozen=True` normally imposes.

**Example -- Creating a ContactState**

```python
import numpy as np
from src.robotics.core.types import ContactState, ContactType

contact = ContactState(
    contact_id=0,
    body_a="robot_foot",
    body_b="ground",
    position=np.array([0.1, -0.05, 0.0]),
    normal=np.array([0.0, 0.0, 1.0]),
    penetration=0.001,
    normal_force=490.5,
    friction_force=np.array([12.0, -3.0, 0.0]),
    friction_coefficient=0.6,
    contact_type=ContactType.PATCH,
    is_active=True,
)
```

### 13.1.3 The get_wrench() Method

`get_wrench()` assembles the full 6D spatial wrench at the contact point:

$$
\mathbf{w} = \begin{bmatrix} \mathbf{f} \\ \boldsymbol{\tau} \end{bmatrix}
= \begin{bmatrix} f_n \hat{\mathbf{n}} + \mathbf{f}_t \\ \mathbf{0} \end{bmatrix}
$$

The total contact force $\mathbf{f}$ is the sum of the normal component
$f_n \hat{\mathbf{n}}$ and the tangential friction force $\mathbf{f}_t$. Under
the point-contact assumption, the torque at the contact point is zero because
all forces act through a single point. The method returns a length-6 NumPy array
with layout `[fx, fy, fz, tx, ty, tz]`.

```python
wrench = contact.get_wrench()  # shape (6,)
force = wrench[:3]   # net contact force in world frame
torque = wrench[3:]  # zero for point contacts
```

To compute the torque contribution at a different reference point $\mathbf{p}$,
apply the standard cross-product transport:

$$
\boldsymbol{\tau}_p = (\mathbf{r}_c - \mathbf{p}) \times \mathbf{f}
$$

where $\mathbf{r}_c$ is `contact.position`.

### 13.1.4 The is_sliding() Method

`is_sliding(tolerance=1e-6)` checks whether the contact is at the Coulomb
friction limit and hence in a sliding (kinetic) state:

$$
\text{sliding} \iff \|\mathbf{f}_t\| \geq \mu \cdot f_n - \epsilon
$$

where $\epsilon$ is the `tolerance` parameter. The method returns `True` when
the friction force magnitude reaches the maximum value allowed by the friction
cone, indicating that the contact surfaces are sliding relative to each other
(or are on the verge of doing so).

```python
if contact.is_sliding():
    print("Contact is at friction limit -- kinetic friction active")
```

### 13.1.5 The with_force() Method

Because `ContactState` is frozen, force updates are performed by creating a new
instance with modified values. The `with_force(normal_force, friction_force=None)`
convenience method copies every field except the two force fields:

```python
updated = contact.with_force(
    normal_force=600.0,
    friction_force=np.array([15.0, -4.0, 0.0]),
)
# updated is a new ContactState; the original is unchanged
```

If `friction_force` is `None`, the original friction force is preserved.

### 13.1.6 Supporting Protocols

The `src/robotics/core/protocols.py` module defines runtime-checkable protocols
that physics engines must implement to participate in the contact subsystem.

**RoboticsCapable** -- the minimal protocol. Provides `get_state()`,
`set_state()`, `compute_mass_matrix()`, `compute_bias_forces()`,
`compute_gravity_forces()`, `compute_jacobian()`, and `get_time()`.

**ContactCapable** -- extends the engine with contact-specific queries:

| Method                      | Return Type                    | Description                                                                            |
| --------------------------- | ------------------------------ | -------------------------------------------------------------------------------------- |
| `get_contact_count()`       | `int`                          | Number of active contacts detected by the engine.                                      |
| `get_contact_info(idx)`     | `dict[str, Any]`               | Dictionary with keys `body_a`, `body_b`, `position`, `normal`, `penetration`, `force`. |
| `get_contact_jacobian(idx)` | `NDArray` (3, n_v) or (6, n_v) | Linear or spatial Jacobian at the contact point.                                       |

Both protocols use `@runtime_checkable` so they can be validated at runtime via
`isinstance` checks. The `ContactManager` (Section 13.3) verifies that the
engine satisfies these protocols during initialization.

---

## 13.2 Friction Cone Analysis

The friction cone module (`src/robotics/contact/friction_cone.py`) provides tools
for representing, testing, linearizing, and projecting forces with respect to
Coulomb friction constraints. All functions employ Design by Contract decorators
from `src/shared/python/contracts`.

### 13.2.1 The FrictionCone Frozen Dataclass

```python
@dataclass(frozen=True)
class FrictionCone:
    mu: float                        # Coulomb friction coefficient
    normal: NDArray[np.float64]      # Contact normal (unit vector)
    num_sides: int = 8               # Sides for linearized approximation
```

**Construction-time validation:**

- `mu >= 0` (non-negative friction).
- `num_sides >= 3` (a pyramid needs at least three faces).
- `normal` is coerced to `float64` and normalized to unit length. A zero-length
  normal raises `ValueError`.

The Coulomb friction model constrains the tangential component of the contact
force to lie within a cone:

$$
\|\mathbf{f}_t\| \leq \mu \cdot f_n
$$

where $\mathbf{f}_t$ is the component of force perpendicular to the contact
normal and $f_n = \mathbf{f} \cdot \hat{\mathbf{n}}$ is the component along
the contact normal. This defines a second-order cone (ice-cream cone) in 3D
force space with half-angle $\alpha = \arctan(\mu)$.

### 13.2.2 The contains() Method

`contains(force, tolerance=1e-6)` checks whether a given force vector lies
inside the friction cone:

1. Decompose the force into normal and tangential components:

$$
f_n = \mathbf{f} \cdot \hat{\mathbf{n}}, \qquad
\mathbf{f}_t = \mathbf{f} - f_n \hat{\mathbf{n}}
$$

2. Check the unilateral constraint: if $f_n < -\epsilon$, the force is tensile
   (pulling the surfaces apart), which violates the contact model. Return
   `False`.

3. Check the friction constraint: return
   $\|\mathbf{f}_t\| \leq \mu \cdot f_n + \epsilon$.

```python
cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]))
force = np.array([3.0, 0.0, 10.0])
assert cone.contains(force)  # 3.0 <= 0.5 * 10.0 = 5.0, within cone
```

### 13.2.3 The get_generators() Method

`get_generators()` returns a polyhedral approximation of the friction cone as a
set of $N$ edge directions (generators), where $N$ = `num_sides`. The
generators are arranged in a circular pattern around the contact normal.

**Algorithm.** Given the contact normal $\hat{\mathbf{n}}$, the code first
constructs an orthonormal tangent basis $\{\hat{\mathbf{t}}_1, \hat{\mathbf{t}}_2\}$
using Gram-Schmidt orthogonalization. The initial vector is chosen to avoid
near-parallel alignment with the normal:

$$
\mathbf{v}_0 =
\begin{cases}
[1, 0, 0]^T & \text{if } |n_x| < 0.9 \\
[0, 1, 0]^T & \text{otherwise}
\end{cases}
$$

Then:

$$
\hat{\mathbf{t}}_1 = \frac{\mathbf{v}_0 - (\mathbf{v}_0 \cdot \hat{\mathbf{n}})\hat{\mathbf{n}}}
  {\|\mathbf{v}_0 - (\mathbf{v}_0 \cdot \hat{\mathbf{n}})\hat{\mathbf{n}}\|},
\qquad
\hat{\mathbf{t}}_2 = \hat{\mathbf{n}} \times \hat{\mathbf{t}}_1
$$

For $i = 0, 1, \ldots, N-1$ the angular position around the cone is:

$$
\theta_i = \frac{2\pi i}{N}
$$

and the $i$-th generator direction is:

$$
\mathbf{g}_i = \hat{\mathbf{n}} + \mu \bigl(\cos\theta_i \cdot \hat{\mathbf{t}}_1
  + \sin\theta_i \cdot \hat{\mathbf{t}}_2\bigr)
$$

Each generator lies on the surface of the friction cone. The returned matrix has
shape `(3, num_sides)`, with each column being one generator.

**Preconditions** (enforced by `@precondition` decorators on the internal
`_compute_cone_generators` function):

- $\mu \geq 0$
- `num_sides >= 3`

**Postcondition:** the result has at least 3 columns.

**Example:**

```python
cone = FrictionCone(mu=0.7, normal=np.array([0, 0, 1]), num_sides=8)
G = cone.get_generators()  # shape (3, 8)
# Each column G[:, i] is a generator on the cone surface
```

### 13.2.4 Linearizing the Friction Cone

The function `linearize_friction_cone(mu, normal, num_faces=8)` converts the
nonlinear second-order cone constraint into a polyhedral (linear) approximation
of the form:

$$
A \mathbf{f} \leq \mathbf{b}
$$

This approximation is essential for QP-based controllers that require linear
constraints.

**Derivation.** The exact Coulomb constraint is:

$$
\|\mathbf{f}_t\| \leq \mu f_n
$$

This is approximated by $N$ half-space constraints, one for each face of the
inscribed polyhedron. For face $i$, define the tangent direction:

$$
\mathbf{d}_i = \cos\theta_i \cdot \hat{\mathbf{t}}_1 + \sin\theta_i \cdot \hat{\mathbf{t}}_2,
\qquad \theta_i = \frac{2\pi i}{N}
$$

The $i$-th linear constraint is:

$$
\mathbf{d}_i^T \mathbf{f} \leq \mu \, \hat{\mathbf{n}}^T \mathbf{f}
$$

which rearranges to:

$$
(\mathbf{d}_i - \mu \hat{\mathbf{n}})^T \mathbf{f} \leq 0
$$

The function returns `(A, b)` where $A$ has shape `(num_faces, 3)` and $b$ is a
zero vector of length `num_faces`. Each row of $A$ is $\mathbf{d}_i - \mu \hat{\mathbf{n}}$.

**Accuracy vs. performance trade-off.** The polyhedral approximation is
conservative (inscribed): every force satisfying the linear constraints also
satisfies the true cone constraint. The approximation tightens as `num_faces`
increases. Common choices:

| `num_faces` | Approximation quality  | Typical use case       |
| ----------- | ---------------------- | ---------------------- |
| 4           | Coarse pyramid         | Real-time MPC          |
| 8           | Good balance (default) | Whole-body control     |
| 16          | High fidelity          | Grasp quality analysis |

The `FrictionConeType` enum in `src/robotics/core/types.py` provides named
presets: `EXACT`, `LINEARIZED_4`, `LINEARIZED_8`, `LINEARIZED_16`.

### 13.2.5 The compute_friction_cone_constraint() Function

`compute_friction_cone_constraint(contact_normal, contact_position,
friction_coeff, num_faces=8)` bundles the linearization with the non-negative
normal force constraint and the cone generators into a single dictionary for
direct consumption by optimization-based controllers:

```python
result = compute_friction_cone_constraint(
    contact_normal=np.array([0, 0, 1]),
    contact_position=np.array([0.1, 0.0, 0.0]),
    friction_coeff=0.5,
    num_faces=8,
)
# result['A']          shape (num_faces + 1, 3)
# result['b']          shape (num_faces + 1,)
# result['normal']     shape (3,)
# result['generators'] shape (3, num_faces)
```

The extra row in `A` encodes the non-negative normal force constraint
$-\hat{\mathbf{n}}^T \mathbf{f} \leq 0$, i.e., $f_n \geq 0$.

### 13.2.6 Projecting onto the Friction Cone

`project_to_friction_cone(force, cone)` finds the closest point inside the
friction cone to a given force vector. This is useful for enforcing friction
constraints in force control or for "clamping" forces computed by unconstrained
solvers.

**Algorithm.** The projection considers three cases:

**Case 1: Force is inside the cone.** If `cone.contains(force)` returns `True`,
return the force unchanged.

**Case 2: Normal force is negative** ($f_n < 0$). The force is pulling the
surfaces apart. Two sub-cases arise:

- If the tangential magnitude is small ($\|\mathbf{f}_t\| \leq -\mu f_n$), the
  force lies in the dual cone; the closest point on the friction cone is the
  origin (zero force).

- Otherwise, project onto the cone surface. The cone edge direction in the
  tangent plane of the force is:

$$
\hat{\mathbf{e}} = \cos\alpha \cdot \hat{\mathbf{n}} + \sin\alpha \cdot \hat{\mathbf{t}}_f
$$

where $\alpha = \arctan(\mu)$ and $\hat{\mathbf{t}}_f = \mathbf{f}_t / \|\mathbf{f}_t\|$.
The projection length along this edge is:

$$
\ell = f_n \cos\alpha + \|\mathbf{f}_t\| \sin\alpha
$$

If $\ell \leq 0$, project to the origin. Otherwise, the projected force is
$\ell \cdot \hat{\mathbf{e}}$.

**Case 3: Tangential force exceeds friction limit** ($\|\mathbf{f}_t\| > \mu f_n$,
with $f_n \geq 0$). Scale the tangential component down to the friction limit
while preserving direction:

$$
\mathbf{f}_t \leftarrow \mathbf{f}_t \cdot \frac{\mu f_n}{\|\mathbf{f}_t\|}
$$

The projected force is $f_n \hat{\mathbf{n}} + \mathbf{f}_t^{\text{proj}}$.

```python
from src.robotics.contact.friction_cone import project_to_friction_cone, FrictionCone

cone = FrictionCone(mu=0.5, normal=np.array([0.0, 0.0, 1.0]))
raw_force = np.array([10.0, 0.0, 5.0])  # tangential exceeds limit
projected = project_to_friction_cone(raw_force, cone)
# projected tangential magnitude will be 0.5 * 5.0 = 2.5
```

---

## 13.3 Contact Management

The `ContactManager` class (`src/robotics/contact/contact_manager.py`) acts as
the central coordinator for contact detection, caching, Jacobian computation,
and support polygon analysis. It inherits from `ContractChecker` to gain
automatic invariant verification.

### 13.3.1 Construction and Invariants

```python
manager = ContactManager(engine, default_friction=0.5)
```

The constructor verifies that `engine` implements `RoboticsCapable` (raising
`TypeError` otherwise) and probes whether it also implements `ContactCapable`.
Two invariants are enforced throughout the object's lifetime:

1. `default_friction >= 0` -- the fallback friction coefficient is non-negative.
2. All contact IDs in the cache are unique -- no two `ContactState` objects
   share the same `contact_id`.

The class maintains an auto-incrementing counter (`_next_contact_id`) that
assigns monotonically increasing IDs to newly created contacts. This guarantees
uniqueness even across successive calls to `detect_contacts`.

### 13.3.2 Detecting Contacts

```python
contacts = manager.detect_contacts(q=None)
```

If a configuration `q` is provided, the engine state is temporarily set to that
configuration (preserving the current velocity). The method then delegates to the
engine's contact detection:

1. Query `engine.get_contact_count()` for the number of active contacts.
2. For each index $i \in [0, n)$, call `engine.get_contact_info(i)` to retrieve
   the raw contact data dictionary.
3. Decompose the reported force into normal and tangential components:

$$
f_n = \max\bigl(0,\; \mathbf{f} \cdot \hat{\mathbf{n}}\bigr),
\qquad
\mathbf{f}_t = \mathbf{f} - f_n \hat{\mathbf{n}}
$$

4. Construct a `ContactState` with a fresh unique ID.

If the engine does not implement `ContactCapable`, an empty list is returned.

**Postcondition** (enforced by `@postcondition`): every element of the returned
list is an instance of `ContactState`.

If `get_contact_info` raises for any index, the error is wrapped in a
`ContactError` with the offending `contact_id` attached.

### 13.3.3 Contact Jacobians

**Single contact:**

```python
J = manager.get_contact_jacobian(contact)  # (3, n_v) or (6, n_v) or None
```

The method looks up the contact's position in the internal cache by matching
`contact_id`, then delegates to `engine.get_contact_jacobian(idx)`. The Jacobian
maps joint velocities to contact-point velocities:

$$
\dot{\mathbf{p}}_c = J_c \dot{\mathbf{q}}
$$

A `(3, n_v)` Jacobian captures linear velocity only (point contacts). A
`(6, n_v)` Jacobian includes angular velocity as well (patch or wrench contacts).

**Stacked Jacobians:**

```python
J_stack = manager.get_contact_jacobian_stack(contacts)  # (3*n_c, n_v) or None
```

For $n_c$ contacts, the method vertically stacks the linear parts of all contact
Jacobians. If a `(6, n_v)` Jacobian is returned by the engine, only the first 3
rows (linear portion) are retained. The result has shape $(3 n_c, n_v)$:

$$
J_{\text{stack}} =
\begin{bmatrix}
J_{c,1} \\ J_{c,2} \\ \vdots \\ J_{c,n_c}
\end{bmatrix}
$$

**Postcondition:** the result is either `None` or a 2D matrix.

### 13.3.4 Computing the Support Polygon

```python
polygon = manager.compute_support_polygon(contacts)  # (m, 2) or None
```

The support polygon is the convex hull of the ground-plane projections of all
active contact points. It is the region within which the zero-moment point (ZMP)
must lie for static stability.

**Algorithm.** The method projects each contact's 3D position onto the $xy$-plane
by taking `position[:2]`. It then computes the 2D convex hull using one of two
strategies:

1. **SciPy path** (preferred): `scipy.spatial.ConvexHull` is used when available.
   Vertices are returned in counter-clockwise order.

2. **Graham scan fallback**: when SciPy is not installed, a manual implementation
   is used. The Graham scan algorithm proceeds as follows:

   **Step 1 -- Pivot selection.** Find the point with the smallest $y$-coordinate
   (breaking ties by smallest $x$). This point is guaranteed to be on the hull.

   **Step 2 -- Polar angle sorting.** Sort all remaining points by their polar
   angle relative to the pivot. Ties in angle are broken by distance from the
   pivot (closer points first).

   **Step 3 -- Stack-based construction.** Initialize a stack with the pivot.
   For each sorted point $p$:

   - While the stack has at least 2 elements and the cross product of
     $\overrightarrow{s_{-2} s_{-1}}$ and $\overrightarrow{s_{-1} p}$ is
     negative (clockwise turn), pop the stack.
   - Push $p$.

   The cross product test is:

   $$
   \text{cross}(\mathbf{o}, \mathbf{a}, \mathbf{b})
   = (a_x - o_x)(b_y - o_y) - (a_y - o_y)(b_x - o_x)
   $$

   Positive values indicate a counter-clockwise turn; negative values indicate
   clockwise. Points causing clockwise turns are discarded.

The function returns `None` if fewer than 3 contacts are provided, since a
polygon requires at least 3 vertices.

### 13.3.5 Point-in-Support-Polygon Test

```python
inside = manager.point_in_support_polygon(point, contacts)
```

This method first computes the support polygon (Section 13.3.4), then checks
whether the given point lies inside it using the **ray casting algorithm**.

**Ray casting algorithm.** Cast a horizontal ray from the test point toward
positive $x$-infinity. Count the number of polygon edges that the ray crosses.
If the count is odd, the point is inside; if even, outside.

For each edge from vertex $(x_i, y_i)$ to $(x_j, y_j)$:

1. Check if the ray's $y$-coordinate is between $y_i$ and $y_j$ (exclusive on
   one side to handle corner cases).
2. Compute the $x$-coordinate of the intersection:
   $x_{\text{int}} = x_i + (y_p - y_i) \cdot (x_j - x_i) / (y_j - y_i)$.
3. If $x_p < x_{\text{int}}$, toggle the inside flag.

The method accepts 2D or 3D input points; the $z$-component is ignored if present.

---

## 13.4 Grasp Analysis

The grasp analysis module (`src/robotics/contact/grasp_analysis.py`) provides
tools for evaluating multi-fingered grasps in terms of force closure, quality
metrics, and wrench-space coverage.

### 13.4.1 The Grasp Matrix

`compute_grasp_matrix(contacts, object_frame=None)` constructs the matrix $G$
that maps stacked contact forces to the resultant object wrench:

$$
\mathbf{w} = G \mathbf{f}
$$

where $\mathbf{w} \in \mathbb{R}^6$ is the wrench (force + torque) applied to
the object at its reference frame, and $\mathbf{f} \in \mathbb{R}^{3 n_c}$ is
the concatenation of all contact forces.

**Structure.** For $n_c$ point contacts, the grasp matrix has shape
$(6, 3 n_c)$:

$$
G = \begin{bmatrix}
I_3 & I_3 & \cdots & I_3 \\
[\mathbf{r}_1]_\times & [\mathbf{r}_2]_\times & \cdots & [\mathbf{r}_{n_c}]_\times
\end{bmatrix}
$$

where $\mathbf{r}_i = \mathbf{p}_i - \mathbf{p}_{\text{obj}}$ is the vector
from the object frame to the $i$-th contact point, and $[\mathbf{r}]_\times$ is
the skew-symmetric matrix:

$$
[\mathbf{r}]_\times =
\begin{bmatrix}
0 & -r_z & r_y \\
r_z & 0 & -r_x \\
-r_y & r_x & 0
\end{bmatrix}
$$

This matrix satisfies the identity $[\mathbf{r}]_\times \mathbf{f} = \mathbf{r} \times \mathbf{f}$.

The top block of $G$ sums all contact forces to produce the net force. The
bottom block computes the net torque by summing the moment arm contributions
$\mathbf{r}_i \times \mathbf{f}_i$.

If `object_frame` is `None`, the centroid of all contact positions is used as
the reference:

$$
\mathbf{p}_{\text{obj}} = \frac{1}{n_c} \sum_{i=1}^{n_c} \mathbf{p}_i
$$

**Precondition:** at least 1 contact must be provided.

**Postcondition:** the result has shape `(6, 3 * len(contacts))`.

```python
from src.robotics.contact.grasp_analysis import compute_grasp_matrix

G = compute_grasp_matrix(contacts, object_frame=np.array([0, 0, 0.05]))
# G.shape == (6, 3 * len(contacts))
```

### 13.4.2 Force Closure Check

`check_force_closure(contacts, num_cone_faces=8)` determines whether a grasp
can resist arbitrary external wrenches. A grasp has **force closure** if and only
if the origin of wrench space lies strictly inside the convex hull of the
achievable wrenches (the contact wrench cone).

**Algorithm.** The check proceeds in three stages:

1. **Build wrench generators.** For each contact, create a `FrictionCone` with
   the contact's friction coefficient and normal, then compute the cone
   generators. Each force generator $\mathbf{g}_j$ at contact $i$ produces a
   wrench generator:

$$
\mathbf{w}_{ij} = \begin{bmatrix} \mathbf{g}_j \\ [\mathbf{r}_i]_\times \mathbf{g}_j \end{bmatrix}
$$

The full set of wrench generators is stored as columns of a $(6, N)$ matrix,
where $N = n_c \cdot n_{\text{faces}}$.

2. **Check generator span.** If there are fewer than 6 generators, the wrench
   space cannot be spanned and force closure is impossible.

3. **Linear programming feasibility test.** Solve the LP:

$$
\min \; \mathbf{0}^T \boldsymbol{\alpha}
\quad \text{s.t.} \quad
\sum_{i=1}^{N} \alpha_i \mathbf{w}_i = \mathbf{0}, \quad
\sum_{i=1}^{N} \alpha_i = 1, \quad
\alpha_i \geq 0
$$

This checks whether the origin can be expressed as a convex combination of
the wrench generators. If the LP is feasible, the grasp has force closure.
The quality margin is reported as the minimum $\alpha_i$ in the solution --
a larger value indicates the origin is deeper inside the convex hull.

The implementation uses `scipy.optimize.linprog` with the HiGHS solver. If
SciPy is unavailable, a heuristic fallback checks the rank of the generator
matrix via SVD: if all 6 singular values exceed $10^{-6}$, force closure is
assumed likely.

**Precondition:** at least 2 contacts are required.

```python
from src.robotics.contact.grasp_analysis import check_force_closure

has_closure, margin = check_force_closure(contacts, num_cone_faces=8)
if has_closure:
    print(f"Force closure achieved, quality margin: {margin:.4f}")
```

### 13.4.3 Grasp Quality Metrics

`compute_grasp_quality(contacts, metric="min_singular_value")` quantifies the
grasp's ability to transmit forces to the object using the singular values of the
grasp matrix.

The grasp matrix $G$ is decomposed via SVD:

$$
G = U \Sigma V^T
$$

where $\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_k)$ with
$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_k > 0$ (singular values
exceeding $10^{-10}$ are retained).

Three metrics are available:

| Metric                 | Formula                         | Interpretation                                                                             |
| ---------------------- | ------------------------------- | ------------------------------------------------------------------------------------------ |
| `"min_singular_value"` | $\sigma_{\min}(G)$              | Worst-case force transmission. Larger is better.                                           |
| `"volume"`             | $\prod_{i=1}^{k} \sigma_i$      | Volume of the grasp wrench ellipsoid. Larger indicates broader wrench coverage.            |
| `"isotropy"`           | $\sigma_{\min} / \sigma_{\max}$ | Uniformity of force transmission across directions. A value of 1.0 is perfectly isotropic. |

A degenerate grasp (all singular values below threshold) returns quality 0.0.

```python
from src.robotics.contact.grasp_analysis import compute_grasp_quality

q_min = compute_grasp_quality(contacts, metric="min_singular_value")
q_vol = compute_grasp_quality(contacts, metric="volume")
q_iso = compute_grasp_quality(contacts, metric="isotropy")
```

### 13.4.4 Contact Wrench Cone

`compute_contact_wrench_cone(contacts, num_faces=8)` returns the full set of
wrench generators without performing the force-closure check. The result is a
$(6, N)$ matrix where each column is a wrench generator:

$$
\mathbf{w}_{ij} =
\begin{bmatrix}
\mathbf{g}_j \\
\mathbf{r}_i \times \mathbf{g}_j
\end{bmatrix}
$$

This is useful for custom wrench-space analyses beyond the built-in metrics.

### 13.4.5 Required Contact Forces

`required_contact_forces(contacts, desired_wrench)` solves for the minimum-norm
contact forces that produce a specified object wrench:

$$
\min_{\mathbf{f}} \; \|\mathbf{f}\|^2
\quad \text{s.t.} \quad G \mathbf{f} = \mathbf{w}_d
$$

The implementation uses `scipy.optimize.minimize` with the SLSQP method. The
initial guess is the least-squares pseudoinverse solution
$\mathbf{f}_0 = G^+ \mathbf{w}_d$. Box bounds approximate friction-cone
feasibility by limiting each force component to
$\pm f_n (1 + \mu)$.

Returns the optimal force vector $(3 n_c,)$ or `None` if the problem is
infeasible.

```python
from src.robotics.contact.grasp_analysis import required_contact_forces

desired_wrench = np.array([0, 0, -9.81 * mass, 0, 0, 0])
forces = required_contact_forces(contacts, desired_wrench)
if forces is not None:
    print(f"Required forces: {forces}")
```

---

# Chapter 14: Whole-Body Control

The whole-body control module (`src/robotics/control/whole_body/`) implements a
hierarchical, optimization-based controller that resolves multiple competing
objectives for complex robotic systems. The module comprises three primary
files: task descriptors (`task.py`), QP solvers (`qp_solver.py`), and the
controller itself (`wbc_controller.py`).

---

## 14.1 Task-Based Control

### 14.1.1 Task Types

The `TaskType` enum classifies how a task participates in the optimization:

| Member       | Semantics                                                                                                  |
| ------------ | ---------------------------------------------------------------------------------------------------------- |
| `EQUALITY`   | Hard equality constraint: $J \ddot{\mathbf{q}} = \ddot{\mathbf{x}}_d$. Used for contact constraints.       |
| `INEQUALITY` | Bounded constraint: $\mathbf{lb} \leq J \ddot{\mathbf{q}} \leq \mathbf{ub}$. Used for joint limits.        |
| `SOFT`       | Weighted objective: minimize $\|J \ddot{\mathbf{q}} - \ddot{\mathbf{x}}_d\|_W^2$. Used for tracking tasks. |

### 14.1.2 Task Priority Levels

The `TaskPriority` enum (from `src/robotics/core/types.py`) defines five
priority levels, with lower numeric values indicating higher priority:

| Level             | Value | Typical usage                           |
| ----------------- | ----- | --------------------------------------- |
| `HARD_CONSTRAINT` | 0     | Contact constraints, equation of motion |
| `SAFETY`          | 1     | Joint limit avoidance                   |
| `PRIMARY`         | 2     | CoM tracking, balance                   |
| `SECONDARY`       | 3     | End-effector tracking                   |
| `TERTIARY`        | 4     | Posture regularization                  |

In hierarchical mode, tasks at priority level $k$ are solved in the null space
of all tasks at levels $0$ through $k-1$, ensuring that lower-priority tasks
cannot interfere with higher-priority objectives.

### 14.1.3 The Task Dataclass

```python
@dataclass
class Task:
    name: str
    task_type: TaskType
    priority: int
    jacobian: NDArray[np.float64]     # (task_dim, n_v)
    target: NDArray[np.float64]       # (task_dim,)
    weight: NDArray[np.float64] | None = None
    lower_bound: NDArray[np.float64] | None = None
    upper_bound: NDArray[np.float64] | None = None
    gain_p: float = 100.0
    gain_d: float = 20.0
```

**Validation.** `__post_init__` verifies:

- `jacobian` is 2D with shape `(task_dim, n_v)`.
- `target` has shape `(task_dim,)` consistent with the Jacobian row count.
- `weight` (if provided) has shape `(task_dim,)`.
- Inequality tasks have at least one bound.
- All values are finite (no NaN or Inf).

**Properties:**

- `task_dim`: the number of task-space dimensions (rows of the Jacobian).
- `config_dim`: the number of configuration-space dimensions (columns of the Jacobian).

**`get_weight_matrix()`** returns a `(task_dim, task_dim)` diagonal matrix. If
`weight` is provided, it becomes the diagonal; otherwise, the identity matrix is
returned.

**`compute_error_feedback(position_error, velocity_error)`** computes the desired
task-space acceleration using PD control:

$$
\ddot{\mathbf{x}}_d = \ddot{\mathbf{x}}_{\text{ff}} + K_p \mathbf{e}_p + K_d \mathbf{e}_v
$$

where $\ddot{\mathbf{x}}_{\text{ff}}$ is the feedforward target (stored in
`self.target`), $\mathbf{e}_p$ is the position error, and $\mathbf{e}_v$ is the
velocity error. The gains $K_p$ and $K_d$ are scalar (applied uniformly across
all task dimensions).

### 14.1.4 Task Creation Utilities

Four factory functions create properly configured tasks:

**`create_com_task()`** -- Center-of-mass tracking.

```python
task = create_com_task(
    jacobian_com,           # (3, n_v) CoM Jacobian
    com_current,            # (3,) current CoM position
    com_target,             # (3,) desired CoM position
    com_velocity,           # (3,) current CoM velocity
    com_velocity_target=None,  # (3,) desired CoM velocity (default: zero)
    weight=1.0,
    priority=2,             # PRIMARY
    gain_p=100.0,
    gain_d=20.0,
)
```

The target acceleration is computed as:

$$
\ddot{\mathbf{x}}_{\text{CoM}} = K_p (\mathbf{x}_d - \mathbf{x}) + K_d (\dot{\mathbf{x}}_d - \dot{\mathbf{x}})
$$

**`create_posture_task()`** -- Joint-space posture regulation.

```python
task = create_posture_task(
    n_v,                    # number of velocity DOFs
    q_current,              # current joint positions
    q_target,               # desired joint positions
    v_current,              # current joint velocities
    weight=0.1,
    priority=4,             # TERTIARY
    gain_p=50.0,
    gain_d=10.0,
    mask=None,              # boolean mask for active joints
)
```

The Jacobian is an identity-like selector matrix that picks out the active
joints: `jacobian = np.eye(n_v)[mask]`.

**`create_ee_task()`** -- End-effector tracking.

```python
task = create_ee_task(
    jacobian_ee,            # (6, n_v) or (3, n_v)
    ee_current,             # current EE pose or position
    ee_target,              # desired EE pose or position
    ee_velocity,            # current EE twist or velocity
    ee_velocity_target=None,
    weight=1.0,
    priority=3,             # SECONDARY
    gain_p=100.0,
    gain_d=20.0,
    position_only=False,    # if True, use only position (3 DOF)
)
```

When `position_only=True`, the Jacobian is truncated to its first 3 rows and
only position components of the pose/velocity are used.

**`create_contact_constraint()`** -- Zero-velocity contact constraint.

```python
task = create_contact_constraint(
    jacobian_contact,       # (3, n_v) or (6, n_v)
    contact_velocity,       # should be ~0
    priority=0,             # HARD_CONSTRAINT
)
```

Creates an equality task with target zero acceleration, enforcing
$J_c \ddot{\mathbf{q}} = \mathbf{0}$ (assuming $\dot{J}_c \dot{\mathbf{q}} \approx \mathbf{0}$).

**`create_joint_limit_task()`** -- Repulsive joint-limit avoidance.

Only creates a task when at least one joint is within `margin` radians of its
limit. The repulsive acceleration pushes the joint away from the limit with gain
proportional to the proximity.

---

## 14.2 QP Solver

### 14.2.1 Problem Formulation

The `QPProblem` dataclass encodes a standard quadratic program:

$$
\min_{\mathbf{x}} \; \frac{1}{2} \mathbf{x}^T H \mathbf{x} + \mathbf{g}^T \mathbf{x}
$$

subject to:

$$
A_{\text{eq}} \mathbf{x} = \mathbf{b}_{\text{eq}}
$$

$$
\mathbf{lb} \leq A_{\text{ineq}} \mathbf{x} \leq \mathbf{ub}
$$

$$
\mathbf{x}_{\text{lb}} \leq \mathbf{x} \leq \mathbf{x}_{\text{ub}}
$$

| Field     | Shape                   | Description                  |
| --------- | ----------------------- | ---------------------------- |
| `H`       | `(n, n)`                | Hessian (must be PSD)        |
| `g`       | `(n,)`                  | Linear cost vector           |
| `A_eq`    | `(m_eq, n)` or `None`   | Equality constraint matrix   |
| `b_eq`    | `(m_eq,)` or `None`     | Equality constraint RHS      |
| `A_ineq`  | `(m_ineq, n)` or `None` | Inequality constraint matrix |
| `lb_ineq` | `(m_ineq,)` or `None`   | Inequality lower bounds      |
| `ub_ineq` | `(m_ineq,)` or `None`   | Inequality upper bounds      |
| `x_lb`    | `(n,)` or `None`        | Variable lower bounds        |
| `x_ub`    | `(n,)` or `None`        | Variable upper bounds        |

**Validation:**

- $H$ must be square.
- $\mathbf{g}$ must have the same dimension as $H$.
- If $A_{\text{eq}}$ is provided, $\mathbf{b}_{\text{eq}}$ is also required.
- Column counts of constraint matrices must equal $n$.

### 14.2.2 QPSolution

The `QPSolution` dataclass carries the solver output:

| Field        | Type                | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `success`    | `bool`              | Did the solver converge?       |
| `x`          | `NDArray` or `None` | Optimal solution               |
| `cost`       | `float`             | Optimal cost (`inf` if failed) |
| `iterations` | `int`               | Iteration count                |
| `solve_time` | `float`             | Wall-clock time in seconds     |
| `status`     | `str`               | Solver-specific message        |
| `dual_eq`    | `NDArray` or `None` | Equality dual variables        |
| `dual_ineq`  | `NDArray` or `None` | Inequality dual variables      |

### 14.2.3 Solver Implementations

**Abstract base:** `QPSolver` defines the interface with two methods:
`solve(problem) -> QPSolution` and `is_available() -> bool`.

**ScipyQPSolver** -- uses `scipy.optimize.minimize` with SLSQP or trust-constr:

- Constructs the objective function $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T H \mathbf{x} + \mathbf{g}^T \mathbf{x}$ and its gradient $\nabla f = H\mathbf{x} + \mathbf{g}$.
- Equality constraints become `{'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq}`.
- Double-sided inequality $\mathbf{lb} \leq A\mathbf{x} \leq \mathbf{ub}$ is split into two one-sided SciPy inequality constraints for each row: $A_i \mathbf{x} - lb_i \geq 0$ and $ub_i - A_i \mathbf{x} \geq 0$.
- Variable bounds are passed as a `scipy.optimize.Bounds` object.

**NullspaceQPSolver** -- a lightweight solver using the KKT system directly:

For equality-constrained QPs, the KKT conditions are:

$$
\begin{bmatrix}
H + \epsilon I & A_{\text{eq}}^T \\
A_{\text{eq}} & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf{x} \\
\boldsymbol{\lambda}
\end{bmatrix}
=
\begin{bmatrix}
-\mathbf{g} \\
\mathbf{b}_{\text{eq}}
\end{bmatrix}
$$

where $\epsilon$ is the regularization parameter (default $10^{-6}$). This
system is solved directly via `np.linalg.solve`. For unconstrained QPs, the
system reduces to $(H + \epsilon I)\mathbf{x} = -\mathbf{g}$.

The `NullspaceQPSolver` is always available (NumPy-only dependency) and is fast
for small problems but does not handle inequality constraints.

**`create_default_solver()`** returns `ScipyQPSolver` if available, otherwise
`NullspaceQPSolver`.

### 14.2.4 Priority-Based Task Stacking

In weighted mode, all tasks are combined into a single QP. The Hessian and
gradient accumulate contributions from each task:

$$
H \mathrel{+}= J_i^T W_i J_i, \qquad \mathbf{g} \mathrel{+}= -J_i^T W_i \ddot{\mathbf{x}}_{d,i}
$$

This is equivalent to minimizing:

$$
\sum_i \|J_i \ddot{\mathbf{q}} - \ddot{\mathbf{x}}_{d,i}\|_{W_i}^2
$$

### 14.2.5 Nullspace Projection for Hierarchical Control

In hierarchical (HQP) mode, tasks are grouped by priority and solved sequentially.
At each level $k$, the accumulated Jacobians from all higher-priority tasks
define a constraint subspace. The nullspace projector is:

$$
N_k = I - J_{\leq k-1}^+ J_{\leq k-1}
$$

where $J_{\leq k-1}$ is the vertical stack of all Jacobians at priorities
$0, 1, \ldots, k-1$, and $J^+$ denotes the Moore-Penrose pseudoinverse. The
current-level task Jacobian is projected:

$$
J_k^{\text{proj}} = J_k N_k
$$

This ensures that the optimization at level $k$ operates only in the subspace
that does not disturb higher-priority tasks. The general recursion for the
resulting torque at each level is:

$$
\boldsymbol{\tau}_k = \boldsymbol{\tau}_{k-1}
  + N_{k-1}\bigl(\boldsymbol{\tau}_k^* - \boldsymbol{\tau}_{k-1}\bigr)
$$

where $\boldsymbol{\tau}_k^*$ is the optimal torque for level $k$ in isolation.

---

## 14.3 Control Modes

### 14.3.1 ControlMode Enumeration

The `ControlMode` enum defines six operating modes:

| Mode         | Description                                                                       |
| ------------ | --------------------------------------------------------------------------------- |
| `TORQUE`     | Direct torque commands sent to joint actuators. Requires model-based computation. |
| `POSITION`   | Joint position setpoints tracked by low-level PD controllers.                     |
| `VELOCITY`   | Joint velocity commands.                                                          |
| `IMPEDANCE`  | Virtual spring-damper behavior at the end-effector.                               |
| `ADMITTANCE` | External forces modulate a reference trajectory.                                  |
| `HYBRID`     | Selective force/position control along different task-space axes.                 |

### 14.3.2 Impedance Control

Impedance control makes the robot behave as a virtual mass-spring-damper system.
The commanded torque is:

$$
\boldsymbol{\tau} = K_p (\mathbf{x}_d - \mathbf{x})
  + B_d (\dot{\mathbf{x}}_d - \dot{\mathbf{x}})
  + J^T \mathbf{f}_{\text{ff}}
$$

where:

- $K_p$ is the stiffness matrix (proportional gain in task space).
- $B_d$ is the damping matrix (derivative gain in task space).
- $\mathbf{x}_d, \dot{\mathbf{x}}_d$ are the desired position and velocity.
- $\mathbf{x}, \dot{\mathbf{x}}$ are the actual position and velocity.
- $\mathbf{f}_{\text{ff}}$ is a feedforward force term.
- $J$ is the task-space Jacobian.

Impedance control is passive by design: the robot absorbs energy from
interactions rather than injecting it, making it inherently safe for
human-robot collaboration.

### 14.3.3 Admittance Control

Admittance control inverts the impedance relationship. Rather than commanding
torques, it modifies the position reference based on measured external forces:

$$
M_d \ddot{\mathbf{x}} + B_d \dot{\mathbf{x}} + K_d \mathbf{x} = \mathbf{f}_{\text{ext}}
$$

where $M_d$, $B_d$, $K_d$ are the desired inertia, damping, and stiffness
matrices, and $\mathbf{f}_{\text{ext}}$ is the measured external force. The
solution $\mathbf{x}(t)$ is used as the reference for a position controller.

Admittance control is preferred when the robot has stiff low-level position
controllers that cannot be bypassed.

### 14.3.4 Computed Torque Control

Computed torque (also called inverse dynamics control) uses the full robot
dynamics model to achieve precise trajectory tracking:

$$
\boldsymbol{\tau} = M(\mathbf{q})\ddot{\mathbf{q}}_d
  + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}
  + \mathbf{g}(\mathbf{q})
$$

where:

- $M(\mathbf{q})$ is the joint-space inertia matrix.
- $C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}$ is the Coriolis/centrifugal
  force vector.
- $\mathbf{g}(\mathbf{q})$ is the gravity torque vector.
- $\ddot{\mathbf{q}}_d$ is the desired joint acceleration, typically computed
  from PD feedback:
  $\ddot{\mathbf{q}}_d = \ddot{\mathbf{q}}_{\text{ref}} + K_p (\mathbf{q}_d - \mathbf{q}) + K_d (\dot{\mathbf{q}}_d - \dot{\mathbf{q}})$.

When applied perfectly, the closed-loop system reduces to
$\ddot{\mathbf{q}} = \ddot{\mathbf{q}}_d$ (linear decoupled dynamics).

### 14.3.5 Operational Space Control

Operational space control extends computed torque to task space, enabling
Cartesian control of end-effectors without requiring explicit inverse kinematics:

$$
\boldsymbol{\tau} = J^T\bigl[\Lambda(\ddot{\mathbf{x}}_d - \dot{J}\dot{\mathbf{q}})
  + \boldsymbol{\mu} + \mathbf{p}\bigr]
$$

where:

- $\Lambda = (J M^{-1} J^T)^{-1}$ is the operational-space inertia matrix.
- $\boldsymbol{\mu} = \Lambda J M^{-1} C \dot{\mathbf{q}} - \Lambda \dot{J}\dot{\mathbf{q}}$
  is the operational-space Coriolis/centrifugal force.
- $\mathbf{p} = \Lambda J M^{-1} \mathbf{g}$ is the operational-space gravity
  compensation.
- $\dot{J}\dot{\mathbf{q}}$ accounts for the time derivative of the Jacobian.

The operational-space inertia $\Lambda$ provides the mapping between task-space
accelerations and task-space forces. When the Jacobian is not full rank
(kinematic singularity), $\Lambda$ becomes ill-conditioned and regularization
or damped pseudoinverses are needed.

### 14.3.6 The WholeBodyController Class

`WholeBodyController` is the top-level orchestrator. It manages a task list,
builds the QP, and extracts the optimal joint accelerations and torques.

**Initialization:**

```python
wbc = WholeBodyController(
    engine,                 # RoboticsCapable physics engine
    config=WBCConfig(
        dt=0.001,
        regularization=1e-6,
        torque_limits=None,
        velocity_limits=None,
        acceleration_limits=None,
        contact_force_regularization=1e-4,
        use_hierarchical=True,
    ),
    solver=None,            # Uses create_default_solver() if None
)
```

**Task management:**

- `add_task(task)` -- adds a task, sorted by priority. Rejects duplicate names.
- `remove_task(name)` -- removes by name.
- `get_task(name)` -- retrieves by name.
- `clear_tasks()` -- removes all.
- `set_contact_jacobians(jacobians)` -- provides contact Jacobians for the
  equation-of-motion constraint.

**Solving:**

```python
solution = wbc.solve()  # returns WBCSolution
```

The solver builds the following QP:

**Decision variables:** $\mathbf{x} = [\ddot{\mathbf{q}}^T, \mathbf{f}_c^T]^T$
where $\ddot{\mathbf{q}} \in \mathbb{R}^{n_v}$ are joint accelerations and
$\mathbf{f}_c \in \mathbb{R}^{3 n_c}$ are contact forces.

**Objective (weighted mode):**

$$
\min_{\mathbf{x}} \sum_i \|J_i \ddot{\mathbf{q}} - \ddot{\mathbf{x}}_{d,i}\|_{W_i}^2
  + \epsilon_q \|\ddot{\mathbf{q}}\|^2
  + \epsilon_f \|\mathbf{f}_c\|^2
$$

**Equation of motion constraint:**

$$
M(\mathbf{q}) \ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}
  + \mathbf{g}(\mathbf{q})
  = \boldsymbol{\tau} + J_c^T \mathbf{f}_c
$$

Rearranged for the QP: $M \ddot{\mathbf{q}} - J_c^T \mathbf{f}_c = -(\text{bias forces})$.

**Friction cone constraints:** For each contact $i$, four pyramid constraints
plus one non-negative normal force constraint:

$$
f_{t,x} \leq \mu f_n, \quad -f_{t,x} \leq \mu f_n, \quad
f_{t,y} \leq \mu f_n, \quad -f_{t,y} \leq \mu f_n, \quad
f_n \geq 0
$$

**Variable bounds:** acceleration limits, velocity limits (translated to
acceleration via $\ddot{q}_{\max} = (v_{\max} - \dot{q})/\Delta t$), and
contact force bounds ($\pm 10000$ N for numerical stability).

The `WBCSolution` contains:

| Field                 | Description                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| `joint_accelerations` | Optimal $\ddot{\mathbf{q}}$                                                 |
| `joint_torques`       | $\boldsymbol{\tau} = M\ddot{\mathbf{q}} + \text{bias} - J_c^T \mathbf{f}_c$ |
| `contact_forces`      | Optimal $\mathbf{f}_c$                                                      |
| `task_errors`         | Per-task weighted squared error                                             |

---

# Chapter 15: Locomotion

The locomotion module (`src/robotics/locomotion/`) implements the gait management
pipeline for bipedal robots: gait parameterization, finite-state-machine
transitions, footstep planning, and balance analysis via the Zero Moment Point.

---

## 15.1 Gait Types and Parameters

### 15.1.1 GaitType Enumeration

The `GaitType` enum (`src/robotics/locomotion/gait_types.py`) enumerates the
supported gait patterns:

| Member   | Description                                              |
| -------- | -------------------------------------------------------- |
| `STAND`  | Static standing with both feet on the ground.            |
| `WALK`   | Normal walking gait with a double-support phase.         |
| `TROT`   | Trot gait using diagonal leg pairs (quadruped or biped). |
| `RUN`    | Running gait with a flight phase (no ground contact).    |
| `CRAWL`  | Slow, maximally stable crawl.                            |
| `BOUND`  | Bounding gait (synchronous front/rear pairs).            |
| `GALLOP` | Galloping with asymmetric leg phasing.                   |

### 15.1.2 GaitPhase Enumeration

`GaitPhase` describes the phase within a single gait cycle:

| Member           | Description                         |
| ---------------- | ----------------------------------- |
| `DOUBLE_SUPPORT` | Both feet are on the ground.        |
| `LEFT_SUPPORT`   | Only the left foot is in contact.   |
| `RIGHT_SUPPORT`  | Only the right foot is in contact.  |
| `FLIGHT`         | Both feet are in the air (running). |
| `LEFT_SWING`     | Left leg is in swing phase.         |
| `RIGHT_SWING`    | Right leg is in swing phase.        |

### 15.1.3 LegState Enumeration

`LegState` tracks the state of an individual leg through its cycle:

| Member          | Description                        |
| --------------- | ---------------------------------- |
| `STANCE`        | Foot on ground, supporting weight. |
| `SWING`         | Foot in air, moving forward.       |
| `EARLY_CONTACT` | Just touched down.                 |
| `LATE_CONTACT`  | About to lift off.                 |
| `LOADING`       | Weight shifting onto this leg.     |
| `UNLOADING`     | Weight shifting off this leg.      |

### 15.1.4 GaitParameters Dataclass

The `GaitParameters` dataclass bundles all parameters that define a gait:

| Field                  | Type       | Default  | Description                                         |
| ---------------------- | ---------- | -------- | --------------------------------------------------- |
| `gait_type`            | `GaitType` | `WALK`   | Selected gait pattern.                              |
| `step_length`          | `float`    | `0.3`    | Forward step length in meters.                      |
| `step_width`           | `float`    | `0.2`    | Lateral step width in meters.                       |
| `step_height`          | `float`    | `0.05`   | Maximum foot lift height in meters.                 |
| `step_duration`        | `float`    | `0.5`    | Total duration of one step in seconds.              |
| `double_support_ratio` | `float`    | `0.2`    | Fraction of step spent in double support.           |
| `swing_height_profile` | `str`      | `"sine"` | Height profile during swing (`"sine"` or `"trap"`). |
| `com_height`           | `float`    | `0.9`    | Target center-of-mass height in meters.             |
| `max_foot_velocity`    | `float`    | `1.0`    | Maximum foot velocity in m/s.                       |
| `settling_time`        | `float`    | `0.1`    | Time to settle after a step in seconds.             |

**Construction-time validation:**

- `step_length >= 0`
- `step_width >= 0`
- `step_height >= 0`
- `step_duration > 0`
- `0 <= double_support_ratio <= 1`
- `com_height > 0`

**Derived properties:**

- **Swing duration:**

$$
t_{\text{swing}} = t_{\text{step}} \times (1 - r_{\text{ds}})
$$

where $t_{\text{step}}$ is `step_duration` and $r_{\text{ds}}$ is
`double_support_ratio`.

- **Double-support duration:**

$$
t_{\text{ds}} = t_{\text{step}} \times r_{\text{ds}}
$$

- **Step frequency:**

$$
f_{\text{step}} = \frac{1}{t_{\text{step}}}
$$

**Factory functions** provide pre-configured parameter sets:

- `create_walk_parameters(step_length=0.3, step_duration=0.5, com_height=0.9)` --
  Standard walking with 20% double support.
- `create_run_parameters(step_length=0.6, step_duration=0.3, com_height=0.85)` --
  Running with zero double support ratio (flight phase).
- `create_stand_parameters(step_width=0.2, com_height=0.9)` -- Standing with
  100% double support, zero step length and height.

---

## 15.2 Gait State Machine

### 15.2.1 Events

The `GaitEvent` enum defines the events that can trigger state transitions:

| Event             | Description                              |
| ----------------- | ---------------------------------------- |
| `STEP_COMPLETE`   | The current step has finished.           |
| `FOOT_CONTACT`    | A foot has touched the ground.           |
| `FOOT_LIFTOFF`    | A foot has left the ground.              |
| `BALANCE_LOST`    | The balance criterion has been violated. |
| `STOP_REQUESTED`  | The user has requested a stop.           |
| `START_REQUESTED` | The user has requested to start walking. |
| `SPEED_CHANGE`    | The velocity command has changed.        |
| `EMERGENCY_STOP`  | An emergency stop has been triggered.    |

### 15.2.2 GaitState

The `GaitState` dataclass holds the complete instantaneous state of the gait
controller:

| Field              | Type           | Default                                |
| ------------------ | -------------- | -------------------------------------- |
| `gait_type`        | `GaitType`     | `STAND`                                |
| `phase`            | `GaitPhase`    | `DOUBLE_SUPPORT`                       |
| `support_state`    | `SupportState` | `DOUBLE_SUPPORT_CENTERED`              |
| `phase_time`       | `float`        | `0.0` -- time elapsed in current phase |
| `cycle_time`       | `float`        | `0.0` -- time elapsed in current cycle |
| `step_count`       | `int`          | `0` -- total steps taken               |
| `is_walking`       | `bool`         | `False`                                |
| `stance_foot`      | `str`          | `"both"`                               |
| `next_stance_foot` | `str`          | `"left"`                               |

### 15.2.3 The GaitStateMachine Class

The FSM manages phase transitions for bipedal walking. The state diagram for
walking is:

```
DOUBLE_SUPPORT --> RIGHT_SWING --> DOUBLE_SUPPORT --> LEFT_SWING --> ...
```

**Initialization:**

```python
gait = GaitStateMachine(parameters=GaitParameters())
```

**Starting and stopping:**

- `start_walking()` -- transitions from standing to walking. Starts in
  `DOUBLE_SUPPORT` with `DOUBLE_SUPPORT_LEFT_LEADING` support state, preparing
  to lift the right foot. Invokes the `gait_change` callbacks.

- `stop_walking()` -- transitions back to standing. Only takes effect during
  `DOUBLE_SUPPORT` phase to avoid stopping mid-swing.

- `emergency_stop()` -- immediately forces the state to `DOUBLE_SUPPORT` /
  `STAND` regardless of current phase.

**Updating:**

```python
state = gait.update(dt=0.001)
```

Each call advances `phase_time` and `cycle_time` by `dt`. If the accumulated
phase time exceeds the current phase duration, a transition is triggered. Excess
time carries over to the new phase. A safety limit of 20 iterations per update
prevents infinite transition loops.

**Phase transitions:**

| From             | To               | Condition                     | Side effects                                    |
| ---------------- | ---------------- | ----------------------------- | ----------------------------------------------- |
| `DOUBLE_SUPPORT` | `RIGHT_SWING`    | `next_stance_foot == "left"`  | `stance_foot = "left"`                          |
| `DOUBLE_SUPPORT` | `LEFT_SWING`     | `next_stance_foot == "right"` | `stance_foot = "right"`                         |
| `RIGHT_SWING`    | `DOUBLE_SUPPORT` | Phase timer expires           | `step_count += 1`, `next_stance_foot = "left"`  |
| `LEFT_SWING`     | `DOUBLE_SUPPORT` | Phase timer expires           | `step_count += 1`, `next_stance_foot = "right"` |

Phase durations are computed from the gait parameters:

- `DOUBLE_SUPPORT` duration: `step_duration * double_support_ratio`
- Swing duration: `step_duration * (1 - double_support_ratio)`

**Callbacks:**

Three callback channels are supported: `"phase_change"`, `"step_complete"`, and
`"gait_change"`. Register with:

```python
gait.register_callback("step_complete", lambda state, event: print(f"Step {state.step_count}"))
```

Callbacks receive the current `GaitState` and the triggering `GaitEvent`.
Exceptions in callbacks are silently caught to prevent breaking the state
machine.

**External events:**

`handle_event(event)` processes external sensor events:

- `FOOT_CONTACT` -- forces transition from swing to double support (early
  touchdown).
- `FOOT_LIFTOFF` -- forces transition from double support to swing (early
  liftoff).

**Phase progress:**

`phase_progress` returns a float in $[0, 1]$ indicating how far through the
current phase the system has progressed:

$$
p = \min\left(1, \frac{t_{\text{phase}}}{T_{\text{phase}}}\right)
$$

**Foot trajectory phase:**

`get_foot_trajectory_phase(foot)` returns $p \in [0, 1]$ for trajectory
interpolation. During swing, it equals `phase_progress`; during stance, it
returns $1.0$.

---

## 15.3 Footstep Planning

### 15.3.1 Footstep and FootstepPlan Data Structures

A `Footstep` represents a single foot placement:

| Field         | Type                    | Description                          |
| ------------- | ----------------------- | ------------------------------------ |
| `position`    | `NDArray[float64]` (3,) | World-frame position of foot center. |
| `orientation` | `NDArray[float64]` (4,) | Quaternion $[w, x, y, z]$.           |
| `foot`        | `str`                   | `"left"` or `"right"`.               |
| `step_index`  | `int`                   | Sequential index in plan.            |
| `is_support`  | `bool`                  | Whether this is a support position.  |
| `timing`      | `float`                 | Planned start time of this step.     |
| `duration`    | `float`                 | Duration allocated for this step.    |

The `yaw` property extracts the heading angle from the quaternion using the
standard atan2 decomposition:

$$
\psi = \arctan2\bigl(2(w z + x y),\; 1 - 2(y^2 + z^2)\bigr)
$$

A `FootstepPlan` is an iterable container of `Footstep` objects with:

- `n_steps` -- total number of steps.
- `total_duration` -- sum of all step durations.
- `get_footsteps_for_foot(foot)` -- filter by foot.
- `get_footstep_at_time(t)` -- find the active footstep at time $t$.

### 15.3.2 The FootstepPlanner Class

```python
planner = FootstepPlanner(
    parameters=GaitParameters(),
    max_step_length=0.5,     # kinematic limit [m]
    max_step_width=0.4,      # kinematic limit [m]
    max_step_rotation=0.5,   # kinematic limit [rad]
)
```

**Yaw-to-quaternion conversion.** The internal `_yaw_to_quat` method converts a
yaw angle $\psi$ to a quaternion representing pure rotation about the $z$-axis:

$$
\mathbf{q} = \bigl[\cos(\psi/2),\; 0,\; 0,\; \sin(\psi/2)\bigr]
$$

**Angle normalization.** The `_normalize_angle` method maps an angle to the
range $[-\pi, \pi]$.

### 15.3.3 Planning to a Goal

```python
plan = planner.plan_to_goal(
    start=np.zeros(3),
    goal=np.array([2.0, 0.0, 0.0]),
    start_yaw=0.0,
    goal_yaw=None,      # defaults to direction toward goal
    start_foot="left",
)
```

The algorithm:

1. Compute the direction vector and distance in the $xy$-plane.
2. If distance $< 10^{-3}$ m, return an empty plan (already at goal).
3. If `goal_yaw` is `None`, set it to $\arctan2(\Delta y, \Delta x)$.
4. Determine the number of steps:
   $n = \max\bigl(1, \lceil d / \min(l_{\text{step}}, l_{\max})\rceil\bigr)$.
5. For each step $i$, interpolate position along the straight path at
   fraction $t = (i+1)/n$ and orientation by linear interpolation of yaw.
6. Add lateral offsets to place left and right feet at
   $\pm w_{\text{nom}}/2$ perpendicular to the walking direction.
7. Assign timing: each step starts at $i \cdot t_{\text{step}}$.

### 15.3.4 Planning from Velocity

```python
plan = planner.plan_from_velocity(
    current_position=np.zeros(3),
    current_yaw=0.0,
    velocity_command=np.array([0.5, 0.0, 0.1]),  # [vx, vy, omega]
    n_steps=4,
    start_foot="left",
)
```

The velocity command $[v_x, v_y, \omega]$ is integrated over one step duration to
compute the displacement:

$$
\Delta x = v_x \cdot \Delta t, \qquad
\Delta y = v_y \cdot \Delta t, \qquad
\Delta\psi = \omega \cdot \Delta t
$$

Each displacement is clamped to the kinematic limits:

$$
|\Delta x| \leq l_{\max}, \qquad
|\Delta y| \leq w_{\max}, \qquad
|\Delta\psi| \leq \psi_{\max}
$$

The yaw rotation is applied in two halves (before and after position update) for
smoother curvature. The displacement is then transformed to the world frame:

$$
\Delta x_w = \cos\psi \cdot \Delta x - \sin\psi \cdot \Delta y
$$

$$
\Delta y_w = \sin\psi \cdot \Delta x + \cos\psi \cdot \Delta y
$$

Lateral foot offsets are computed relative to the current heading:

- Left foot: offset $= (-\sin\psi, \cos\psi) \cdot w_{\text{nom}}/2$
- Right foot: offset $= (\sin\psi, -\cos\psi) \cdot w_{\text{nom}}/2$

### 15.3.5 Planning an In-Place Turn

```python
plan = planner.plan_in_place_turn(
    current_position=np.zeros(3),
    current_yaw=0.0,
    target_yaw=np.pi / 2,
    start_foot="left",
)
```

The total rotation $\Delta\psi = \text{normalize}(\psi_{\text{target}} - \psi_{\text{current}})$
is divided into steps of at most `max_step_rotation`:

$$
n = \max\bigl(1, \lceil|\Delta\psi| / \psi_{\max}\rceil\bigr),
\qquad
\delta\psi = \Delta\psi / n
$$

Each step rotates the feet around the current position by $\delta\psi$, with
alternating left/right placement.

---

## 15.4 Zero Moment Point

### 15.4.1 ZMP Theory

The Zero Moment Point is the point on the ground plane where the net moment of
the inertial forces and gravity has zero horizontal components. For a bipedal
robot to remain dynamically balanced, the ZMP must stay within the support
polygon.

### 15.4.2 ZMP Computation

The `ZMPComputer` class (`src/robotics/locomotion/zmp_computer.py`) computes the
ZMP from the robot's centroidal dynamics.

The ZMP position on a flat ground plane at height $z_g$ is:

$$
x_{\text{ZMP}} = x_{\text{CoM}} -
  \frac{z_{\text{CoM}} - z_g}{\ddot{z}_{\text{CoM}} + g}
  \left(\ddot{x}_{\text{CoM}} + \frac{\dot{L}_y}{m}\right)
$$

$$
y_{\text{ZMP}} = y_{\text{CoM}} -
  \frac{z_{\text{CoM}} - z_g}{\ddot{z}_{\text{CoM}} + g}
  \left(\ddot{y}_{\text{CoM}} - \frac{\dot{L}_x}{m}\right)
$$

where:

- $(x_{\text{CoM}}, y_{\text{CoM}}, z_{\text{CoM}})$ is the center of mass position.
- $(\ddot{x}_{\text{CoM}}, \ddot{y}_{\text{CoM}}, \ddot{z}_{\text{CoM}})$ is the
  CoM acceleration.
- $g = 9.81 \text{ m/s}^2$ is gravitational acceleration.
- $m$ is the total robot mass.
- $\dot{L}_x, \dot{L}_y$ are the rates of change of angular momentum about the
  $x$ and $y$ axes at the CoM.

The denominator $\ddot{z}_{\text{CoM}} + g$ represents the total vertical
acceleration. When this is near zero (free fall), the ZMP is undefined and the
method returns an invalid result with `is_valid=False`.

**Implementation.** The `compute_zmp` method accepts optional overrides for all
state quantities. If not provided:

- `com_position` is queried from the engine via `get_com_position()` (requires
  `HumanoidCapable`).
- `com_acceleration` defaults to zero (quasi-static assumption).
- `angular_momentum_rate` defaults to zero.

**Free-fall detection.** If $|\ddot{z}_{\text{CoM}} + g| < 10^{-6}$, the robot
is in free fall. The returned `ZMPResult` has `is_valid=False` and
`total_normal_force=0`.

### 15.4.3 ZMPResult

The `ZMPResult` dataclass contains:

| Field                | Type                    | Description                                       |
| -------------------- | ----------------------- | ------------------------------------------------- |
| `zmp_position`       | `NDArray[float64]` (3,) | ZMP on ground plane.                              |
| `cop_position`       | `NDArray[float64]` (3,) | Center of Pressure (same as ZMP for flat ground). |
| `is_valid`           | `bool`                  | Whether ZMP is within support polygon.            |
| `support_margin`     | `float`                 | Signed distance to boundary (negative = outside). |
| `total_normal_force` | `float`                 | Total vertical ground reaction force.             |
| `ground_height`      | `float`                 | Height of ground plane.                           |

### 15.4.4 Capture Point and Divergent Component of Motion

The **Capture Point** (also called the Instantaneous Capture Point or Divergent
Component of Motion, DCM) is the point on the ground where the robot should
place its foot to come to a complete stop from its current state:

$$
\mathbf{p}_{\text{CP}} = \mathbf{p}_{\text{CoM}} + \frac{\dot{\mathbf{p}}_{\text{CoM}}}{\omega_0}
$$

where $\omega_0 = \sqrt{g / z_{\text{CoM}}}$ is the natural frequency of the
Linear Inverted Pendulum Model (LIPM).

The `compute_capture_point` and `compute_dcm` methods both compute this
quantity (they are equivalent for the LIPM). Only the horizontal components
($x, y$) are meaningful; the $z$ component is set to the ground height.

**Physical interpretation.** If the capture point lies outside the support
polygon, the robot cannot stop without taking a step. The distance between the
capture point and the support polygon boundary indicates the urgency of stepping.

```python
zmp_computer = ZMPComputer(engine, ground_height=0.0)
capture = zmp_computer.compute_capture_point()
# capture[0:2] gives the horizontal capture point
```

### 15.4.5 Stability Margin

The `compute_stability_margin(zmp_position, support_polygon)` method returns the
**signed distance** from the ZMP to the nearest edge of the support polygon:

- **Positive:** ZMP is inside the polygon, with the value indicating the
  distance to the closest boundary.
- **Negative:** ZMP is outside the polygon, with the magnitude indicating the
  distance to the closest boundary.

The boundary distance is computed by finding the minimum point-to-segment
distance over all polygon edges. For a polygon with vertices
$\mathbf{v}_0, \mathbf{v}_1, \ldots, \mathbf{v}_{n-1}$, the distance from
point $\mathbf{p}$ to edge $(\mathbf{v}_i, \mathbf{v}_{i+1 \bmod n})$ is:

$$
d_i = \|\mathbf{p} - \text{closest}(\mathbf{p}, \overline{\mathbf{v}_i \mathbf{v}_{i+1}})\|
$$

where the closest point on the segment is:

$$
\text{closest} = \mathbf{v}_i + \text{clamp}\left(\frac{(\mathbf{p} - \mathbf{v}_i) \cdot (\mathbf{v}_{i+1} - \mathbf{v}_i)}{\|\mathbf{v}_{i+1} - \mathbf{v}_i\|^2}, 0, 1\right) (\mathbf{v}_{i+1} - \mathbf{v}_i)
$$

The stability margin is $\min_i d_i$ with sign determined by the point-in-polygon
test.

---

# Chapter 16: Sensing

The sensing module (`src/robotics/sensing/`) provides configurable sensor
simulations for realistic noise modeling. This is critical for sim-to-real
transfer: controllers developed in simulation must be robust to the noise
characteristics of physical sensors.

---

## 16.1 Noise Models

The noise model framework (`src/robotics/sensing/noise_models.py`) defines an
abstract `NoiseModel` base class with two methods:

- `apply(signal) -> noisy_signal` -- transforms a clean signal array into a
  noisy one.
- `reset()` -- resets any internal state (e.g., accumulated bias).

All noise models preserve the shape of the input: if `signal.shape == (3,)`,
the output also has shape `(3,)`.

### 16.1.1 GaussianNoise

Additive white Gaussian noise:

$$
y = x + \mathcal{N}(\mu, \sigma^2)
$$

| Parameter | Type          | Default | Description                      |
| --------- | ------------- | ------- | -------------------------------- |
| `std`     | `float`       | `0.01`  | Standard deviation $\sigma$.     |
| `mean`    | `float`       | `0.0`   | Bias $\mu$.                      |
| `seed`    | `int or None` | `None`  | Random seed for reproducibility. |

The noise is drawn independently for each element of the signal array using
NumPy's `default_rng`. Calling `reset()` re-seeds the generator, producing the
same noise sequence again.

```python
from src.robotics.sensing.noise_models import GaussianNoise

noise = GaussianNoise(std=0.1, mean=0.0, seed=42)
clean = np.array([1.0, 2.0, 3.0])
noisy = noise.apply(clean)  # each element perturbed by N(0, 0.01)
```

### 16.1.2 BrownianNoise (Bias Drift)

Models slowly-varying sensor bias that accumulates over time via a random walk.
This is the dominant error source in MEMS gyroscopes and accelerometers over
long time horizons.

**Update equation:**

$$
b_{t+1} = \text{clip}\bigl(b_t + \mathcal{N}(0, \sigma_{\text{drift}}^2),\; [-b_{\max}, b_{\max}]\bigr)
$$

$$
y = x + b_t
$$

| Parameter      | Type          | Default | Description                                     |
| -------------- | ------------- | ------- | ----------------------------------------------- |
| `drift_rate`   | `float`       | `0.001` | Std of drift increment $\sigma_{\text{drift}}$. |
| `initial_bias` | `float`       | `0.0`   | Starting bias value.                            |
| `max_bias`     | `float`       | `1.0`   | Clipping bounds $b_{\max}$.                     |
| `seed`         | `int or None` | `None`  | Random seed.                                    |

The bias is a scalar that applies uniformly to all elements of the signal. The
clipping prevents unbounded growth. The `current_bias` property provides
read access to the current bias state for monitoring.

```python
from src.robotics.sensing.noise_models import BrownianNoise

drift = BrownianNoise(drift_rate=0.0001, max_bias=0.05, seed=42)
# After many calls to drift.apply(...), the bias will random-walk
# within [-0.05, 0.05]
```

### 16.1.3 QuantizationNoise

Models the discrete nature of analog-to-digital conversion:

$$
y = \text{round}\!\left(\frac{x - x_0}{r}\right) \cdot r + x_0
$$

where $r$ is the quantization resolution (least significant bit value) and $x_0$
is a DC offset.

| Parameter    | Type    | Default | Description                 |
| ------------ | ------- | ------- | --------------------------- |
| `resolution` | `float` | `0.001` | Quantization step size $r$. |
| `offset`     | `float` | `0.0`   | Offset before quantization. |

This noise model is stateless -- `reset()` is a no-op.

For a 16-bit ADC with a full-scale range of $R$, the resolution is:

$$
r = \frac{R}{2^{16}} = \frac{R}{65536}
$$

```python
from src.robotics.sensing.noise_models import QuantizationNoise

quant = QuantizationNoise(resolution=0.001)
clean = np.array([1.2345, 2.3456])
quantized = quant.apply(clean)  # [1.235, 2.346] approximately
```

### 16.1.4 BandwidthLimitedNoise

Models the finite bandwidth of physical sensors using a first-order IIR
(infinite impulse response) low-pass filter:

$$
y_t = \alpha x_t + (1 - \alpha) y_{t-1}
$$

where the filter coefficient is derived from the cutoff frequency:

$$
\tau = \frac{1}{2\pi f_c}, \qquad
\alpha = \frac{\Delta t}{\tau + \Delta t}
$$

Here $f_c$ is the cutoff frequency, $\Delta t = 1/f_s$ is the sample period,
and $\tau$ is the time constant.

| Parameter          | Type    | Default  | Description                           |
| ------------------ | ------- | -------- | ------------------------------------- |
| `cutoff_frequency` | `float` | `100.0`  | Filter cutoff frequency $f_c$ [Hz].   |
| `sample_rate`      | `float` | `1000.0` | Sampling rate $f_s$ [Hz].             |
| `order`            | `int`   | `2`      | Filter order (currently first-order). |

The filter state (`_filter_state`) is initialized to the first input signal and
accumulates thereafter. Calling `reset()` clears the state.

**Frequency response.** For a first-order IIR filter, the magnitude response at
frequency $f$ is:

$$
|H(f)| = \frac{\alpha}{\sqrt{1 - 2(1-\alpha)\cos(2\pi f / f_s) + (1-\alpha)^2}}
$$

At the cutoff frequency, the attenuation is approximately $-3$ dB.

```python
from src.robotics.sensing.noise_models import BandwidthLimitedNoise

bw_filter = BandwidthLimitedNoise(cutoff_frequency=50.0, sample_rate=1000.0)
for t in range(1000):
    signal = np.array([np.sin(2 * np.pi * 200 * t / 1000)])  # 200 Hz
    filtered = bw_filter.apply(signal)
    # High-frequency content will be attenuated
```

### 16.1.5 CompositeNoise

Chains multiple noise models in sequence:

$$
y = N_k\bigl(\cdots N_2\bigl(N_1(x)\bigr)\cdots\bigr)
$$

| Parameter | Type               | Default | Description            |
| --------- | ------------------ | ------- | ---------------------- |
| `models`  | `list[NoiseModel]` | `[]`    | Noise models in order. |

The `apply` method iterates through `models` in order, feeding each output into
the next model's input. The `reset` method resets all constituent models.

The `add_model(model)` method appends a noise model to the chain.

```python
from src.robotics.sensing.noise_models import CompositeNoise, GaussianNoise, BrownianNoise

composite = CompositeNoise(models=[
    BrownianNoise(drift_rate=0.0001),
    GaussianNoise(std=0.01),
])
noisy = composite.apply(clean_signal)
```

### 16.1.6 Factory: create_realistic_sensor_noise()

`create_realistic_sensor_noise(noise_std, bias_drift_rate, quantization_bits,
signal_range, seed)` creates a `CompositeNoise` with three stages:

1. **BrownianNoise** -- bias drift with rate `bias_drift_rate`.
2. **GaussianNoise** -- white noise with std `noise_std`.
3. **QuantizationNoise** -- resolution $r = R / 2^b$ where $R$ is `signal_range`
   and $b$ is `quantization_bits`.

---

## 16.2 IMU Sensor

The `IMUSensor` class (`src/robotics/sensing/imu_sensor.py`) simulates a 6-axis
inertial measurement unit with 3-axis accelerometer and 3-axis gyroscope.

### 16.2.1 IMUSensorConfig

| Field              | Type                    | Default         | Description                                            |
| ------------------ | ----------------------- | --------------- | ------------------------------------------------------ |
| `sensor_id`        | `str`                   | `"imu"`         | Unique identifier.                                     |
| `accel_range`      | `float`                 | `160.0`         | Max measurable acceleration [m/s^2] (~16g).            |
| `gyro_range`       | `float`                 | `35.0`          | Max measurable angular velocity [rad/s] (~2000 deg/s). |
| `accel_noise_std`  | `float`                 | `0.01`          | Accelerometer white noise std [m/s^2].                 |
| `gyro_noise_std`   | `float`                 | `0.001`         | Gyroscope white noise std [rad/s].                     |
| `accel_bias_drift` | `float`                 | `0.0001`        | Accelerometer bias drift rate [m/s^2/step].            |
| `gyro_bias_drift`  | `float`                 | `0.00001`       | Gyroscope bias drift rate [rad/s/step].                |
| `gravity`          | `NDArray[float64]` (3,) | `[0, 0, -9.81]` | Gravity vector in world frame.                         |
| `cutoff_frequency` | `float`                 | `200.0`         | Sensor bandwidth [Hz].                                 |
| `sample_rate`      | `float`                 | `1000.0`        | Sampling rate [Hz].                                    |
| `seed`             | `int or None`           | `None`          | Random seed.                                           |

### 16.2.2 Noise Pipeline

Each measurement channel (accelerometer and gyroscope) passes through a
composite noise model containing:

1. **BrownianNoise** -- bias drift with `max_bias = noise_std * 10`.
2. **GaussianNoise** -- white noise.

After the noise stage, a `BandwidthLimitedNoise` filter is applied to model the
sensor's finite bandwidth. Finally, the result is clipped to the sensor's
measurement range:

$$
y_{\text{accel}} = \text{clip}\bigl(\text{filter}(\text{noise}(a_{\text{true}})),\; -a_{\max},\; a_{\max}\bigr)
$$

### 16.2.3 Reading the Sensor

```python
reading = imu.read(
    linear_accel=np.array([0.0, 0.0, 9.81]),
    angular_vel=np.array([0.0, 0.0, 0.1]),
    timestamp=0.001,
    include_orientation=True,
)
# reading.linear_acceleration  -- noisy accel (3,)
# reading.angular_velocity     -- noisy gyro (3,)
# reading.orientation          -- quaternion (4,) or None
```

The `read` method:

1. Validates that inputs are 3-vectors.
2. Applies noise to acceleration and angular velocity independently.
3. Applies bandwidth filters.
4. Clips to sensor ranges.
5. If `include_orientation=True` and a previous timestamp exists, integrates
   the gyroscope to update the orientation estimate.

### 16.2.4 Quaternion Integration

The orientation is updated by integrating the angular velocity using first-order
quaternion kinematics. Given angular velocity $\boldsymbol{\omega}$ and timestep
$\Delta t$:

1. Compute the rotation angle $\theta = \|\boldsymbol{\omega}\| \cdot \Delta t$
   and the rotation axis $\hat{\boldsymbol{\omega}} = \boldsymbol{\omega} / \|\boldsymbol{\omega}\|$.

2. Form the incremental rotation quaternion:

$$
\Delta\mathbf{q} = \left[\cos\!\left(\frac{\theta}{2}\right),\;
  \sin\!\left(\frac{\theta}{2}\right) \hat{\boldsymbol{\omega}}\right]
$$

3. Update via quaternion multiplication:

$$
\mathbf{q}_{\text{new}} = \mathbf{q}_{\text{old}} \otimes \Delta\mathbf{q}
$$

4. Normalize to maintain unit length:

$$
\mathbf{q}_{\text{new}} \leftarrow \frac{\mathbf{q}_{\text{new}}}{\|\mathbf{q}_{\text{new}}\|}
$$

The quaternion multiplication $\mathbf{q}_1 \otimes \mathbf{q}_2$ with
$\mathbf{q}_i = [w_i, x_i, y_i, z_i]$ is computed via the Hamilton product:

$$
\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix}
w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
\end{bmatrix}
$$

If $\|\boldsymbol{\omega}\| < 10^{-10}$, the integration is skipped to avoid
division by zero.

### 16.2.5 Gravity in Sensor Frame

`get_gravity_in_sensor_frame()` transforms the world-frame gravity vector into
the current sensor frame using the orientation quaternion:

$$
\mathbf{g}' = \mathbf{q}^{-1} \otimes [0, \mathbf{g}] \otimes \mathbf{q}
$$

where $\mathbf{q}^{-1} = [w, -x, -y, -z]$ is the quaternion conjugate (inverse
for unit quaternions). The operation `_rotate_vector_by_quaternion(v, q)` wraps
the 3-vector as a pure quaternion $[0, v_x, v_y, v_z]$, performs the
sandwich product $q \otimes v \otimes q^{-1}$, and extracts the vector part.

This is essential for separating gravitational acceleration from linear
acceleration in the sensor reading:

$$
a_{\text{measured}} = a_{\text{linear}} + \mathbf{g}'
$$

### 16.2.6 Quality Presets

`create_realistic_imu(sensor_id, quality, seed)` provides three quality levels:

| Quality        | `accel_noise_std` | `gyro_noise_std` | `accel_bias_drift` | `gyro_bias_drift` |
| -------------- | ----------------- | ---------------- | ------------------ | ----------------- |
| `"mems"`       | 0.1               | 0.01             | 0.001              | 0.0001            |
| `"industrial"` | 0.01              | 0.001            | 0.0001             | 0.00001           |
| `"tactical"`   | 0.001             | 0.0001           | 0.00001            | 0.000001          |

Each tier represents roughly an order-of-magnitude improvement in all noise
parameters, reflecting the progression from consumer MEMS devices through
industrial units to tactical/navigation-grade sensors.

`create_ideal_imu(sensor_id)` sets all noise parameters to zero, producing a
perfect sensor useful for debugging control algorithms in isolation.

### 16.2.7 Sensor Reset and State Management

- `reset()` -- resets all noise models, filters, orientation (to identity
  quaternion), and timestamp.
- `set_orientation(quaternion)` -- overrides the current orientation estimate.
  The quaternion is normalized after assignment.

---

## 16.3 Force-Torque Sensor

The `ForceTorqueSensor` class (`src/robotics/sensing/force_torque_sensor.py`)
simulates a 6-axis force/torque transducer that measures three force components
and three torque components.

### 16.3.1 ForceTorqueSensorConfig

| Field               | Type          | Default       | Description                  |
| ------------------- | ------------- | ------------- | ---------------------------- |
| `sensor_id`         | `str`         | `"ft_sensor"` | Unique identifier.           |
| `force_range`       | `float`       | `1000.0`      | Max measurable force [N].    |
| `torque_range`      | `float`       | `100.0`       | Max measurable torque [Nm].  |
| `force_noise_std`   | `float`       | `0.1`         | Force noise std [N].         |
| `torque_noise_std`  | `float`       | `0.01`        | Torque noise std [Nm].       |
| `force_bias_drift`  | `float`       | `0.001`       | Force bias drift [N/step].   |
| `torque_bias_drift` | `float`       | `0.0001`      | Torque bias drift [Nm/step]. |
| `cutoff_frequency`  | `float`       | `100.0`       | Sensor bandwidth [Hz].       |
| `sample_rate`       | `float`       | `1000.0`      | Sampling rate [Hz].          |
| `seed`              | `int or None` | `None`        | Random seed.                 |

### 16.3.2 Noise Pipeline

The force and torque channels have independent noise models, each containing:

1. **BrownianNoise** -- bias drift with `max_bias = noise_std * 10`.
2. **GaussianNoise** -- white noise.

After applying noise independently to the force (first 3 components) and torque
(last 3 components), the full 6D wrench is passed through a single
`BandwidthLimitedNoise` filter. A tare offset is then subtracted, and the result
is clipped to the sensor's measurement range:

$$
y_{\text{force}} = \text{clip}\bigl(w_f - \text{tare}_f,\; -F_{\max},\; F_{\max}\bigr)
$$

$$
y_{\text{torque}} = \text{clip}\bigl(w_\tau - \text{tare}_\tau,\; -\tau_{\max},\; \tau_{\max}\bigr)
$$

### 16.3.3 Reading the Sensor

```python
reading = sensor.read(
    true_wrench=np.array([10.0, 0.0, 50.0, 0.0, 0.0, 0.0]),
    timestamp=0.001,
)
# reading.wrench  -- noisy 6D wrench
# reading.force   -- first 3 components
# reading.torque  -- last 3 components
```

The `read_raw` method applies noise but skips the bandwidth filter, useful for
diagnostic purposes.

### 16.3.4 Tare Offset Handling

The `tare(current_wrench=None)` method zeros the sensor by recording the current
reading as a bias offset. Subsequent measurements subtract this offset:

```python
# Zero the sensor under no-load conditions
sensor.tare(current_wrench=np.zeros(6))

# Or tare using the last reading
sensor.tare()  # uses self._last_reading
```

This models the common laboratory practice of "taring" or "zeroing" a
force/torque sensor before use.

### 16.3.5 Contact Location Estimation

`estimate_contact_location(wrench)` uses the measured wrench to estimate the
point of application of a single contact force:

$$
\mathbf{r} = \frac{\mathbf{f} \times \boldsymbol{\tau}}{\|\mathbf{f}\|^2}
$$

**Derivation.** For a single point contact at position $\mathbf{r}$ applying
force $\mathbf{f}$, the resulting torque is:

$$
\boldsymbol{\tau} = \mathbf{r} \times \mathbf{f}
$$

Taking the cross product of $\mathbf{f}$ with both sides:

$$
\mathbf{f} \times \boldsymbol{\tau} = \mathbf{f} \times (\mathbf{r} \times \mathbf{f})
= \mathbf{r} (\mathbf{f} \cdot \mathbf{f}) - \mathbf{f} (\mathbf{f} \cdot \mathbf{r})
= \|\mathbf{f}\|^2 \mathbf{r} - (\mathbf{f} \cdot \mathbf{r}) \mathbf{f}
$$

The formula $\mathbf{r} = (\mathbf{f} \times \boldsymbol{\tau}) / \|\mathbf{f}\|^2$
gives the component of $\mathbf{r}$ perpendicular to $\mathbf{f}$. This is the
unique minimum-distance solution for $\mathbf{r}$ consistent with the measured
wrench.

The method returns `None` if the force magnitude is below $10^{-6}$ N (too small
for reliable estimation).

```python
wrench = np.array([0.0, 0.0, 10.0, 0.5, -0.3, 0.0])
location = sensor.estimate_contact_location(wrench)
# location gives approximate contact point relative to sensor frame
```

### 16.3.6 Quality Presets

`create_realistic_sensor(sensor_id, quality, seed)` provides three quality levels:

| Quality        | `force_noise_std` | `torque_noise_std` | `force_bias_drift` | `torque_bias_drift` |
| -------------- | ----------------- | ------------------ | ------------------ | ------------------- |
| `"research"`   | 0.01              | 0.001              | 0.0001             | 0.00001             |
| `"industrial"` | 0.1               | 0.01               | 0.001              | 0.0001              |
| `"consumer"`   | 1.0               | 0.1                | 0.01               | 0.001               |

Research-grade sensors (e.g., ATI Nano17) achieve sub-Newton force resolution
and are used in precision manipulation studies. Industrial sensors are suitable
for most factory automation. Consumer-grade sensors offer wider measurement
ranges at the cost of increased noise.

`create_ideal_sensor(sensor_id)` sets all noise parameters to zero.

### 16.3.7 Sensor Reset

`reset()` clears all internal state: noise models, bandwidth filter, tare offset,
and last reading cache. This restores the sensor to its initial condition.

---

## Summary of Key Imports

For quick reference, here are the primary imports for each chapter:

**Chapter 13 -- Contact Dynamics:**

```python
from src.robotics.core.types import ContactState, ContactType
from src.robotics.contact.friction_cone import (
    FrictionCone,
    linearize_friction_cone,
    compute_friction_cone_constraint,
    project_to_friction_cone,
)
from src.robotics.contact.contact_manager import ContactManager
from src.robotics.contact.grasp_analysis import (
    compute_grasp_matrix,
    check_force_closure,
    compute_grasp_quality,
    compute_contact_wrench_cone,
    required_contact_forces,
)
```

**Chapter 14 -- Whole-Body Control:**

```python
from src.robotics.core.types import TaskPriority, ControlMode, TaskDescriptor
from src.robotics.control.whole_body.task import (
    Task, TaskType,
    create_com_task, create_posture_task,
    create_ee_task, create_contact_constraint,
    create_joint_limit_task,
)
from src.robotics.control.whole_body.qp_solver import (
    QPProblem, QPSolution, QPSolver,
    ScipyQPSolver, NullspaceQPSolver,
    create_default_solver,
)
from src.robotics.control.whole_body.wbc_controller import (
    WholeBodyController, WBCConfig, WBCSolution,
)
```

**Chapter 15 -- Locomotion:**

```python
from src.robotics.locomotion.gait_types import (
    GaitType, GaitPhase, LegState, SupportState,
    GaitParameters,
    create_walk_parameters, create_run_parameters, create_stand_parameters,
)
from src.robotics.locomotion.gait_state_machine import (
    GaitEvent, GaitState, GaitStateMachine,
)
from src.robotics.locomotion.footstep_planner import (
    Footstep, FootstepPlan, FootstepPlanner,
)
from src.robotics.locomotion.zmp_computer import (
    ZMPResult, ZMPComputer,
)
```

**Chapter 16 -- Sensing:**

```python
from src.robotics.sensing.noise_models import (
    NoiseModel, GaussianNoise, BrownianNoise,
    QuantizationNoise, BandwidthLimitedNoise,
    CompositeNoise,
    create_realistic_sensor_noise,
)
from src.robotics.sensing.imu_sensor import (
    IMUSensorConfig, IMUSensor,
    create_ideal_imu, create_realistic_imu,
)
from src.robotics.sensing.force_torque_sensor import (
    ForceTorqueSensorConfig, ForceTorqueSensor,
    create_ideal_sensor, create_realistic_sensor,
)
```

---

# UpstreamDrift User Manual -- Part 5: Machine Learning and Deployment

---

# Chapter 17: Reinforcement Learning

The `learning/rl/` module provides a complete reinforcement learning framework for
training robot control policies in simulation. It builds on the Gymnasium API standard
and integrates tightly with the UpstreamDrift physics engines through the
`PhysicsEngineProtocol` interface. This chapter covers the configuration system,
the base environment class, and the concrete humanoid and manipulation environments.

Source files referenced in this chapter:

- `src/learning/rl/configs.py`
- `src/learning/rl/base_env.py`
- `src/learning/rl/humanoid_envs.py`
- `src/learning/rl/manipulation_envs.py`

---

## 17.1 Configuration System

The RL module uses a layered dataclass-based configuration system defined in
`src/learning/rl/configs.py`. Every aspect of the environment -- observations,
actions, rewards, and task parameters -- is governed by its own configuration
object. This design allows users to compose environments from independent,
reusable configuration blocks.

### 17.1.1 ActionMode Enum

The `ActionMode` enum selects the interface between the policy output and the
physics engine. It has four members:

| Member      | Value         | Description                                        |
| ----------- | ------------- | -------------------------------------------------- |
| `TORQUE`    | `"torque"`    | Direct joint torque commands in Nm.                |
| `POSITION`  | `"position"`  | Target joint positions in radians.                 |
| `VELOCITY`  | `"velocity"`  | Target joint velocities in rad/s.                  |
| `IMPEDANCE` | `"impedance"` | Position targets with stiffness and damping gains. |

The action mode determines how the raw policy output is interpreted by the
`_apply_action` method of each environment subclass. Torque mode provides the
most direct control but requires the policy to learn inverse dynamics implicitly.
Position mode is the most forgiving because low-level joint PD controllers track
the target, but it hides dynamics from the policy. Impedance mode is a hybrid
that exposes compliance parameters.

Usage:

```python
from src.learning.rl.configs import ActionMode

mode = ActionMode.TORQUE
print(mode.value)  # "torque"
```

### 17.1.2 TaskType Enum

The `TaskType` enum categorizes environments by the high-level task objective:

| Member         | Value            | Typical environments                   |
| -------------- | ---------------- | -------------------------------------- |
| `LOCOMOTION`   | `"locomotion"`   | Walking, running, stair climbing       |
| `MANIPULATION` | `"manipulation"` | Pick-and-place, assembly, tool use     |
| `BALANCE`      | `"balance"`      | Standing, balancing under perturbation |
| `TRACKING`     | `"tracking"`     | End-effector trajectory tracking       |
| `REACHING`     | `"reaching"`     | Reaching to target positions           |

The task type is stored in `TaskConfig` and is available to reward functions and
termination conditions. It can also be used by curriculum learning schedulers to
organize training progressions.

### 17.1.3 ObservationConfig

`ObservationConfig` is a dataclass that specifies which sensor channels to
include in the observation vector and what noise to apply to each channel.

**Fields and defaults:**

| Field                    | Type    | Default | Description                               |
| ------------------------ | ------- | ------- | ----------------------------------------- |
| `include_joint_pos`      | `bool`  | `True`  | Include joint position readings.          |
| `include_joint_vel`      | `bool`  | `True`  | Include joint velocity readings.          |
| `include_joint_torque`   | `bool`  | `False` | Include measured joint torques.           |
| `include_ee_pos`         | `bool`  | `False` | Include end-effector Cartesian position.  |
| `include_ee_vel`         | `bool`  | `False` | Include end-effector Cartesian velocity.  |
| `include_contact_forces` | `bool`  | `False` | Include ground reaction / contact forces. |
| `include_imu`            | `bool`  | `False` | Include IMU accelerometer + gyroscope.    |
| `include_privileged`     | `bool`  | `False` | Include privileged info (teacher only).   |
| `position_noise_std`     | `float` | `0.0`   | Gaussian noise on position readings.      |
| `velocity_noise_std`     | `float` | `0.0`   | Gaussian noise on velocity readings.      |
| `history_length`         | `int`   | `1`     | Number of stacked observation frames.     |

The `include_privileged` flag is intended for asymmetric actor-critic training.
When set to `True`, the observation includes ground-truth information (such as
terrain height maps or exact object poses) that would not be available on real
hardware. A teacher policy is trained with privileged information, then a student
policy is distilled from it using only the non-privileged observations.

**Computing the observation dimension.** The `get_obs_dim` method calculates the
total flat observation size:

```python
def get_obs_dim(self, n_joints: int, n_ee: int = 0) -> int:
```

The formula accumulates dimensions channel by channel:

$$
d_{obs} = \Bigl(\underbrace{n_q}_{\text{pos}} + \underbrace{n_q}_{\text{vel}} + \underbrace{n_q}_{\text{torque}} + \underbrace{3 n_{ee}}_{\text{ee pos}} + \underbrace{6 n_{ee}}_{\text{ee vel}} + \underbrace{6}_{\text{IMU}}\Bigr) \times H
$$

where each term is included only if its corresponding flag is `True`, $n_q$ is
`n_joints`, $n_{ee}$ is the number of end-effectors, the IMU contributes 6
dimensions (3 accelerometer + 3 gyroscope), and $H$ is `history_length`. Note
that end-effector velocity is 6-dimensional (linear + angular), while
end-effector position is 3-dimensional (Cartesian xyz only).

Contact forces are excluded from the `get_obs_dim` calculation because their
dimensionality varies by robot model. Environments that include contact forces
override `_build_observation_space` to account for them.

Example:

```python
from src.learning.rl.configs import ObservationConfig

obs_config = ObservationConfig(
    include_joint_pos=True,
    include_joint_vel=True,
    include_imu=True,
    position_noise_std=0.005,
    velocity_noise_std=0.01,
    history_length=3,
)
dim = obs_config.get_obs_dim(n_joints=21, n_ee=2)
# dim = (21 + 21 + 6) * 3 = 144
```

### 17.1.4 ActionConfig

`ActionConfig` governs the action space and the processing pipeline that maps raw
policy outputs to actuator commands.

**Fields and defaults:**

| Field             | Type         | Default             | Description                               |
| ----------------- | ------------ | ------------------- | ----------------------------------------- |
| `mode`            | `ActionMode` | `ActionMode.TORQUE` | Control mode for the action space.        |
| `action_scale`    | `float`      | `1.0`               | Multiplicative scaling for actions.       |
| `action_clip`     | `float`      | `1.0`               | Symmetric clipping bound for raw actions. |
| `smoothing_alpha` | `float`      | `0.0`               | Exponential smoothing coefficient.        |

**The `process_action` pipeline.** Every raw action from the policy passes through
a three-stage pipeline implemented in `process_action`:

```python
def process_action(
    self,
    action: NDArray[np.floating],
    prev_action: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
```

The three stages are:

1. **Clipping.** The raw action is clamped to $[-c, c]$ where $c$ is
   `action_clip`:

$$
a^{clip}_i = \text{clip}(a^{raw}_i,\; -c,\; c)
$$

2. **Scaling.** The clipped action is multiplied by `action_scale`:

$$
a^{scaled}_i = s \cdot a^{clip}_i
$$

where $s$ is `action_scale`. This is useful when the policy outputs are
normalized to $[-1, 1]$ and must be mapped to physical torque ranges.

3. **Exponential smoothing.** If `smoothing_alpha` $\alpha > 0$ and a previous
   action is available, the output is a weighted blend of the previous and
   current actions:

$$
a_t = \alpha \, a_{t-1} + (1 - \alpha) \, a_t^{raw}
$$

where $a_{t-1}$ is the previous processed action and $a_t^{raw}$ is the
current scaled action. When $\alpha = 0$ there is no smoothing. Higher values
of $\alpha$ produce smoother but more sluggish motion, which can be beneficial
for sim-to-real transfer by filtering out high-frequency oscillations.

Example:

```python
from src.learning.rl.configs import ActionConfig, ActionMode

action_config = ActionConfig(
    mode=ActionMode.POSITION,
    action_scale=0.5,
    action_clip=1.0,
    smoothing_alpha=0.2,
)

import numpy as np
raw = np.array([1.5, -0.3, 0.8])
prev = np.array([0.1, 0.2, -0.1])
processed = action_config.process_action(raw, prev)
# Step 1: clip to [-1, 1] -> [1.0, -0.3, 0.8]
# Step 2: scale by 0.5 -> [0.5, -0.15, 0.4]
# Step 3: smooth: 0.2 * prev + 0.8 * scaled
```

### 17.1.5 RewardConfig

`RewardConfig` specifies the weighting of reward components. The total reward at
each timestep is assembled by the environment's `_compute_reward` method using
these weights.

**Fields and defaults:**

| Field                       | Type    | Default  | Description                           |
| --------------------------- | ------- | -------- | ------------------------------------- |
| `task_reward_weight`        | `float` | `1.0`    | Weight for the task-specific reward.  |
| `energy_penalty_weight`     | `float` | `0.001`  | Weight for the energy penalty.        |
| `smoothness_penalty_weight` | `float` | `0.0001` | Weight for the smoothness penalty.    |
| `contact_penalty_weight`    | `float` | `0.0`    | Weight for undesired contacts.        |
| `use_potential_shaping`     | `bool`  | `False`  | Enable potential-based shaping.       |
| `alive_bonus`               | `float` | `0.0`    | Constant bonus per non-terminal step. |

**Energy penalty.** The `compute_energy_penalty` method penalizes high torques to
encourage efficient motions:

$$
R_{energy} = -w_e \sum_{i=1}^{n} \tau_i^2
$$

where $w_e$ is `energy_penalty_weight` and $\tau_i$ is the torque applied to
joint $i$. The method computes `float(np.sum(torques**2)) * self.energy_penalty_weight`.
This quadratic penalty strongly discourages large torques while allowing moderate
ones, which tends to produce smoother, more energy-efficient gaits.

**Smoothness penalty.** The `compute_smoothness_penalty` method penalizes abrupt
changes in the action signal:

$$
R_{smooth} = -w_s \sum_{i=1}^{n} (a_i^t - a_i^{t-1})^2
$$

where $w_s$ is `smoothness_penalty_weight`, $a_i^t$ is the current action for
joint $i$, and $a_i^{t-1}$ is the previous action. When `prev_action` is `None`
(at the start of an episode), the penalty is zero. This penalty complements the
exponential smoothing in `ActionConfig` -- smoothing operates on the action
signal directly, while the smoothness penalty provides a gradient signal to the
policy.

**Alive bonus.** The constant `alive_bonus` is added at every non-terminal
timestep. This encourages the policy to avoid early termination conditions (such
as falling). A typical value for locomotion tasks is 0.5 to 1.0.

**Potential shaping.** When `use_potential_shaping` is `True`, the environment
adds a potential-based shaping term:

$$
\Delta\Phi = \gamma \, \Phi(s') - \Phi(s)
$$

where $\Phi(s)$ is a potential function defined by the environment subclass.
Potential shaping preserves the optimal policy under certain conditions
(Ng et al., 1999) while providing denser learning signal. For the walking
environment, $\Phi(s) = x_{forward}$, so the shaping reward is proportional to
the forward displacement per step.

Example:

```python
from src.learning.rl.configs import RewardConfig

reward_config = RewardConfig(
    task_reward_weight=2.0,
    energy_penalty_weight=0.005,
    smoothness_penalty_weight=0.001,
    alive_bonus=0.5,
    use_potential_shaping=True,
)
```

### 17.1.6 TaskConfig

`TaskConfig` defines the task objective and episode parameters.

**Fields and defaults:**

| Field                | Type                   | Default               | Description                      |
| -------------------- | ---------------------- | --------------------- | -------------------------------- | ------------------------------- |
| `task_type`          | `TaskType`             | `TaskType.LOCOMOTION` | Task category.                   |
| `target_velocity`    | `NDArray[np.floating]` | `[1.0, 0.0, 0.0]`     | Target velocity (locomotion).    |
| `target_position`    | `NDArray[np.floating]  | None`                 | `None`                           | Target position (manipulation). |
| `target_orientation` | `NDArray[np.floating]  | None`                 | `None`                           | Target orientation quaternion.  |
| `max_episode_steps`  | `int`                  | `1000`                | Maximum timesteps per episode.   |
| `early_termination`  | `bool`                 | `True`                | Terminate on failure conditions. |
| `success_threshold`  | `float`                | `0.05`                | Error threshold for success.     |

The `is_success(error)` method returns `True` when the task error falls below
`success_threshold`, providing a binary success metric for evaluation.

---

## 17.2 Gymnasium Environment Base

The `RoboticsGymEnv` class in `src/learning/rl/base_env.py` is the abstract base
for all RL environments. It follows the Gymnasium `Env` interface but does not
inherit from `gymnasium.Env` directly -- instead it implements the same API to
avoid a hard dependency on Gymnasium at import time. If Gymnasium is not
installed, importing the module raises a clear error message at instantiation.

### 17.2.1 Constructor

```python
class RoboticsGymEnv:
    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        model_path: str | None = None,
        task_config: TaskConfig | None = None,
        obs_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
```

**Parameters:**

- `engine` -- A physics engine implementing `PhysicsEngineProtocol`. This can be
  any of the UpstreamDrift backends (MuJoCo, PyBullet, Drake, etc.).
- `model_path` -- Optional path to a robot model file (URDF, MJCF). If provided
  and the engine has a `load_from_path` method, the model is loaded
  automatically.
- `task_config`, `obs_config`, `action_config`, `reward_config` -- Configuration
  objects. If `None`, default configurations are used.
- `render_mode` -- Either `"human"` (on-screen display) or `"rgb_array"` (return
  pixel buffer). Matches the Gymnasium render mode convention.

During construction the class:

1. Loads the model into the engine if `model_path` is provided.
2. Instantiates default configs for any that are `None`.
3. Queries the engine for `n_joints` (from `engine.n_q`), `n_actuators` (from
   `engine.n_v`), and `n_end_effectors` (default 1, overridden by subclasses).
4. Builds the observation and action spaces as `gymnasium.spaces.Box` objects.
5. Initializes episode state variables: step counter, previous action, previous
   potential, and the random number generator.

### 17.2.2 Observation and Action Spaces

The observation space is a `Box` of shape `(obs_dim,)` with bounds
$(-\infty, +\infty)$ and `dtype=np.float32`. The observation dimension is
computed from `obs_config.get_obs_dim(n_joints, n_ee)`.

The action space is a `Box` of shape `(n_actuators,)` with bounds
$[-c, c]$ where $c$ is `action_config.action_clip`, and `dtype=np.float32`.

### 17.2.3 The `step` Method

```python
def step(
    self, action: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float, bool, bool, dict[str, Any]]:
```

The `step` method follows the standard Gymnasium 5-tuple return signature and
executes the following sequence:

1. **Process action.** The raw action passes through
   `action_config.process_action(action, self._prev_action)` for clipping,
   scaling, and smoothing.

2. **Apply action.** Calls the abstract `_apply_action(processed_action)` which
   the subclass implements to write commands to the physics engine.

3. **Step simulation.** Calls `_step_simulation()` to advance the physics by one
   control timestep (which may correspond to multiple physics substeps).

4. **Get observation.** Calls `_get_observation()` to read sensor data from the
   engine and assemble the observation vector.

5. **Compute reward.** Calls `_compute_reward(processed_action)` to compute the
   scalar reward.

6. **Check termination.** Calls `_check_termination()` for failure-based
   termination (`terminated`), and compares `_step_count` against
   `task_config.max_episode_steps` for time-based truncation (`truncated`).

7. **Build info dict.** Calls `_get_info()` for environment-specific metrics.

8. **Update state.** Increments `_step_count` and stores the processed action as
   `_prev_action`.

### 17.2.4 The `reset` Method

```python
def reset(
    self,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
```

The `reset` method:

1. Seeds or initializes the random number generator via
   `np.random.default_rng(seed)`.
2. Calls `_reset_simulation(options)` to reset the physics state.
3. Resets `_step_count` to 0, `_prev_action` to `None`, and
   `_prev_potential` to `_compute_potential()`.
4. Returns the initial observation and info dict.

### 17.2.5 Abstract Methods

Subclasses must implement six abstract methods:

| Method               | Signature                    | Purpose                               |
| -------------------- | ---------------------------- | ------------------------------------- | --------------------------------------- |
| `_apply_action`      | `(action: NDArray) -> None`  | Write commands to the physics engine. |
| `_step_simulation`   | `() -> None`                 | Advance the simulation forward.       |
| `_get_observation`   | `() -> NDArray`              | Read and assemble the observation.    |
| `_compute_reward`    | `(action: NDArray) -> float` | Compute the scalar reward.            |
| `_check_termination` | `() -> bool`                 | Check for early termination.          |
| `_reset_simulation`  | `(options: dict              | None) -> None`                        | Reset the physics to the initial state. |

Additionally, two optional hook methods can be overridden:

- `_compute_potential() -> float` -- Returns the potential function value for
  reward shaping. Default returns 0.0.
- `_get_info() -> dict[str, Any]` -- Returns additional info. Default returns
  `{"step_count": self._step_count}`.

### 17.2.6 Rendering

The `render()` method dispatches to `_render_frame()` (returns an RGB `uint8`
array of shape `(480, 640, 3)`) or `_render_human()` (on-screen display) based
on `render_mode`. The `close()` method calls `engine.close()` if available.

---

## 17.3 Humanoid Environments

The humanoid environments in `src/learning/rl/humanoid_envs.py` provide
locomotion and balance tasks for full-body humanoid robots. Both environments
inherit from `RoboticsGymEnv` and implement all six abstract methods.

### 17.3.1 HumanoidWalkEnv

`HumanoidWalkEnv` trains a humanoid to walk at a commanded forward velocity.

**Constructor parameters (beyond the base class):**

| Parameter         | Type    | Default | Description                  |
| ----------------- | ------- | ------- | ---------------------------- |
| `target_velocity` | `float` | `1.0`   | Target forward speed in m/s. |

The constructor creates a `TaskConfig` with `task_type=TaskType.LOCOMOTION`,
`target_velocity=[target_velocity, 0.0, 0.0]`, and `max_episode_steps=1000`.
Two termination thresholds are defined:

- `_base_height_threshold = 0.3` -- Minimum base height in meters before
  the episode terminates (the robot has fallen).
- `_base_tilt_threshold = 0.8` -- Minimum vertical component of the torso's
  up-vector. This corresponds to a maximum tilt angle of approximately
  $\cos^{-1}(0.8) \approx 37^{\circ}$.

**Observation assembly.** The `_get_observation` method concatenates the enabled
sensor channels in order: joint positions, joint velocities, joint torques, IMU
data, and contact forces. Gaussian noise is added to positions and velocities
according to `obs_config.position_noise_std` and `obs_config.velocity_noise_std`:

$$
\tilde{q} = q + \mathcal{N}(0, \sigma_q^2 I)
$$

$$
\tilde{\dot{q}} = \dot{q} + \mathcal{N}(0, \sigma_v^2 I)
$$

**Reward function.** The total reward at each step is:

$$
R = w_{task} \exp\bigl(-\|v - v_{target}\|\bigr) + R_{alive} + R_{energy} + R_{smooth} + \Delta\Phi
$$

where:

- $v$ is the current base velocity (first two components, i.e., forward and
  lateral), and $v_{target}$ is the target velocity.
- $w_{task}$ is `reward_config.task_reward_weight`.
- $R_{alive}$ is `reward_config.alive_bonus`.
- $R_{energy} = -w_e \sum_i \tau_i^2$ as described in Section 17.1.5.
- $R_{smooth} = -w_s \sum_i (a_i^t - a_i^{t-1})^2$ as described in Section 17.1.5.
- $\Delta\Phi = \gamma \Phi(s') - \Phi(s)$ with $\Phi(s) = x_{forward}$, the
  x-coordinate of the base position. This is only added when
  `reward_config.use_potential_shaping` is `True`.

The exponential velocity reward $\exp(-\|v - v_{target}\|)$ is maximized at 1.0
when the robot matches the target velocity exactly, and decays smoothly toward 0
as the error increases. This is preferable to a linear penalty because it is
bounded and provides strong signal near the optimum.

**Termination conditions.** The episode terminates early if
`task_config.early_termination` is `True` and either:

1. The base height drops below 0.3 m (the robot has fallen):

$$
z_{base} < 0.3
$$

2. The base tilts excessively. The vertical component of the up-vector is
   computed from the orientation quaternion as:

$$
up_z = 1 - 2(q_y^2 + q_z^2)
$$

and the episode terminates if $up_z < 0.8$.

**Reset with perturbation.** The `_reset_simulation` method resets the engine and
optionally applies a small random perturbation to the initial joint positions.
The `options` dictionary supports:

- `randomize_init` (bool, default `True`) -- Whether to perturb the initial state.
- `init_noise_scale` (float, default `0.01`) -- Standard deviation of the
  Gaussian perturbation added to joint positions.

**Info dictionary.** The returned info includes:

- `forward_velocity` -- Current forward speed.
- `lateral_velocity` -- Current lateral speed.
- `base_height` -- Current base height.
- `forward_distance` -- Total forward distance traveled.

### 17.3.2 HumanoidStandEnv

`HumanoidStandEnv` trains a humanoid to maintain a stable standing posture,
optionally under external perturbations.

**Constructor parameters (beyond the base class):**

| Parameter            | Type    | Default | Description                      |
| -------------------- | ------- | ------- | -------------------------------- |
| `perturbation_force` | `float` | `0.0`   | Maximum perturbation force in N. |

The constructor creates a `TaskConfig` with `task_type=TaskType.BALANCE`,
zero target velocity, and `max_episode_steps=500`. The target standing height
is `_target_height = 0.9` meters.

**External perturbations.** When `perturbation_force > 0`, the `_step_simulation`
method applies random horizontal forces to the torso with a 1% probability per
timestep. The force is sampled uniformly from
$[-F_{max}, F_{max}]$ in the x and y components, with zero vertical component:

```python
force = rng.uniform(-perturbation_force, perturbation_force, size=3)
force[2] = 0  # No vertical force
engine.apply_external_force("torso", force)
```

**Reward function.** The standing reward combines four components:

$$
R = w_h \, e^{-10\,|z - 0.9|} + w_u \, \frac{up_z + 1}{2} + w_s \, e^{-\|v\|} + R_{alive} + R_{energy}
$$

where:

- $w_h$ is `reward_config.task_reward_weight`, and $z$ is the base height.
  The exponential term $e^{-10|z - 0.9|}$ rewards maintaining the target height
  of 0.9 m with a sharp peak.

- $w_u = 0.5$ (hardcoded) weights the upright reward. The quantity
  $up_z = 1 - 2(q_y^2 + q_z^2)$ is the vertical component of the orientation,
  normalized to $[0, 1]$ by $(up_z + 1)/2$.

- $w_s = 0.3$ (hardcoded) weights the stillness reward. The term
  $e^{-\|v\|}$ rewards low base velocity, encouraging the robot to stand in
  place rather than drift.

- $R_{alive}$ and $R_{energy}$ are as previously defined.

**Termination.** The episode terminates if the base height drops below 0.3 m.

**Info dictionary.** Returns `base_height` and `upright_score` (the raw $up_z$
value before normalization).

---

## 17.4 Manipulation Environments

The manipulation environments in `src/learning/rl/manipulation_envs.py` provide
object interaction tasks for robotic arms.

### 17.4.1 ManipulationPickPlaceEnv

`ManipulationPickPlaceEnv` implements a three-phase pick-and-place task where the
robot must reach for an object, grasp it, and place it at a target location.

**Constructor parameters (beyond the base class):**

| Parameter            | Type     | Default | Description       |
| -------------------- | -------- | ------- | ----------------- | -------------------------- |
| `object_initial_pos` | `NDArray | None`   | `[0.5, 0.0, 0.1]` | Initial object position.   |
| `target_pos`         | `NDArray | None`   | `[0.5, 0.3, 0.1]` | Target placement position. |

The environment uses `TaskType.MANIPULATION`, `max_episode_steps=500`,
`early_termination=False`, and `success_threshold=0.05`.

**Grasp state machine.** The task proceeds through three logical phases tracked
by the boolean `_object_grasped`:

1. **Reaching phase** (`_object_grasped = False`). The robot must move its
   end-effector toward the object. The grasp threshold is 0.05 m.

2. **Grasping transition.** When the end-effector is within `_grasp_threshold`
   distance of the object AND the gripper is closed (gripper state < 0.1), the
   `_update_grasp_state` method sets `_object_grasped = True`. The object then
   follows the end-effector position.

3. **Placing phase** (`_object_grasped = True`). The robot must transport the
   object to the target location. Success is declared when the object-to-target
   distance is less than `_place_threshold` (0.05 m).

**Observation space.** The observation extends the base observation (joint
positions, velocities, end-effector state) with:

- Object position (3D) -- either the engine's reported position or the
  end-effector position if grasped.
- Target position (3D).
- Grasp state (1D binary).

The total observation dimension is `base_dim + 7`.

**Reward function.** The reward is phase-dependent:

- **Reaching phase:**

$$
R_{reach} = e^{-5 \, d_{hand \to obj}} \cdot w_{task} \cdot 0.5
$$

where $d_{hand \to obj} = \|p_{ee} - p_{obj}\|$ is the Euclidean distance
from the end-effector to the object.

- **Placing phase:**

$$
R_{place} = e^{-5 \, d_{obj \to target}} \cdot w_{task}
$$

where $d_{obj \to target} = \|p_{obj} - p_{target}\|$ is the distance from
the (grasped) object to the target.

- **Success bonus:** A reward of $+10$ is awarded when
  $d_{obj \to target} < 0.05$.

Both phases also include energy and smoothness penalties.

**Termination.** The episode terminates (success) when the grasped object is
within `_place_threshold` of the target. There is no failure termination since
`early_termination=False` in the task config.

**Reset options.** The `options` dictionary supports `randomize_positions`
(bool), which adds uniform noise in $[-0.1, 0.1]$ to both the object and target
positions for curriculum-style training.

### 17.4.2 DualArmManipulationEnv

`DualArmManipulationEnv` implements a bimanual coordination task where two robot
arms must cooperate to lift a heavy object.

**Constructor parameters (beyond the base class):**

| Parameter     | Type    | Default | Description               |
| ------------- | ------- | ------- | ------------------------- |
| `object_mass` | `float` | `5.0`   | Mass of the object in kg. |

This environment overrides `_get_n_end_effectors` to return 2 and doubles the
action space to accommodate both arms. The action vector is split in half:
`action[:n]` controls the left arm and `action[n:]` controls the right arm.

**Bimanual grasp state.** Three boolean flags track the task state:

- `_left_grasped` -- Left gripper has grasped the object (distance < 0.08 m and
  gripper closed).
- `_right_grasped` -- Right gripper has grasped the object.
- `_object_lifted` -- Both arms are grasping simultaneously. This is the only
  condition under which the object can be lifted, reflecting the requirement
  that a single arm cannot support the heavy object.

**Observation.** The observation concatenates:

1. Left arm joint positions and velocities.
2. Right arm joint positions and velocities.
3. Object position (3D).
4. Target position (3D).
5. Grasp states (3D): `[left_grasped, right_grasped, object_lifted]`.

**Reward function.** The reward structure encourages coordination:

- **Before lifting** ($\text{object\_lifted} = \text{False}$):

$$
R_{reach} = e^{-3(d_{left} + d_{right})} \cdot w_{task} \cdot 0.3
$$

where $d_{left}$ and $d_{right}$ are the distances from each end-effector to
the object. Additionally, a bonus of $+1.0$ is awarded for each arm that
achieves a grasp.

- **During lifting** ($\text{object\_lifted} = \text{True}$):

$$
R_{lift} = e^{-3 \, d_{obj \to target}} \cdot w_{task}
$$

A large bonus of $+20$ is awarded when $d_{obj \to target} < 0.1$.

- **Coordination penalty:** The velocity difference between the two
  end-effectors is penalized to encourage synchronized motion:

$$
R_{coord} = -0.1 \, \|v_{left} - v_{right}\|
$$

This term prevents one arm from jerking the object away from the other.

- Energy penalty is also applied.

**Termination.** The episode terminates if the object has been lifted
(`_object_lifted = True`) but then drops below $z = 0.05$ m, indicating the
arms lost their grip.

---

# Chapter 18: Imitation Learning

The `learning/imitation/` module provides tools for learning robot control
policies from expert demonstrations. It includes a flexible demonstration dataset
format, and three imitation learning algorithms: Behavior Cloning (BC), Dataset
Aggregation (DAgger), and Generative Adversarial Imitation Learning (GAIL).
All implementations use pure NumPy for portability, with no dependency on
PyTorch or TensorFlow.

Source files referenced in this chapter:

- `src/learning/imitation/dataset.py`
- `src/learning/imitation/learners.py`

---

## 18.1 Demonstration Dataset

### 18.1.1 The Demonstration Dataclass

The `Demonstration` dataclass in `src/learning/imitation/dataset.py` represents a
single recorded trajectory of robot motion.

**Fields:**

| Field                | Type                       | Default           | Description                               |
| -------------------- | -------------------------- | ----------------- | ----------------------------------------- | ------------------------------ |
| `timestamps`         | `NDArray` shape `(T,)`     | required          | Time values for each frame in seconds.    |
| `joint_positions`    | `NDArray` shape `(T, n_q)` | required          | Joint positions at each timestep.         |
| `joint_velocities`   | `NDArray` shape `(T, n_v)` | required          | Joint velocities at each timestep.        |
| `actions`            | `NDArray` shape `(T, n_u)` | `None`            | Control actions applied at each timestep. |
| `end_effector_poses` | `NDArray` shape `(T, 7)`   | `None`            | EE poses [x,y,z,qw,qx,qy,qz].             |
| `contact_states`     | `list[list[dict]]`         | `None`            | Contact info per timestep.                |
| `task_id`            | `str                       | None`             | `None`                                    | Task identifier for filtering. |
| `success`            | `bool`                     | `True`            | Whether the demonstration was successful. |
| `source`             | `str`                      | `"teleoperation"` | Source of the demonstration.              |
| `metadata`           | `dict[str, Any]`           | `{}`              | Arbitrary additional metadata.            |

**Validation.** The `__post_init__` method enforces that `timestamps`,
`joint_positions`, and `joint_velocities` all share the same first dimension
$T$, raising `ValueError` if they differ.

**Properties:**

- `duration` -- Total time span: `timestamps[-1] - timestamps[0]`.
- `n_frames` -- Number of frames: `len(timestamps)`.
- `n_joints` -- Number of joints: `joint_positions.shape[1]`.

**Frame access.** The `get_frame(idx)` method returns a dictionary with
the data at a single timestep, including the action and end-effector pose
if available.

**Subsampling.** The `subsample(factor)` method returns a new `Demonstration`
keeping every $k$-th frame, useful for adapting high-frequency recordings to
lower control frequencies.

**Serialization.** The `to_dict()` method converts all arrays to nested Python
lists for JSON serialization. The `from_dict(data)` class method reconstructs a
`Demonstration` from a dictionary, converting lists back to NumPy arrays.

### 18.1.2 The DemonstrationDataset Class

`DemonstrationDataset` is a container for multiple demonstrations with methods
for filtering, conversion, augmentation, and I/O.

**Construction.**

```python
dataset = DemonstrationDataset()           # Empty
dataset = DemonstrationDataset([demo1, demo2])  # From list
```

**Standard container interface.** The class supports `len()`, indexing with
`[]`, and iteration, making it usable in standard Python loops and comprehensions.

**Adding demonstrations.**

- `add(demo)` -- Append a single `Demonstration`.
- `extend(demos)` -- Append a list of `Demonstration` objects.

**Filtering.**

- `filter_successful()` -- Returns a new `DemonstrationDataset` containing only
  demonstrations with `success=True`.
- `filter_by_task(task_id)` -- Returns a new dataset containing only
  demonstrations whose `task_id` matches the given string.

**Aggregate properties.**

- `total_frames` -- Sum of `n_frames` across all demonstrations.
- `total_transitions` -- Sum of `max(0, n_frames - 1)` across all
  demonstrations. Each demonstration of length $T$ contributes $T - 1$
  state transitions.

**Conversion to training data.** Two methods convert the dataset into NumPy
arrays suitable for supervised learning:

1. `to_transitions()` returns a tuple `(states, actions, next_states)` where
   each state is the concatenation of joint positions and velocities
   $s = [q, \dot{q}]$. Each demonstration of $T$ frames produces $T - 1$
   transitions. Demonstrations without actions are skipped.

2. `to_state_action_pairs()` returns `(states, actions)`, discarding the
   next-state information. This is the primary input format for behavior
   cloning.

**Data augmentation.** The `augment` method creates additional training data by
adding Gaussian noise to joint positions and velocities:

$$
\tilde{q} = q + \mathcal{N}(0, \sigma^2 I)
$$

$$
\tilde{\dot{q}} = \dot{q} + \mathcal{N}(0, \sigma^2 I)
$$

where $\sigma$ is the `noise_std` parameter. For each original demonstration,
`num_augmentations` noisy copies are created (default 5). The original
demonstration is preserved. Augmented demonstrations have their `source` field
appended with `"_augmented"` and their metadata includes `{"augmented": True}`.

Actions are NOT perturbed during augmentation -- only the state trajectory is
modified. This teaches the policy to recover from states slightly off the
demonstrated trajectory, improving robustness.

```python
augmented = dataset.augment(noise_std=0.01, num_augmentations=5)
# If dataset had 10 demos, augmented has 10 * (1 + 5) = 60 demos
```

**Sampling.** The `sample(n)` method returns a random subset of `n`
demonstrations, useful for creating mini-datasets for quick experiments.

**Persistence.** The `save(path)` and `load(path)` class method serialize the
dataset to and from JSON format. The file includes a version string and the
count of demonstrations for quick inspection.

**Statistics.** The `get_statistics()` method computes and returns:

- `n_demonstrations` -- Number of demonstrations.
- `total_frames` -- Total frame count.
- `total_transitions` -- Total transition count.
- `success_rate` -- Fraction of successful demonstrations.
- `mean_duration` -- Average demonstration length in seconds.
- `position_mean`, `position_std` -- Per-joint mean and standard deviation of
  positions across all frames.
- `velocity_mean`, `velocity_std` -- Per-joint mean and standard deviation of
  velocities across all frames.

These statistics are essential for normalizing inputs before training.

---

## 18.2 Behavior Cloning

Behavior cloning (BC) is the simplest imitation learning approach: it frames
the problem as supervised regression from states to actions. The
`BehaviorCloning` class in `src/learning/imitation/learners.py` implements BC
with a NumPy-based MLP.

### 18.2.1 TrainingConfig

All imitation learners share a common `TrainingConfig` dataclass:

| Field           | Type        | Default      | Description                         |
| --------------- | ----------- | ------------ | ----------------------------------- |
| `epochs`        | `int`       | `100`        | Number of training epochs.          |
| `batch_size`    | `int`       | `256`        | Mini-batch size.                    |
| `learning_rate` | `float`     | `1e-3`       | Learning rate for SGD.              |
| `weight_decay`  | `float`     | `1e-5`       | L2 regularization coefficient.      |
| `hidden_sizes`  | `list[int]` | `[256, 256]` | Hidden layer dimensions.            |
| `activation`    | `str`       | `"relu"`     | Activation function name.           |
| `dropout`       | `float`     | `0.0`        | Dropout probability (not yet used). |

### 18.2.2 Network Architecture

The policy is a fully-connected multi-layer perceptron (MLP) stored as a list
of dictionaries, where each dictionary contains a weight matrix `W` and a bias
vector `b`. The network has `len(hidden_sizes)` hidden layers plus one output
layer.

For a network with hidden sizes $[h_1, h_2]$, the architecture is:

$$
\text{Input } (d_{obs}) \xrightarrow{W_1, b_1} h_1 \xrightarrow{\text{ReLU}} \xrightarrow{W_2, b_2} h_2 \xrightarrow{\text{ReLU}} \xrightarrow{W_3, b_3} \text{Output } (d_{act})
$$

Weight initialization uses small random Gaussian values (`np.random.randn * 0.01`)
and zero biases.

### 18.2.3 Forward Pass

The `_forward` method computes the network output:

$$
h_0 = x
$$

$$
h_l = \text{ReLU}(W_l \, h_{l-1} + b_l) \quad \text{for } l = 1, \ldots, L-1
$$

$$
\hat{a} = W_L \, h_{L-1} + b_L
$$

where $x$ is the observation, $L$ is the total number of layers, and ReLU is
defined as $\text{ReLU}(z) = \max(0, z)$. The output layer has no activation
function (linear output), which is appropriate for continuous action prediction.

### 18.2.4 Loss Function

The loss is the mean squared error (MSE) between predicted and expert actions:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \|\hat{a}_i - a_i^*\|^2
$$

where $N$ is the batch size, $\hat{a}_i$ is the predicted action, and $a_i^*$
is the expert action. This is implemented as:

```python
np.mean((predictions - actions) ** 2)
```

### 18.2.5 Backpropagation

The `_backward` method implements manual backpropagation through the network.
The algorithm proceeds as follows:

**Forward pass with caching.** All intermediate activations
$h_0, h_1, \ldots, h_L$ are stored for use in the backward pass.

**Output gradient.** The gradient of the MSE loss with respect to the output is:

$$
\delta_L = \frac{2}{N}(\hat{a} - a^*)
$$

**Backward recursion.** For each layer $l$ from $L$ down to 1:

1. Compute weight gradient:

$$
\frac{\partial \mathcal{L}}{\partial W_l} = h_{l-1}^T \, \delta_l
$$

2. Compute bias gradient:

$$
\frac{\partial \mathcal{L}}{\partial b_l} = \sum_{i} \delta_l^{(i)}
$$

where the sum is over the batch dimension.

3. Propagate error to previous layer (for $l > 1$):

$$
\delta_{l-1} = (\delta_l \, W_l^T) \odot \mathbb{1}[h_{l-1} > 0]
$$

The term $\mathbb{1}[h_{l-1} > 0]$ is the ReLU gradient, which passes the
error through only for units that were active (positive) during the forward
pass.

### 18.2.6 Training Loop

The `train` method implements the full training procedure:

1. **Data preparation.** Calls `dataset.to_state_action_pairs()` to get
   `(observations, actions)` arrays. Splits into training and validation sets
   based on `validation_split` (default 10%).

2. **Epoch loop.** For each epoch:
   a. Shuffle training data with a random permutation.
   b. Iterate over mini-batches of size `config.batch_size`.
   c. For each mini-batch, compute gradients via `_backward` and update weights.

3. **Weight update with decay.** The update rule is mini-batch SGD with L2
   weight decay (also known as L2 regularization or ridge regression):

$$
W \leftarrow W - \eta \left(\frac{\partial \mathcal{L}}{\partial W} + \lambda W\right)
$$

$$
b \leftarrow b - \eta \, \frac{\partial \mathcal{L}}{\partial b}
$$

where $\eta$ is the learning rate and $\lambda$ is the weight decay
coefficient. Note that weight decay is applied to weights only, not biases,
which is standard practice to avoid biasing the network toward zero output.

4. **Metrics.** Training and validation loss are recorded at each epoch and
   returned as the training history dictionary.

### 18.2.7 Inference

The `predict` method runs a single forward pass:

```python
action = bc.predict(observation, deterministic=True)
```

If the input is 1-dimensional (a single observation), it is reshaped to a
batch of size 1, processed, and the output is flattened back to 1D.

### 18.2.8 Persistence

The `save` and `load` methods serialize the policy weights and configuration to
a NumPy `.npz` file. This includes the observation and action dimensions, the
training configuration, and all layer weights and biases.

### 18.2.9 Complete Example

```python
from src.learning.imitation.dataset import DemonstrationDataset
from src.learning.imitation.learners import BehaviorCloning, TrainingConfig

# Load demonstrations
dataset = DemonstrationDataset.load("demonstrations.json")

# Filter and augment
dataset = dataset.filter_successful()
dataset = dataset.augment(noise_std=0.01, num_augmentations=5)

# Configure and train
config = TrainingConfig(
    epochs=200,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-5,
    hidden_sizes=[256, 256, 128],
)
bc = BehaviorCloning(
    observation_dim=42,
    action_dim=7,
    config=config,
)
history = bc.train(dataset, validation_split=0.15)

# Evaluate
obs = env.reset()[0]
action = bc.predict(obs)

# Save
bc.save("bc_policy.npz")
```

---

## 18.3 DAgger (Dataset Aggregation)

DAgger addresses the distribution shift problem inherent in behavior cloning.
In BC, the policy is trained on expert states, but at test time it visits states
generated by its own (imperfect) actions. DAgger iteratively collects on-policy
data, labels it with expert actions, and retrains.

### 18.3.1 Algorithm Overview

The DAgger algorithm proceeds as follows:

1. **Initialize.** Train an initial policy $\pi_0$ via behavior cloning on the
   expert dataset $\mathcal{D}_0$.

2. **For each iteration** $i = 1, \ldots, N$:

   a. Compute a mixing parameter $\beta_i$ that determines the probability of
   using the expert policy for execution.

   b. Collect trajectories using the mixed policy:

$$
\pi_{mix} = \beta_i \, \pi_{expert} + (1 - \beta_i) \, \pi_{learner}
$$

At each timestep, with probability $\beta_i$ the expert action is executed,
otherwise the learned policy's action is executed.

c. Regardless of which action was executed, label every visited state with the
expert's action: $a^* = \pi_{expert}(s)$.

d. Aggregate the new data into the dataset: $\mathcal{D}_{i} = \mathcal{D}_{i-1} \cup \mathcal{D}_{new}$.

e. Retrain the policy on the aggregated dataset.

### 18.3.2 Beta Schedules

The `train_online` method supports two $\beta$ schedules:

- **Linear decay** (`beta_schedule="linear"`):

$$
\beta_i = 1 - \frac{i}{N}
$$

This starts with pure expert execution and linearly transitions to pure
policy execution.

- **Exponential decay** (any other value):

$$
\beta_i = 0.5^i
$$

This decays more rapidly, reaching $\beta = 0.0625$ by iteration 4.

### 18.3.3 Implementation

The `DAgger` class wraps a `BehaviorCloning` instance internally. The workflow
is:

```python
from src.learning.imitation.learners import DAgger, TrainingConfig

config = TrainingConfig(epochs=50, batch_size=256)
dagger = DAgger(
    observation_dim=42,
    action_dim=7,
    config=config,
)

# Phase 1: Offline training on initial dataset
dagger.train(initial_dataset)

# Phase 2: Online training with expert queries
def expert_policy(obs):
    """Expert policy callable."""
    return compute_expert_action(obs)

results = dagger.train_online(
    env=gym_env,
    expert=expert_policy,
    iterations=10,
    trajectories_per_iter=10,
    max_steps=500,
    beta_schedule="linear",
)
```

The `train_online` method returns a dictionary with:

- `iteration_rewards` -- Mean trajectory reward at each iteration.
- `dataset_size` -- Growing dataset size at each iteration.

### 18.3.4 Trajectory Collection

At each DAgger iteration, `trajectories_per_iter` trajectories are collected.
For each trajectory:

1. The environment is reset.
2. At each timestep:
   - Both the learner's action and the expert's action are computed.
   - The expert action is always stored in the demonstration (for labeling).
   - The executed action is chosen based on $\beta$: expert with probability
     $\beta$, learner otherwise.
3. The trajectory is stored as a `Demonstration` with `source="dagger"`.
4. All new demonstrations are added to the aggregated dataset.
5. The BC policy is retrained on the full aggregated dataset.

---

## 18.4 GAIL (Generative Adversarial Imitation Learning)

GAIL frames imitation learning as a game between a policy (generator) and a
discriminator. The discriminator learns to distinguish expert trajectories from
policy trajectories, while the policy learns to fool the discriminator.

### 18.4.1 Network Architecture

GAIL maintains two neural networks:

1. **Policy network** $\pi_\theta(s)$: Maps states to actions. Architecture is
   identical to the BC policy (MLP with ReLU activations and linear output).

2. **Discriminator network** $D_\phi(s, a)$: Maps state-action pairs to a
   probability in $[0, 1]$. The input is the concatenation of state and action
   vectors. The architecture is an MLP with ReLU hidden layers and a sigmoid
   output:

$$
D_\phi(s, a) = \sigma\bigl(W_L \, h_{L-1} + b_L\bigr)
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

### 18.4.2 Discriminator Training

The discriminator is trained to output values close to 1 for expert data and
close to 0 for policy data. The discriminator loss is the binary cross-entropy:

$$
\mathcal{L}_D = -\mathbb{E}_{(s,a) \sim \pi_{expert}}\bigl[\log D_\phi(s, a)\bigr] - \mathbb{E}_{(s,a) \sim \pi_\theta}\bigl[\log(1 - D_\phi(s, a))\bigr]
$$

In practice this is computed as:

```python
disc_loss = -np.mean(np.log(expert_preds + eps) + np.log(1 - policy_preds + eps))
```

where `eps = 1e-8` prevents numerical overflow in the logarithm.

### 18.4.3 Policy Reward

Instead of using the environment's reward function, GAIL defines the policy
reward as:

$$
r(s, a) = -\log\bigl(1 - D_\phi(s, a)\bigr)
$$

This reward is high when the discriminator believes the state-action pair is
from the expert (i.e., $D_\phi(s, a)$ is close to 1), and low when the
discriminator recognizes it as policy-generated.

The `get_reward` method computes this reward for a single state-action pair:

```python
gail = GAIL(observation_dim=42, action_dim=7)
reward = gail.get_reward(state, action)
```

### 18.4.4 Policy Update

In the full GAIL algorithm, the policy is updated using a reinforcement learning
algorithm (typically PPO or TRPO) with the discriminator reward replacing the
environment reward. The current UpstreamDrift implementation provides a
simplified training loop suitable for initial experiments; full RL integration
should use the RL environments from Chapter 17 with the GAIL reward.

### 18.4.5 Training

```python
from src.learning.imitation.learners import GAIL, TrainingConfig

config = TrainingConfig(
    epochs=500,
    batch_size=256,
    learning_rate=1e-4,
    hidden_sizes=[256, 256],
)
gail = GAIL(
    observation_dim=42,
    action_dim=7,
    config=config,
)

# Train on expert demonstrations
history = gail.train(dataset)
# history contains 'discriminator_loss' and 'policy_loss'

# Use as reward function in RL
reward = gail.get_reward(state, action)

# Predict action
action = gail.predict(observation, deterministic=False)
```

When `deterministic=False`, Gaussian noise with standard deviation 0.1 is added
to the action for exploration.

### 18.4.6 Persistence

GAIL saves both the policy and discriminator networks via `save(path)` and
restores them with `load(path)`, using NumPy `.npz` format.

---

# Chapter 19: Sim-to-Real Transfer

The `learning/sim2real/` module provides tools for bridging the gap between
simulated training environments and real robot hardware. It includes domain
randomization for training robust policies and system identification for
calibrating simulation parameters.

Source files referenced in this chapter:

- `src/learning/sim2real/domain_randomization.py`
- `src/learning/sim2real/system_identification.py`

---

## 19.1 Domain Randomization

Domain randomization is a technique that trains policies in simulation with
randomly varied physical parameters. The intuition is that if the policy
performs well across a wide distribution of simulated dynamics, the real-world
dynamics are likely to fall within that distribution, enabling zero-shot
transfer.

### 19.1.1 DomainRandomizationConfig

The `DomainRandomizationConfig` dataclass defines the randomization ranges for
each physical parameter. Ranges are specified as `(min, max)` tuples,
interpreted as multipliers relative to nominal values for most parameters.

**Dynamics parameters:**

| Field                  | Default         | Description                    |
| ---------------------- | --------------- | ------------------------------ |
| `mass_range`           | `(0.8, 1.2)`    | Link mass scale factor.        |
| `friction_range`       | `(0.5, 1.5)`    | Friction coefficient scale.    |
| `damping_range`        | `(0.8, 1.2)`    | Joint damping scale.           |
| `motor_strength_range` | `(0.9, 1.1)`    | Motor strength scale.          |
| `inertia_range`        | `(0.9, 1.1)`    | Link inertia scale.            |
| `center_of_mass_range` | `(-0.01, 0.01)` | CoM position offset in meters. |

**Delays and noise:**

| Field                     | Default  | Description                     |
| ------------------------- | -------- | ------------------------------- |
| `action_delay_range`      | `(0, 3)` | Action delay in timesteps.      |
| `observation_delay_range` | `(0, 2)` | Observation delay in timesteps. |
| `observation_noise_std`   | `0.01`   | Gaussian noise on observations. |
| `action_noise_std`        | `0.01`   | Gaussian noise on actions.      |

**Environment parameters:**

| Field                  | Default       | Description                 |
| ---------------------- | ------------- | --------------------------- |
| `gravity_range`        | `(9.5, 10.1)` | Gravity magnitude in m/s^2. |
| `floor_friction_range` | `(0.5, 1.5)`  | Floor friction scale.       |

**Actuator parameters:**

| Field                 | Default         | Description                       |
| --------------------- | --------------- | --------------------------------- |
| `joint_offset_range`  | `(-0.02, 0.02)` | Joint position offset in radians. |
| `actuator_gain_range` | `(0.95, 1.05)`  | Actuator gain scale.              |

**Enable flags.** Each category of randomization can be enabled or disabled
independently:

| Flag                 | Default | Controls                      |
| -------------------- | ------- | ----------------------------- |
| `randomize_mass`     | `True`  | Mass randomization.           |
| `randomize_friction` | `True`  | Friction randomization.       |
| `randomize_damping`  | `True`  | Damping randomization.        |
| `randomize_motor`    | `True`  | Motor strength randomization. |
| `randomize_gravity`  | `True`  | Gravity randomization.        |
| `randomize_delays`   | `True`  | Delay randomization.          |
| `randomize_noise`    | `True`  | Noise randomization.          |

### 19.1.2 The DomainRandomizer Class

`DomainRandomizer` applies randomization to a physics engine and manages the
delay and noise pipelines.

**Construction.**

```python
from src.learning.sim2real.domain_randomization import (
    DomainRandomizer,
    DomainRandomizationConfig,
)

config = DomainRandomizationConfig(
    mass_range=(0.7, 1.3),
    friction_range=(0.3, 2.0),
    randomize_delays=True,
)
randomizer = DomainRandomizer(engine, config)
```

During construction, the randomizer:

1. Stores the engine reference.
2. Calls `_store_nominal_params()` to read and cache the engine's current
   parameter values (masses, damping, friction, motor strength, gravity,
   actuator gains). These nominal values serve as the baseline for
   multiplicative randomization.
3. Initializes empty delay buffers and a random number generator.

**The `randomize` method.** This is the main entry point, typically called at
the start of each episode:

```python
randomization_values = randomizer.randomize(seed=42)
```

The method applies the following randomizations in order:

1. **Mass.** A single scale factor is sampled uniformly from `mass_range` and
   applied to all link masses:

$$
m_i^{new} = m_i^{nom} \cdot \text{Uniform}(m_{lo}, m_{hi})
$$

2. **Friction.** A scale factor is sampled and applied to all friction
   coefficients.

3. **Damping.** A scale factor is sampled and applied to all joint damping
   values.

4. **Motor strength.** A scale factor is sampled and applied to all motor
   strength values.

5. **Gravity.** The gravity magnitude is sampled uniformly from `gravity_range`
   and set as the z-component (assuming z-up convention):

$$
g^{new} = [0, 0, -\text{Uniform}(g_{lo}, g_{hi})]
$$

6. **Delays.** Integer delays are sampled for both actions and observations:

$$
k_{action} \sim U\{0, 1, \ldots, k_{max}\}
$$

$$
k_{obs} \sim U\{0, 1, \ldots, k_{max}^{obs}\}
$$

Delay buffers are cleared when new delays are sampled.

The method returns a dictionary recording the sampled values, which is useful
for logging and debugging.

**The `reset_to_nominal` method.** Restores all engine parameters to their
original values and clears delay buffers. Call this before `randomize` to
ensure a clean starting state.

### 19.1.3 Action Delay Buffer

Action delays simulate the communication latency between the control computer
and the robot actuators. The `apply_action_with_delay` method implements a
queue-based delay:

$$
a_{applied}(t) = a_{commanded}(t - k)
$$

where $k \sim U[0, k_{max}]$ is the sampled delay in timesteps.

The implementation maintains a FIFO buffer. At each timestep:

1. The current action is appended to the buffer.
2. If the buffer has accumulated more entries than the delay, the oldest entry
   is popped and returned.
3. During the initial fill period (when the buffer has fewer entries than the
   delay), a zero action is returned.

After the delayed action is retrieved, action noise is applied if enabled:

$$
a_{noisy} = a_{delayed} + \mathcal{N}(0, \sigma_a^2 I)
$$

### 19.1.4 Observation Delay

Observation delays simulate sensor processing latency. The
`get_observation_with_delay` method works analogously to the action delay:

$$
o_{received}(t) = o_{true}(t - k_{obs})
$$

During the initial fill period, the first observation in the buffer is returned
(rather than zeros), since returning a zero observation would be unrealistic.
After retrieval, observation noise is added:

$$
o_{noisy} = o_{delayed} + \mathcal{N}(0, \sigma_o^2 I)
$$

### 19.1.5 Batch Sampling for Parallel Environments

The `sample_randomization_batch` method generates a batch of randomization
configurations for use with vectorized environments:

```python
configs = randomizer.sample_randomization_batch(batch_size=64)
# configs is a list of 64 dictionaries, each with different randomization values
```

For each sample, the method calls `randomize` with a unique seed, records the
configuration, and then calls `reset_to_nominal` to restore the engine state
before sampling the next configuration. This allows parallel environments to
each apply a different configuration from the batch.

### 19.1.6 Integration with RL Training

A typical training loop with domain randomization:

```python
randomizer = DomainRandomizer(engine, config)

for episode in range(n_episodes):
    # Randomize at episode start
    randomizer.randomize()
    obs, info = env.reset()

    done = False
    while not done:
        action = policy(obs)

        # Apply action with delay and noise
        delayed_action = randomizer.apply_action_with_delay(action)
        obs, reward, terminated, truncated, info = env.step(delayed_action)

        # Apply observation delay and noise
        obs = randomizer.get_observation_with_delay(obs)

        done = terminated or truncated

    # Reset to nominal before next randomization
    randomizer.reset_to_nominal()
```

---

## 19.2 System Identification

While domain randomization trains policies that are robust to parameter
uncertainty, system identification takes the complementary approach of
calibrating the simulation to match the real robot as closely as possible.

### 19.2.1 IdentificationResult

The `IdentificationResult` dataclass holds the output of the identification
process:

| Field               | Type             | Description                     |
| ------------------- | ---------------- | ------------------------------- | ---------------------------- |
| `identified_params` | `dict[str, float | NDArray]`                       | Identified parameter values. |
| `residual_error`    | `float`          | Final optimization residual.    |
| `iterations`        | `int`            | Number of iterations performed. |
| `converged`         | `bool`           | Whether optimization converged. |

### 19.2.2 The SystemIdentifier Class

`SystemIdentifier` finds simulation parameters that minimize the discrepancy
between simulated and real trajectories.

**Construction.**

```python
from src.learning.sim2real.system_identification import SystemIdentifier

identifier = SystemIdentifier(
    model=engine,
    param_bounds={
        "mass_scale": (0.5, 2.0),
        "friction_scale": (0.2, 3.0),
        "damping_scale": (0.5, 2.0),
        "motor_scale": (0.5, 1.5),
        "com_offset_x": (-0.05, 0.05),
        "com_offset_y": (-0.05, 0.05),
        "com_offset_z": (-0.05, 0.05),
    },
)
```

If `param_bounds` is not provided, sensible defaults are used. The constructor
also caches the nominal engine parameters.

**Default parameter bounds:**

| Parameter        | Lower | Upper | Description                 |
| ---------------- | ----- | ----- | --------------------------- |
| `mass_scale`     | 0.5   | 2.0   | Overall mass multiplier.    |
| `friction_scale` | 0.2   | 3.0   | Friction coefficient scale. |
| `damping_scale`  | 0.5   | 2.0   | Joint damping scale.        |
| `motor_scale`    | 0.5   | 1.5   | Motor strength scale.       |
| `com_offset_x`   | -0.05 | 0.05  | CoM x-offset in meters.     |
| `com_offset_y`   | -0.05 | 0.05  | CoM y-offset in meters.     |
| `com_offset_z`   | -0.05 | 0.05  | CoM z-offset in meters.     |

### 19.2.3 Identification Objective

The identification problem is formulated as:

$$
\min_\theta \frac{1}{N} \sum_{i=1}^{N} \text{MSE}\bigl(\text{sim}(\theta, s_0^{(i)}, a^{(i)}),\; x_{real}^{(i)}\bigr)
$$

where:

- $\theta$ is the parameter vector being optimized.
- $N$ is the number of real-robot trajectories.
- $\text{sim}(\theta, s_0, a)$ simulates a trajectory from initial state $s_0$
  under action sequence $a$ with parameters $\theta$.
- $x_{real}^{(i)}$ is the recorded real-robot state trajectory.
- MSE is the mean squared error between the simulated and real trajectories.

The MSE is computed over the full state vector $[q, \dot{q}]$ (positions and
velocities concatenated):

$$
\text{MSE} = \frac{1}{T \cdot 2n_q} \sum_{t=0}^{T} \|x_{sim}^{(t)} - x_{real}^{(t)}\|^2
$$

An optional weight vector can emphasize certain state components (e.g.,
weighting positions more than velocities).

### 19.2.4 Optimization Algorithm

The implementation uses coordinate descent with numerical perturbations. This
gradient-free approach is chosen for robustness with non-smooth simulation
dynamics.

**Algorithm:**

```
Initialize: theta = [1, 1, ..., 1] (nominal scales)
best_error = objective(theta)

For iteration = 1 to max_iterations:
    improved = False
    For each parameter i:
        For delta in [+0.1, -0.1, +0.05, -0.05, +0.01, -0.01]:
            theta_test = theta.copy()
            theta_test[i] = clip(theta_test[i] + delta, lower[i], upper[i])
            error = objective(theta_test)
            If error < best_error - tolerance:
                best_error = error
                theta = theta_test
                improved = True
    If not improved:
        converged = True
        Break
```

For each parameter, six perturbation magnitudes are tried:
$\{+0.1, -0.1, +0.05, -0.05, +0.01, -0.01\}$. The coarse perturbations
($\pm 0.1$) allow rapid initial progress, while the fine perturbations
($\pm 0.01$) refine the solution near the optimum.

The algorithm converges when no parameter perturbation improves the objective
by more than `tolerance` (default $10^{-6}$).

### 19.2.5 Trajectory Simulation

The `_simulate_trajectory` method runs a forward simulation from an initial
state under a given action sequence:

1. Set initial joint positions $q_0$ and velocities $v_0$ in the engine.
2. For each timestep:
   a. Apply the torque action to the engine.
   b. Step the simulation by $\Delta t$.
   c. Record the resulting state $[q, \dot{q}]$.
3. Return the full state trajectory as a `(T+1, 2n_q)` array.

The timestep $\Delta t$ is computed from the demonstration timestamps:

$$
\Delta t = \frac{1}{T-1} \sum_{t=1}^{T-1} (t_{k+1} - t_k)
$$

### 19.2.6 Reality Gap Metrics

The `compute_reality_gap` method provides a comprehensive set of metrics
quantifying the simulation-to-reality discrepancy:

| Metric                   | Formula                                                |
| ------------------------ | ------------------------------------------------------ | ---------------------------------------------- | --- |
| `total_mse`              | $\frac{1}{T \cdot 2n_q} \sum \|x_{sim} - x_{real}\|^2$ |
| `position_mse`           | MSE over position components only.                     |
| `velocity_mse`           | MSE over velocity components only.                     |
| `max_position_error`     | $\max\_{t,j}                                           | q*{sim}^{(t,j)} - q*{real}^{(t,j)}             | $   |
| `max_velocity_error`     | $\max\_{t,j}                                           | \dot{q}_{sim}^{(t,j)} - \dot{q}_{real}^{(t,j)} | $   |
| `mean_position_error`    | Mean absolute position error.                          |
| `mean_velocity_error`    | Mean absolute velocity error.                          |
| `trajectory_length`      | Number of timesteps compared.                          |
| `joint_{j}_position_mse` | Per-joint position MSE.                                |
| `joint_{j}_velocity_mse` | Per-joint velocity MSE.                                |

### 19.2.7 Validation

The `validate_identification` method evaluates the identified parameters on
held-out test trajectories. This is essential for checking that the identified
parameters generalize and are not overfit to the training trajectories.

```python
# Split trajectories
train_trajs = trajectories[:8]
test_trajs = trajectories[8:]

# Identify
result = identifier.identify_from_trajectories(train_trajs)

# Validate
val_metrics = identifier.validate_identification(
    test_trajectories=test_trajs,
    identified_params=result.identified_params,
)
print(f"Validation error: {val_metrics['mean_error']:.6f}")
print(f"Std error: {val_metrics['std_error']:.6f}")
print(f"Max error: {val_metrics['max_error']:.6f}")
```

The validation metrics include `mean_error`, `std_error`, `max_error`,
`min_error`, and `n_trajectories`.

### 19.2.8 Combined Workflow

A recommended workflow combining system identification with domain
randomization:

1. Collect real-robot trajectories (10-20 short trajectories).
2. Run system identification to find the best parameter vector $\theta^*$.
3. Set the nominal simulation parameters to $\theta^*$.
4. Apply domain randomization centered around $\theta^*$, using narrower ranges
   than the defaults since the nominal parameters are now closer to reality.
5. Train the RL policy with domain randomization.

---

# Chapter 20: Deployment Infrastructure

The `deployment/` module provides the complete infrastructure for deploying
trained policies to real robot hardware. It covers real-time control loops,
digital twin synchronization, safety monitoring, and teleoperation.

Source files referenced in this chapter:

- `src/deployment/realtime/controller.py`
- `src/deployment/realtime/state.py`
- `src/deployment/digital_twin/twin.py`
- `src/deployment/digital_twin/estimator.py`
- `src/deployment/safety/monitor.py`
- `src/deployment/safety/collision.py`
- `src/deployment/teleoperation/interface.py`
- `src/deployment/teleoperation/devices.py`

---

## 20.1 Real-Time Control

The real-time control subsystem manages the high-frequency control loop that
interfaces with robot hardware. It is designed around a callback model where
user-defined control functions are executed at a fixed frequency.

### 20.1.1 Communication Protocols

The `CommunicationType` enum defines supported communication protocols:

| Member       | Value          | Description                             |
| ------------ | -------------- | --------------------------------------- |
| `ETHERCAT`   | `"ethercat"`   | Industrial real-time fieldbus.          |
| `ROS2`       | `"ros2"`       | ROS 2 middleware.                       |
| `UDP`        | `"udp"`        | Raw UDP sockets.                        |
| `SIMULATION` | `"simulation"` | Simulated interface (no real hardware). |

The `SIMULATION` protocol is always available and is the default. It allows
testing the full deployment pipeline without hardware.

### 20.1.2 RobotConfig

`RobotConfig` specifies the hardware parameters of the robot:

| Field                | Type                | Default        | Description              |
| -------------------- | ------------------- | -------------- | ------------------------ | ---------------------------- |
| `name`               | `str`               | required       | Robot identifier string. |
| `n_joints`           | `int`               | required       | Number of joints.        |
| `joint_names`        | `list[str]`         | auto-generated | Names for each joint.    |
| `joint_limits_lower` | `NDArray            | None`          | `None`                   | Lower position limits (rad). |
| `joint_limits_upper` | `NDArray            | None`          | `None`                   | Upper position limits (rad). |
| `velocity_limits`    | `NDArray            | None`          | `None`                   | Maximum velocities (rad/s).  |
| `torque_limits`      | `NDArray            | None`          | `None`                   | Maximum torques (Nm).        |
| `communication_type` | `CommunicationType` | `SIMULATION`   | Communication protocol.  |
| `ip_address`         | `str`               | `"127.0.0.1"`  | Robot IP (if networked). |
| `port`               | `int`               | `5000`         | Communication port.      |

If `joint_names` is not provided, they are auto-generated as
`["joint_0", "joint_1", ...]`.

### 20.1.3 ControlMode and ControlCommand

The `ControlMode` enum (in `src/deployment/realtime/state.py`) defines the
hardware-level control modes:

| Member      | Value         | Required fields                            |
| ----------- | ------------- | ------------------------------------------ |
| `POSITION`  | `"position"`  | `position_targets`                         |
| `VELOCITY`  | `"velocity"`  | `velocity_targets`                         |
| `TORQUE`    | `"torque"`    | `torque_commands`                          |
| `IMPEDANCE` | `"impedance"` | `position_targets`, `stiffness`, `damping` |
| `HYBRID`    | `"hybrid"`    | Mixed mode (implementation-specific).      |

The `ControlCommand` dataclass packages a control command to send to the robot:

| Field                | Type          | Description                    |
| -------------------- | ------------- | ------------------------------ | ------------------------------------ |
| `timestamp`          | `float`       | Command timestamp (seconds).   |
| `mode`               | `ControlMode` | Control mode for this command. |
| `position_targets`   | `NDArray      | None`                          | Target joint positions (rad).        |
| `velocity_targets`   | `NDArray      | None`                          | Target joint velocities (rad/s).     |
| `torque_commands`    | `NDArray      | None`                          | Commanded joint torques (Nm).        |
| `feedforward_torque` | `NDArray      | None`                          | Feed-forward torque term (Nm).       |
| `stiffness`          | `NDArray      | None`                          | Joint stiffness (impedance mode).    |
| `damping`            | `NDArray      | None`                          | Joint damping (impedance mode).      |
| `gripper_command`    | `float        | None`                          | Gripper position (0=closed, 1=open). |

**Validation.** The `validate(n_joints)` method checks that the required fields
for the current mode are present and have the correct dimension. It raises
`ValueError` on any inconsistency.

**Factory methods.** Three convenience constructors simplify command creation:

```python
cmd = ControlCommand.position_command(timestamp, positions, feedforward=None)
cmd = ControlCommand.torque_command(timestamp, torques)
cmd = ControlCommand.impedance_command(timestamp, positions, stiffness, damping)
```

### 20.1.4 RobotState

`RobotState` encapsulates all sensor data received from the robot at a single
timestep:

| Field                | Type                | Description                      |
| -------------------- | ------------------- | -------------------------------- | ------------------------------------- |
| `timestamp`          | `float`             | Measurement timestamp (seconds). |
| `joint_positions`    | `NDArray`           | Joint positions (rad).           |
| `joint_velocities`   | `NDArray`           | Joint velocities (rad/s).        |
| `joint_torques`      | `NDArray`           | Measured joint torques (Nm).     |
| `ft_wrenches`        | `dict[str, NDArray] | None`                            | Force/torque sensor readings by name. |
| `imu_data`           | `IMUReading         | None`                            | IMU sensor data.                      |
| `contact_states`     | `list[bool]         | None`                            | Binary contact flags.                 |
| `motor_temperatures` | `NDArray            | None`                            | Motor temperatures (Celsius).         |
| `battery_level`      | `float              | None`                            | Battery level (0.0 to 1.0).           |

The `get_state_vector()` method returns the concatenated state $[q, \dot{q}]$.
The `get_ft_wrench(sensor_name)` method looks up a specific force/torque sensor.

The `IMUReading` dataclass stores inertial measurement unit data:

| Field                 | Shape  | Description                        |
| --------------------- | ------ | ---------------------------------- |
| `timestamp`           | scalar | Measurement time.                  |
| `linear_acceleration` | `(3,)` | Accelerometer [ax, ay, az] m/s^2.  |
| `angular_velocity`    | `(3,)` | Gyroscope [wx, wy, wz] rad/s.      |
| `orientation`         | `(4,)` | Orientation quaternion (optional). |

### 20.1.5 TimingStatistics

`TimingStatistics` records control loop performance:

| Field             | Description                                           |
| ----------------- | ----------------------------------------------------- |
| `mean_cycle_time` | Average time per control cycle (seconds).             |
| `max_cycle_time`  | Worst-case cycle time.                                |
| `min_cycle_time`  | Best-case cycle time.                                 |
| `std_cycle_time`  | Standard deviation of cycle times.                    |
| `jitter`          | Maximum deviation from target period.                 |
| `overruns`        | Number of times the cycle exceeded the target period. |
| `total_cycles`    | Total number of control cycles executed.              |
| `uptime`          | Total running time in seconds.                        |

Jitter is computed as:

$$
\text{jitter} = \max_t |T_{cycle}(t) - T_{target}|
$$

where $T_{target} = 1/f_{control}$ is the target period.

### 20.1.6 The RealTimeController Class

`RealTimeController` orchestrates the control loop.

**Construction.**

```python
from src.deployment.realtime.controller import RealTimeController

controller = RealTimeController(
    control_frequency=1000.0,  # 1 kHz
    communication_type="simulation",
)
```

The controller stores the frequency and computes `dt = 1.0 / control_frequency`.
It initializes thread locks for state and command buffers, ensuring thread-safe
access from the main thread while the control loop runs in a background thread.

**Connection lifecycle.**

```python
config = RobotConfig(name="my_robot", n_joints=7)

# Connect
success = controller.connect(config)  # Returns True on success

# ... use controller ...

# Disconnect (stops loop if running)
controller.disconnect()
```

The `connect` method dispatches to protocol-specific connection handlers
(`_connect_ros2`, `_connect_udp`, `_connect_ethercat`) or succeeds immediately
for the `SIMULATION` protocol.

**Setting the control callback.**

```python
def my_controller(state: RobotState) -> ControlCommand:
    """User-defined control law."""
    # Compute desired torques from state
    error = desired_q - state.joint_positions
    tau = kp * error - kd * state.joint_velocities
    return ControlCommand.torque_command(
        timestamp=state.timestamp,
        torques=tau,
    )

controller.set_control_callback(my_controller)
```

The callback function receives the current `RobotState` and must return a
`ControlCommand`. It is called at `control_frequency` Hz.

**Starting and stopping.**

```python
controller.start()   # Starts background control thread
# ... robot is running ...
controller.stop()    # Stops loop, sends zero torque command
```

The `start` method raises `RuntimeError` if the controller is not connected or
no callback is set. The `stop` method sets a flag, waits for the control thread
to finish (with a 2-second timeout), and sends a zero-torque safety command.

**The control loop.** The `_control_loop` method runs in a daemon thread and
implements a fixed-rate loop:

```
next_cycle_time = now()

while not should_stop:
    cycle_start = now()

    state = read_state()
    command = control_callback(state)
    command.validate(n_joints)
    send_command(command)

    cycle_time = now() - cycle_start
    record_timing(cycle_time)

    if cycle_time > dt:
        overruns += 1

    sleep_until(next_cycle_time + dt)
```

If the sleep time is negative (a deadline was missed), the timing reference is
reset to prevent cascading overruns. Cycle times are recorded for the
`get_timing_stats()` method.

**State and command access.** Thread-safe access to the most recent state and
command is provided by:

```python
state = controller.get_last_state()
command = controller.get_last_command()
state = controller.wait_for_state(timeout=1.0)  # Blocks until new state
```

The `wait_for_state` method polls at 1 kHz until a new state arrives or the
timeout expires.

---

## 20.2 Digital Twin

The digital twin system maintains a synchronized simulation that mirrors the
physical robot's state, enabling prediction, anomaly detection, and contact
estimation.

### 20.2.1 The DigitalTwin Class

```python
from src.deployment.digital_twin.twin import DigitalTwin

twin = DigitalTwin(
    sim_engine=physics_engine,
    real_interface=real_time_controller,
)
```

The constructor takes a physics engine (for simulation) and a real-time
controller (for reading hardware state). It initializes a `StateEstimator` for
sensor fusion.

**Synchronization.** The `synchronize()` method reads the latest hardware state
and updates the simulation to match:

```python
error = twin.synchronize()
```

The synchronization error is computed as:

$$
e_{sync} = \|q_{real} - q_{sim}\| + 0.1 \, \|\dot{q}_{real} - \dot{q}_{sim}\|
$$

The velocity error is weighted by 0.1 because velocities are typically noisier
than positions.

**Prediction.** The `predict` method rolls out the simulation forward under a
given control sequence:

```python
trajectory = twin.predict(
    horizon=0.5,         # 0.5 seconds ahead
    control_sequence=u,  # Shape (n_steps, n_controls)
    dt=0.001,            # 1 kHz simulation
)
# trajectory has shape (n_steps + 1, 2 * n_dof)
```

The method saves the current simulation state, performs the rollout, and then
restores the original state. This allows prediction without disturbing the
twin's synchronized state.

### 20.2.2 Anomaly Detection

The `detect_anomaly` method compares simulation predictions with actual hardware
readings to detect deviations:

$$
\|x_{sim} - x_{real}\| > \epsilon
$$

where $\epsilon$ is the anomaly threshold (default 0.1 rad).

**AnomalyType enum.** Detected anomalies are categorized:

| Type                  | Description                                    |
| --------------------- | ---------------------------------------------- |
| `COLLISION`           | Unexpected contact detected from torque spike. |
| `SLIP`                | Slip detected (contact loss).                  |
| `STUCK`               | Joint is stuck (no motion despite torque).     |
| `MODEL_MISMATCH`      | Simulation-reality discrepancy.                |
| `SENSOR_FAULT`        | Sensor reading is out of range.                |
| `COMMUNICATION_ERROR` | Communication timeout or error.                |
| `JOINT_LIMIT`         | Joint approaching or exceeding limits.         |
| `VELOCITY_LIMIT`      | Velocity exceeding safe limits.                |
| `TORQUE_LIMIT`        | Torque exceeding safe limits.                  |

**AnomalyReport dataclass.**

| Field                | Type          | Description                       |
| -------------------- | ------------- | --------------------------------- | ------------------------------ |
| `timestamp`          | `float`       | Detection time.                   |
| `anomaly_type`       | `AnomalyType` | Category of anomaly.              |
| `severity`           | `float`       | Severity level, 0.0 to 1.0.       |
| `affected_joints`    | `list[int]`   | Indices of affected joints.       |
| `description`        | `str`         | Human-readable description.       |
| `recommended_action` | `str`         | Suggested response.               |
| `confidence`         | `float`       | Detection confidence, 0.0 to 1.0. |
| `raw_data`           | `dict         | None`                             | Raw sensor data for debugging. |

The `detect_anomaly` method checks two conditions:

1. **Position mismatch.** If any per-joint position error exceeds the threshold,
   a `MODEL_MISMATCH` anomaly is reported. The severity is normalized by 0.5 rad
   and capped at 1.0.

2. **Torque spike.** If any per-joint torque discrepancy exceeds 10 Nm, a
   `COLLISION` anomaly is reported with severity 0.8.

**Anomaly history.** All detected anomalies are stored in an internal list.
The `get_anomaly_history(max_age=None)` method retrieves anomalies, optionally
filtering by age. `clear_anomaly_history()` resets the history.

**Threshold tuning.** The `set_anomaly_threshold(threshold)` method adjusts the
detection sensitivity. Lower thresholds increase sensitivity but may produce
more false positives.

### 20.2.3 Contact Estimation

The `get_estimated_contacts` method estimates contact states from sensor data:

- Force/torque sensors: A contact is reported when the force magnitude exceeds
  1.0 N.
- Binary contact sensors: Contacts are reported directly.

The returned list contains dictionaries with contact location, force vector,
torque vector, and magnitude.

### 20.2.4 Virtual Force Estimation

The `compute_virtual_forces` method estimates external forces from the torque
discrepancy between simulation and reality:

$$
\tau_{virtual} = \tau_{real} - \tau_{sim}
$$

In the simplified implementation, the first 6 components of the torque
difference are interpreted as a Cartesian wrench. A full implementation would
use the Jacobian transpose mapping.

### 20.2.5 State Estimator

The `StateEstimator` in `src/deployment/digital_twin/estimator.py` implements
an Extended Kalman Filter (EKF) for fusing noisy sensor data into smooth state
estimates.

**State model.** The estimator tracks a $3n$-dimensional state vector:

$$
x = \begin{bmatrix} q \\ \dot{q} \\ \ddot{q} \end{bmatrix}
$$

consisting of joint positions, velocities, and accelerations for $n$ degrees
of freedom.

**EstimatorConfig:**

| Field                   | Default | Description                            |
| ----------------------- | ------- | -------------------------------------- |
| `process_noise`         | `0.001` | Process noise covariance.              |
| `measurement_noise`     | `0.01`  | Measurement noise covariance.          |
| `use_velocity_filter`   | `True`  | Apply low-pass filter to velocities.   |
| `velocity_filter_alpha` | `0.3`   | Filter coefficient (lower = smoother). |
| `outlier_threshold`     | `3.0`   | Sigma threshold for outlier rejection. |

**Prediction step.** The state transition model assumes constant acceleration:

$$
\hat{x}_{k|k-1} = F \, \hat{x}_{k-1|k-1}
$$

where the state transition matrix is:

$$
F = \begin{bmatrix}
I & I \Delta t & \frac{1}{2} I \Delta t^2 \\
0 & I & I \Delta t \\
0 & 0 & I
\end{bmatrix}
$$

The covariance prediction is:

$$
P_{k|k-1} = F \, P_{k-1|k-1} \, F^T + Q
$$

where $Q = \sigma_p^2 I_{3n}$ is the process noise covariance.

**Measurement model.** The measurement is:

$$
z = \begin{bmatrix} q_{meas} \\ \dot{q}_{meas} \end{bmatrix}
$$

with measurement matrix:

$$
H = \begin{bmatrix} I_n & 0 & 0 \\ 0 & I_n & 0 \end{bmatrix}
$$

**Update step.** The standard Kalman update is:

$$
y = z - H \hat{x}_{k|k-1}
$$

$$
S = H P_{k|k-1} H^T + R
$$

$$
K = P_{k|k-1} H^T S^{-1}
$$

$$
\hat{x}_{k|k} = \hat{x}_{k|k-1} + K y
$$

$$
P_{k|k} = (I - K H) P_{k|k-1}
$$

where $R = \sigma_m^2 I_{2n}$ is the measurement noise covariance.

**Outlier rejection.** Before the update, the `_reject_outliers` method checks
each measurement component against the predicted value. If the residual exceeds
`outlier_threshold` standard deviations (computed from the innovation
covariance), the measurement is replaced with the predicted value:

$$
\text{if } |z_i - \hat{z}_i| > \kappa \, \sigma_i \implies z_i \leftarrow \hat{z}_i
$$

where $\kappa$ is the outlier threshold (default 3.0).

**Velocity filtering.** After the Kalman update, an additional exponential
smoothing filter is applied to the velocity estimate if
`use_velocity_filter=True`:

$$
\hat{\dot{q}}_{filtered} = \alpha \, \hat{\dot{q}}_{KF} + (1 - \alpha) \, \hat{\dot{q}}_{prev}
$$

where $\alpha$ is `velocity_filter_alpha`. This provides additional smoothing
beyond what the Kalman filter achieves.

**API methods:**

```python
estimator = StateEstimator(n_dof=7)

# Update with new measurement
result = estimator.update(robot_state, dt=0.001)
# result = {"position": ..., "velocity": ..., "acceleration": ...}

# Query current estimates
q = estimator.get_position()
qd = estimator.get_velocity()
qdd = estimator.get_acceleration()

# Get uncertainty
pos_std = estimator.get_position_uncertainty()
vel_std = estimator.get_velocity_uncertainty()

# Predict future state
predicted = estimator.predict(dt=0.01, control=qdd_desired)

# Reset
estimator.reset(position=q0, velocity=v0)
```

---

## 20.3 Safety System

The safety system provides real-time monitoring and enforcement of safety
constraints, following guidelines from ISO 10218-1 (industrial robots) and
ISO/TS 15066 (collaborative robots).

### 20.3.1 SafetyLimits

The `SafetyLimits` dataclass defines the safety boundaries:

| Field                    | Default  | Description                               |
| ------------------------ | -------- | ----------------------------------------- |
| `max_joint_velocity`     | required | Per-joint velocity limits (rad/s).        |
| `max_joint_torque`       | required | Per-joint torque limits (Nm).             |
| `max_cartesian_velocity` | `0.25`   | Max EE velocity for collaborative mode.   |
| `max_cartesian_force`    | `150.0`  | Maximum contact force (N).                |
| `workspace_bounds`       | `None`   | Workspace limits [xmin, xmax, ymin, ...]. |
| `forbidden_zones`        | `[]`     | Boxes where the robot must not enter.     |
| `max_contact_force`      | `150.0`  | ISO 10218-1 contact force limit (N).      |
| `max_pressure`           | `110.0`  | ISO/TS 15066 pressure limit (N/cm^2).     |
| `joint_limits_lower`     | `None`   | Per-joint lower position limits (rad).    |
| `joint_limits_upper`     | `None`   | Per-joint upper position limits (rad).    |

The `from_config(robot_config)` class method creates `SafetyLimits` from a
`RobotConfig`, using the robot's declared limits or defaults (2.0 rad/s
velocity, 50 Nm torque).

### 20.3.2 SafetyMonitor

`SafetyMonitor` performs real-time safety checking at the control frequency.

**Construction.**

```python
from src.deployment.safety.monitor import SafetyMonitor, SafetyLimits

monitor = SafetyMonitor(
    robot_config=config,
    limits=SafetyLimits.from_config(config),
)
```

**SafetyStatusLevel enum:**

| Level       | Description                                   |
| ----------- | --------------------------------------------- |
| `OK`        | All checks passed, operation is nominal.      |
| `WARNING`   | Approaching limits, operation can continue.   |
| `ERROR`     | Limits violated, commands should be modified. |
| `EMERGENCY` | Critical violation, emergency stop required.  |

**State checking.** The `check_state(state)` method validates the current robot
state against all safety limits:

1. **Joint velocity limits:**

$$
|\dot{q}_i| \leq \dot{q}_{max,i} \quad \forall i
$$

2. **Joint torque limits:**

$$
|\tau_i| \leq \tau_{max,i} \quad \forall i
$$

3. **Joint position limits (lower):**

$$
q_i \geq q_{min,i} \quad \forall i
$$

4. **Joint position limits (upper):**

$$
q_i \leq q_{max,i} \quad \forall i
$$

5. **Approach warnings.** If any joint is within 0.1 rad of its upper limit,
   a warning is issued.

6. **Emergency stop.** If the emergency stop is active, a violation is reported.

The method returns a `SafetyStatus` with the overall level, a boolean `is_safe`
flag, lists of violations and warnings, and the current `speed_override`.

**Command checking.** The `check_command(command)` method validates a control
command before it is sent to the robot:

- Torque commands are checked against `max_joint_torque`.
- Position targets are checked against joint limits.

**Safe command computation.** The `compute_safe_command(desired, state)` method
modifies a command to ensure it satisfies all safety limits:

1. Apply speed override: scale velocity and torque commands by the speed factor.
2. Clip torque commands to $[-\tau_{max}, \tau_{max}]$.
3. Clip position targets to $[q_{min}, q_{max}]$.

This method is designed to be called in the control loop between the policy
output and the hardware command, providing a safety layer that cannot be
bypassed by the policy.

**Stopping distance.** The `get_stopping_distance(state, body)` method estimates
the minimum distance needed to stop:

$$
d_{stop} = \frac{v_{max}^2}{2 \, a_{max}}
$$

where $v_{max}$ is the maximum joint velocity magnitude and $a_{max} = 2.0$ m/s^2
is a conservative deceleration estimate.

**Speed override.** The `set_speed_override(factor)` method sets a global speed
reduction factor in $[0, 1]$ that scales all velocity and torque commands.

**Human proximity.** When `set_human_nearby(True)` is called, the speed override
is automatically capped at 0.5 (50% speed), per collaborative robot safety
guidelines.

**Emergency stop.** The emergency stop cascade is:

```python
# Detection
monitor.trigger_emergency_stop()
# This sets speed_override to 0.0 and emergency_stop flag to True

# All subsequent check_state() calls will report a violation
# compute_safe_command() will output zero commands due to speed_override = 0.0

# Recovery (manual reset required)
monitor.clear_emergency_stop()
# Restores speed_override to 1.0
```

### 20.3.3 CollisionAvoidance

The `CollisionAvoidance` class in `src/deployment/safety/collision.py`
implements real-time collision checking using artificial potential fields.

**Obstacle types.** The `ObstacleType` enum defines supported obstacle
geometries:

| Type       | Description                        | Dimensions               |
| ---------- | ---------------------------------- | ------------------------ |
| `BOX`      | Axis-aligned box.                  | `[width, depth, height]` |
| `SPHERE`   | Sphere.                            | `[radius]`               |
| `CYLINDER` | Z-axis aligned cylinder.           | `[radius, height]`       |
| `HUMAN`    | Human bounding box (extra safety). | `[width, depth, height]` |
| `DYNAMIC`  | Moving obstacle with velocity.     | (geometry-dependent)     |

**Obstacle dataclass.** Each obstacle has:

- `name` -- Identifier string.
- `obstacle_type` -- Geometry type.
- `position` -- Center position [x, y, z].
- `dimensions` -- Size (type-dependent).
- `velocity` -- For dynamic obstacles.
- `inflation` -- Safety inflation radius (default 0.05 m, 0.3 m for humans).

**Signed distance computation.** The `get_distance(point)` method computes the
signed distance from a query point to the obstacle surface. Negative values
indicate the point is inside the obstacle:

- **Sphere:** $d = \|p - c\| - r - r_{inflate}$
- **Box/Human:** Clamped distance from point to box surface minus inflation.
- **Cylinder:** Combined radial and axial distance.

**Distance gradient.** The `get_gradient(point)` method computes the normalized
gradient of the distance function via central finite differences with step
$\epsilon = 10^{-6}$. The gradient points away from the obstacle.

**HumanState dataclass.** Represents a detected human with position, velocity,
bounding box, optional skeleton joints, detection confidence, and motion flag.
The `to_obstacle()` method converts to an `Obstacle` with `ObstacleType.HUMAN`
and an inflation radius of 0.3 m.

**The CollisionAvoidance class.**

```python
from src.deployment.safety.collision import CollisionAvoidance

collision = CollisionAvoidance(
    robot_model=engine,
    safety_distance=0.1,  # 10 cm minimum clearance
)
```

**Obstacle management:**

```python
collision.add_obstacle(obstacle)
collision.remove_obstacle("obstacle_name")
collision.clear_obstacles()
collision.update_human_position(human_state)
```

**Repulsive potential field.** The `compute_repulsive_field(state)` method
computes artificial potential field repulsive forces in joint space:

For each robot link and each obstacle, if the distance is less than
`_repulsion_distance` (default 0.5 m):

- If inside the obstacle ($d \leq 0$): maximum repulsion force.
- If outside but within range: inverse-square repulsion:

$$
F_{rep} = k_{rep} \left(\frac{1}{d} - \frac{1}{d_{max}}\right) \frac{1}{d^2}
$$

where $k_{rep}$ is `_repulsion_gain` (default 1.0) and $d_{max}$ is
`_repulsion_distance`. The magnitude is capped at `_max_repulsion` (default 10.0).

The Cartesian repulsion force is mapped to joint space and averaged over all
links.

**Path clearance checking.** The `check_path_clearance(trajectory, min_distance)`
method verifies that an entire trajectory maintains sufficient clearance:

```python
is_clear, min_dist = collision.check_path_clearance(
    trajectory=joint_trajectory,  # Shape (n_steps, n_joints)
    min_distance=0.1,
)
```

Returns `(True, min_distance_found)` if all waypoints are clear, or
`(False, min_distance_found)` on the first violation.

**Velocity scaling.** The `get_safe_velocity_scaling(state)` method computes a
scaling factor in $[0, 1]$ based on proximity to obstacles:

- Distance $\leq 0$ (inside obstacle): scale = 0.0
- Distance $< d_{safety}$: scale = $d / d_{safety}$
- Distance $< d_{repulsion}$: scale = $0.5 + 0.5 \cdot \frac{d - d_{safety}}{d_{repulsion} - d_{safety}}$
- Distance $\geq d_{repulsion}$: scale = 1.0

**Minimum distance query.** The `get_minimum_distance(state)` method returns the
closest distance to any obstacle from any robot link.

### 20.3.4 Integrated Safety Pipeline

A complete safety-aware control loop integrates all safety components:

```python
controller = RealTimeController(control_frequency=1000.0)
monitor = SafetyMonitor(robot_config, limits)
collision = CollisionAvoidance(engine, safety_distance=0.1)
twin = DigitalTwin(engine, controller)

def safe_control_callback(state: RobotState) -> ControlCommand:
    # 1. Update state estimator
    estimated = twin._state_estimator.update(state)

    # 2. Check current state safety
    status = monitor.check_state(state)
    if status.level == SafetyStatusLevel.EMERGENCY:
        return ControlCommand.torque_command(state.timestamp, np.zeros(7))

    # 3. Compute desired command from policy
    obs = build_observation(estimated)
    desired_action = policy(obs)
    desired_cmd = ControlCommand.torque_command(state.timestamp, desired_action)

    # 4. Add collision avoidance
    repulsion = collision.compute_repulsive_field(state)
    desired_cmd.torque_commands += repulsion

    # 5. Apply velocity scaling
    vel_scale = collision.get_safe_velocity_scaling(state)
    monitor.set_speed_override(vel_scale)

    # 6. Enforce safety limits
    safe_cmd = monitor.compute_safe_command(desired_cmd, state)

    # 7. Check for anomalies
    anomaly = twin.detect_anomaly()
    if anomaly and anomaly.severity > 0.8:
        monitor.trigger_emergency_stop()

    return safe_cmd
```

---

## 20.4 Teleoperation

The teleoperation subsystem enables remote control of robots through various
input devices, with support for demonstration recording.

### 20.4.1 Input Device Protocol

The `InputDevice` protocol in `src/deployment/teleoperation/devices.py` defines
the interface that all input devices must implement:

| Method                 | Return type       | Description                          |
| ---------------------- | ----------------- | ------------------------------------ |
| `get_pose()`           | `NDArray` (7,)    | Device pose [x,y,z,qw,qx,qy,qz].     |
| `get_twist()`          | `NDArray` (6,)    | Device velocity [vx,vy,vz,wx,wy,wz]. |
| `get_gripper_state()`  | `float`           | Gripper openness (0=closed, 1=open). |
| `set_force_feedback()` | `None`            | Send haptic feedback wrench.         |
| `get_buttons()`        | `dict[str, bool]` | Button press states.                 |

All devices also support `connect()`, `disconnect()`, and `update()` for
lifecycle management.

### 20.4.2 Supported Devices

Four device implementations are provided:

**KeyboardMouseInput.** A fallback device for testing without specialized
hardware. Keyboard keys map to discrete velocity commands:

| Key        | Direction | Velocity |
| ---------- | --------- | -------- |
| `forward`  | +x        | 0.1 m/s  |
| `backward` | -x        | 0.1 m/s  |
| `left`     | +y        | 0.1 m/s  |
| `right`    | -y        | 0.1 m/s  |
| `up`       | +z        | 0.1 m/s  |
| `down`     | -z        | 0.1 m/s  |

Gripper is controlled by `open_gripper` and `close_gripper` keys. Key states
are set externally via `set_key_state(key, pressed)`.

**SpaceMouseInput.** Interface for 3Dconnexion SpaceMouse 6-DOF controllers.
Provides continuous position and rotation input with configurable sensitivity
via `set_sensitivity(value)`. Buttons `button_1` and `button_2` are available
for clutch and mode switching.

**VRControllerInput.** Interface for VR controllers (Oculus Touch, Vive Wands,
etc.). Supports left or right hand configuration and multiple tracking systems
(SteamVR, Oculus). Provides analog trigger and grip values for proportional
gripper control. The gripper state is derived from the trigger:
`gripper = 1.0 - trigger_value`. Available buttons include `trigger`, `grip`,
`thumbstick_click`, `a`, and `b`.

**HapticDeviceInput.** Interface for force-feedback devices (Phantom Omni,
Force Dimension Sigma.7, etc.). Provides high-fidelity position input and
bidirectional force feedback. The `set_force_feedback` method clips forces to
the device's maximum (default 3.3 N for Phantom Omni). Workspace scaling is
configurable via `set_workspace_scale(scale)`.

### 20.4.3 TeleoperationMode

The `TeleoperationMode` enum defines how device input is mapped to robot
commands:

| Mode        | Description                                       |
| ----------- | ------------------------------------------------- |
| `POSITION`  | Device pose maps to robot end-effector position.  |
| `VELOCITY`  | Device twist maps to robot end-effector velocity. |
| `WRENCH`    | Device twist interpreted as force/torque command. |
| `IMPEDANCE` | Position control with compliance.                 |

### 20.4.4 WorkspaceMapping

The `WorkspaceMapping` dataclass configures the coordinate transformation
between device and robot:

| Field            | Default  | Description                             |
| ---------------- | -------- | --------------------------------------- |
| `leader_frame`   | `eye(4)` | Device reference frame (4x4 transform). |
| `follower_frame` | `eye(4)` | Robot reference frame (4x4 transform).  |
| `position_scale` | `1.0`    | Position scaling factor.                |
| `rotation_scale` | `1.0`    | Rotation scaling factor.                |
| `deadband`       | `0.001`  | Motion deadband (meters).               |
| `rate_limit`     | `0.5`    | Maximum velocity (m/s).                 |

### 20.4.5 The TeleoperationInterface Class

`TeleoperationInterface` is the main class for teleoperation. It bridges input
devices and robot control.

**Construction.**

```python
from src.deployment.teleoperation.interface import TeleoperationInterface
from src.deployment.teleoperation.devices import SpaceMouseInput

device = SpaceMouseInput()
device.connect()

teleop = TeleoperationInterface(
    robot=engine,
    input_device=device,
)
teleop.set_control_mode(TeleoperationMode.POSITION)
```

**Clutch mechanism.** The clutch allows the operator to reposition the input
device without moving the robot:

```python
teleop.engage_clutch()    # Motion enabled
teleop.disengage_clutch() # Motion disabled, robot holds position
```

When the clutch is engaged, the current device pose is stored as the reference.
Subsequent device motion is computed relative to this reference. When the clutch
is disengaged, all commands are zero (the robot holds its current position).

Button 1 engages the clutch; button 2 disengages it.

**The `update` method.** This is the main method called at control frequency:

```python
command = teleop.update()
```

It performs the following sequence:

1. Read device pose, twist, gripper state, and buttons.
2. Handle clutch state from button input.
3. Dispatch to mode-specific command computation.
4. Return the `ControlCommand`.

**Position mode.** Computes position delta from the reference pose, applies
scaling and deadband, adds to current robot EE position, solves inverse
kinematics, and returns joint position targets:

$$
\Delta p = (p_{device} - p_{ref}) \cdot s
$$

$$
p_{target} = p_{ee} + \Delta p
$$

$$
q_{target} = \text{IK}(p_{target})
$$

The reference pose is updated after each command to provide relative (incremental)
control.

**Velocity mode.** Scales the device twist, applies rate limiting, and maps
to joint velocities using the Jacobian pseudo-inverse:

$$
\dot{q}_{target} = J^{\dagger} \cdot (v_{device} \cdot s)
$$

where $J^{\dagger} = J^T(JJ^T)^{-1}$ is the Moore-Penrose pseudo-inverse.

**Wrench mode.** Interprets the device twist as a desired end-effector wrench
(scaled by 10x) and maps to joint torques using the Jacobian transpose:

$$
\tau = J^T \cdot (v_{device} \cdot 10)
$$

**Impedance mode.** Combines position targeting with compliance parameters:
position targets from IK, with stiffness of 100 N/rad and damping of 10 Ns/rad
per joint.

### 20.4.6 Haptic Feedback

The `get_haptic_feedback` method computes force feedback for haptic devices:

```python
wrench = teleop.get_haptic_feedback()
device.set_force_feedback(wrench)
```

If the robot has contact force sensors, the measured forces are scaled down
by 0.1 and returned as a 6D wrench.

### 20.4.7 Demonstration Recording

The teleoperation interface integrates with the imitation learning module for
recording expert demonstrations:

```python
# Start recording
teleop.start_demonstration_recording()

# Operate the robot...
for step in range(1000):
    command = teleop.update()
    controller.send_command(command)
    state = controller.get_last_state()

    # Record state at each step
    teleop.record_state(
        joint_positions=state.joint_positions,
        joint_velocities=state.joint_velocities,
        action=command.torque_commands,
    )

# Stop and retrieve demonstration
demo = teleop.stop_demonstration_recording()
# demo is a Demonstration object ready for the imitation learning pipeline
```

The recording stores timestamps (relative to recording start), joint positions,
joint velocities, and actions. The returned `Demonstration` object has
`source="teleoperation"` and `success=True`.

### 20.4.8 Complete Teleoperation Pipeline

A full teleoperation deployment integrating all subsystems:

```python
from src.deployment.realtime.controller import RealTimeController, RobotConfig
from src.deployment.safety.monitor import SafetyMonitor
from src.deployment.safety.collision import CollisionAvoidance, HumanState
from src.deployment.teleoperation.interface import TeleoperationInterface
from src.deployment.teleoperation.devices import HapticDeviceInput

# Setup
config = RobotConfig(name="franka", n_joints=7)
controller = RealTimeController(control_frequency=1000.0)
controller.connect(config)

monitor = SafetyMonitor(config)
collision = CollisionAvoidance(engine, safety_distance=0.15)

device = HapticDeviceInput(device_name="sigma7")
device.connect()
teleop = TeleoperationInterface(engine, device)

# Control loop callback
def teleop_callback(state: RobotState) -> ControlCommand:
    # Update safety
    safety = monitor.check_state(state)
    if not safety.is_safe:
        return ControlCommand.torque_command(state.timestamp, np.zeros(7))

    # Get teleop command
    cmd = teleop.update()

    # Enforce safety
    safe_cmd = monitor.compute_safe_command(cmd, state)

    # Send haptic feedback
    feedback = teleop.get_haptic_feedback()
    device.set_force_feedback(feedback)

    # Record if active
    if teleop.is_recording:
        teleop.record_state(
            state.joint_positions,
            state.joint_velocities,
            safe_cmd.torque_commands,
        )

    return safe_cmd

controller.set_control_callback(teleop_callback)
controller.start()

# Record a demonstration
teleop.start_demonstration_recording()
input("Press Enter to stop recording...")
demo = teleop.stop_demonstration_recording()

# Save demonstration
from src.learning.imitation.dataset import DemonstrationDataset
dataset = DemonstrationDataset([demo])
dataset.save("teleop_demo.json")

# Cleanup
controller.stop()
controller.disconnect()
device.disconnect()
```

---

## Summary

Chapters 17 through 20 cover the complete machine learning and deployment
pipeline in UpstreamDrift:

- **Chapter 17** describes the reinforcement learning framework, including
  the configuration system (observation, action, reward, and task configs),
  the Gymnasium-compatible base environment, and concrete environments for
  humanoid locomotion/balance and manipulation tasks.

- **Chapter 18** covers imitation learning from demonstrations, including
  the demonstration dataset format, behavior cloning with NumPy-based MLPs,
  DAgger for addressing distribution shift, and GAIL for adversarial imitation.

- **Chapter 19** addresses sim-to-real transfer through domain randomization
  (randomizing masses, friction, delays, noise, and other parameters) and
  system identification (coordinate descent optimization to match simulated
  trajectories to real-robot data).

- **Chapter 20** presents the deployment infrastructure: real-time control
  with hardware interfaces, digital twin synchronization with Kalman filtering,
  comprehensive safety monitoring with collision avoidance, and teleoperation
  with multiple input device types and demonstration recording.

Together, these modules provide a path from initial policy training in
simulation to safe deployment on physical robot hardware.

---

# UpstreamDrift User Manual -- Part 6

# Research Modules, Visualization, REST API, Motion Retargeting, and Tools

---

# Chapter 21: Research Modules

The `src/research/` package contains four experimental subsystems that
extend the core simulation framework into advanced control, differentiable
simulation, deformable body dynamics, and multi-robot coordination.
Each module follows the same `PhysicsEngineProtocol` conventions used by
the main engines so that they compose naturally with the rest of the
codebase.

```
src/research/
    mpc/
        controller.py      # Model Predictive Controller
        specialized.py      # CentroidalMPC, WholeBodyMPC
    differentiable/
        engine.py           # DifferentiableEngine, ContactDifferentiableEngine
    deformable/
        objects.py          # SoftBody, Cable, Cloth
    multi_robot/
        system.py           # MultiRobotSystem, TaskCoordinator
        coordination.py     # FormationController, CooperativeManipulation
```

---

## 21.1 Model Predictive Control (MPC)

### 21.1.1 Overview

Model Predictive Control (MPC) solves an open-loop optimal control problem
at every control step, applies the first element of the resulting control
sequence, and re-solves after the system has advanced by one time step.
This **receding-horizon** strategy provides closed-loop feedback despite
relying on an open-loop optimization internally.

The controller lives in `src/research/mpc/controller.py` and is
implemented by the class `ModelPredictiveController`.

### 21.1.2 Mathematical Formulation

At each time step the controller solves:

$$\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \ell(x_k, u_k) + V_f(x_N)$$

subject to the constraints:

$$x_{k+1} = f(x_k, u_k), \quad x_0 = x_{\text{current}}$$

$$x_k \in \mathcal{X}, \quad u_k \in \mathcal{U} \quad \forall\, k$$

where:

- $N$ is the **prediction horizon** (attribute `horizon`, default 20).
- $f$ is the discrete-time dynamics model, evaluated by calling
  `model.step(dt)` on the wrapped `PhysicsEngineProtocol`.
- $\ell$ is the **stage cost** (running cost).
- $V_f$ is the **terminal cost** evaluated at the final predicted state.
- $\mathcal{X}$ and $\mathcal{U}$ are the state and control constraint
  sets, represented by the `Constraint` dataclass.

### 21.1.3 Cost Function

The cost function is specified through the `CostFunction` dataclass. It
supports both quadratic and linear terms.

**Stage cost** (running cost evaluated at each step $k$):

$$\ell(x, u) = (x - x_{\text{ref}})^T Q\, (x - x_{\text{ref}}) + q^T (x - x_{\text{ref}}) + (u - u_{\text{ref}})^T R\, (u - u_{\text{ref}}) + r^T (u - u_{\text{ref}})$$

When no linear terms are provided ($q = 0$, $r = 0$), this reduces to the
purely quadratic form:

$$\ell(x, u) = (x - x_{\text{ref}})^T Q\, (x - x_{\text{ref}}) + u^T R\, u$$

**Terminal cost** (evaluated at the final predicted state $x_N$):

$$V_f(x) = (x - x_{\text{ref}})^T P\, (x - x_{\text{ref}}) + p^T (x - x_{\text{ref}})$$

If the terminal cost matrix $P$ is `None`, the terminal cost is zero.

The `CostFunction` fields are:

| Field   | Type                 | Description                |
| ------- | -------------------- | -------------------------- | ----------------------------------------- |
| `Q`     | `NDArray (n_x, n_x)` | State cost weight matrix   |
| `R`     | `NDArray (n_u, n_u)` | Control cost weight matrix |
| `P`     | `NDArray (n_x, n_x)  | None`                      | Terminal cost weight matrix               |
| `q`     | `NDArray (n_x,)      | None`                      | Linear state cost (optional)              |
| `r`     | `NDArray (n_u,)      | None`                      | Linear control cost (optional)            |
| `p`     | `NDArray (n_x,)      | None`                      | Linear terminal cost (optional)           |
| `x_ref` | `NDArray             | None`                      | Reference state (1-D or 2-D trajectory)   |
| `u_ref` | `NDArray             | None`                      | Reference control (1-D or 2-D trajectory) |

When `x_ref` is two-dimensional (shape `(T, n_x)`), the reference at
step $k$ is `x_ref[k]`; if $k$ exceeds the length the last entry is
used. The same convention applies to `u_ref`.

```python
from src.research.mpc.controller import CostFunction
import numpy as np

Q = np.diag([100, 100, 10, 10])   # position-heavy
R = np.eye(2) * 0.01              # cheap control
P = Q * 10                        # stabilising terminal cost
cost = CostFunction(Q=Q, R=R, P=P, x_ref=np.array([1, 0, 0, 0]))
```

### 21.1.4 Constraint Specification

Constraints are expressed in the form:

$$\text{lb} \leq A\, x + B\, u \leq \text{ub}$$

and are managed through the `Constraint` dataclass:

| Field             | Type     | Description                          |
| ----------------- | -------- | ------------------------------------ | -------------------------- |
| `A`               | `NDArray | None`                                | State coefficient matrix   |
| `B`               | `NDArray | None`                                | Control coefficient matrix |
| `lb`              | `NDArray | None`                                | Lower bound vector         |
| `ub`              | `NDArray | None`                                | Upper bound vector         |
| `constraint_type` | `str`    | `"state"`, `"control"`, or `"mixed"` |

Setting only `A` with `lb` and `ub` yields a pure state constraint;
setting only `B` yields a pure control constraint. The controller
accumulates constraints via `add_constraint()` and enforces them during
the forward rollout.

```python
from src.research.mpc.controller import Constraint

# Joint limits: -pi <= q <= pi
A_pos = np.eye(n_x)
ctrl.add_constraint(Constraint(
    A=A_pos, lb=-np.pi * np.ones(n_x), ub=np.pi * np.ones(n_x),
    constraint_type="state",
))

# Torque limits: |u| <= 50
ctrl.add_constraint(Constraint(
    B=np.eye(n_u), lb=-50 * np.ones(n_u), ub=50 * np.ones(n_u),
    constraint_type="control",
))
```

### 21.1.5 Solver: iterative Linear Quadratic Regulator (iLQR)

The `solve()` method implements the **iLQR** algorithm, an iterative
local-approximation method that alternates between:

1. **Backward pass** -- Linearise dynamics around the current trajectory
   and compute the Q-function approximation. At each step $k$ from $N-1$
   down to 0:

   $$
   A_k = \frac{\partial f}{\partial x}\bigg|_{(x_k, u_k)}, \qquad
     B_k = \frac{\partial f}{\partial u}\bigg|_{(x_k, u_k)}
   $$

   The Jacobians $A_k$ and $B_k$ are obtained via forward finite
   differences with perturbation $\epsilon = 10^{-5}$. The Q-function
   matrices are:

   $$Q_x = \ell_x + A_k^T V_x, \qquad Q_u = \ell_u + B_k^T V_x$$

   $$Q_{xx} = \ell_{xx} + A_k^T V_{xx} A_k, \qquad Q_{uu} = \ell_{uu} + B_k^T V_{xx} B_k$$

   $$Q_{ux} = \ell_{ux} + B_k^T V_{xx} A_k$$

   The optimal feedback gain $K_k$ and feed-forward term $d_k$ are:

   $$K_k = -Q_{uu}^{-1} Q_{ux}, \qquad d_k = -Q_{uu}^{-1} Q_u$$

   A Tikhonov regularisation of $10^{-6} I$ is added to $Q_{uu}$ before
   inversion for numerical stability.

2. **Forward pass** -- Roll out new trajectory using the gains with a
   backtracking line search over parameter $\alpha$:

   $$\delta x_k = x_k^{\text{new}} - x_k^{\text{old}}$$

   $$u_k^{\text{new}} = u_k^{\text{old}} + \alpha\, d_k + K_k\, \delta x_k$$

   The line search tries up to 10 values of $\alpha$, halving on each
   iteration ($\alpha \leftarrow 0.5\, \alpha$), and accepts the $\alpha$
   that yields the lowest total cost.

Convergence is declared when the absolute change in cost between
consecutive iterations drops below the tolerance (default $10^{-6}$).
The maximum number of iterations defaults to 100.

### 21.1.6 Solver Settings

Settings are exposed as private attributes that can be changed before
calling `solve()`:

| Attribute            | Default   | Description                          |
| -------------------- | --------- | ------------------------------------ |
| `_max_iterations`    | 100       | Maximum iLQR iterations              |
| `_tolerance`         | $10^{-6}$ | Convergence tolerance on cost change |
| `_line_search_alpha` | 0.5       | Initial line-search step size        |
| `_line_search_beta`  | 0.5       | Line-search back-off factor          |

### 21.1.7 Result Object

`solve()` returns an `MPCResult` dataclass:

| Field                   | Type                 | Description                            |
| ----------------------- | -------------------- | -------------------------------------- |
| `success`               | `bool`               | Whether iLQR converged                 |
| `optimal_states`        | `NDArray (N+1, n_x)` | Predicted state trajectory             |
| `optimal_controls`      | `NDArray (N, n_u)`   | Optimal control sequence               |
| `cost`                  | `float`              | Total cost of the solution             |
| `solve_time`            | `float`              | Wall-clock solver time in seconds      |
| `iterations`            | `int`                | Number of iLQR iterations completed    |
| `constraint_violations` | `float`              | Maximum constraint violation magnitude |

### 21.1.8 Receding-Horizon Loop

In a closed-loop application only the first control action $u_0^*$ is
applied. The helper method `get_first_control(result)` extracts it:

```python
ctrl = ModelPredictiveController(engine, horizon=20, dt=0.01)
ctrl.set_cost_function(cost)

for step in range(simulation_steps):
    result = ctrl.solve(current_state)
    u0 = ctrl.get_first_control(result)
    engine.set_joint_torques(u0)
    engine.step(dt)
    current_state = get_state(engine)
```

### 21.1.9 Specialised MPC Variants

The file `src/research/mpc/specialized.py` provides two domain-specific
subclasses.

#### 21.1.9.1 CentroidalMPC

`CentroidalMPC` uses a simplified **centroidal dynamics** model for
locomotion planning. The state is nine-dimensional:

$$x = \begin{bmatrix} c_x & c_y & c_z & \dot c_x & \dot c_y & \dot c_z & L_x & L_y & L_z \end{bmatrix}^T$$

where $c$ is the centre-of-mass position and $L$ is the angular momentum.
Controls are the 3-D contact forces at each of $n_c$ contact points,
giving $n_u = 3 n_c$.

The dynamics are:

$$
\ddot c = \frac{1}{m}\sum_{i=1}^{n_c} f_i + g, \qquad
  \dot L = \sum_{i=1}^{n_c} (p_i - c) \times f_i
$$

integrated with forward Euler. Default cost weights emphasise CoM
position tracking ($Q_{\text{pos}} = 100 I_3$) with a small control
regulariser ($R = 0.001 I_{n_u}$).

Key methods:

- `set_mass(mass)` -- Set the robot total mass in kilograms.
- `update_contact_positions(positions)` -- Provide world-frame contact
  point locations.
- `add_friction_cone_constraints()` -- Add linearised Coulomb friction
  constraints:
  $f_z \geq 0$, $|f_x| \leq \mu f_z$, $|f_y| \leq \mu f_z$.
- `set_gait_reference(target_velocity, target_height)` -- Generate a
  reference CoM trajectory at constant velocity and height.

#### 21.1.9.2 WholeBodyMPC

`WholeBodyMPC` uses the full rigid-body dynamics model for manipulation
and whole-body control tasks. The state dimension equals twice the number
of generalised coordinates ($n_x = 2 n_q$, positions and velocities),
and controls are joint torques.

Key methods:

- `set_end_effector_target(ee_name, target_pose)` -- Specify a desired
  7-D pose (position + quaternion) for an end effector.
- `set_joint_targets(target_positions, target_velocities)` -- Provide
  reference joint angles and velocities.
- `add_joint_limit_constraints(lower, upper)` -- Enforce position limits.
- `add_torque_limit_constraints(torque_limits)` -- Enforce torque limits:
  $-\tau_{\max} \leq u \leq \tau_{\max}$.
- `solve_with_ee_tracking(initial_state)` -- Converts end-effector
  targets to joint targets via IK and then calls `solve()`.

---

## 21.2 Differentiable Physics

The differentiable-physics subsystem in
`src/research/differentiable/engine.py` wraps any `PhysicsEngineProtocol`
to enable **gradient-based trajectory optimisation**. Two classes are
provided: `DifferentiableEngine` for smooth dynamics and
`ContactDifferentiableEngine` for contact-rich scenarios.

### 21.2.1 DifferentiableEngine

#### Constructor

```python
DifferentiableEngine(
    engine: PhysicsEngineProtocol,
    backend: str = "numpy",       # "jax", "torch", or "numpy"
)
```

The `backend` parameter selects the automatic-differentiation framework.
When set to `"numpy"` (the default), all gradients and Jacobians are
computed via **numerical finite differences** -- no AD library is
required.

#### AutodiffBackend Enum

```python
class AutodiffBackend(Enum):
    JAX   = "jax"
    TORCH = "torch"
    NUMPY = "numpy"
```

#### Forward Simulation

```python
trajectory = engine.simulate_trajectory(initial_state, controls, dt=0.01)
# trajectory.shape == (T+1, n_x)
```

Rolls out the physics for `T = len(controls)` steps, recording
the full state at each instant. The state is partitioned as
$x = [q, v]$ where $q$ has dimension `n_q` and $v$ has dimension `n_v`.

#### Gradient Computation

```python
gradient = engine.compute_gradient(initial_state, controls, loss_fn, dt)
# gradient.shape == controls.shape == (T, n_u)
```

Computes the gradient of a scalar loss with respect to the control
sequence using forward finite differences:

$$\frac{\partial L}{\partial u_{t,i}} \approx \frac{L(u + \epsilon\, e_{t,i}) - L(u)}{\epsilon}$$

where $\epsilon = 10^{-5}$ and $e_{t,i}$ is the unit vector in the
$(t, i)$ direction of the flattened control array. Each partial
derivative requires one additional full forward simulation, making the
cost $\mathcal{O}(T \cdot n_u)$ simulations per gradient evaluation.

#### Jacobian Computation

```python
A, B = engine.compute_jacobian(state, control, dt)
# A.shape == (n_x, n_x)   -- state Jacobian  df/dx
# B.shape == (n_x, n_u)   -- control Jacobian df/du
```

Returns the linearised dynamics around a single operating point
$(x, u)$, computed column-by-column via finite differences:

$$
A_{:,i} = \frac{f(x + \epsilon\, e_i, u) - f(x, u)}{\epsilon}, \qquad
  B_{:,j} = \frac{f(x, u + \epsilon\, e_j) - f(x, u)}{\epsilon}
$$

These Jacobians are the building blocks for LQR, Kalman filtering, and
sensitivity analysis.

#### Trajectory Optimisation

```python
result = engine.optimize_trajectory(
    initial_state, goal_state,
    horizon=50, dt=0.01,
    method="adam",           # or "sgd"
    max_iterations=100,
    learning_rate=0.01,
)
```

Finds a control sequence that drives the system from `initial_state` to
`goal_state` by minimising the squared state error at the final time
step. The loss function is:

$$L(u) = \| x_T - x_{\text{goal}} \|^2$$

When `method="adam"`, the Adam optimiser is used with the standard update
rules. At iteration $t$ the first and second moment estimates are:

$$m_t = \beta_1\, m_{t-1} + (1 - \beta_1)\, \nabla L$$

$$v_t = \beta_2\, v_{t-1} + (1 - \beta_2)\, (\nabla L)^2$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$u \leftarrow u - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

with $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$.
When `method="sgd"`, a simple gradient-descent step is used:
$u \leftarrow u - \eta\, \nabla L$.

The result is an `OptimizationResult` dataclass:

| Field              | Type                 | Description                    |
| ------------------ | -------------------- | ------------------------------ |
| `success`          | `bool`               | True if final cost < 0.1       |
| `optimal_states`   | `NDArray (T+1, n_x)` | Best trajectory found          |
| `optimal_controls` | `NDArray (T, n_u)`   | Best control sequence          |
| `final_cost`       | `float`              | Loss at the best iterate       |
| `iterations`       | `int`                | Number of iterations performed |
| `gradient_norm`    | `float`              | Gradient norm at termination   |

### 21.2.2 ContactDifferentiableEngine

`ContactDifferentiableEngine` extends `DifferentiableEngine` to handle
the **non-smooth** gradient landscape that arises from contact and
collision dynamics. Three smoothing strategies are available:

#### Randomised Smoothing

In `"randomized"` mode, the gradient is estimated as the average of $K$
perturbed gradient evaluations:

$$\nabla L \approx \frac{1}{K} \sum_{k=1}^{K} \nabla L(u + \xi_k), \qquad \xi_k \sim \mathcal{N}(0, \sigma^2 I)$$

The default uses $K = 10$ samples and the `smoothing_factor` attribute
sets $\sigma$. This is the recommended mode for contact-rich problems.

#### Stochastic Smoothing

In `"stochastic"` mode, a single-sample approximation is used (i.e.,
$K = 1$), which is faster but noisier.

#### Standard Smoothing

In the default `"smoothed"` mode, the gradient is computed without
perturbation, relying on the underlying finite-difference scheme. This
is appropriate when contacts are not dominant.

#### Optimisation Through Contact

```python
result = engine.optimize_through_contact(
    initial_state, goal_state,
    contact_schedule=[False, False, True, True, True, False, ...],
    horizon=50, dt=0.01,
    contact_smoothing_multiplier=5.0,
    contact_penalty_weight=0.1,
)
```

This method implements **phase-aware smoothing**: during time steps where
`contact_schedule[t]` is `True`, the smoothing factor is increased by
`contact_smoothing_multiplier` to broaden the non-smooth contact
landscape.

An additional contact-transition penalty discourages velocity
discontinuities at the boundaries between contact and flight phases:

$$C_{\text{contact}} = w_c \sum_{t \in \mathcal{T}_{\text{transition}}} \| v_{t+1} - v_t \|^2$$

where $\mathcal{T}_{\text{transition}}$ is the set of time steps where
the contact flag changes between consecutive steps, and $w_c$ is
`contact_penalty_weight`.

The total loss minimised is:

$$L_{\text{total}} = \| x_T - x_{\text{goal}} \|^2 + C_{\text{contact}}$$

Internally the method uses the Adam optimiser (same parameters as
`optimize_trajectory`) and runs for up to 100 iterations.

---

## 21.3 Deformable Objects

The deformable objects module (`src/research/deformable/objects.py`)
provides finite-element and mass-spring simulations for soft bodies,
cables, and cloth. All classes inherit from the abstract base class
`DeformableObject`.

### 21.3.1 Material Properties

The `MaterialProperties` dataclass specifies physical material
parameters:

| Field               | Type    | Default | Description                    |
| ------------------- | ------- | ------- | ------------------------------ | ------------------------------------ |
| `youngs_modulus`    | `float` | $10^6$  | Young's modulus $E$ in Pascals |
| `poisson_ratio`     | `float` | 0.3     | Poisson's ratio $\nu$          |
| `density`           | `float` | 1000.0  | Material density in kg/m$^3$   |
| `damping`           | `float` | 0.01    | Viscous damping coefficient    |
| `bending_stiffness` | `float  | None`   | `None`                         | Bending stiffness (for shells/cloth) |
| `shear_stiffness`   | `float  | None`   | `None`                         | Shear stiffness (for cloth)          |

Derived elastic moduli are available as computed properties:

$$G = \frac{E}{2(1+\nu)}, \qquad K = \frac{E}{3(1-2\nu)}$$

where $G$ is the **shear modulus** and $K$ is the **bulk modulus**.

### 21.3.2 DeformableObject Base Class

Every deformable object manages:

- `_mesh` / `_rest_mesh` -- Current and rest-configuration node positions,
  each of shape `(N, 3)`.
- `_velocities` -- Node velocities `(N, 3)`.
- `_external_forces` -- Accumulated external forces `(N, 3)`.
- `_fixed_nodes` -- Set of node indices held at their rest positions
  (Dirichlet boundary conditions).

Public interface:

| Method                                  | Description                                        |
| --------------------------------------- | -------------------------------------------------- |
| `get_node_positions()`                  | Returns copy of current node positions             |
| `get_node_velocities()`                 | Returns copy of current node velocities            |
| `set_node_positions(pos)`               | Directly set node positions                        |
| `apply_external_force(indices, forces)` | Apply forces to specified nodes                    |
| `clear_external_forces()`               | Zero out all applied external forces               |
| `fix_nodes(indices)`                    | Pin nodes to rest positions                        |
| `unfix_nodes(indices)`                  | Release pinned nodes                               |
| `reset()`                               | Restore rest configuration, zero velocities/forces |
| `step(dt)` (abstract)                   | Advance simulation by `dt` seconds                 |
| `compute_internal_forces()` (abstract)  | Compute elastic restoring forces                   |

### 21.3.3 SoftBody (Volumetric FEM)

`SoftBody` implements volumetric soft-body simulation using tetrahedral
finite elements with a **neo-Hookean** constitutive model.

#### Constructor

```python
body = SoftBody(
    mesh=node_positions,       # (N, 3)
    tetrahedra=connectivity,   # (M, 4) integer indices
    material=MaterialProperties(youngs_modulus=1e5, poisson_ratio=0.45),
)
```

#### Constitutive Model

For each tetrahedron $e$ with deformation gradient $F$:

1. Compute the deformation gradient:
   $F = D_s\, B_e$ where $D_s$ is the matrix of deformed edge vectors
   and $B_e$ is the inverse of the rest-configuration edge matrix.

2. Compute the First Piola-Kirchhoff stress (neo-Hookean):

   $$P = \mu\,(F - F^{-T}) + \lambda\, \ln(J)\, F^{-T}$$

   where $J = \det(F)$, $\mu$ is the shear modulus, and $\lambda$ is
   the first Lame parameter:

   $$\lambda = \frac{E\,\nu}{(1 + \nu)(1 - 2\nu)}$$

3. Compute nodal forces from the stress:
   $H = -V_0\, P\, B_e^T$
   where $V_0$ is the rest volume of the tetrahedron, computed as:

   $$V_0 = \frac{|\det([v_1 - v_0,\; v_2 - v_0,\; v_3 - v_0])|}{6}$$

4. Distribute $H$ to the four nodes of the tetrahedron.

#### Time Integration

`step(dt)` uses explicit (forward) Euler integration:

$$v_{n+1} = v_n + \frac{f_{\text{internal}} + f_{\text{external}} - c\, v_n}{m_{\text{node}}}\, \Delta t$$

$$x_{n+1} = x_n + v_{n+1}\, \Delta t$$

Node mass is computed as $m_{\text{node}} = \rho \sum V_0 / N$. Fixed
nodes are reset to their rest positions with zero velocity after each
step.

### 21.3.4 Cable

`Cable` simulates a one-dimensional deformable body (rope, wire, shaft)
using a mass-spring model with bending resistance.

```python
cable = Cable(
    mesh=np.linspace([0,0,0], [1,0,0], 20),  # 20 nodes
    material=MaterialProperties(youngs_modulus=1e4, damping=0.1),
)
cable.fix_nodes([0])   # fix one end
```

**Stretch forces.** Between consecutive nodes $i$ and $i+1$:

$$f_{\text{stretch}} = E \cdot \frac{l - l_0}{l_0} \cdot \hat{d}$$

where $l$ is the current distance, $l_0$ is the rest length, and
$\hat{d}$ is the unit direction from node $i$ to node $i+1$.

**Bending forces.** At each interior node $i$ the angle between adjacent
segments is penalised:

$$f_{\text{bend}} = k_b\,(1 - \cos\theta)\, \hat{n}$$

where $\theta$ is the angle between the two segment vectors and
$k_b$ defaults to $0.1 E$ when `bending_stiffness` is not set.

Gravity ($g = -9.81\,\hat{z}$) is applied automatically.

Useful properties:

- `rest_length` -- Total rest length of the cable.
- `get_length()` -- Current total length.
- `get_tension()` -- Average tension (mean force magnitude across nodes).

### 21.3.5 Cloth

`Cloth` simulates a two-dimensional deformable surface using a
grid-based mass-spring system with three spring types:

| Spring Type | Connectivity         | Stiffness Default |
| ----------- | -------------------- | ----------------- |
| Stretch     | Horizontal, vertical | $E$               |
| Shear       | Diagonal             | $0.5\, E$         |
| Bend        | Skip-one (h, v)      | $0.1\, E$         |

```python
# Create 10x10 cloth
w, h = 10, 10
mesh = np.zeros((w * h, 3))
for y in range(h):
    for x in range(w):
        mesh[y * w + x] = [x * 0.1, y * 0.1, 1.0]

cloth = Cloth(mesh, width=w, height=h, material=MaterialProperties())
cloth.fix_nodes([0, w - 1])   # pin top corners
```

The `attach_to_body(body_id, nodes, positions)` method connects
cloth nodes to a rigid body, setting their positions and pinning them.

---

## 21.4 Multi-Robot Systems

The multi-robot subsystem in `src/research/multi_robot/` provides
infrastructure for managing, coordinating, and controlling teams of
robots within a shared simulation environment.

### 21.4.1 Task Model

Tasks are the atomic unit of work assigned to robots. The `Task`
dataclass carries:

| Field                | Type          | Description                                        |
| -------------------- | ------------- | -------------------------------------------------- | ---------------------------------- |
| `task_id`            | `str`         | Unique identifier                                  |
| `task_type`          | `TaskType`    | MOVE_TO, PICK, PLACE, MANIPULATE, INSPECT, or WAIT |
| `target_position`    | `NDArray (3,) | None`                                              | Spatial target for motion tasks    |
| `target_object`      | `str          | None`                                              | Object identifier for manipulation |
| `priority`           | `int`         | Higher values = higher priority                    |
| `status`             | `TaskStatus`  | PENDING, ASSIGNED, IN_PROGRESS, COMPLETED, FAILED  |
| `assigned_robot`     | `str          | None`                                              | Robot currently assigned           |
| `dependencies`       | `list[str]`   | Task IDs that must complete first                  |
| `estimated_duration` | `float`       | Expected completion time (seconds)                 |

The `is_ready(completed_tasks)` method checks whether all
`dependencies` have been satisfied.

### 21.4.2 TaskCoordinator

`TaskCoordinator` is a centralised task scheduler that:

1. Maintains a pool of tasks indexed by `task_id`.
2. Tracks which tasks have been completed.
3. Manages robot-to-task assignments (one task per robot at a time).

Key methods:

| Method                           | Description                                                               |
| -------------------------------- | ------------------------------------------------------------------------- |
| `add_task(task)`                 | Register a task                                                           |
| `remove_task(task_id)`           | Remove a task from the pool                                               |
| `get_ready_tasks()`              | Return pending tasks with satisfied deps, sorted by priority (descending) |
| `assign_task(task_id, robot_id)` | Bind a task to a robot                                                    |
| `start_task(task_id)`            | Transition ASSIGNED to IN_PROGRESS                                        |
| `complete_task(task_id)`         | Transition to COMPLETED                                                   |
| `fail_task(task_id)`             | Transition to FAILED                                                      |
| `get_robot_task(robot_id)`       | Return the task currently assigned to a robot                             |
| `get_available_robots(all_ids)`  | Return robots not currently busy                                          |

### 21.4.3 MultiRobotSystem

`MultiRobotSystem` is the top-level manager for a team of robots. Each
robot is identified by a string ID and is backed by its own
`PhysicsEngineProtocol` instance.

```python
system = MultiRobotSystem()
system.add_robot("arm_1", engine_1, base_pose=np.array([0,0,0, 1,0,0,0]))
system.add_robot("arm_2", engine_2, base_pose=np.array([2,0,0, 1,0,0,0]))
```

Key methods:

- `step_all(dt)` -- Step all robots synchronously.
- `check_inter_robot_collision(safety_distance=0.2)` -- Bounding-sphere
  collision check returning pairs of robot IDs whose base positions are
  closer than `safety_distance`.
- `allocate_tasks(tasks)` -- Greedy allocation: the highest-priority
  ready task is assigned to the closest available robot.
- `get_system_state()` -- Return a serialisable dictionary of all robot
  states (base poses, joint positions, joint velocities).

### 21.4.4 Formation Control

`FormationController` in `coordination.py` computes velocity commands to
maintain a geometric formation among robots.

#### Formation Presets

Three factory methods create common formations:

| Factory                              | Layout                   | Parameters               |
| ------------------------------------ | ------------------------ | ------------------------ |
| `line_formation(n, spacing)`         | Robots in a line along y | `spacing` (m)            |
| `circle_formation(n, radius)`        | Robots on a circle       | `radius` (m)             |
| `wedge_formation(n, spacing, angle)` | V-formation              | `spacing`, `angle` (rad) |

#### Control Law

Given a leader pose $(p_L, R_L)$ and per-robot desired offsets
$\delta_i$ from the `FormationConfig`, the desired world-frame position
for robot $i$ is:

$$p_i^d = p_L + R_L\, \delta_i$$

A proportional-derivative controller computes the velocity command:

$$v_i = K_p\,(p_i^d - p_i) - K_v\, \dot{p}_i$$

Gains default to $K_p = 2.0$ and $K_v = 1.0$. Use `set_gains()` to
adjust.

The formation error is the sum of Euclidean distances:

$$e = \sum_{i} \| p_i^d - p_i \|$$

available via `get_formation_error()`.

### 21.4.5 Cooperative Manipulation

`CooperativeManipulation` handles the case where multiple robots jointly
manipulate a single object.

#### Grasp Matrix

The grasp matrix $G$ maps contact forces to the resultant wrench on the
object:

$$G^T f = w_{\text{object}}$$

where $G \in \mathbb{R}^{6 \times 3 n_c}$, $f$ is the stacked vector of
contact forces, and $w = [f_x, f_y, f_z, \tau_x, \tau_y, \tau_z]^T$ is
the object wrench. The torque contribution uses the skew-symmetric
cross-product matrix:

$$[r]_\times = \begin{bmatrix} 0 & -r_z & r_y \\ r_z & 0 & -r_x \\ -r_y & r_x & 0 \end{bmatrix}$$

#### Load Sharing

`compute_load_sharing(desired_wrench, object_pose)` distributes forces
among the contacts to achieve the desired object wrench while minimising
the total force magnitude. The minimum-norm solution is:

$$f^* = G^T (G\, G^T + \epsilon I)^{-1}\, w$$

with $\epsilon = 10^{-6}$ for regularisation.

#### Force Closure

`check_force_closure(object_pose, friction_coeff)` checks whether the
grasp can resist arbitrary external wrenches by verifying that the grasp
matrix has rank 6. A quality metric (smallest singular value of $G$) is
also returned.

#### Cooperative Motion Planning

`plan_cooperative_motion(goal_pose, current_pose, dt, duration)` generates
smooth end-effector trajectories for all robots using:

1. **Smoothstep** position interpolation:
   $s(t) = 3t^2 - 2t^3$

2. **SLERP** quaternion interpolation:

   $$q(t) = \frac{\sin((1-t)\theta)}{\sin\theta}\, q_0 + \frac{\sin(t\theta)}{\sin\theta}\, q_1$$

   where $\theta = \arccos(q_0 \cdot q_1)$. When $q_0 \cdot q_1 > 0.9995$,
   linear interpolation is used as a numerically stable fallback.

---

# Chapter 22: Visualization System

UpstreamDrift provides a layered visualization architecture. At the
lowest level, the MuJoCo 3-D viewer renders force arrows, trajectory
traces, and torque spheres directly into the scene. Above that, a
comprehensive matplotlib-based plotting framework generates publication-
quality figures and animations. A Qt-based dashboard ties everything
together for interactive use.

```
Visualization layers
====================
Layer 3:  Dashboard (Qt)      -- UnifiedDashboardWindow, LivePlotWidget
Layer 2:  Plotting (matplotlib) -- 11 specialised renderers, animation, export
Layer 1:  3-D Overlays (MuJoCo) -- arrows, spheres, trajectory lines in-scene
```

---

## 22.1 MuJoCo 3-D Visualization

The file
`src/engines/physics_engines/mujoco/docker/src/humanoid_golf/visualization.py`
implements real-time 3-D overlays rendered through the native MuJoCo
`mjvGeom` API. No external renderer is needed; geometry primitives are
injected directly into the MuJoCo scene buffer.

### 22.1.1 TrajectoryTracer

`TrajectoryTracer` records body positions over time in bounded deques
(default `max_points=1000`):

```python
tracer = TrajectoryTracer(max_points=1000)
tracer.add_point("club_head", np.array([0.5, 0.1, 1.2]))
```

The tracer also supports a **desired-trajectory overlay** via
`set_desired_trajectory(body_name, positions)`. This registers a
reference path that is drawn alongside the actual trace for comparison.
`get_desired_trace(body_name)` retrieves the reference.

`clear(body_name=None)` removes traces for one body (or all bodies if
`body_name` is omitted).

### 22.1.2 ForceVisualizer

`ForceVisualizer` extracts simulation state from a MuJoCo model/data
pair:

| Method                 | Returns                                                           |
| ---------------------- | ----------------------------------------------------------------- |
| `get_contact_forces()` | List of dicts with position, normal, force magnitudes, body names |
| `get_joint_torques()`  | Dict mapping actuator name to torque value                        |
| `get_center_of_mass()` | Dict with CoM position and velocity vectors                       |

Contact forces are extracted via `mujoco.mj_contactForce(model, data, i, force)`,
which populates a 6-D vector: the first element is the normal force
magnitude and elements 1-2 are tangential friction forces.

### 22.1.3 Geometry Helpers

Three low-level functions populate `mjvGeom` structures for the 3-D
viewer:

#### `_init_arrow_geom(geom, start, end, radius, rgba)`

Creates a capsule (arrow shaft) aligned from `start` to `end`. The
rotation matrix is constructed so that the capsule's z-axis aligns with
the direction vector $\mathbf{d} = \text{end} - \text{start}$:

$$
\hat{z} = \frac{\mathbf{d}}{\|\mathbf{d}\|}, \qquad
  \hat{x} = \frac{\hat{z} \times \hat{e}}{\|\hat{z} \times \hat{e}\|}, \qquad
  \hat{y} = \hat{z} \times \hat{x}
$$

where $\hat{e}$ is an arbitrary reference vector chosen to avoid
degeneracy (the x-axis if $|\hat{z} \cdot \hat{x}| < 0.9$, otherwise
the y-axis).

The geometry is initialised via `mujoco.mjv_initGeom()` with type
`mjGEOM_CAPSULE`.

#### `_init_sphere_geom(geom, pos, radius, rgba)`

Creates a sphere at position `pos`. Used for torque visualisation where
the radius is proportional to torque magnitude: $r \propto |\tau|$.

#### `_init_line_geom(geom, p0, p1, radius, rgba)`

Creates a thin cylinder between two trajectory points. Implemented as
a delegate to `_init_arrow_geom` with a small radius.

### 22.1.4 Overlay Configuration

The `add_visualization_overlays()` entry point (not shown in full)
renders the following:

| Overlay                  | Geometry      | Colour Convention             |
| ------------------------ | ------------- | ----------------------------- | ---- | --- |
| Contact normal forces    | Arrow         | Red (`rgba=[1,0,0,0.8]`)      |
| Contact friction forces  | Arrow         | Orange (`rgba=[1,0.5,0,0.6]`) |
| Joint torques (positive) | Sphere        | Green, radius $\propto        | \tau | $   |
| Joint torques (negative) | Sphere        | Blue, radius $\propto         | \tau | $   |
| Actual trajectory        | Line segments | Per-body colour cycle         |
| Desired trajectory       | Line segments | Cyan, alternating-gap style   |

Toggles (`show_contacts`, `show_torques`, `show_trajectories`, etc.) and
scale factors (`force_scale`, `torque_scale`) are configurable at
runtime.

---

## 22.2 Matplotlib Plotting Framework

The plotting framework lives under `src/shared/python/plotting/` and is
structured as follows:

```
plotting/
    __init__.py
    config.py          # PlotConfig, ColorPalette
    base.py            # Base classes and interfaces
    core.py            # GolfSwingPlotter orchestrator
    animation.py       # SwingAnimator, AnimationConfig
    export.py          # ExportConfig, export_figure, export_plot_data
    kinematics.py      # Kinematics-specific helpers
    energy.py          # Energy-specific helpers
    transforms.py      # Coordinate transforms
    renderers/
        base.py        # BaseRenderer
        kinematics.py  # KinematicsRenderer
        kinetics.py    # KineticsRenderer
        energy.py      # EnergyRenderer
        club.py        # ClubRenderer
        signal.py      # SignalRenderer
        coordination.py# CoordinationRenderer
        stability.py   # StabilityRenderer
        comparison.py  # ComparisonRenderer
        dashboard.py   # DashboardRenderer
        vectors.py     # VectorOverlayRenderer
```

### 22.2.1 PlotConfig

`PlotConfig` is the single source of truth for all figure styling.
Every plotting function accepts an optional `PlotConfig`; the module
default `DEFAULT_CONFIG` is used when none is supplied.

| Attribute      | Type           | Default | Description                 |
| -------------- | -------------- | ------- | --------------------------- | ---------------------- |
| `width`        | `float`        | 10.0    | Figure width (inches)       |
| `height`       | `float`        | 6.0     | Figure height (inches)      |
| `dpi`          | `int`          | 100     | Dots per inch               |
| `line_width`   | `float`        | 1.5     | Default line width          |
| `marker_size`  | `float`        | 20.0    | Default scatter marker size |
| `title_size`   | `float`        | 14.0    | Title font size             |
| `label_size`   | `float`        | 12.0    | Axis label font size        |
| `tick_size`    | `float`        | 10.0    | Tick label font size        |
| `legend_size`  | `float`        | 10.0    | Legend font size            |
| `show_grid`    | `bool`         | `True`  | Display grid lines          |
| `grid_alpha`   | `float`        | 0.3     | Grid line opacity           |
| `tight_layout` | `bool`         | `True`  | Apply tight-layout          |
| `colors`       | `ColorPalette` | default | Colour palette instance     |
| `style`        | `str           | None`   | `None`                      | Matplotlib style sheet |

Helper methods:

- `create_figure(nrows, ncols, **kwargs)` -- Returns a pre-styled
  `(Figure, Axes)` tuple.
- `apply_to_axes(ax)` -- Apply tick sizes and grid settings to an
  existing Axes.

#### Preset Factories

| Factory                     | Purpose               | Key Differences                  |
| --------------------------- | --------------------- | -------------------------------- |
| `PlotConfig.presentation()` | Slides / posters      | 14x8 in, DPI 150, large fonts    |
| `PlotConfig.publication()`  | Journal figures       | 7x4.5 in, DPI 300, small fonts   |
| `PlotConfig.dashboard()`    | Multi-panel real-time | 5x3.5 in, DPI 100, compact fonts |

### 22.2.2 ColorPalette

`ColorPalette` is a frozen (immutable) dataclass providing named colours
and an indexed cycle:

```python
palette = ColorPalette()
palette.primary     # "#1f77b4"
palette.secondary   # "#ff7f0e"
palette.get_color(0)  # "#1f77b4"
palette.get_color(8)  # wraps to "#1f77b4"
```

The cycle contains eight colours: primary through senary, plus accent
and dark. `get_color(index)` wraps around automatically.

### 22.2.3 Specialised Renderers

All renderers inherit from `BaseRenderer` and receive recorded
simulation data and a colour dictionary. The following eleven renderers
are available:

#### KinematicsRenderer

Generates plots of joint-space motion:

- Joint angles vs. time (degrees)
- Joint angular velocities vs. time
- Joint angular accelerations vs. time
- Phase diagrams (angle vs. angular velocity)
- 3-D phase-space trajectories
- Poincare maps (stroboscopic sections)

#### KineticsRenderer

Generates dynamics-related plots:

- Joint torques vs. time (per actuator)
- Joint powers vs. time ($P = \tau \cdot \dot{q}$)
- Work loops (torque vs. angle, closed curves)
- Joint stiffness profiles
- 3-D angular momentum vector

#### EnergyRenderer

Plots energy flow through the system:

- Kinetic energy (KE), potential energy (PE), and total energy vs. time
- Power flow diagrams (rate of energy change)
- Segmental energy breakdown by body segment

#### ClubRenderer

Golf-specific club analysis:

- Club head speed vs. time (m/s and mph)
- 3-D club head trajectory
- Swing plane fitting (least-squares plane fit to trajectory points)

#### SignalRenderer

Advanced signal analysis visualisations:

- Jerk profiles (time derivative of acceleration)
- Power Spectral Density (PSD) using Welch's method
- Spectrograms (time-frequency representation)
- Wavelet scalograms (continuous wavelet transform)
- Lyapunov exponent estimation (orbital divergence)

#### CoordinationRenderer

Inter-joint coordination analysis:

- Coupling angle plots (relative phase between joints)
- Cross-correlation matrix (heatmap)
- Synergy trajectory plots (principal component projections)

#### StabilityRenderer

Balance and stability analysis:

- Centre of Pressure (CoP) trajectory
- CoM-CoP distance vs. time
- Ground Reaction Force (GRF) butterfly plots
- Local dynamic stability metrics
- 3-D vector field plots (velocity and force fields)

#### ComparisonRenderer

Counterfactual analysis tools:

- ZTCF (Zero-Torque Counterfactual) vs. ZVCF (Zero-Velocity
  Counterfactual) comparison plots
- Side-by-side actual vs. counterfactual trajectories

#### DashboardRenderer

Summary visualisation:

- 6-subplot summary grid (club speed, energy, angular momentum, joint
  angles, CoP, ground reaction forces)
- Radar/spider charts for multi-metric performance profiles

#### VectorOverlayRenderer

Spatial vector overlays:

- Force arrow plots in 2-D/3-D
- Torque vector plots
- Trajectory comparison with desired overlay
- GRF vector visualisation at foot contacts

---

## 22.3 Animation System

The animation subsystem in `src/shared/python/plotting/animation.py`
creates matplotlib `FuncAnimation` objects from simulation recordings.

### 22.3.1 AnimationConfig

| Attribute        | Type               | Default     | Description                              |
| ---------------- | ------------------ | ----------- | ---------------------------------------- |
| `fps`            | `int`              | 30          | Frames per second                        |
| `interval_ms`    | `int`              | 0           | Override ms between frames (0 = use fps) |
| `trail_length`   | `int`              | 60          | Number of past positions to keep visible |
| `figsize`        | `(float, float)`   | (10.0, 8.0) | Figure size in inches                    |
| `dpi`            | `int`              | 100         | Output resolution                        |
| `show_vectors`   | `bool`             | `True`      | Overlay force/velocity vectors           |
| `vector_scale`   | `float`            | 0.01        | Vector magnitude scale factor            |
| `skeleton_links` | `list[(str, str)]` | `[]`        | Body pairs for stick-figure links        |
| `desired_color`  | `str`              | `"#00CED1"` | Desired trajectory colour (cyan)         |
| `actual_color`   | `str`              | `"#FF4500"` | Actual trajectory colour (red-orange)    |

The `effective_interval` property computes `max(1, int(1000 / fps))`
when `interval_ms` is zero.

### 22.3.2 SwingAnimator

`SwingAnimator` is constructed with a `RecorderInterface` and an optional
`AnimationConfig`.

#### `create_trajectory_animation(body_names, desired_positions)`

Creates a 3-D trajectory playback animation. For each named body, the
recorded positions are retrieved via `recorder.get_time_series()` and
plotted with a trailing line of length `trail_length`.

If `desired_positions` is provided (a dictionary mapping body names to
`(N, 3)` arrays), the desired paths are drawn as semi-transparent static
lines for overlay comparison.

Axis limits are computed from the data extents with a 10% margin. A
timestamp annotation updates on each frame.

#### `create_stick_figure_animation(body_positions, links)`

Animates a skeleton by connecting body positions via straight lines.
Each entry in `links` is a `(body_a, body_b)` pair. The animation
updates all link lines on each frame.

#### `create_vector_field_animation(positions, vectors, times, label)`

Animates time-varying vectors (forces, velocities) at spatial positions
using matplotlib's `quiver` plotting. The old quiver is removed and
replaced on each frame, and a time annotation is displayed.

#### `save_animation(anim, path, writer, fps, dpi)` (static)

Convenience wrapper around `anim.save()`. Supports:

| Writer     | File Extension | Requirement      |
| ---------- | -------------- | ---------------- |
| `"ffmpeg"` | `.mp4`         | FFmpeg installed |
| `"pillow"` | `.gif`         | Pillow installed |

The output directory is created automatically if it does not exist.

---

## 22.4 Export System

The export module (`src/shared/python/plotting/export.py`) provides
functions for persisting figures and their underlying data.

### 22.4.1 ExportConfig

| Attribute          | Type   | Default   | Description                     |
| ------------------ | ------ | --------- | ------------------------------- | --------------------- |
| `output_dir`       | `str   | Path`     | `"exports"`                     | Root export directory |
| `image_format`     | `str`  | `"png"`   | Default raster format           |
| `vector_format`    | `str`  | `"pdf"`   | Default vector format           |
| `dpi`              | `int`  | 300       | Raster export resolution        |
| `transparent`      | `bool` | `False`   | Transparent background          |
| `bbox_inches`      | `str`  | `"tight"` | Bounding box mode               |
| `include_metadata` | `bool` | `True`    | Embed timestamp and source info |

### 22.4.2 export_figure

```python
paths = export_figure(fig, name="club_speed", config=cfg)
# Returns list of Paths: [exports/club_speed.png, exports/club_speed.pdf]
```

Saves a matplotlib `Figure` to one or more formats. By default both the
raster and vector formats from the config are used. Custom formats can
be passed via the `formats` parameter.

### 22.4.3 export_plot_data

```python
path = export_plot_data(
    data={"time": times, "speed": speeds},
    name="club_speed_data",
    config=cfg,
    fmt="json",   # or "csv"
)
```

Exports the raw numerical data behind a plot.

- **JSON** mode: wraps the data in a dictionary, converting NumPy arrays
  to lists. If `include_metadata` is set, a `_meta` key is added with
  the export timestamp and source identifier (`"UpstreamDrift"`).

- **CSV** mode: flattens arrays into columns. Two-dimensional arrays are
  split into separate columns named `key_0`, `key_1`, etc. The first
  row is the header.

### 22.4.4 export_all_figures

```python
results = export_all_figures(
    figures={"speed": fig1, "energy": fig2},
    config=cfg,
)
# results == {"speed": [Path(...), ...], "energy": [Path(...), ...]}
```

Batch export of multiple named figures, calling `export_figure()` for
each entry.

---

## 22.5 Dashboard

The Qt-based dashboard in `src/shared/python/dashboard/` provides an
interactive GUI for simulation control, live plotting, post-hoc analysis,
and data export.

### 22.5.1 Architecture

```
dashboard/
    window.py         # UnifiedDashboardWindow (QMainWindow)
    widgets.py        # LivePlotWidget, ControlPanel, FrequencyAnalysisDialog
    runner.py         # SimulationRunner (QThread)
    recorder.py       # GenericPhysicsRecorder
    launcher.py       # launch_dashboard() entry point
    advanced_analysis.py  # Counterfactual analysis dialog
```

### 22.5.2 UnifiedDashboardWindow

The main application window (`QMainWindow`) is divided into two panels:

**Left panel:**

- `ControlPanel` -- Start, Stop, Reset buttons for simulation control.
- `LivePlotWidget` -- Real-time metric plotting with a metric selector
  (positions, velocities, torques, ZTCF, induced accelerations, frequency
  analysis).

**Right panel** (tabbed):

- **Plotting** tab -- Drop-down menu for selecting plot types (summary
  dashboard, joint angles, torques, energies, club analysis, etc.) with
  a `MplCanvas` rendering area.
- **Counterfactuals** tab -- ZTCF/ZVCF counterfactual analysis dialog.
- **Export** tab -- Export simulation data to multiple formats (CSV, JSON,
  MAT, C3D, HDF5).

The window resolves joint names from the engine if available and passes
them to both the live plot widget and the plotter for labelling.

### 22.5.3 LivePlotWidget

`LivePlotWidget` (`QWidget`) displays real-time plots that update on each
simulation frame. It supports:

- Multiple metric types via combo-box selection.
- Per-joint plotting with individual joint selectors.
- Parametric (X-Y) plotting.
- Frequency-domain analysis via `FrequencyAnalysisDialog` (PSD computed
  using Welch's method, displayed in dB/Hz with semilog scaling).

The widget embeds a `MplCanvas` (matplotlib `FigureCanvasQTAgg`
subclass) for rendering.

### 22.5.4 ControlPanel

`ControlPanel` provides three buttons:

| Button | Action                          |
| ------ | ------------------------------- |
| Start  | Start `SimulationRunner` thread |
| Stop   | Stop simulation loop            |
| Reset  | Reset engine state and recorder |

### 22.5.5 SimulationRunner

`SimulationRunner` is a `QThread` that runs the physics simulation loop
in a background thread, preventing the GUI from freezing.

Signals emitted:

| Signal                | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `frame_ready`         | Emitted after each physics step; triggers GUI update |
| `simulation_finished` | Emitted when simulation ends or is stopped           |
| `status_message(str)` | Status text for the status bar                       |

The runner targets a configurable frame rate (default 60 FPS) by sleeping
between steps. A maximum step count (default 10,000) prevents infinite
runs.

### 22.5.6 GenericPhysicsRecorder

`GenericPhysicsRecorder` records simulation data from any engine
implementing the `PhysicsEngine` interface.

| Parameter          | Default | Description                                       |
| ------------------ | ------- | ------------------------------------------------- |
| `max_samples`      | 100,000 | Hard ceiling on buffer size                       |
| `initial_capacity` | 1,000   | Starting buffer size                              |
| `growth_factor`    | 1.5     | Buffer growth multiplier when capacity is reached |

When the buffer is full, recording stops automatically with a warning.
The recorder supports real-time counterfactual analysis configuration
via `analysis_config` (ZTCF, ZVCF, drift tracking, induced
accelerations).

### 22.5.7 Launching the Dashboard

Engine-specific launcher scripts in `src/launchers/` provide a
one-command entry point:

```python
# src/launchers/mujoco_dashboard.py
from src.shared.python.dashboard.launcher import launch_dashboard
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
    MuJoCoPhysicsEngine,
)

launch_dashboard(engine_class=MuJoCoPhysicsEngine, title="MuJoCo Dashboard")
```

Similar launchers exist for Drake and Pinocchio engines.

---

# Chapter 23: REST API Reference

The UpstreamDrift REST API is a FastAPI application that exposes the
simulation, analysis, and video processing capabilities over HTTP and
WebSocket protocols.

---

## 23.1 Server Architecture

### 23.1.1 Framework

The server is built on **FastAPI** (v2.1.0) with the following middleware
stack:

| Middleware              | Purpose                                |
| ----------------------- | -------------------------------------- |
| `TrustedHostMiddleware` | Reject requests from untrusted hosts   |
| `CORSMiddleware`        | Cross-Origin Resource Sharing control  |
| `slowapi` rate limiter  | Request rate limiting per IP           |
| Security headers        | HSTS, X-Content-Type-Options, etc.     |
| Upload size validation  | Reject oversized uploads               |
| Request tracing         | Attach correlation IDs for diagnostics |

CORS is configured with restricted origins (from `get_cors_origins()`)
and explicit allowed headers (`Content-Type`, `Authorization`,
`X-API-Key`). Wildcard headers are never used.

### 23.1.2 Database

An SQLite database (`golf_modeling_suite.db`) stores user accounts, API
keys, and usage tracking data. The database is initialised on startup
via `init_db()`.

### 23.1.3 Dependency Injection

Services are stored in `app.state` and injected into route handlers via
FastAPI's `Depends()` mechanism:

| Dependency          | Provider Function          |
| ------------------- | -------------------------- |
| `SimulationService` | `get_simulation_service()` |
| `AnalysisService`   | `get_analysis_service()`   |
| `EngineManager`     | `get_engine_manager()`     |
| `TaskManager`       | `get_task_manager()`       |
| Logger              | `get_logger()`             |

### 23.1.4 Task Manager

Background simulation tasks are tracked by `TaskManager`, a thread-safe
dictionary with automatic TTL-based cleanup:

| Setting       | Value | Description                                  |
| ------------- | ----- | -------------------------------------------- |
| `TTL_SECONDS` | 3600  | Tasks expire after 1 hour                    |
| `MAX_TASKS`   | 1000  | Maximum stored tasks; LRU eviction on excess |

### 23.1.5 Error Handling

Errors are returned as structured JSON with HTTP status codes.
Application-specific error codes follow the pattern `GMS-XXX-YYY`.
All requests are tagged with a correlation ID via the `RequestTracer`
middleware for end-to-end tracing.

### 23.1.6 Environment Configuration

| Environment Variable  | Description                                 |
| --------------------- | ------------------------------------------- |
| `ENVIRONMENT`         | `"development"` or `"production"`           |
| `GOLF_API_SECRET_KEY` | JWT signing secret (required in production) |
| `SECRET_KEY`          | Alias for the above                         |

In development mode, auto-reload is enabled. In production, the server
refuses to start if no secret key is configured.

---

## 23.2 Core Endpoints

### 23.2.1 Health and Information

#### `GET /`

Root endpoint returning API metadata.

**Response:**

```json
{
  "message": "Golf Modeling Suite API",
  "version": "1.0.0",
  "docs": "/docs",
  "status": "running"
}
```

#### `GET /health`

Health check endpoint returning server status and engine availability.

**Response:**

```json
{
  "status": "healthy",
  "engines_available": 3,
  "timestamp": "2026-02-05T12:00:00Z"
}
```

### 23.2.2 Engine Management

#### `GET /engines`

List all physics engines with their status and capabilities.

**Response model:** `EngineListResponse`

```json
{
  "engines": [
    {
      "name": "MUJOCO",
      "available": true,
      "loaded": true,
      "version": null,
      "capabilities": ["physics", "contacts", "muscles", "tendons"],
      "engine_type": "MUJOCO",
      "status": "loaded",
      "is_available": true,
      "description": "MUJOCO physics engine"
    }
  ],
  "mode": "local"
}
```

Each engine reports its capabilities:

| Engine    | Capabilities                        |
| --------- | ----------------------------------- |
| MUJOCO    | physics, contacts, muscles, tendons |
| DRAKE     | physics, optimization, control      |
| PINOCCHIO | kinematics, dynamics, collision     |
| OPENSIM   | musculoskeletal, biomechanics       |
| MYOSIM    | muscle, tendon, control             |
| MATLAB_2D | 2d-simulation, simscape             |
| MATLAB_3D | 3d-simulation, simscape             |
| PENDULUM  | pendulum, educational               |

#### `POST /engines/{engine_type}/load`

Load a specific engine with an optional model file path.

**Parameters:**

- `engine_type` (path) -- Engine type string (e.g., `"mujoco"`)
- `model_path` (query, optional) -- Path to a model file

Model paths undergo security validation to prevent path traversal
attacks.

### 23.2.3 Simulation

#### `POST /simulate`

Execute a synchronous simulation.

**Request model:** `SimulationRequest`

| Field             | Type        | Required | Description                   |
| ----------------- | ----------- | -------- | ----------------------------- | ---------------------------------- |
| `engine_type`     | `str`       | Yes      | Physics engine to use         |
| `model_path`      | `str        | None`    | No                            | Path to model file                 |
| `duration`        | `float`     | No (1.0) | Simulation duration (seconds) |
| `timestep`        | `float      | None`    | No                            | Simulation time step               |
| `initial_state`   | `dict       | None`    | No                            | Initial joint positions/velocities |
| `control_inputs`  | `list[dict] | None`    | No                            | Control sequence                   |
| `analysis_config` | `dict       | None`    | No                            | Analysis configuration             |

**Response model:** `SimulationResponse`

| Field              | Type       | Description                          |
| ------------------ | ---------- | ------------------------------------ | ----------------------------- |
| `success`          | `bool`     | Whether simulation completed         |
| `duration`         | `float`    | Actual simulation duration           |
| `frames`           | `int`      | Number of simulation frames          |
| `data`             | `dict`     | States, controls, derived quantities |
| `analysis_results` | `dict      | None`                                | Analysis results if requested |
| `export_paths`     | `list[str] | None`                                | Paths to exported files       |

**Error responses:**

| Status | Detail                    | Cause                     |
| ------ | ------------------------- | ------------------------- |
| 400    | Invalid parameters: ...   | Validation error          |
| 504    | Simulation timed out      | Duration exceeded timeout |
| 500    | Simulation failed: ...    | Runtime error             |
| 500    | Internal simulation error | Unexpected exception      |

#### `POST /simulate/async`

Start an asynchronous simulation. Returns immediately with a task ID.

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started"
}
```

#### `GET /simulate/status/{task_id}`

Poll the status of an asynchronous simulation.

**Response:**

```json
{
  "status": "completed",
  "result": { ... }
}
```

Returns 404 if the task ID is not found.

### 23.2.4 WebSocket Simulation Streaming

#### `WS /ws/simulate/{engine_type}`

Real-time bidirectional simulation streaming.

**Client sends:**

```json
{"action": "start", "config": {
    "duration": 3.0,
    "timestep": 0.002,
    "initial_state": { ... },
    "live_analysis": true
}}
```

**Server streams:**

```json
{
  "frame": 42,
  "time": 0.084,
  "state": { ... },
  "analysis": {
    "joint_angles": [...],
    "velocities": [...]
  }
}
```

**Control commands:**

| Action     | Effect                                |
| ---------- | ------------------------------------- |
| `"start"`  | Begin simulation with provided config |
| `"stop"`   | Terminate simulation immediately      |
| `"pause"`  | Pause simulation loop                 |
| `"resume"` | Resume from pause                     |

Frame throttling targets 60 FPS for UI responsiveness:
`frame_skip = max(1, int(steps_per_second / 60))`.

**Completion message:**

```json
{
  "status": "complete",
  "total_frames": 1500,
  "total_time": 3.0
}
```

### 23.2.5 Analysis

#### `POST /analysis`

Run biomechanical analysis on simulation data.

**Request model:** `AnalysisRequest`

| Field           | Type   | Description                              |
| --------------- | ------ | ---------------------------------------- |
| `analysis_type` | `str`  | Type: kinematics, kinetics, energy, etc. |
| `data_source`   | `str`  | Source: simulation, c3d, video           |
| `parameters`    | `dict` | Analysis-specific parameters             |
| `export_format` | `str`  | Output format (default `"json"`)         |

**Response model:** `AnalysisResponse`

| Field            | Type       | Description                        |
| ---------------- | ---------- | ---------------------------------- | ---------------------------------- |
| `analysis_type`  | `str`      | Type of analysis performed         |
| `success`        | `bool`     | Whether analysis completed         |
| `results`        | `dict`     | Analysis results (metrics, values) |
| `visualizations` | `list[str] | None`                              | Generated visualisation file paths |
| `export_path`    | `str       | None`                              | Path to exported results           |

### 23.2.6 Video Pose Estimation

#### `POST /video/pose`

Process a video file for pose estimation.

**Request model:** `VideoAnalysisRequest`

| Field                       | Type    | Default       | Description                  |
| --------------------------- | ------- | ------------- | ---------------------------- | ----------------------- |
| `estimator_type`            | `str`   | `"mediapipe"` | Pose estimator backend       |
| `min_confidence`            | `float` | 0.5           | Minimum detection confidence |
| `enable_temporal_smoothing` | `bool`  | `True`        | Smooth pose over time        |
| `max_frames`                | `int    | None`         | `None`                       | Limit frames to process |
| `export_keypoints`          | `bool`  | `True`        | Include raw keypoints        |
| `export_joint_angles`       | `bool`  | `True`        | Compute joint angles         |

**Response model:** `VideoAnalysisResponse`

| Field                | Type         | Description                       |
| -------------------- | ------------ | --------------------------------- |
| `filename`           | `str`        | Original video filename           |
| `total_frames`       | `int`        | Total frames in video             |
| `valid_frames`       | `int`        | Frames with valid pose detection  |
| `average_confidence` | `float`      | Mean confidence across frames     |
| `quality_metrics`    | `dict`       | Quality assessment metrics        |
| `pose_data`          | `list[dict]` | Per-frame pose estimation results |

### 23.2.7 Data Export

#### `GET /export/{task_id}`

Export results for a completed task.

**Query parameters:**

- `format` -- Export format (`"json"`, `"csv"`, etc.). Must be one of the
  valid export formats defined in `config.VALID_EXPORT_FORMATS`.

**Error responses:**

| Status | Detail               | Cause                        |
| ------ | -------------------- | ---------------------------- |
| 400    | Invalid format '...' | Unsupported export format    |
| 404    | Task not found       | Unknown task ID              |
| 400    | Task not completed   | Task still running or failed |

---

## 23.3 Authentication

The authentication system in `src/api/auth/` implements JWT-based access
control with role-based authorisation and API key support.

### 23.3.1 User Registration

#### `POST /auth/register`

**Rate limit:** 3 requests per hour per IP (protects against account
farming).

**Request model:** `UserCreate`

| Field          | Type  | Description              |
| -------------- | ----- | ------------------------ |
| `email`        | `str` | User email (unique)      |
| `password`     | `str` | Plain-text password      |
| `full_name`    | `str` | Display name             |
| `organization` | `str` | Organisation affiliation |

Passwords are hashed with **bcrypt** (cost factor 12). New users
receive the `FREE` role.

### 23.3.2 User Login

#### `POST /auth/login`

**Rate limit:** 5 requests per minute per IP (protects against brute
force).

**Request model:** `LoginRequest`

| Field      | Type  | Description      |
| ---------- | ----- | ---------------- |
| `email`    | `str` | Registered email |
| `password` | `str` | Account password |

**Response model:** `LoginResponse`

Contains `access_token` and `refresh_token` (both JWT strings).

### 23.3.3 JWT Tokens

The `SecurityManager` class handles token creation and verification.

**Access tokens:**

- Algorithm: HS256
- Default expiry: 30 minutes
- Payload includes `{"sub": user_id, "type": "access", "exp": ...}`

**Refresh tokens:**

- Default expiry: 30 days
- Payload includes `{"type": "refresh"}`

Token verification checks:

1. Signature validity (using the server secret key).
2. Expiration (`exp` claim).
3. Token type matches the expected type (`"access"` or `"refresh"`).

### 23.3.4 API Key Authentication

API keys are generated with a `gms_` prefix:

```
gms_<44-character-url-safe-random-string>
```

Keys are stored as bcrypt hashes (not raw SHA256) for brute-force
resistance. An `AuthCache` provides a 5-minute TTL cache of validated
keys to avoid repeated bcrypt computations.

For fast database lookup, a SHA256 hash of the first 8 characters of the
key is stored as an index. This prefix hash is distinct from the full
bcrypt-hashed key used for actual verification.

### 23.3.5 Role-Based Access Control

Four subscription tiers are defined:

| Role           | Level | Description               |
| -------------- | ----- | ------------------------- |
| `FREE`         | 0     | Basic access              |
| `PROFESSIONAL` | 1     | Extended quotas           |
| `ENTERPRISE`   | 2     | Full access               |
| `ADMIN`        | 3     | Administrative privileges |

The `RoleChecker` class enforces minimum-role requirements on endpoints.
The `UsageTracker` tracks and enforces per-month quotas for API calls,
video analyses, and simulations.

### 23.3.6 Authorised Requests

Protected endpoints require a Bearer token in the `Authorization` header:

```
Authorization: Bearer <access_token>
```

or an API key in the `X-API-Key` header:

```
X-API-Key: gms_<key>
```

In local mode (`is_local_mode()` returns `True`), authentication is
optional for most endpoints to support development workflows.

---

## 23.4 Request/Response Models

All request and response models are defined using **Pydantic** `BaseModel`
with `Field()` descriptors for validation and documentation.

### 23.4.1 SimulationRequest

```python
class SimulationRequest(BaseModel):
    engine_type: str = Field(..., description="Physics engine to use")
    model_path: str | None = Field(None, description="Path to model file")
    duration: float = Field(1.0, gt=0, description="Simulation duration")
    timestep: float | None = Field(None, gt=0, description="Time step")
    initial_state: dict | None = Field(None, description="Initial state")
    control_inputs: list[dict] | None = Field(None, description="Controls")
    analysis_config: dict | None = Field(None, description="Analysis config")
```

The `engine_type` field is required (`...` sentinel). The `duration`
and `timestep` fields enforce `gt=0` (greater than zero).

### 23.4.2 SimulationResponse

```python
class SimulationResponse(BaseModel):
    success: bool
    duration: float
    frames: int
    data: dict
    analysis_results: dict | None = None
    export_paths: list[str] | None = None
```

### 23.4.3 AnalysisRequest

```python
class AnalysisRequest(BaseModel):
    analysis_type: str
    data_source: str
    parameters: dict = Field(default_factory=dict)
    export_format: str = "json"
```

### 23.4.4 AnalysisResponse

```python
class AnalysisResponse(BaseModel):
    analysis_type: str
    success: bool
    results: dict
    visualizations: list[str] | None = None
    export_path: str | None = None
```

### 23.4.5 VideoAnalysisRequest and Response

See Section 23.2.6 for field details.

### 23.4.6 TaskStatusResponse

```python
class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float | None = None
    result: dict | None = None
    error: str | None = None
```

### 23.4.7 Model Fitting Request

```python
class ModelFittingRequest(BaseModel):
    model_path: str
    data_path: str
    fitting_method: str = "least_squares"
    parameters: dict = Field(default_factory=dict)
```

Used for fitting biomechanical models to experimental data (C3D, etc.).

---

# Chapter 24: Motion Retargeting

The motion retargeting module in `src/learning/retargeting/retargeter.py`
transfers motion data between skeletal embodiments with different
topologies, bone lengths, and joint conventions.

---

## 24.1 Skeleton Configuration

### 24.1.1 SkeletonConfig

The `SkeletonConfig` dataclass describes the kinematic structure of a
skeleton:

| Field             | Type             | Description                          |
| ----------------- | ---------------- | ------------------------------------ | ------------------------------------------ |
| `name`            | `str`            | Skeleton identifier                  |
| `joint_names`     | `list[str]`      | Ordered list of joint names          |
| `parent_indices`  | `list[int]`      | Parent index per joint (-1 for root) |
| `joint_offsets`   | `NDArray (n, 3)` | T-pose offset from parent in metres  |
| `joint_axes`      | `NDArray (n, 3)  | None`                                | Rotation axis per joint (default z)        |
| `joint_limits`    | `NDArray (n, 2)  | None`                                | [min, max] angle limits (default $\pm\pi$) |
| `semantic_labels` | `dict[str, str]` | Maps semantic names to joint names   |
| `end_effectors`   | `list[str]`      | Names of end-effector joints         |

Post-initialisation validation ensures that `parent_indices` and
`joint_offsets` have the same length as `joint_names`. Missing
`joint_axes` default to the z-axis; missing `joint_limits` default to
$[-\pi, \pi]$.

### 24.1.2 Utility Methods

| Method                           | Description                              |
| -------------------------------- | ---------------------------------------- |
| `n_joints` (property)            | Number of joints                         |
| `get_joint_index(name)`          | Name to index lookup (raises if missing) |
| `get_semantic_joint(semantic)`   | Semantic label to joint name             |
| `get_kinematic_chain(end_joint)` | Root-to-end joint name list              |

### 24.1.3 Standard Humanoid Configuration

`SkeletonConfig.create_humanoid()` returns a standard 22-joint humanoid
skeleton suitable for whole-body motion:

```
Spine chain (6 joints):  pelvis -> spine_1 -> spine_2 -> spine_3 -> neck -> head
Left leg (4 joints):     pelvis -> left_hip -> left_knee -> left_ankle -> left_foot
Right leg (4 joints):    pelvis -> right_hip -> right_knee -> right_ankle -> right_foot
Left arm (4 joints):     spine_3 -> left_shoulder -> left_elbow -> left_wrist -> left_hand
Right arm (4 joints):    spine_3 -> right_shoulder -> right_elbow -> right_wrist -> right_hand
```

All 22 joints have semantic labels (e.g., `"left_shoulder"` maps to
joint `"left_shoulder"`), and five end effectors are designated: `head`,
`left_hand`, `right_hand`, `left_foot`, `right_foot`.

T-pose offsets use approximate human proportions in metres (e.g.,
upper leg length 0.4 m, forearm length 0.25 m).

### 24.1.4 Kinematic Chain Extraction

`get_kinematic_chain(end_joint)` traverses the parent hierarchy from
`end_joint` to the root and returns the full chain:

```python
skel = SkeletonConfig.create_humanoid()
chain = skel.get_kinematic_chain("left_hand")
# ['pelvis', 'spine_1', 'spine_2', 'spine_3',
#  'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand']
```

---

## 24.2 Retargeting Methods

The `MotionRetargeter` class handles motion transfer between a source
and target skeleton.

### 24.2.1 Initialisation

```python
retargeter = MotionRetargeter(source_skeleton, target_skeleton)
```

During construction, two mappings are computed:

1. **Joint mapping** -- Established via semantic labels. For each
   semantic name in the source skeleton, the corresponding target joint
   is looked up. Only joints with matching semantic labels are mapped.

2. **Scale factors** -- The ratio of target bone length to source bone
   length for each mapped joint:

   $$s_j = \frac{\| \text{offset}_{\text{target},j} \|}{\| \text{offset}_{\text{source},j} \|}$$

   Scale factors are used for positional retargeting (motion capture).

### 24.2.2 Direct Mapping

```python
target_motion = retargeter.retarget(source_motion, method="direct")
```

The simplest retargeting method: source joint angles are copied directly
to the corresponding target joints. For each mapped pair
$(s_j, t_j)$:

$$q_{t_j}(t) = q_{s_j}(t)$$

After copying, joint limits of the target skeleton are enforced by
clipping.

This method works well when the source and target skeletons have similar
topologies and proportions.

### 24.2.3 Optimisation-Based Retargeting

```python
target_motion = retargeter.retarget(source_motion, method="optimization")
```

For each frame, the target joint angles are optimised to minimise the
squared error between source and target end-effector positions:

$$\min_{\mathbf{q}} \sum_{e \in \text{EE}} \| \text{FK}_e(\mathbf{q}) - \mathbf{p}_e^{\text{source}} \|^2$$

where $\text{FK}_e(\mathbf{q})$ is the forward kinematics of end
effector $e$ and $\mathbf{p}_e^{\text{source}}$ is the source
end-effector position.

The optimisation uses gradient descent with numerical gradients
($\epsilon = 10^{-4}$, step size 0.01, up to 50 iterations per frame).
The initial guess is the direct-mapping result.

Forward kinematics is computed by traversing the kinematic chain,
applying z-axis rotations at each joint:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

$$p_e = R(\theta_{n}) \cdots R(\theta_1) \, \delta_1 + \cdots + \delta_n$$

Joint limits are enforced at each iteration.

### 24.2.4 IK-Based Retargeting

```python
target_motion = retargeter.retarget(source_motion, method="ik")
```

Currently delegates to the optimisation-based method. The architecture
is designed to allow integration with engine-specific IK solvers (e.g.,
MuJoCo or Pinocchio IK) in the future.

---

## 24.3 Motion Capture Integration

### 24.3.1 retarget_from_mocap

```python
target_motion = retargeter.retarget_from_mocap(
    marker_positions,        # (T, n_markers, 3)
    marker_names,            # list of marker names
    marker_to_joint_mapping, # optional dict
)
```

This method bridges from marker-based motion capture data to joint
angles on the target skeleton:

1. **Marker-to-joint mapping.** If no explicit mapping is provided,
   `_infer_marker_mapping()` uses standard C3D naming conventions:

   | Marker Name | Semantic Joint |
   | ----------- | -------------- |
   | LSHO        | left_shoulder  |
   | RSHO        | right_shoulder |
   | LELB        | left_elbow     |
   | RELB        | right_elbow    |
   | LWRI        | left_wrist     |
   | RWRI        | right_wrist    |
   | LHIP        | left_hip       |
   | RHIP        | right_hip      |
   | LKNE        | left_knee      |
   | RKNE        | right_knee     |
   | LANK        | left_ankle     |
   | RANK        | right_ankle    |

2. **Position extraction.** For each frame, marker positions are
   associated with the target joints via the mapping.

3. **Analytical IK.** For two-link chains, joint angles are computed
   using the **law of cosines**:

   $$\cos\theta = \frac{a^2 + b^2 - d^2}{2ab}$$

   where $a$ and $b$ are the bone lengths of the two links and $d$ is
   the distance to the target position. The joint angle is then:

   $$\theta = \arccos\left(\text{clip}\left(\frac{a^2 + b^2 - d^2}{2ab + \epsilon}, -1, 1\right)\right)$$

   The small $\epsilon = 10^{-6}$ prevents division by zero.

### 24.3.2 Mapping Visualisation

`visualize_mapping()` returns a human-readable text summary:

```
Motion Retargeting: source_humanoid -> target_robot
==================================================

Joint Mapping:
  left_elbow           -> left_elbow           (scale: 0.83)
  left_shoulder        -> left_shoulder        (scale: 1.00)
  ...

Mapped joints: 18
Source joints: 22
Target joints: 20
```

---

# Chapter 25: Tools and Utilities

The `src/tools/` package and `src/shared/python/` provide utility
modules for model generation, model exploration, video analysis, injury
risk assessment, signal processing, and spatial algebra.

---

## 25.1 Model Generation

The model generation subsystem (`src/tools/model_generation/`) is
organised into eight sub-packages:

```
model_generation/
    builders/     # Parametric model building
    humanoid/     # Humanoid-specific generation
    mesh/         # Mesh generation and assembly
    inertia/      # Inertia tensor computation
    export/       # URDF, MJCF XML export
    core/         # Core data types and utilities
    library/      # Pre-built model templates
    converters/   # Format conversion (URDF <-> MJCF)
    plugins/      # Extension points
    editor/       # Interactive model editing
    cli/          # Command-line interface
    api/          # Programmatic generation API
```

### 25.1.1 Inertia Computation

The `inertia/` sub-package provides analytical formulae for computing
inertia tensors of standard geometric primitives.

#### Cylinder

For a solid cylinder of mass $m$, radius $r$, and height $h$ aligned
along the z-axis:

$$I_{xx} = I_{yy} = \frac{m}{12}(3r^2 + h^2), \qquad I_{zz} = \frac{mr^2}{2}$$

```python
from model_generation.inertia import cylinder_inertia
result = cylinder_inertia(mass=2.0, radius=0.05, length=0.3, axis="z")
# result == {"ixx": ..., "iyy": ..., "izz": ..., "ixy": 0, "ixz": 0, "iyz": 0}
```

When the cylinder axis is changed (e.g., `axis="x"`), the diagonal
entries are permuted accordingly so that the axial moment of inertia
aligns with the specified axis.

#### Sphere

For a solid sphere of mass $m$ and radius $r$:

$$I_{xx} = I_{yy} = I_{zz} = \frac{2mr^2}{5}$$

```python
from model_generation.inertia import sphere_inertia
result = sphere_inertia(mass=1.0, radius=0.1)
```

#### Box (Cuboid)

For a solid box of mass $m$ and full side lengths $a$, $b$, $c$:

$$I_{xx} = \frac{m}{12}(b^2 + c^2), \qquad I_{yy} = \frac{m}{12}(a^2 + c^2), \qquad I_{zz} = \frac{m}{12}(a^2 + b^2)$$

```python
from model_generation.inertia import box_inertia
result = box_inertia(mass=5.0, size_x=0.3, size_y=0.2, size_z=0.1)
```

#### Capsule and Ellipsoid

`capsule_inertia(mass, radius, length)` and
`ellipsoid_inertia(mass, semi_x, semi_y, semi_z)` compute the inertia
for capsule (cylinder with hemispherical caps) and ellipsoid primitives
respectively.

#### Unified InertiaCalculator

`InertiaCalculator` provides a high-level interface supporting four
computation modes:

| Mode (`InertiaMode`) | Description                                   |
| -------------------- | --------------------------------------------- |
| `PRIMITIVE`          | Use analytical formulas for standard shapes   |
| `MESH_DENSITY`       | Compute from mesh volume with uniform density |
| `MESH_MASS`          | Compute from mesh volume with specified mass  |
| `MANUAL`             | User provides inertia tensor directly         |

Returns an `InertiaResult` containing `ixx`, `iyy`, `izz`, `ixy`,
`ixz`, `iyz`, and `mass`.

#### Spatial Inertia

The `spatial.py` module provides tools for spatial (6-D) inertia:

- `mcI(mass, com, inertia)` -- Construct the $6 \times 6$ spatial
  inertia matrix.
- `transform_spatial_inertia(I_spatial, T)` -- Transform inertia to a
  new reference frame using a homogeneous transformation matrix.
- `spatial_inertia_to_urdf(I_spatial)` -- Convert to URDF-compatible
  inertia representation.

### 25.1.2 Export

The `export/` sub-package generates model files in standard formats:

- **URDF** (Unified Robot Description Format) -- XML format for ROS
  and many physics engines.
- **MJCF** (MuJoCo XML) -- Native MuJoCo model format.

Both exporters accept the internal model representation and produce
validated XML output.

### 25.1.3 Builders

The `builders/` sub-package provides parametric model construction:

- Segment-by-segment body assembly.
- Joint type selection (revolute, prismatic, ball, free).
- Collision geometry attachment.
- Visual geometry attachment.
- Actuator definition.

### 25.1.4 Humanoid Generation

The `humanoid/` sub-package specialises model generation for humanoid
characters:

- Anthropometric scaling from body measurements.
- Standard joint configurations for human motion.
- Segment mass and inertia estimation from body mass (de Leva tables).

---

## 25.2 Model Explorer

The model explorer (`src/tools/model_explorer/`) provides an interactive
browser for inspecting robot and humanoid models:

- URDF file loading and parsing.
- Joint tree visualisation with kinematic structure display.
- Interactive joint angle sliders for posing.
- Collision and visual geometry rendering.
- Inertia property inspection.
- Bundled human models and reference meshes.

The visualisation widget (`visualization_widget.py`) renders 3-D model
previews and supports camera manipulation (orbit, pan, zoom).

---

## 25.3 Video Analyzer

The video analysis pipeline (`src/tools/video_analyzer/` and
`src/shared/python/video_pose_pipeline.py`) extracts pose data from
video recordings.

### 25.3.1 Pipeline Architecture

```
video_analyzer/
    video_processor.py     # Frame extraction, pre-processing
    pose_estimator.py      # MediaPipe/OpenPose integration
    analyzer.py            # Swing detection, joint angle computation
    types.py               # Data types for video analysis results
```

### 25.3.2 Video Processing Config

```python
config = VideoProcessingConfig(
    estimator_type="mediapipe",   # or "openpose"
    min_confidence=0.5,
    enable_temporal_smoothing=True,
)
pipeline = VideoPosePipeline(config)
```

### 25.3.3 Pose Estimation

The pipeline supports two backends:

| Backend   | Library   | Landmarks | Description                   |
| --------- | --------- | --------- | ----------------------------- |
| MediaPipe | mediapipe | 33 body   | Google's real-time pose model |
| OpenPose  | openpose  | 25 body   | CMU's multi-person detector   |

Each frame is processed to extract 2-D and 3-D landmark positions with
per-landmark confidence scores. Frames with average confidence below
`min_confidence` are marked as invalid.

### 25.3.4 Temporal Smoothing

When `enable_temporal_smoothing` is set, landmark positions are smoothed
across frames using a one-dimensional low-pass filter to reduce jitter.

### 25.3.5 Swing Sequence Detection

The analyzer (`analyzer.py`) includes heuristics for detecting golf
swing phases (address, backswing, transition, downswing, impact,
follow-through) from the sequence of body landmarks.

---

## 25.4 Injury Risk Assessment

The spinal load analysis module
(`src/shared/python/analysis/` area) provides injury risk
assessment for golf swing motions.

### 25.4.1 Spinal Load Analysis

Spinal loading is estimated by modelling the lumbar spine as a rigid
column subjected to the forces and torques generated during the swing.
Key metrics computed:

| Metric               | Unit | Description                       |
| -------------------- | ---- | --------------------------------- |
| Compressive force    | N    | Axial load on lumbar vertebrae    |
| Anterior shear force | N    | Forward-directed shear at L4/L5   |
| Lateral shear force  | N    | Side-directed shear at L4/L5      |
| Resultant moment     | N m  | Net moment about the lumbar spine |

### 25.4.2 Risk Thresholds

Risk thresholds are based on NIOSH (National Institute for Occupational
Safety and Health) guidelines and biomechanical literature:

| Threshold               | Value  | Interpretation                  |
| ----------------------- | ------ | ------------------------------- |
| Compressive force (AL)  | 3400 N | Action limit -- increased risk  |
| Compressive force (MPL) | 6400 N | Maximum permissible limit       |
| Shear force limit       | 1000 N | Threshold for shear injury risk |

Loads exceeding the action limit trigger a warning; loads exceeding the
maximum permissible limit indicate a high injury risk.

### 25.4.3 Usage

```python
from src.shared.python.analysis import spinal_load_analysis

results = spinal_load_analysis.analyze_spinal_loads(
    joint_angles=joint_angles,
    joint_torques=joint_torques,
    body_masses=segment_masses,
)
# results contains peak forces, risk flags, and time-series data
```

---

## 25.5 Signal Processing Toolkit

The signal processing module (`src/shared/python/signal_processing.py`)
provides a comprehensive set of functions for processing biomechanical
time-series data.

### 25.5.1 Filtering

| Function              | Description                                              |
| --------------------- | -------------------------------------------------------- |
| Low-pass filter       | Butterworth IIR filter for removing high-frequency noise |
| Band-pass filter      | Retain signal components within a frequency band         |
| Savitzky-Golay filter | Polynomial smoothing preserving peak shapes              |

The `savgol_filter` from SciPy is used for Savitzky-Golay smoothing,
which fits successive sub-sets of data points with a low-degree
polynomial by the method of linear least squares.

### 25.5.2 Numerical Differentiation

Two methods are available:

**Central differences:**

$$\dot{x}(t) \approx \frac{x(t + \Delta t) - x(t - \Delta t)}{2\,\Delta t}$$

$$\ddot{x}(t) \approx \frac{x(t + \Delta t) - 2\,x(t) + x(t - \Delta t)}{(\Delta t)^2}$$

**Savitzky-Golay differentiation:** Fits a polynomial to a window of
data points and differentiates the polynomial analytically. This
produces smoother derivatives than finite differences, especially for
noisy data.

### 25.5.3 Frequency Analysis

| Function          | Description                                  |
| ----------------- | -------------------------------------------- |
| `compute_psd()`   | Power Spectral Density via Welch's method    |
| Spectrogram       | Time-frequency representation via STFT       |
| Coherence         | Cross-spectral coherence between two signals |
| Cross-correlation | Sliding dot-product between signals          |

The PSD function returns frequencies and power values; the dashboard
converts to decibels: $P_{\text{dB}} = 10 \log_{10}(P + \epsilon)$ with
$\epsilon = 10^{-12}$.

### 25.5.4 Time Normalisation

Movement data can be normalised to a percentage scale (0-100%) to enable
comparison across trials of different durations. This is essential for
ensemble averaging and inter-subject comparison.

### 25.5.5 Feature Extraction

| Feature        | Description                                                   |
| -------------- | ------------------------------------------------------------- |
| Peak detection | Local maxima and minima with configurable prominence          |
| Zero crossings | Time points where signal changes sign                         |
| RMS            | Root mean square: $\text{RMS} = \sqrt{\frac{1}{N}\sum x_i^2}$ |
| DTW distance   | Dynamic Time Warping for shape comparison                     |

The DTW implementation includes an optional **Numba JIT** acceleration
(approximately 100x speedup) and an $\mathcal{O}(M)$ space optimisation
that stores only two rows of the cost matrix instead of the full
$\mathcal{O}(NM)$ matrix.

### 25.5.6 Continuous Wavelet Transform

The CWT decomposes a signal into time-scale components using Morlet
wavelets. Wavelet generation is cached with `functools.lru_cache` for
performance.

---

## 25.6 Spatial Algebra

Spatial algebra utilities are distributed across several modules in the
codebase (primarily `src/shared/python/reference_frames.py` and the
plotting transforms module).

### 25.6.1 Rotation Matrices

Elementary rotation matrices about the coordinate axes:

$$R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}$$

$$R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$$

$$R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### 25.6.2 Euler Angle Conventions

Conversion between Euler angles (ZYX, XYZ, ZXZ, etc.) and $3 \times 3$
rotation matrices. The ZYX convention is the most common in the
codebase:

$$R_{\text{ZYX}}(\alpha, \beta, \gamma) = R_z(\alpha)\, R_y(\beta)\, R_x(\gamma)$$

Extraction of Euler angles from a rotation matrix handles the
gimbal-lock singularity when $\cos\beta = 0$ by falling back to a
two-angle decomposition.

### 25.6.3 Quaternion Operations

Quaternions are represented as $q = [w, x, y, z]$ (scalar-first
convention, consistent with MuJoCo).

| Operation          | Formula                                              |
| ------------------ | ---------------------------------------------------- |
| Multiply           | Hamilton product: $q_1 \otimes q_2$                  |
| Inverse            | $q^{-1} = \bar{q} / \|q\|^2$                         |
| To rotation matrix | See Section 21.4.4 (implemented in multiple modules) |
| SLERP              | Spherical linear interpolation (see Section 21.4.5)  |

**Quaternion to rotation matrix:**

$$R = \begin{bmatrix} 1 - 2(y^2 + z^2) & 2(xy - zw) & 2(xz + yw) \\ 2(xy + zw) & 1 - 2(x^2 + z^2) & 2(yz - xw) \\ 2(xz - yw) & 2(yz + xw) & 1 - 2(x^2 + y^2) \end{bmatrix}$$

This conversion appears in `FormationController`, `CooperativeManipulation`,
and the MuJoCo visualization module.

**SLERP (Spherical Linear Interpolation):**

$$q(t) = \frac{\sin((1 - t)\theta)}{\sin\theta}\, q_0 + \frac{\sin(t\theta)}{\sin\theta}\, q_1$$

where $\theta = \arccos(q_0 \cdot q_1)$. When the quaternions are
nearly identical ($\text{dot} > 0.9995$), linear interpolation with
normalisation is used as a numerically stable fallback. When the dot
product is negative, one quaternion is negated to take the short path.

### 25.6.4 Homogeneous Transformations

A rigid-body transformation is represented as a $4 \times 4$ homogeneous
matrix:

$$T = \begin{bmatrix} R & t \\ 0_{1 \times 3} & 1 \end{bmatrix}$$

where $R \in \text{SO}(3)$ is the rotation matrix and
$t \in \mathbb{R}^3$ is the translation vector.

Transformation of a point $p$ in the body frame to the world frame:

$$p_{\text{world}} = R\, p + t$$

Inverse transformation:

$$T^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$$

Composition of transformations:

$$T_{AC} = T_{AB}\, T_{BC}$$

### 25.6.5 Skew-Symmetric Matrix

The skew-symmetric (hat) operator converts a 3-D vector to a
$3 \times 3$ matrix that encodes the cross product:

$$[\mathbf{v}]_\times = \begin{bmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{bmatrix}$$

This satisfies $[\mathbf{v}]_\times\, \mathbf{w} = \mathbf{v} \times \mathbf{w}$
for any vector $\mathbf{w}$.

The skew-symmetric matrix is used extensively in the grasp matrix
computation (Section 21.4.5), the MuJoCo arrow geometry alignment
(Section 22.1.3), and the soft-body FEM stress computation
(Section 21.3.3).

### 25.6.6 Rodrigues' Rotation Formula

Given a unit axis $\hat{k}$ and angle $\theta$, the rotation matrix is:

$$R = I + \sin\theta\, [\hat{k}]_\times + (1 - \cos\theta)\, [\hat{k}]_\times^2$$

This is equivalent to the matrix exponential $R = \exp(\theta\, [\hat{k}]_\times)$
and is used for axis-angle to rotation matrix conversion.

### 25.6.7 Parallel Axis Theorem

When transferring an inertia tensor from the centre of mass to a
parallel axis displaced by vector $d$:

$$I' = I_{\text{cm}} + m\, (d^T d\, I_3 - d\, d^T)$$

This is implemented in the spatial inertia module
(`transform_spatial_inertia`) and used during model generation to express
link inertias in joint frames.

---

## Summary of Key File Paths

| Component                | Path                                                                           |
| ------------------------ | ------------------------------------------------------------------------------ |
| MPC Controller           | `src/research/mpc/controller.py`                                               |
| MPC Specialisations      | `src/research/mpc/specialized.py`                                              |
| Differentiable Engine    | `src/research/differentiable/engine.py`                                        |
| Deformable Objects       | `src/research/deformable/objects.py`                                           |
| Multi-Robot System       | `src/research/multi_robot/system.py`                                           |
| Multi-Robot Coordination | `src/research/multi_robot/coordination.py`                                     |
| Plot Config              | `src/shared/python/plotting/config.py`                                         |
| Plot Renderers           | `src/shared/python/plotting/renderers/`                                        |
| Animation                | `src/shared/python/plotting/animation.py`                                      |
| Export                   | `src/shared/python/plotting/export.py`                                         |
| Dashboard Window         | `src/shared/python/dashboard/window.py`                                        |
| Dashboard Widgets        | `src/shared/python/dashboard/widgets.py`                                       |
| Simulation Runner        | `src/shared/python/dashboard/runner.py`                                        |
| Physics Recorder         | `src/shared/python/dashboard/recorder.py`                                      |
| MuJoCo 3-D Visualization | `src/engines/physics_engines/mujoco/docker/src/humanoid_golf/visualization.py` |
| API Server               | `src/api/server.py`                                                            |
| API Routes (Simulation)  | `src/api/routes/simulation.py`                                                 |
| API Routes (WebSocket)   | `src/api/routes/simulation_ws.py`                                              |
| API Routes (Engines)     | `src/api/routes/engines.py`                                                    |
| API Routes (Auth)        | `src/api/routes/auth.py`                                                       |
| API Routes (Export)      | `src/api/routes/export.py`                                                     |
| API Request Models       | `src/api/models/requests.py`                                                   |
| API Response Models      | `src/api/models/responses.py`                                                  |
| JWT Security             | `src/api/auth/security.py`                                                     |
| Motion Retargeting       | `src/learning/retargeting/retargeter.py`                                       |
| Model Generation         | `src/tools/model_generation/`                                                  |
| Inertia Primitives       | `src/tools/model_generation/inertia/primitives.py`                             |
| Model Explorer           | `src/tools/model_explorer/`                                                    |
| Video Analyzer           | `src/tools/video_analyzer/`                                                    |
| Video Pose Pipeline      | `src/shared/python/video_pose_pipeline.py`                                     |
| Signal Processing        | `src/shared/python/signal_processing.py`                                       |

---

# UpstreamDrift User Manual -- Part 7: Appendices

---

## Appendix A: Mathematical Reference

This appendix consolidates the mathematical notation, equations, and derivations
referenced throughout the UpstreamDrift (Golf Modeling Suite) documentation.
All notation follows standard conventions from robotics, biomechanics, and
computational mechanics literature.

---

### A.1 Notation Conventions

The following symbols are used consistently across the codebase, API
documentation, and this manual.

#### A.1.1 State Variables

| Symbol                          | Domain              | Description                                         |
| ------------------------------- | ------------------- | --------------------------------------------------- |
| $q \in \mathbb{R}^{n_q}$        | Configuration space | Generalized coordinates (joint angles, positions)   |
| $\dot{q} \in \mathbb{R}^{n_v}$  | Tangent space       | Generalized velocities (joint angular/linear rates) |
| $\ddot{q} \in \mathbb{R}^{n_v}$ | Acceleration space  | Generalized accelerations                           |
| $\tau \in \mathbb{R}^{n_u}$     | Actuation space     | Applied torques and forces                          |
| $t \in \mathbb{R}_{\geq 0}$     | Time domain         | Simulation time in seconds                          |

Note: In general $n_q = n_v$ for systems without quaternion joints. For models
using quaternion orientation representation, $n_q = n_v + 1$ per quaternion
joint because the quaternion has four components but only three degrees of
freedom.

#### A.1.2 Dynamic Matrices

| Symbol                                          | Dimensions  | Description                                        |
| ----------------------------------------------- | ----------- | -------------------------------------------------- |
| $M(q) \in \mathbb{R}^{n_v \times n_v}$          | Square, SPD | Mass (inertia) matrix; symmetric positive-definite |
| $C(q, \dot{q}) \in \mathbb{R}^{n_v \times n_v}$ | Square      | Coriolis and centrifugal effects matrix            |
| $g(q) \in \mathbb{R}^{n_v}$                     | Vector      | Gravitational forces vector                        |
| $B \in \mathbb{R}^{n_v \times n_u}$             | Rectangular | Actuation selection matrix                         |
| $J(q) \in \mathbb{R}^{m \times n_v}$            | Rectangular | Task-space Jacobian                                |
| $J_c(q) \in \mathbb{R}^{n_c \times n_v}$        | Rectangular | Contact Jacobian                                   |
| $\Lambda(q) \in \mathbb{R}^{m \times m}$        | Square, SPD | Operational space inertia matrix                   |

The operational space inertia is defined as:

$$\Lambda(q) = \left( J(q) \, M(q)^{-1} \, J(q)^T \right)^{-1}$$

#### A.1.3 Task-Space Variables

| Symbol                         | Domain                   | Description                              |
| ------------------------------ | ------------------------ | ---------------------------------------- |
| $\mathbf{x} \in SE(3)$         | Special Euclidean group  | Task-space pose (position + orientation) |
| $\mathbf{p} \in \mathbb{R}^3$  | Euclidean space          | Cartesian position                       |
| $R \in SO(3)$                  | Special orthogonal group | Rotation matrix                          |
| $\mathcal{V} \in \mathbb{R}^6$ | Spatial velocity         | Twist (angular + linear velocity)        |
| $\mathcal{F} \in \mathbb{R}^6$ | Spatial force            | Wrench (moment + force)                  |
| $f_c \in \mathbb{R}^{n_c}$     | Contact forces           | Contact force magnitudes                 |

#### A.1.4 Muscle Model Variables

| Symbol                        | Domain        | Description                                       |
| ----------------------------- | ------------- | ------------------------------------------------- |
| $a \in [0, 1]$                | Bounded       | Muscle activation level                           |
| $u \in [0, 1]$                | Bounded       | Neural excitation signal                          |
| $l_M \in \mathbb{R}_{>0}$     | Positive real | Muscle fiber length                               |
| $l_T \in \mathbb{R}_{>0}$     | Positive real | Tendon length                                     |
| $l_{MT} \in \mathbb{R}_{>0}$  | Positive real | Musculotendon path length                         |
| $F_M \in \mathbb{R}_{\geq 0}$ | Non-negative  | Total muscle force                                |
| $F_0 \in \mathbb{R}_{>0}$     | Positive      | Maximum isometric muscle force                    |
| $\tilde{l}_M$                 | Normalized    | Normalized fiber length $= l_M / l_M^{opt}$       |
| $\tilde{v}_M$                 | Normalized    | Normalized fiber velocity $= \dot{l}_M / v_{max}$ |
| $r_j$                         | Real          | Muscle moment arm about joint $j$                 |

#### A.1.5 Ball Flight Variables

| Symbol                           | Domain        | Description                                             |
| -------------------------------- | ------------- | ------------------------------------------------------- |
| $\mathbf{r} \in \mathbb{R}^3$    | Position      | Ball position vector (x=downrange, y=lateral, z=height) |
| $\mathbf{v} \in \mathbb{R}^3$    | Velocity      | Ball velocity vector                                    |
| $\omega \in \mathbb{R}_{\geq 0}$ | Scalar        | Ball spin rate (rad/s)                                  |
| $\hat{s} \in \mathbb{R}^3$       | Unit vector   | Spin axis direction                                     |
| $C_D$                            | Dimensionless | Drag coefficient                                        |
| $C_L$                            | Dimensionless | Lift (Magnus) coefficient                               |
| $\rho$                           | kg/m^3        | Air density                                             |
| $A$                              | m^2           | Ball cross-sectional area                               |
| $e$                              | Dimensionless | Coefficient of restitution                              |

#### A.1.6 Subscript and Superscript Conventions

| Notation          | Meaning                                    |
| ----------------- | ------------------------------------------ |
| $(\cdot)_k$       | Value at time step $k$                     |
| $(\cdot)^{opt}$   | Optimal or reference value                 |
| $(\cdot)^{slack}$ | Slack (rest) length                        |
| $(\cdot)^{max}$   | Maximum value                              |
| $[\cdot]_\times$  | Skew-symmetric (cross-product) matrix form |
| $\hat{(\cdot)}$   | Unit vector or bias-corrected estimate     |
| $\tilde{(\cdot)}$ | Normalized quantity                        |

---

### A.2 Equations of Motion

#### A.2.1 Manipulator Equation

The fundamental equation governing the rigid-body dynamics of an articulated
system with $n_v$ velocity degrees of freedom:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = B\tau + J_c(q)^T f_c$$

where:

- $M(q)\ddot{q}$ represents inertial forces,
- $C(q, \dot{q})\dot{q}$ represents velocity-dependent (Coriolis and centrifugal) forces,
- $g(q)$ represents gravitational forces,
- $B\tau$ represents applied actuator forces/torques,
- $J_c(q)^T f_c$ represents contact forces mapped to joint space.

**Properties of $M(q)$:**

1. Symmetric: $M(q) = M(q)^T$
2. Positive-definite: $x^T M(q) x > 0$ for all $x \neq 0$
3. Bounded: $\lambda_{min} I \preceq M(q) \preceq \lambda_{max} I$ for finite $\lambda_{min}, \lambda_{max} > 0$

**Property of $\dot{M} - 2C$:**
The matrix $\dot{M}(q) - 2C(q, \dot{q})$ is skew-symmetric when $C$ is computed
using Christoffel symbols:

$$C_{ij} = \sum_k \frac{1}{2}\left( \frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i} \right) \dot{q}_k$$

This property is essential for passivity-based control and energy analysis.

#### A.2.2 Forward Dynamics

Given $q, \dot{q}, \tau$, compute $\ddot{q}$:

$$\ddot{q} = M(q)^{-1}\left( B\tau + J_c^T f_c - C(q, \dot{q})\dot{q} - g(q) \right)$$

In practice, rather than explicitly forming $M^{-1}$, one solves the linear
system $M(q)\ddot{q} = b$ using Cholesky decomposition ($O(n_v^2)$ for
dense, $O(n_v)$ for sparse/articulated using the Articulated Body Algorithm).

#### A.2.3 Inverse Dynamics

Given $q, \dot{q}, \ddot{q}$, compute $\tau$:

$$\tau = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q)$$

Efficiently computed using the Recursive Newton-Euler Algorithm (RNEA) in
$O(n_v)$ time.

#### A.2.4 Articulated Body Algorithm (ABA)

The ABA computes forward dynamics in $O(n_v)$ for tree-structured kinematic
chains. The algorithm proceeds in three passes:

**Pass 1 (outward):** Compute velocities and bias forces for each body $i$:

$$v_i = {}^i X_{\lambda(i)} \, v_{\lambda(i)} + S_i \dot{q}_i$$

$$a_i^{bias} = {}^i X_{\lambda(i)} \, a_{\lambda(i)}^{bias} + S_i c_i + v_i \times S_i \dot{q}_i$$

**Pass 2 (inward):** Compute articulated body inertias $I_i^A$ and bias forces $p_i^A$:

$$I_i^A = I_i + \sum_{j \in \text{children}(i)} I_j^a$$

$$p_i^A = p_i^{bias} + \sum_{j \in \text{children}(i)} p_j^a$$

where the projected quantities are:

$$U_i = I_i^A S_i, \quad D_i = S_i^T U_i, \quad u_i = \tau_i - S_i^T p_i^A$$

**Pass 3 (outward):** Compute accelerations:

$$\ddot{q}_i = D_i^{-1}(u_i - U_i^T a_{\lambda(i)})$$

$$a_i = a_{\lambda(i)}^{bias} + S_i \ddot{q}_i$$

#### A.2.5 Composite Rigid Body Algorithm (CRBA)

The CRBA computes the mass matrix $M(q)$ in $O(n_v^2)$. This is used
when the full mass matrix is needed (e.g., for operational space control):

$$M_{ij} = S_i^T \, I_i^c \, S_j$$

where $I_i^c$ is the composite rigid body inertia of body $i$ and all its
descendants.

---

### A.3 Spatial Algebra

Spatial algebra provides a compact, coordinate-invariant notation for
rigid body dynamics. UpstreamDrift uses spatial algebra internally in its
Pinocchio and Drake engine backends.

#### A.3.1 Spatial Velocity (Twist)

A spatial velocity combines angular and linear velocity into a single 6D vector:

$$\mathcal{V} = \begin{bmatrix} \omega \\ v \end{bmatrix} \in \mathbb{R}^6$$

where $\omega \in \mathbb{R}^3$ is the angular velocity and $v \in \mathbb{R}^3$
is the linear velocity of the body-fixed frame origin.

#### A.3.2 Spatial Force (Wrench)

A spatial force combines moment and force:

$$\mathcal{F} = \begin{bmatrix} n \\ f \end{bmatrix} \in \mathbb{R}^6$$

where $n \in \mathbb{R}^3$ is the moment (torque) and $f \in \mathbb{R}^3$
is the linear force. The power generated by a wrench acting through a twist is:

$$P = \mathcal{F}^T \mathcal{V} = n^T \omega + f^T v$$

#### A.3.3 Spatial Inertia

The $6 \times 6$ spatial inertia matrix of a rigid body:

$$\mathcal{I} = \begin{bmatrix} I + m[c]_\times [c]_\times^T & m[c]_\times \\ m[c]_\times^T & mE_3 \end{bmatrix}$$

where:

- $m$ is the body mass,
- $c \in \mathbb{R}^3$ is the center of mass offset from the frame origin,
- $I \in \mathbb{R}^{3 \times 3}$ is the rotational inertia about the center of mass,
- $E_3$ is the $3 \times 3$ identity matrix,
- $[c]_\times$ is the skew-symmetric matrix corresponding to cross product with $c$.

**Properties:**

- $\mathcal{I}$ is symmetric positive-definite (for $m > 0$)
- $\mathcal{I}$ has exactly 10 independent parameters: $m$, $c$ (3), $I$ (6 due to symmetry)

#### A.3.4 Spatial Cross Product Operators

**Motion cross product:**

$$\mathcal{V} \times = \begin{bmatrix} [\omega]_\times & 0_{3 \times 3} \\ [v]_\times & [\omega]_\times \end{bmatrix}$$

**Force cross product (dual):**

$$\mathcal{V} \times^* = -(\mathcal{V} \times)^T = \begin{bmatrix} [\omega]_\times^T & [v]_\times^T \\ 0_{3 \times 3} & [\omega]_\times^T \end{bmatrix}$$

#### A.3.5 Skew-Symmetric Matrix

For a vector $a = [a_1, a_2, a_3]^T$, the skew-symmetric form is:

$$[a]_\times = \begin{bmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{bmatrix}$$

such that $[a]_\times b = a \times b$ for any $b \in \mathbb{R}^3$.

#### A.3.6 Plucker Transforms

A Plucker transform ${}^A X_B$ maps spatial quantities from frame $B$ to
frame $A$. Given rotation $R \in SO(3)$ and translation $p \in \mathbb{R}^3$:

**Motion transform:**

$${}^A X_B = \begin{bmatrix} R & 0 \\ [p]_\times R & R \end{bmatrix}$$

**Force transform:**

$${}^A X_B^{*} = \begin{bmatrix} R & [p]_\times R \\ 0 & R \end{bmatrix}$$

These satisfy ${}^A X_B^{*} = ({}^A X_B)^{-T}$.

---

### A.4 Rotation Representations

UpstreamDrift supports multiple rotation representations. Each has
trade-offs between compactness, singularity avoidance, and computational
efficiency.

#### A.4.1 Rotation Matrix

A rotation matrix $R \in SO(3)$ satisfies:

$$R^T R = I, \quad \det(R) = 1$$

**Properties:**

- 9 parameters with 6 constraints (3 DOF)
- No singularities
- Composition: $R_{AC} = R_{AB} \, R_{BC}$
- Inverse: $R^{-1} = R^T$

#### A.4.2 Euler Angles (ZYX Convention)

UpstreamDrift uses the ZYX (yaw-pitch-roll) convention by default:

$$R = R_z(\psi) \, R_y(\theta) \, R_x(\phi)$$

where:

$$R_x(\phi) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi & \cos\phi \end{bmatrix}$$

$$R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$$

$$R_z(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Gimbal lock** occurs when $\theta = \pm \pi/2$, causing loss of one degree
of freedom. UpstreamDrift detects gimbal lock proximity and issues warnings
when $|\theta - \pi/2| < 0.01$ rad.

#### A.4.3 Axis-Angle (Rodrigues' Formula)

A rotation of angle $\theta$ about unit axis $\hat{k}$:

$$R = I + \sin\theta \, [\hat{k}]_\times + (1 - \cos\theta) \, [\hat{k}]_\times^2$$

This is the matrix exponential of the skew-symmetric matrix:

$$R = \exp(\theta \, [\hat{k}]_\times) = \exp([\omega]_\times)$$

where $\omega = \theta \hat{k}$ is the rotation vector. The inverse mapping
(logarithm) is:

$$\theta = \arccos\!\left(\frac{\text{tr}(R) - 1}{2}\right), \quad [\hat{k}]_\times = \frac{R - R^T}{2\sin\theta}$$

#### A.4.4 Unit Quaternion

A unit quaternion $\mathbf{q} = [w, x, y, z]^T$ with $\|\mathbf{q}\| = 1$ represents
a rotation. Writing $\mathbf{q} = [w, \mathbf{v}]$ where $\mathbf{v} = [x, y, z]^T$:

**Quaternion multiplication:**

$$\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix} w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2 \\ w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2 \end{bmatrix}$$

**Rotating a point:**

$$\mathbf{p}' = \mathbf{q} \otimes [0, \mathbf{p}]^T \otimes \mathbf{q}^{-1}$$

where $\mathbf{q}^{-1} = [w, -\mathbf{v}]^T$ for unit quaternions (conjugate).

**Quaternion to rotation matrix:**

$$R = \begin{bmatrix} 1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\ 2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\ 2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2) \end{bmatrix}$$

**Quaternion integration (angular velocity $\omega$ over timestep $\Delta t$):**

$$\Delta \mathbf{q} = \begin{bmatrix} \cos(\|\omega\| \Delta t / 2) \\ \sin(\|\omega\| \Delta t / 2) \, \hat{\omega} \end{bmatrix}$$

$$\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \Delta \mathbf{q}$$

**Advantages over Euler angles:**

- No gimbal lock
- Smooth interpolation via SLERP
- Compact (4 parameters, 1 constraint)
- Efficient composition

**Quaternion SLERP (Spherical Linear Interpolation):**

$$\text{slerp}(\mathbf{q}_0, \mathbf{q}_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} \mathbf{q}_0 + \frac{\sin(t\Omega)}{\sin\Omega} \mathbf{q}_1$$

where $\cos\Omega = \mathbf{q}_0 \cdot \mathbf{q}_1$ and $t \in [0, 1]$.

---

### A.5 Optimization Methods

UpstreamDrift employs several optimization methods for trajectory
optimization, parameter identification, and motion planning.

#### A.5.1 SLSQP (Sequential Least Squares Programming)

Used for small-to-medium scale constrained optimization. Solves a sequence of
quadratic programming subproblems:

$$\min_{d} \quad \frac{1}{2} d^T H_k d + \nabla f(x_k)^T d$$

$$\text{s.t.} \quad \nabla c_i(x_k)^T d + c_i(x_k) = 0, \quad i \in \mathcal{E}$$

$$\quad\quad\; \nabla c_i(x_k)^T d + c_i(x_k) \geq 0, \quad i \in \mathcal{I}$$

where $H_k$ is a BFGS approximation to the Hessian of the Lagrangian.

UpstreamDrift uses SciPy's `minimize(method='SLSQP')` for problems with
fewer than approximately 100 decision variables.

#### A.5.2 Adam Optimizer

Used for neural network training in reinforcement learning policies:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$\theta_{t+1} = \theta_t - \frac{\eta \, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Default hyperparameters in UpstreamDrift:**

- Learning rate: $\eta = 10^{-3}$
- First moment decay: $\beta_1 = 0.9$
- Second moment decay: $\beta_2 = 0.999$
- Numerical stability: $\epsilon = 10^{-8}$

#### A.5.3 SNOPT and IPOPT

For large-scale nonlinear programming (trajectory optimization with Drake):

$$\min_{x} \quad f(x)$$

$$\text{s.t.} \quad c_L \leq c(x) \leq c_U$$

$$\quad\quad\; x_L \leq x \leq x_U$$

**SNOPT** uses a sparse sequential quadratic programming (SQP) method with
limited-memory quasi-Newton Hessian updates. Preferred when the constraint
Jacobian is sparse.

**IPOPT** uses a primal-dual interior-point method with line-search or
trust-region globalization. Preferred for problems with many inequality
constraints.

#### A.5.4 Direct Collocation

Trajectory optimization by transcribing continuous dynamics into a finite-
dimensional nonlinear program. UpstreamDrift supports Hermite-Simpson
collocation.

**Decision variables:** States $x_k$ and controls $u_k$ at $N$ knot points.

**Hermite-Simpson collocation defect constraints:**

Midpoint state interpolation:

$$x_{k+1/2} = \frac{x_k + x_{k+1}}{2} + \frac{h}{8}\left(f_k - f_{k+1}\right)$$

Simpson quadrature constraint:

$$x_{k+1} = x_k + \frac{h}{6}\left(f_k + 4 f_{k+1/2} + f_{k+1}\right)$$

where $f_k = f(x_k, u_k)$ is the dynamics evaluated at knot point $k$,
$f_{k+1/2} = f(x_{k+1/2}, u_{k+1/2})$ is the dynamics at the midpoint,
and $h$ is the time step between knot points.

**Typical problem sizes in UpstreamDrift:**

- Swing optimization (3 DOF, 100 knot points): ~600 decision variables, ~300 constraints
- Full-body swing (15 DOF, 200 knot points): ~6000 decision variables, ~3000 constraints

#### A.5.5 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

Used for derivative-free optimization of swing parameters:

$$\mathbf{x}_k^{(i)} \sim \mathcal{N}\!\left(\mathbf{m}_k, \sigma_k^2 \, C_k\right), \quad i = 1, \ldots, \lambda$$

where $\mathbf{m}_k$ is the distribution mean, $\sigma_k$ is the step size,
and $C_k$ is the covariance matrix, all adapted over generations $k$.

---

### A.6 Hill-Type Muscle Model -- Complete Derivation

UpstreamDrift implements Hill-type muscle models for musculoskeletal
simulation via the MyoSuite integration and the `activation_dynamics` module.

#### A.6.1 Musculotendon Architecture

A musculotendon unit consists of:

1. **Contractile Element (CE):** Active force generation
2. **Parallel Elastic Element (PE):** Passive stiffness of muscle fiber
3. **Series Elastic Element (SE/Tendon):** Tendon compliance

The total musculotendon length:

$$l_{MT} = l_T + l_M \cos\alpha$$

where $\alpha$ is the pennation angle, $l_T$ is the tendon length, and
$l_M$ is the muscle fiber length.

#### A.6.2 Active Muscle Force

The contractile element force depends on activation, force-length, and
force-velocity relationships:

$$F_{CE} = a \cdot F_0 \cdot f_L(\tilde{l}_M) \cdot f_V(\tilde{v}_M)$$

where:

- $a \in [0, 1]$ is the activation level,
- $F_0$ is the maximum isometric force,
- $\tilde{l}_M = l_M / l_M^{opt}$ is the normalized fiber length,
- $\tilde{v}_M = \dot{l}_M / v_{max}$ is the normalized fiber velocity.

#### A.6.3 Force-Length Relationship

The active force-length curve is modeled as a Gaussian:

$$f_L(\tilde{l}) = \exp\!\left( -\frac{(\tilde{l} - 1)^2}{\gamma} \right)$$

where $\gamma \approx 0.45$ controls the width of the bell curve. The muscle
generates peak force at its optimal length ($\tilde{l} = 1$) and diminishing
force at shorter or longer lengths.

#### A.6.4 Force-Velocity Relationship

The force-velocity relationship is asymmetric, with different behavior for
concentric (shortening) and eccentric (lengthening) contractions:

$$f_V(\tilde{v}) = \begin{cases} \displaystyle \frac{1 + \tilde{v}/\tilde{v}_{max}}{1 - \tilde{v}/(K \cdot \tilde{v}_{max})} & \tilde{v} \leq 0 \;\text{(concentric)} \\[10pt] \displaystyle N + (N - 1)\frac{\tilde{v}_{max} + \tilde{v}}{7.56 K \tilde{v} - \tilde{v}_{max}} & \tilde{v} > 0 \;\text{(eccentric)} \end{cases}$$

**Typical parameters:**

- $K \approx 0.25$: Shape factor for concentric curve
- $N \approx 1.5$: Eccentric force enhancement ratio
- $\tilde{v}_{max} \approx 10$: Maximum normalized shortening velocity

#### A.6.5 Passive Elastic Force

The parallel elastic element produces force when the muscle is stretched
beyond its optimal length:

$$F_{PE}(\tilde{l}_M) = k_{PE} \cdot \left[\max(0, \, \tilde{l}_M - 1)\right]^2 \cdot F_0$$

where $k_{PE} \approx 5.0$ is the passive stiffness coefficient.

#### A.6.6 Total Muscle Force

The total muscle force is:

$$F_M = F_{CE} + F_{PE} = a \cdot F_0 \cdot f_L(\tilde{l}_M) \cdot f_V(\tilde{v}_M) + F_{PE}(\tilde{l}_M)$$

#### A.6.7 Tendon Force

The tendon is modeled as a nonlinear elastic element:

$$F_T = F_T^{max} \cdot f_T(\epsilon_T)$$

where the tendon strain is:

$$\epsilon_T = \frac{l_T - l_T^{slack}}{l_T^{slack}}$$

and $f_T(\epsilon_T)$ is an exponential-linear tendon curve:

$$f_T(\epsilon) = \begin{cases} 0 & \epsilon < 0 \\ c_1 \exp(k_T \epsilon) - c_2 & 0 \leq \epsilon < \epsilon_{toe} \\ k_{lin}(\epsilon - \epsilon_{toe}) + F_{toe} & \epsilon \geq \epsilon_{toe} \end{cases}$$

with $\epsilon_{toe} \approx 0.033$ (3.3% strain at the toe region transition).

#### A.6.8 Activation Dynamics

The time delay between neural excitation and muscle activation is modeled
as a first-order differential equation (implemented in
`src/shared/python/activation_dynamics.py`):

$$\dot{a} = \frac{u - a}{\tau(u, a)}$$

where the time constant $\tau$ depends on whether the muscle is activating
or deactivating:

$$\tau = \begin{cases} \tau_{act} \cdot (0.5 + 1.5a) & \text{if } u > a \;\text{(activation)} \\ \tau_{deact} / (0.5 + 1.5a) & \text{if } u \leq a \;\text{(deactivation)} \end{cases}$$

**Typical values:**

- $\tau_{act} \approx 10 \text{ ms}$: Fast activation (calcium release)
- $\tau_{deact} \approx 40 \text{ ms}$: Slower deactivation (calcium reuptake)

This asymmetry is physiologically realistic: calcium release from the
sarcoplasmic reticulum is faster than reuptake by the calcium pump (SERCA).

#### A.6.9 Moment Arm and Joint Torque

The muscle moment arm about joint $j$ is the negative partial derivative of
the musculotendon path length with respect to the joint angle:

$$r_j = -\frac{\partial l_{MT}}{\partial q_j}$$

The total joint torque from all muscles crossing joint $j$:

$$\tau_j = \sum_{m=1}^{N_m} r_{j,m} \cdot F_M^m$$

where $N_m$ is the number of muscles crossing joint $j$.

#### A.6.10 Equilibrium Condition

At each instant, the force in the tendon must equal the component of muscle
force along the tendon direction:

$$F_T = F_M \cos\alpha$$

This equilibrium condition is solved implicitly at each time step to
determine the fiber length and tendon length partition.

---

### A.7 Numerical Integration Methods

#### A.7.1 Runge-Kutta 4th Order (RK4)

The primary integrator for ball flight simulation
(`src/shared/python/ball_flight_physics.py`):

$$k_1 = f(t_n, y_n)$$

$$k_2 = f(t_n + h/2, \; y_n + h k_1 / 2)$$

$$k_3 = f(t_n + h/2, \; y_n + h k_2 / 2)$$

$$k_4 = f(t_n + h, \; y_n + h k_3)$$

$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Local truncation error: $O(h^5)$. Global error: $O(h^4)$.

#### A.7.2 Semi-Implicit Euler (MuJoCo Default)

MuJoCo uses a semi-implicit Euler scheme:

$$v_{n+1} = v_n + h \, M^{-1}(f_n + J_c^T \lambda)$$

$$q_{n+1} = q_n + h \, v_{n+1}$$

This is first-order but symplectic, preserving energy behavior over long
simulations. MuJoCo compensates with small time steps ($h \leq 0.002$ s).

#### A.7.3 Implicit Midpoint Rule

For stiff systems and contact-rich simulation:

$$q_{n+1} = q_n + h \, f\!\left(\frac{q_n + q_{n+1}}{2}\right)$$

Requires solving a nonlinear system at each step (Newton iterations).

---

## Appendix B: Model Specifications

This appendix details the biomechanical models available in UpstreamDrift,
ordered from simplest to most complex.

---

### B.1 Double Pendulum (2 DOF)

The simplest golf swing model, treating the arm-club system as a planar
double pendulum.

| Property                 | Segment 1 (Upper Arm)                      | Segment 2 (Forearm + Club)                 |
| ------------------------ | ------------------------------------------ | ------------------------------------------ |
| Length                   | 0.30 m                                     | 0.60 m                                     |
| Mass                     | 2.0 kg                                     | 1.5 kg                                     |
| MOI (about proximal end) | $\frac{1}{3} m l^2 = 0.060 \text{ kg m}^2$ | $\frac{1}{3} m l^2 = 0.180 \text{ kg m}^2$ |
| COM location             | 0.15 m from shoulder                       | 0.30 m from elbow                          |

**Joints:**

- Shoulder: Revolute, range $[-\pi, \pi]$ rad
- Elbow: Revolute, range $[0, \pi]$ rad (only flexion/extension)

**Degrees of freedom:** 2 ($n_q = n_v = 2$)

**Use cases:** Basic swing mechanics, wrist-lag analysis, teaching demonstrations.

**Configuration:** `q = [\theta_{shoulder}, \theta_{elbow}]^T$

---

### B.2 Triple Pendulum (3 DOF)

Extends the double pendulum with a wrist joint for studying club release
mechanics.

| Property   | Upper Arm     | Forearm    | Hand + Club            |
| ---------- | ------------- | ---------- | ---------------------- |
| Length     | 0.30 m        | 0.27 m     | 0.19 m + 1.14 m (club) |
| Mass       | 2.5 kg        | 1.5 kg     | 0.4 kg + 0.315 kg      |
| Joint      | Shoulder      | Elbow      | Wrist                  |
| Joint type | Revolute      | Revolute   | Revolute               |
| Range      | $[-\pi, \pi]$ | $[0, \pi]$ | $[-\pi/2, \pi/3]$      |

**Degrees of freedom:** 3 ($n_q = n_v = 3$)

**Configuration:** $q = [\theta_{shoulder}, \theta_{elbow}, \theta_{wrist}]^T$

**Use cases:** Wrist cock/release timing, kinematic sequence analysis, club
lag optimization.

---

### B.3 Upper Body Model (10 DOF)

A more realistic model including bilateral arm kinematics and trunk rotation.

| Body Segment   | DOF | Joint Types                            |
| -------------- | --- | -------------------------------------- |
| Torso          | 2   | Axial rotation, lateral bend           |
| Left shoulder  | 2   | Flexion/extension, abduction/adduction |
| Right shoulder | 2   | Flexion/extension, abduction/adduction |
| Left elbow     | 1   | Flexion/extension                      |
| Right elbow    | 1   | Flexion/extension                      |
| Left wrist     | 1   | Flexion/extension                      |
| Right wrist    | 1   | Flexion/extension                      |

**Total degrees of freedom:** 10 ($n_q = n_v = 10$)

**Segment masses (default, kg):**

- Torso: 35.0 (from `TORSO_MASS_KG`)
- Upper arm: 2.5 each (from `UPPER_ARM_MASS_KG`)
- Forearm: 1.5 each (from `FOREARM_MASS_KG`)
- Hand: 0.4 each (from `HAND_MASS_KG`)

**Use cases:** Bilateral coordination, trunk rotation contribution, X-factor
analysis, lead arm vs. trail arm mechanics.

---

### B.4 Full Body Model (15 DOF)

Adds lower body kinematics for ground reaction force modeling and
kinetic chain analysis.

| Body Segment | DOF | Joint Types                 |
| ------------ | --- | --------------------------- |
| Upper body   | 10  | (as in B.3)                 |
| Left hip     | 1   | Internal/external rotation  |
| Right hip    | 1   | Internal/external rotation  |
| Left knee    | 1   | Flexion/extension           |
| Right knee   | 1   | Flexion/extension           |
| Left ankle   | 1   | Dorsiflexion/plantarflexion |

**Total degrees of freedom:** 15 ($n_q = n_v = 15$)

**Additional segment masses (default, kg):**

- Thigh: 7.5 each
- Shank: 3.5 each
- Foot: 1.0 each

**Use cases:** Full kinetic chain, ground reaction forces, weight transfer,
lower body contribution to clubhead speed.

---

### B.5 Advanced Model (28 DOF)

High-fidelity model with scapular kinematics, 3-DOF shoulders, and
flexible club shaft.

| Body Segment   | DOF | Joint Types                                                        |
| -------------- | --- | ------------------------------------------------------------------ |
| Spine          | 4   | Flexion, extension, lateral bend, axial rotation                   |
| Left scapula   | 2   | Protraction/retraction, elevation/depression                       |
| Right scapula  | 2   | Protraction/retraction, elevation/depression                       |
| Left shoulder  | 3   | Flexion/extension, abduction/adduction, internal/external rotation |
| Right shoulder | 3   | Flexion/extension, abduction/adduction, internal/external rotation |
| Left elbow     | 2   | Flexion/extension, pronation/supination                            |
| Right elbow    | 2   | Flexion/extension, pronation/supination                            |
| Left wrist     | 2   | Flexion/extension, radial/ulnar deviation                          |
| Right wrist    | 2   | Flexion/extension, radial/ulnar deviation                          |
| Hips           | 2   | Internal/external rotation (each)                                  |
| Knees          | 2   | Flexion/extension (each)                                           |

**Club shaft flexibility:** Additional DOF from modal or finite element
shaft model (see `src/shared/python/flexible_shaft.py`):

- Modal model: 2-4 mode shapes (bending modes)
- FE model: $2 \times n_{nodes}$ DOF (deflection + rotation per node)

**Total degrees of freedom:** 28 (rigid body) + shaft DOF

**Use cases:** Scapular plane analysis, shaft flexibility effects, pronation/
supination contribution, detailed shoulder mechanics, club path optimization.

---

### B.6 MyoSuite Full Body (52 DOF, 290 Muscles)

Complete musculoskeletal model integrated through the MyoSuite/OpenSim
interface.

| System                 | DOF | Muscles                                                  |
| ---------------------- | --- | -------------------------------------------------------- |
| Lumbar spine           | 3   | 36 (erector spinae, obliques, rectus abdominis, etc.)    |
| Thoracic spine         | 3   | 24                                                       |
| Neck                   | 3   | 16                                                       |
| Left shoulder complex  | 5   | 22 (deltoid, rotator cuff, pectoralis, latissimus, etc.) |
| Right shoulder complex | 5   | 22                                                       |
| Left elbow/forearm     | 2   | 18 (biceps, triceps, brachialis, pronators, supinators)  |
| Right elbow/forearm    | 2   | 18                                                       |
| Left wrist/hand        | 4   | 20 (flexors, extensors, intrinsic hand muscles)          |
| Right wrist/hand       | 4   | 20                                                       |
| Left hip               | 3   | 22 (gluteals, iliopsoas, adductors, etc.)                |
| Right hip              | 3   | 22                                                       |
| Left knee              | 1   | 12 (quadriceps, hamstrings, gastrocnemius)               |
| Right knee             | 1   | 12                                                       |
| Left ankle/foot        | 3   | 14 (tibialis anterior/posterior, peroneals, intrinsic)   |
| Right ankle/foot       | 3   | 14                                                       |
| Subtalar joints        | 2   | 6                                                        |
| Metatarsophalangeal    | 4   | 12                                                       |

**Total:** 52 DOF, 290 muscles

**Muscle properties for each muscle include:**

- Maximum isometric force $F_0$
- Optimal fiber length $l_M^{opt}$
- Tendon slack length $l_T^{slack}$
- Pennation angle $\alpha_0$
- Maximum shortening velocity $v_{max}$
- Activation/deactivation time constants $\tau_{act}, \tau_{deact}$

**Use cases:** Detailed muscle coordination, co-contraction analysis,
injury risk assessment, EMG-driven simulation, metabolic cost estimation,
muscle-level optimization of swing technique.

---

### B.7 Standard Humanoid (66 DOF)

Derived from the human-gazebo `humanSubject06_66dof` model (robotology
project). This high-fidelity model includes full-body articulation with
mesh-based collision and visualization geometries.

**Key properties:**

- Height: 1.75 m
- Mass: 75.0 kg
- Format: URDF with STL meshes
- License: CC-BY-SA 2.0
- Cross-engine compatible: MuJoCo, Drake, Pinocchio

---

## Appendix C: Golf Ball Physics

This appendix documents the ball flight model implemented in
`src/shared/python/ball_flight_physics.py` and the aerodynamics module.

---

### C.1 USGA Ball Specifications

All ball parameters conform to USGA and R&A equipment rules:

| Parameter                | Value                                      | Rule Reference    |
| ------------------------ | ------------------------------------------ | ----------------- |
| Maximum mass             | 45.93 g (1.620 oz)                         | USGA Rule 5-1     |
| Minimum diameter         | 42.67 mm (1.680 in)                        | USGA Rule 5-2     |
| Maximum COR              | 0.830 at 48.8 m/s (109 mph)                | USGA Appendix III |
| Maximum initial velocity | 76.2 m/s (250 ft/s) + 2% tolerance         | USGA Appendix III |
| Symmetry                 | Must not deviate > 0.75% between poles     | USGA Appendix III |
| Overall distance         | Not to exceed 317 yards + 3 yard tolerance | USGA Appendix III |

**Dimple specifications:**

- Typical count: 300-500 dimples
- Dimple depth: 0.127-0.178 mm (0.005-0.007 in)
- Dimple diameter: 3.56-4.45 mm (0.140-0.175 in)
- Coverage: approximately 75-80% of ball surface
- Patterns: icosahedral, octahedral, or proprietary

**Default ball properties in UpstreamDrift** (from `BallProperties` dataclass):

| Property                     | Symbol   | Default Value                          |
| ---------------------------- | -------- | -------------------------------------- |
| Mass                         | $m$      | 0.04593 kg                             |
| Diameter                     | $d$      | 0.04267 m                              |
| Radius                       | $r$      | 0.021335 m                             |
| Cross-sectional area         | $A$      | $\pi r^2 = 1.430 \times 10^{-3}$ m$^2$ |
| Drag coefficient (base)      | $C_{D0}$ | 0.21                                   |
| Drag coefficient (linear)    | $C_{D1}$ | 0.05                                   |
| Drag coefficient (quadratic) | $C_{D2}$ | 0.02                                   |
| Lift coefficient (base)      | $C_{L0}$ | 0.00                                   |
| Lift coefficient (linear)    | $C_{L1}$ | 0.38                                   |
| Lift coefficient (quadratic) | $C_{L2}$ | 0.08                                   |

---

### C.2 Ball Flight Model

#### C.2.1 Governing Equation

The ball is treated as a point mass with aerodynamic forces:

$$m \ddot{\mathbf{r}} = \mathbf{F}_g + \mathbf{F}_D + \mathbf{F}_L$$

where the three force components are gravity, drag, and lift (Magnus).

**Coordinate system:**

- $x$: downrange (positive toward target)
- $y$: lateral (positive to the right from behind the ball)
- $z$: vertical (positive upward)

#### C.2.2 Gravitational Force

$$\mathbf{F}_g = \begin{bmatrix} 0 \\ 0 \\ -mg \end{bmatrix}$$

where $g = 9.80665$ m/s$^2$ (from `GRAVITY_M_S2`).

#### C.2.3 Aerodynamic Drag

$$\mathbf{F}_D = -\frac{1}{2} \rho \, v_{rel} \, C_D \, A \, \mathbf{v}_{rel}$$

where:

- $\rho = 1.225$ kg/m$^3$ is air density at sea level and 15 C (from `AIR_DENSITY_SEA_LEVEL_KG_M3`),
- $\mathbf{v}_{rel} = \mathbf{v} - \mathbf{v}_{wind}$ is the velocity relative to the air,
- $v_{rel} = \|\mathbf{v}_{rel}\|$ is the relative speed,
- $A = \pi r^2$ is the cross-sectional area.

The drag coefficient depends on the spin ratio $S = \omega r / v_{rel}$:

$$C_D(S) = C_{D0} + S \cdot C_{D1} + S^2 \cdot C_{D2}$$

For a typical driver shot ($v = 70$ m/s, $\omega = 280$ rad/s, $r = 0.021$ m):
$S \approx 0.084$, $C_D \approx 0.214$.

#### C.2.4 Magnus Force (Lift)

The Magnus effect creates a force perpendicular to both the velocity and spin
axis:

$$\mathbf{F}_L = \frac{1}{2} \rho \, v_{rel}^2 \, C_L \, A \, \frac{\hat{s} \times \hat{v}_{rel}}{\|\hat{s} \times \hat{v}_{rel}\|}$$

where $\hat{s}$ is the unit spin axis vector and $C_L$ depends on the spin ratio:

$$C_L(S) = \min\!\left(C_{L,max}, \; C_{L0} + S \cdot C_{L1} + S^2 \cdot C_{L2}\right)$$

The maximum lift coefficient is capped at $C_{L,max} = 0.25$ to prevent
unphysical behavior at extreme spin rates.

**Backspin** (spin axis pointing left from behind) creates upward lift,
increasing carry distance and landing angle.

**Sidespin** creates lateral drift (draw/fade/hook/slice).

#### C.2.5 Spin Decay

Ball spin decays exponentially due to air resistance:

$$\omega(t) = \omega_0 \, e^{-t / \tau_{spin}}$$

where $\tau_{spin} \approx 20$ s is the spin decay time constant
(from `SPIN_DECAY_RATE_S = 0.05` 1/s, so $\tau_{spin} = 1/0.05 = 20$ s).

In the `EnhancedBallFlightSimulator`, spin decay is computed per time step:

$$\omega_{n+1} = \omega_n \cdot e^{-\Delta t / \tau_{spin}}$$

#### C.2.6 Environmental Effects

**Altitude correction:** Air density decreases with altitude:

$$\rho(h) = \rho_0 \exp\!\left(-\frac{Mgh}{RT}\right)$$

where $M = 0.029$ kg/mol is the molar mass of air, $R = 8.314$ J/(mol K), and
$T$ is the absolute temperature.

**Temperature correction:** Air density is inversely proportional to temperature:

$$\rho(T) = \rho_0 \cdot \frac{T_0}{T}$$

**Humidity:** Moist air is less dense than dry air (water vapor is lighter
than N$_2$/O$_2$), reducing drag.

**Wind model** (from `aerodynamics.py`):

- Base wind velocity (constant component)
- Gust model (random impulses with configurable frequency and magnitude)
- Turbulence model (Dryden or von Karman spectral models)

---

### C.3 Impact Model

The club-ball impact determines launch conditions from swing delivery
parameters.

#### C.3.1 Coefficient of Restitution

The COR relates separation and approach speeds:

$$e = -\frac{v_{sep}}{v_{approach}} = -\frac{v_{ball}' - v_{club}'}{v_{ball} - v_{club}}$$

For a stationary ball ($v_{ball} = 0$):

$$e = \frac{v_{ball}' - v_{club}'}{v_{club}}$$

USGA maximum COR: $e = 0.830$ (for drivers at standard test speed).

In practice, COR varies with:

- Impact speed (COR decreases at very high speeds due to material nonlinearity)
- Impact location on the face (lower at heel/toe due to reduced face flex)
- Club type (irons have lower COR than drivers)

#### C.3.2 Ball Speed

From conservation of momentum and the COR definition:

$$v_{ball} = \frac{(1 + e) \, m_{club}}{m_{club} + m_{ball}} \cdot v_{club}$$

For typical values ($e = 0.83$, $m_{club} = 0.200$ kg, $m_{ball} = 0.046$ kg):

$$\frac{v_{ball}}{v_{club}} = \frac{(1 + 0.83) \times 0.200}{0.200 + 0.046} = \frac{0.366}{0.246} \approx 1.49$$

This "smash factor" of ~1.49 is consistent with measured driver performance.

#### C.3.3 Launch Angle

The launch angle depends on the club loft $\alpha_{loft}$, the angle of
attack $\alpha_{AoA}$ (dynamic loft), and the impact location:

$$\alpha_{launch} \approx k_1 \cdot \alpha_{dynamic\,loft} + k_2 \cdot \alpha_{AoA}$$

where $k_1 \approx 0.85$ and $k_2 \approx 0.15$ for a driver. The dynamic
loft includes shaft lean and face angle at impact.

#### C.3.4 Spin Generation

Ball spin arises from two mechanisms:

**Loft spin (backspin):** Friction between the ball and the lofted club face:

$$\omega_{back} \propto \frac{\mu \, F_N \, \Delta t}{I_{ball}} \cdot \sin(\alpha_{loft})$$

**Gear effect spin:** For off-center impacts, the club head rotates about its
center of gravity, imparting additional spin to the ball in the opposite
direction. For toe hits on a driver, this produces draw spin (right-to-left
for a right-handed golfer).

**Total spin axis tilt** determines the ratio of backspin to sidespin:

$$\omega_{total} = \sqrt{\omega_{back}^2 + \omega_{side}^2}$$

$$\text{Axis tilt} = \arctan\!\left(\frac{\omega_{side}}{\omega_{back}}\right)$$

---

### C.4 Typical Ball Flight Parameters

Reference values for a well-struck driver shot by a professional golfer:

| Parameter      | Value                     |
| -------------- | ------------------------- |
| Clubhead speed | 50-56 m/s (112-125 mph)   |
| Ball speed     | 74-83 m/s (165-186 mph)   |
| Smash factor   | 1.48-1.51                 |
| Launch angle   | 10-14 degrees             |
| Backspin rate  | 2200-2800 rpm             |
| Carry distance | 255-290 m (280-317 yards) |
| Maximum height | 28-35 m (92-115 ft)       |
| Flight time    | 5.5-6.5 s                 |
| Landing angle  | 38-45 degrees             |

---

## Appendix D: Configuration Reference

This appendix details all configuration files, their format, and available
options in the UpstreamDrift project.

---

### D.1 pyproject.toml

The primary project configuration file, following PEP 621 and using Hatchling
as the build backend.

**Location:** `/pyproject.toml`

#### D.1.1 Project Metadata

```toml
[project]
name = "upstream-drift"
version = "2.1.0"
description = "Biomechanical golf simulation and analysis suite"
requires-python = ">=3.11"
license = { text = "MIT" }
```

#### D.1.2 Core Dependencies

| Package           | Version Constraint | Purpose                    |
| ----------------- | ------------------ | -------------------------- |
| numpy             | >=1.26.4, <3.0.0   | Numerical computation      |
| scipy             | >=1.13.1           | Scientific algorithms      |
| fastapi           | >=0.126.0          | REST API framework         |
| uvicorn[standard] | >=0.30.0           | ASGI server                |
| pydantic          | >=2.5.0            | Data validation            |
| httpx             | >=0.27.0           | HTTP client                |
| mujoco            | >=3.3.0, <4.0.0    | Default physics engine     |
| simpleeval        | >=1.0.0            | Safe expression evaluation |
| structlog         | >=24.1.0           | Structured logging         |

#### D.1.3 Optional Dependency Groups

| Group         | Command                           | Packages                                       |
| ------------- | --------------------------------- | ---------------------------------------------- |
| `drake`       | `pip install -e ".[drake]"`       | drake>=1.22.0                                  |
| `pinocchio`   | `pip install -e ".[pinocchio]"`   | pin>=2.6.0, meshcat>=0.3.0                     |
| `all-engines` | `pip install -e ".[all-engines]"` | drake + pinocchio                              |
| `analysis`    | `pip install -e ".[analysis]"`    | opencv-python>=4.8.0, scikit-learn>=1.3.0      |
| `urdf`        | `pip install -e ".[urdf]"`        | trimesh>=4.0.0, PyYAML>=6.0, defusedxml>=0.7.0 |
| `dev`         | `pip install -e ".[dev]"`         | pytest, ruff, mypy, etc.                       |
| `gui-test`    | `pip install -e ".[gui-test]"`    | PyQt6>=6.5.0, pytest-qt>=4.2.0                 |
| `all`         | `pip install -e ".[all]"`         | All of the above (except gui-test)             |

#### D.1.4 Tool Configurations

**Ruff (linting):**

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501", "B008"]
```

| Rule Set | Description                      |
| -------- | -------------------------------- |
| E        | pycodestyle errors               |
| F        | Pyflakes                         |
| I        | isort (import ordering)          |
| UP       | pyupgrade (Python modernization) |
| B        | flake8-bugbear (common bugs)     |

**Mypy (type checking):**

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
check_untyped_defs = true
disallow_untyped_defs = false
```

**Pytest:**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --strict-markers --strict-config"
markers = [
    "slow",
    "integration",
    "unit",
    "requires_gl",
    "headless_safe",
    "slow_numba",
    "asyncio",
    "benchmark",
]
```

---

### D.2 environment.yml

Conda environment specification for full installation with binary
dependencies.

**Location:** `/environment.yml`

**Environment name:** `golf-suite`

**Channels:**

1. `conda-forge` (primary)
2. `defaults` (fallback)

#### D.2.1 Conda-Managed Dependencies

| Category      | Packages                                                                   |
| ------------- | -------------------------------------------------------------------------- |
| Core Python   | python=3.11, pip>=23.0                                                     |
| Scientific    | numpy>=1.26.4, scipy>=1.13.1, pandas>=2.2.3, sympy>=1.12                   |
| Visualization | matplotlib>=3.8.0, opencv>=4.9.0                                           |
| GUI           | pyqt>=6.6.0                                                                |
| Physics       | mujoco>=3.3.0                                                              |
| Dev tools     | pytest>=8.2.1, pytest-cov, pytest-mock, pytest-qt, black, mypy, pre-commit |

#### D.2.2 Pip-Managed Dependencies

| Category         | Packages                                           |
| ---------------- | -------------------------------------------------- |
| API              | fastapi, uvicorn, slowapi, python-multipart, httpx |
| Database/Auth    | sqlalchemy, pydantic, PyJWT, bcrypt, cryptography  |
| Data formats     | pyyaml, defusedxml, ezc3d                          |
| Logging/Security | structlog, simpleeval                              |
| Linting          | ruff==0.14.10                                      |
| Type stubs       | types-PyYAML, pandas-stubs, types-requests         |

#### D.2.3 Platform-Specific Notes

| Platform                  | Notes                                                                                                     |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Windows**               | Requires Visual C++ Redistributable 2015-2022 for MuJoCo. Update GPU drivers if OpenGL errors occur.      |
| **macOS (Apple Silicon)** | Use `CONDA_SUBDIR=osx-arm64 conda env create ...` for M1/M2/M3 chips. Some engines may lack ARM64 builds. |
| **Linux**                 | Install OpenGL libraries: `sudo apt install libgl1-mesa-glx`. For headless servers: `xvfb-run pytest`.    |

#### D.2.4 Installation Commands

```bash
# Full installation
conda env create -f environment.yml
conda activate golf-suite

# Update existing environment
conda env update -f environment.yml --prune

# Light installation (no heavy engines)
conda env create -f environment.yml --name golf-suite-light
```

---

### D.3 Makefile Targets

**Location:** `/Makefile`

| Target      | Command          | Description                                                                                        |
| ----------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| `help`      | `make help`      | Display available targets and descriptions                                                         |
| `install`   | `make install`   | Install requirements.txt and package in editable mode with dev extras                              |
| `lint`      | `make lint`      | Run ruff check and mypy (mypy errors are advisory)                                                 |
| `format`    | `make format`    | Run black, ruff format, and ruff fix                                                               |
| `test`      | `make test`      | Run full pytest suite with verbose output                                                          |
| `test-unit` | `make test-unit` | Run only `tests/unit/`                                                                             |
| `test-int`  | `make test-int`  | Run only `tests/integration/`                                                                      |
| `check`     | `make check`     | Run lint + test sequentially                                                                       |
| `docs`      | `make docs`      | Build Sphinx documentation (HTML)                                                                  |
| `clean`     | `make clean`     | Remove `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `*.egg-info`, build artifacts |
| `all`       | `make all`       | install, format, lint, test (in order)                                                             |

---

### D.4 Docker Configuration

#### D.4.1 Dockerfile.unified

Multi-stage build for the complete application container.

**Location:** `/Dockerfile.unified`

**Stage 1: UI Builder**

| Property   | Value                                     |
| ---------- | ----------------------------------------- |
| Base image | `node:20-slim`                            |
| Purpose    | Build the web UI (npm ci + npm run build) |
| Output     | `/app/ui/dist` (static assets)            |

**Stage 2: Python Runtime**

| Property     | Value                                                                 |
| ------------ | --------------------------------------------------------------------- |
| Base image   | `mambaorg/micromamba:1.5-jammy`                                       |
| System deps  | libgl1-mesa-glx, libegl1, libglib2.0-0, libsm6, libxrender1, libxext6 |
| Python env   | Created from `environment.yml` via micromamba                         |
| Application  | Installed in editable mode                                            |
| User         | Non-root `golfer` (UID 1000)                                          |
| Exposed port | 8000                                                                  |
| Health check | `curl -f http://localhost:8000/api/health` (30s interval)             |
| Default CMD  | `golf-suite --no-browser`                                             |

#### D.4.2 Running the Container

```bash
# Build
docker build -f Dockerfile.unified -t upstream-drift:latest .

# Run (API server mode)
docker run -p 8000:8000 upstream-drift:latest

# Run with GPU access (NVIDIA)
docker run --gpus all -p 8000:8000 upstream-drift:latest

# Run interactive CLI
docker run -it upstream-drift:latest bash
```

#### D.4.3 Environment Variables

| Variable                    | Default      | Description               |
| --------------------------- | ------------ | ------------------------- |
| `ENV_NAME`                  | `golf-suite` | Conda environment name    |
| `MAMBA_DOCKERFILE_ACTIVATE` | `1`          | Auto-activate environment |
| `GMS_LOG_LEVEL`             | `INFO`       | Logging verbosity         |
| `GMS_API_PORT`              | `8000`       | API server port           |
| `GMS_API_HOST`              | `0.0.0.0`    | API server bind address   |

---

### D.5 Logging Configuration

UpstreamDrift uses `structlog` for structured JSON logging throughout.

| Level      | Usage                                                                  |
| ---------- | ---------------------------------------------------------------------- |
| `DEBUG`    | Detailed computational values, per-step simulation data                |
| `INFO`     | Normal operations: engine loading, simulation start/stop, API requests |
| `WARNING`  | Recoverable issues: gimbal lock proximity, suboptimal convergence      |
| `ERROR`    | Operation failures: simulation divergence, file not found              |
| `CRITICAL` | Unrecoverable: engine crash, data corruption                           |

**Log format (production):**

```json
{
  "timestamp": "2026-02-05T10:30:00.000Z",
  "level": "info",
  "event": "simulation_complete",
  "engine": "mujoco",
  "model": "full_body_15dof",
  "duration_ms": 1523,
  "steps": 10000,
  "request_id": "req_abc123"
}
```

---

### D.6 Numerical Constants

Key numerical constants defined in `src/shared/python/physics_constants.py`
and `src/shared/python/constants.py`:

| Constant                      | Value      | Unit     | Source                   |
| ----------------------------- | ---------- | -------- | ------------------------ |
| `GRAVITY_M_S2`                | 9.80665    | m/s$^2$  | NIST CODATA 2018         |
| `AIR_DENSITY_SEA_LEVEL_KG_M3` | 1.225      | kg/m$^3$ | ISA Standard Atmosphere  |
| `GOLF_BALL_MASS_KG`           | 0.04593    | kg       | USGA Rule 5-1            |
| `GOLF_BALL_DIAMETER_M`        | 0.04267    | m        | USGA Rule 5-2            |
| `DRIVER_COR`                  | 0.78       | --       | USGA Rule Limit          |
| `TYPICAL_CONTACT_DURATION_S`  | 0.0005     | s        | High-speed video studies |
| `GRAPHITE_DENSITY_KG_M3`      | 1750       | kg/m$^3$ | Materials Handbook       |
| `STEEL_DENSITY_KG_M3`         | 7850       | kg/m$^3$ | Materials Handbook       |
| `TITANIUM_DENSITY_KG_M3`      | 4506       | kg/m$^3$ | Materials Handbook       |
| `MAGNUS_COEFFICIENT`          | 0.25       | --       | Bearman & Harvey         |
| `SPIN_DECAY_RATE_S`           | 0.05       | 1/s      | Trackman Data            |
| `EPSILON`                     | $10^{-15}$ | --       | Solver numerical floor   |
| `CONVERGENCE_TOLERANCE`       | $10^{-6}$  | --       | Default solver tolerance |
| `MAX_ITERATIONS`              | 10000      | --       | Default iteration cap    |
| `DEFAULT_RANDOM_SEED`         | 42         | --       | Reproducibility          |

---

## Appendix E: Error Codes Reference

UpstreamDrift uses a structured error code system defined in
`src/api/utils/error_codes.py`. All errors follow a consistent format and
include machine-readable codes for automated error handling.

---

### E.1 Error Code Format

```
GMS-{CATEGORY}-{NUMBER}
```

- **GMS**: Golf Modeling Suite (project prefix)
- **CATEGORY**: Three-letter category code identifying the subsystem
- **NUMBER**: Three-digit sequential error number within the category

#### E.1.1 Error Categories

| Code  | Category   | Description                                                |
| ----- | ---------- | ---------------------------------------------------------- |
| `GEN` | General    | Generic/uncategorized errors                               |
| `ENG` | Engine     | Physics engine loading, initialization, and runtime errors |
| `SIM` | Simulation | Simulation execution, parameter, and state errors          |
| `VID` | Video      | Video processing and pose estimation errors                |
| `ANL` | Analysis   | Swing analysis and computation errors                      |
| `AUT` | Auth       | Authentication, authorization, and quota errors            |
| `VAL` | Validation | Input validation and schema errors                         |
| `RES` | Resource   | Resource lifecycle errors (CRUD operations)                |
| `SYS` | System     | Infrastructure, database, and dependency errors            |

---

### E.2 Complete Error Code Table

#### E.2.1 General Errors (GMS-GEN-xxx)

| Code          | Name                | HTTP Status | Description                                  | Resolution                                                                         |
| ------------- | ------------------- | ----------- | -------------------------------------------- | ---------------------------------------------------------------------------------- |
| `GMS-GEN-001` | INTERNAL_ERROR      | 500         | An unexpected internal server error occurred | Check server logs for stack trace. Report if persistent.                           |
| `GMS-GEN-002` | INVALID_REQUEST     | 400         | The request format or parameters are invalid | Review the API documentation for the correct request format.                       |
| `GMS-GEN-003` | RATE_LIMITED        | 429         | Rate limit exceeded                          | Wait and retry after the indicated cooldown period. Reduce request frequency.      |
| `GMS-GEN-004` | SERVICE_UNAVAILABLE | 503         | The service is temporarily unavailable       | The server may be restarting or under maintenance. Retry with exponential backoff. |

#### E.2.2 Engine Errors (GMS-ENG-xxx)

| Code          | Name                         | HTTP Status | Description                                          | Resolution                                                                           |
| ------------- | ---------------------------- | ----------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `GMS-ENG-001` | ENGINE_NOT_FOUND             | 404         | The specified physics engine was not found           | Verify the engine name. Available engines: mujoco, drake, pinocchio.                 |
| `GMS-ENG-002` | ENGINE_NOT_LOADED            | 400         | No physics engine is currently loaded                | Call the engine loading endpoint before attempting simulation.                       |
| `GMS-ENG-003` | ENGINE_LOAD_FAILED           | 500         | The physics engine failed to load                    | Check that engine dependencies are installed. Verify model file paths.               |
| `GMS-ENG-004` | ENGINE_NOT_AVAILABLE         | 400         | The requested engine is not installed on this system | Install the engine: `pip install -e ".[drake]"` or `pip install -e ".[pinocchio]"`.  |
| `GMS-ENG-005` | ENGINE_INITIALIZATION_FAILED | 500         | Engine initialization failed after loading           | Check model file validity (URDF/MJCF syntax). Verify system resources (GPU, memory). |
| `GMS-ENG-006` | ENGINE_INVALID_STATE         | 400         | The engine is in an invalid state for this operation | Reset the engine state or reload the model before retrying.                          |

#### E.2.3 Simulation Errors (GMS-SIM-xxx)

| Code          | Name                       | HTTP Status | Description                                               | Resolution                                                                                           |
| ------------- | -------------------------- | ----------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `GMS-SIM-001` | SIMULATION_FAILED          | 500         | The simulation execution failed                           | Check simulation parameters for physical validity (time step, joint limits). Review model stability. |
| `GMS-SIM-002` | SIMULATION_TIMEOUT         | 504         | The simulation exceeded the maximum allowed time          | Reduce simulation duration, increase time step, or simplify the model.                               |
| `GMS-SIM-003` | SIMULATION_INVALID_PARAMS  | 400         | One or more simulation parameters are invalid             | Verify parameter ranges: time_step > 0, duration > 0, joint angles within limits.                    |
| `GMS-SIM-004` | SIMULATION_MODEL_NOT_FOUND | 404         | The specified simulation model file was not found         | Verify the model path. Run `golf-suite --setup-models` to download standard models.                  |
| `GMS-SIM-005` | SIMULATION_STATE_ERROR     | 500         | The simulation reached an invalid state (NaN, divergence) | Reduce time step, check for unrealistic forces, verify contact parameters.                           |
| `GMS-SIM-006` | TASK_NOT_FOUND             | 404         | The specified async task ID was not found                 | The task may have expired or the ID is incorrect. Submit a new simulation request.                   |
| `GMS-SIM-007` | TASK_NOT_COMPLETED         | 400         | The task has not completed yet; results not available     | Poll the task status endpoint until completion, or use the WebSocket notification channel.           |

#### E.2.4 Video Processing Errors (GMS-VID-xxx)

| Code          | Name                           | HTTP Status | Description                                       | Resolution                                                                    |
| ------------- | ------------------------------ | ----------- | ------------------------------------------------- | ----------------------------------------------------------------------------- |
| `GMS-VID-001` | VIDEO_PIPELINE_NOT_INITIALIZED | 500         | The video processing pipeline is not initialized  | Ensure analysis dependencies are installed: `pip install -e ".[analysis]"`.   |
| `GMS-VID-002` | VIDEO_INVALID_FORMAT           | 400         | The uploaded file is not a supported video format | Supported formats: MP4, AVI, MOV, MKV. Maximum file size: 500 MB.             |
| `GMS-VID-003` | VIDEO_PROCESSING_FAILED        | 500         | Video processing failed during analysis           | Check video quality (resolution >= 720p recommended). Ensure adequate memory. |
| `GMS-VID-004` | VIDEO_ESTIMATOR_INVALID        | 400         | Invalid pose estimator type specified             | Available estimators: mediapipe, openpose, movenet.                           |
| `GMS-VID-005` | VIDEO_CONFIDENCE_INVALID       | 400         | Confidence threshold out of valid range           | Confidence must be in range [0.0, 1.0]. Default: 0.5.                         |

#### E.2.5 Analysis Errors (GMS-ANL-xxx)

| Code          | Name                             | HTTP Status | Description                                   | Resolution                                                                                          |
| ------------- | -------------------------------- | ----------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `GMS-ANL-001` | ANALYSIS_SERVICE_NOT_INITIALIZED | 500         | The analysis service has not been initialized | The server may need to be restarted. Check service health endpoint.                                 |
| `GMS-ANL-002` | ANALYSIS_FAILED                  | 500         | The analysis computation failed               | Check input data quality. Ensure sufficient trajectory data points (minimum 10).                    |
| `GMS-ANL-003` | ANALYSIS_INVALID_TYPE            | 400         | The requested analysis type is not recognized | Available types: swing_metrics, energy_analysis, kinematic_sequence, phase_detection, grf_analysis. |

#### E.2.6 Authentication Errors (GMS-AUT-xxx)

| Code          | Name                          | HTTP Status | Description                                                 | Resolution                                                     |
| ------------- | ----------------------------- | ----------- | ----------------------------------------------------------- | -------------------------------------------------------------- |
| `GMS-AUT-001` | AUTH_TOKEN_INVALID            | 401         | The authentication token is invalid or malformed            | Ensure the Authorization header contains a valid Bearer token. |
| `GMS-AUT-002` | AUTH_TOKEN_EXPIRED            | 401         | The authentication token has expired                        | Request a new token via the `/api/auth/token` endpoint.        |
| `GMS-AUT-003` | AUTH_INSUFFICIENT_PERMISSIONS | 403         | The authenticated user lacks permissions for this operation | Contact the administrator to request elevated permissions.     |
| `GMS-AUT-004` | AUTH_QUOTA_EXCEEDED           | 429         | The usage quota for this billing period has been exceeded   | Upgrade your plan or wait until the next billing period.       |
| `GMS-AUT-005` | AUTH_USER_NOT_FOUND           | 404         | The specified user account was not found                    | Verify the username/email. Register a new account if needed.   |

#### E.2.7 Validation Errors (GMS-VAL-xxx)

| Code          | Name                     | HTTP Status | Description                                                  | Resolution                                                         |
| ------------- | ------------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| `GMS-VAL-001` | VALIDATION_FAILED        | 422         | General request validation failed                            | Review the error details for specific field violations.            |
| `GMS-VAL-002` | VALIDATION_MISSING_FIELD | 422         | A required field is missing from the request                 | Add the missing field as indicated in the error details.           |
| `GMS-VAL-003` | VALIDATION_INVALID_VALUE | 422         | A field value is outside its valid range or type             | Check the field type and range constraints in the API schema.      |
| `GMS-VAL-004` | VALIDATION_PATH_INVALID  | 400         | A file path is invalid, does not exist, or is not accessible | Verify the path exists and the server process has read permission. |

#### E.2.8 Resource Errors (GMS-RES-xxx)

| Code          | Name                    | HTTP Status | Description                                        | Resolution                                                                |
| ------------- | ----------------------- | ----------- | -------------------------------------------------- | ------------------------------------------------------------------------- |
| `GMS-RES-001` | RESOURCE_NOT_FOUND      | 404         | The requested resource was not found               | Verify the resource ID or path. The resource may have been deleted.       |
| `GMS-RES-002` | RESOURCE_ALREADY_EXISTS | 409         | A resource with the same identifier already exists | Use a different identifier, or update/delete the existing resource first. |
| `GMS-RES-003` | RESOURCE_ACCESS_DENIED  | 403         | Access to the resource is denied                   | Check resource ownership and sharing permissions.                         |

#### E.2.9 System Errors (GMS-SYS-xxx)

| Code          | Name                | HTTP Status | Description                                     | Resolution                                                               |
| ------------- | ------------------- | ----------- | ----------------------------------------------- | ------------------------------------------------------------------------ |
| `GMS-SYS-001` | DATABASE_ERROR      | 500         | A database operation failed                     | Check database connectivity and disk space. Retry the operation.         |
| `GMS-SYS-002` | CONFIGURATION_ERROR | 500         | The server configuration is invalid             | Review environment variables and configuration files.                    |
| `GMS-SYS-003` | DEPENDENCY_ERROR    | 500         | A required external dependency is not available | Install the missing dependency. Check `pip list` for installed packages. |

---

### E.3 Error Response Format

All API error responses follow a consistent JSON structure:

```json
{
  "error": {
    "code": "GMS-ENG-003",
    "message": "Failed to load physics engine",
    "details": {
      "engine_type": "drake",
      "reason": "ImportError: No module named 'pydrake'"
    },
    "request_id": "req_a1b2c3d4",
    "correlation_id": "corr_x7y8z9"
  }
}
```

| Field            | Type   | Description                                      |
| ---------------- | ------ | ------------------------------------------------ |
| `code`           | string | Machine-readable error code (GMS-XXX-YYY format) |
| `message`        | string | Human-readable error description                 |
| `details`        | object | Additional context (optional, varies by error)   |
| `request_id`     | string | Unique identifier for this request (for support) |
| `correlation_id` | string | Trace ID for cross-service correlation           |

---

### E.4 Exception Hierarchy

The internal Python exception hierarchy in `src/shared/python/error_utils.py`:

```
GolfSuiteError (base)
  |-- EngineNotAvailableError
  |-- ConfigurationError
  |     |-- EnvironmentError
  |-- ValidationError
  |     |-- PhysicalValidationError
  |-- ModelError
  |-- SimulationError
  |-- FileOperationError
  |-- IOError
  |     |-- FileNotFoundIOError
  |     |-- FileParseError
  |-- DataFormatError
  |-- TimeoutError
  |-- ResourceError
```

Additional domain-specific exceptions:

```
ContractViolationError (from contracts.py)
  |-- PreconditionError
  |-- PostconditionError
  |-- InvariantError
  |-- StateError

RoboticsError (from robotics/core/exceptions.py)
  |-- ContactError
  |-- ControlError
  |-- SolverError
  |-- LocomotionError
  |-- KinematicsError

AIError (from ai/exceptions.py)
  |-- AIProviderError
  |     |-- AIConnectionError
  |     |-- AIRateLimitError
  |     |-- AITimeoutError
  |-- ScientificValidationError
  |-- WorkflowError
  |-- ToolExecutionError
```

---

### E.5 Contract Violation Errors

Design by Contract violations (from `src/shared/python/contracts.py`) are
a special class of runtime errors that indicate programming logic errors
rather than user input errors.

| Exception            | Trigger                                                    | Example                                      |
| -------------------- | ---------------------------------------------------------- | -------------------------------------------- |
| `PreconditionError`  | Input constraint violated before method execution          | Calling `step()` on an uninitialized engine  |
| `PostconditionError` | Output constraint violated after method execution          | Computed acceleration contains NaN values    |
| `InvariantError`     | Class invariant violated after a state-modifying operation | Mass becomes negative after parameter update |
| `StateError`         | Operation attempted in an invalid state                    | Attempting simulation before model is loaded |

Contracts can be globally disabled for production performance:

```python
from src.shared.python.contracts import disable_contracts
disable_contracts()  # Skip all contract checks
```

---

## Appendix F: Glossary

Definitions of technical terms, acronyms, and domain-specific vocabulary
used throughout the UpstreamDrift documentation and codebase. Terms are
organized alphabetically.

---

### Acronyms

| Acronym    | Full Form                                       | Definition                                                                                                                                                                                                                               |
| ---------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ABA**    | Articulated Body Algorithm                      | $O(n)$ algorithm for computing forward dynamics of articulated rigid-body systems. Computes joint accelerations given joint positions, velocities, and applied forces. Used internally by MuJoCo and Pinocchio.                          |
| **ASGI**   | Asynchronous Server Gateway Interface           | Python web server interface standard. UpstreamDrift uses ASGI via uvicorn for the FastAPI server.                                                                                                                                        |
| **CMA-ES** | Covariance Matrix Adaptation Evolution Strategy | Derivative-free optimization algorithm that adapts a multivariate Gaussian search distribution. Used for swing parameter optimization when gradients are unavailable.                                                                    |
| **COM**    | Center of Mass                                  | The mass-weighted average position of all particles in a body. Critical for balance analysis and ground reaction force computation.                                                                                                      |
| **COP**    | Center of Pressure                              | The point on the ground surface where the resultant ground reaction force acts. Moves between the feet during the golf swing.                                                                                                            |
| **COR**    | Coefficient of Restitution                      | Ratio of relative separation speed to relative approach speed in a collision. The USGA limits driver COR to 0.830.                                                                                                                       |
| **CRBA**   | Composite Rigid Body Algorithm                  | $O(n^2)$ algorithm for computing the joint-space mass matrix $M(q)$ of an articulated system.                                                                                                                                            |
| **DAgger** | Dataset Aggregation                             | An imitation learning algorithm that iteratively collects data under the learned policy and retrains. Addresses distribution shift in behavior cloning.                                                                                  |
| **DCM**    | Divergent Component of Motion                   | A point derived from the COM and its velocity that characterizes dynamic balance. If the DCM stays within the support polygon, the system can maintain balance. Defined as $\xi = c + \dot{c}/\omega_0$ where $\omega_0 = \sqrt{g/z_c}$. |
| **DOF**    | Degrees of Freedom                              | The number of independent parameters that define the configuration of a mechanical system. A free rigid body in 3D has 6 DOF.                                                                                                            |
| **EMG**    | Electromyography                                | Measurement of electrical activity in muscles. Used for validation of muscle activation predictions and EMG-driven simulation.                                                                                                           |
| **FK**     | Forward Kinematics                              | Computing the position and orientation of end-effectors given joint angles. The mapping $q \mapsto \mathbf{x}$.                                                                                                                          |
| **FPS**    | Frames Per Second                               | Rate of simulation visualization or video capture. UpstreamDrift renders at 60 FPS by default.                                                                                                                                           |
| **GAIL**   | Generative Adversarial Imitation Learning       | An imitation learning algorithm that uses a discriminator to distinguish expert demonstrations from learned behavior. Combines adversarial training with reinforcement learning.                                                         |
| **GRF**    | Ground Reaction Force                           | The force exerted by the ground on a body in contact with it. Equal and opposite to the force the body exerts on the ground. Measured by force plates in biomechanics.                                                                   |
| **ICP**    | Instantaneous Capture Point                     | Synonymous with DCM. The point on the ground where a robot (or golfer) must step to come to a complete stop.                                                                                                                             |
| **IK**     | Inverse Kinematics                              | Computing joint angles that achieve a desired end-effector position/orientation. The mapping $\mathbf{x} \mapsto q$. May have zero, one, or many solutions.                                                                              |
| **IMU**    | Inertial Measurement Unit                       | A sensor combining accelerometers and gyroscopes to measure linear acceleration and angular velocity. Used for motion capture and swing analysis.                                                                                        |
| **IPOPT**  | Interior Point Optimizer                        | A large-scale nonlinear programming solver using interior-point/barrier methods. Available through Drake for trajectory optimization.                                                                                                    |
| **MJCF**   | MuJoCo Format                                   | XML-based model description format native to MuJoCo. Supports contacts, actuators, sensors, and tendon routing.                                                                                                                          |
| **MOI**    | Moment of Inertia                               | Resistance to angular acceleration about an axis. The rotational analog of mass.                                                                                                                                                         |
| **MPC**    | Model Predictive Control                        | A control strategy that solves a finite-horizon optimal control problem at each time step, applying only the first control action. Provides feedback by re-solving with updated state information.                                       |
| **NLP**    | Nonlinear Program                               | An optimization problem with a nonlinear objective function and/or nonlinear constraints. Trajectory optimization problems are typically NLPs.                                                                                           |
| **PPO**    | Proximal Policy Optimization                    | A reinforcement learning algorithm that constrains policy updates to a trust region. Used for training swing policies in the MyoSuite integration.                                                                                       |
| **QP**     | Quadratic Program                               | An optimization problem with a quadratic objective and linear constraints. Solved as a subproblem in SQP methods and for contact force resolution.                                                                                       |
| **RL**     | Reinforcement Learning                          | A machine learning paradigm where an agent learns to make decisions by interacting with an environment and receiving rewards.                                                                                                            |
| **RNEA**   | Recursive Newton-Euler Algorithm                | $O(n)$ algorithm for computing inverse dynamics. Given joint positions, velocities, and accelerations, computes the required joint torques.                                                                                              |
| **RK4**    | Runge-Kutta 4th Order                           | A fourth-order numerical integration method. Used by UpstreamDrift for ball flight trajectory integration.                                                                                                                               |
| **SDF**    | Simulation Description Format                   | An XML-based format for describing robots and environments, originating from the Gazebo simulator. Supports world-level descriptions including lighting, terrain, and multiple models.                                                   |
| **SE(3)**  | Special Euclidean Group (3D)                    | The group of rigid body transformations in 3D, combining rotation $SO(3)$ and translation $\mathbb{R}^3$. An element of $SE(3)$ represents a pose.                                                                                       |
| **SLSQP**  | Sequential Least Squares Programming            | A gradient-based constrained optimization algorithm. Used for small-to-medium scale optimization in UpstreamDrift via SciPy.                                                                                                             |
| **SLERP**  | Spherical Linear Interpolation                  | A method for smoothly interpolating between two rotations on the unit quaternion sphere. Constant angular velocity interpolation.                                                                                                        |
| **SNOPT**  | Sparse Nonlinear Optimizer                      | A large-scale SQP-based nonlinear programming solver. Available through Drake for trajectory optimization with sparse constraint Jacobians.                                                                                              |
| **SO(3)**  | Special Orthogonal Group (3D)                   | The group of 3D rotations, represented by $3 \times 3$ orthogonal matrices with determinant +1.                                                                                                                                          |
| **SPD**    | Symmetric Positive-Definite                     | A matrix property. The mass matrix $M(q)$ is always SPD for physical systems with positive masses. Ensures unique solutions to linear systems and positive kinetic energy.                                                               |
| **SQP**    | Sequential Quadratic Programming                | An optimization method that solves a sequence of QP subproblems to find the solution of a general NLP. SNOPT is an SQP solver.                                                                                                           |
| **URDF**   | Unified Robot Description Format                | An XML-based format for describing robot kinematics and dynamics. The primary model exchange format in UpstreamDrift. Supported by MuJoCo, Drake, and Pinocchio.                                                                         |
| **ZMP**    | Zero Moment Point                               | The point on the ground surface where the net moment of ground reaction forces and gravity has zero horizontal component. Used for balance analysis. If ZMP stays within the support polygon, the system is statically balanced.         |

---

### Technical Terms

**Activation dynamics**
: The time-dependent process by which neural excitation signals are transformed into muscle activation levels. Modeled as a first-order ODE with asymmetric time constants for activation (~10 ms) and deactivation (~40 ms). See `src/shared/python/activation_dynamics.py` and Appendix A.6.8.

**Admittance control**
: A control paradigm where the controller accepts force inputs and produces motion outputs. The robot behaves as if it has a specified mass-spring-damper system: $F_{ext} = M_d \ddot{x} + D_d \dot{x} + K_d x$. Complementary to impedance control.

**Articulated body inertia**
: The effective inertia of a body in an articulated chain, accounting for the dynamics of all descendant bodies. Used in the ABA for $O(n)$ forward dynamics.

**Attack angle (angle of attack)**
: The vertical angle of the clubhead's path at impact relative to the horizontal. Positive means upward (hitting up on the ball), negative means downward. Typically negative for irons, slightly positive for driver.

**Behavior cloning**
: An imitation learning technique where a neural network policy is trained via supervised learning on expert demonstration data. Simple but suffers from distribution shift (compounding errors).

**Carry distance**
: The distance the golf ball travels through the air from the point of impact to the first landing point, measured along the ground. Excludes roll. Computed in `BallFlightSimulator.calculate_carry_distance()`.

**Club path**
: The horizontal direction of the clubhead's center of gravity movement at impact, relative to the target line. Positive is to the right (for a right-handed golfer), negative is to the left.

**Co-contraction**
: Simultaneous activation of agonist and antagonist muscles crossing a joint. Increases joint stiffness at the cost of metabolic energy. Important for stability during the golf swing transition.

**Collocation (direct)**
: A trajectory optimization method that simultaneously optimizes states and controls at discrete time points (knot points), enforcing dynamics as equality constraints. See Appendix A.5.4.

**Composite rigid body inertia**
: The total inertia of a body and all its descendant bodies when treated as a single rigid body. Used in the CRBA to compute the mass matrix.

**Contact Jacobian**
: The matrix $J_c(q)$ that maps generalized velocities to contact point velocities: $v_{contact} = J_c(q) \dot{q}$. Its transpose maps contact forces to generalized forces: $\tau_{contact} = J_c^T f_c$.

**Damping ratio**
: A dimensionless measure describing how oscillations decay. For a second-order system: $\zeta = c / (2\sqrt{km})$. Critically damped at $\zeta = 1$.

**Design by Contract (DbC)**
: A software design methodology where interfaces are defined by formal, verifiable specifications (contracts): preconditions, postconditions, and invariants. Implemented in `src/shared/python/contracts.py`.

**Direct collocation**
: See "Collocation (direct)."

**Domain randomization**
: A sim-to-real transfer technique where simulation parameters (friction, mass, sensor noise, etc.) are randomly varied during training so the policy generalizes to the real world.

**Dynamic loft**
: The effective loft angle of the club face at the moment of impact, accounting for shaft lean, face angle, and attack angle. Different from the static loft stamped on the club.

**Face angle**
: The horizontal orientation of the club face at impact, relative to the target line. Open (pointing right of target for right-handed), closed (pointing left), or square.

**Flexible shaft model**
: A model that accounts for golf club shaft bending during the swing. UpstreamDrift supports three approaches: rigid (no flex), modal (2-4 bending mode shapes), and finite element (beam elements with 2 DOF per node). See `src/shared/python/flexible_shaft.py`.

**Force closure**
: A contact configuration where the contact forces can resist any external wrench. Requires contacts that can generate forces in enough independent directions.

**Force-length relationship**
: The relationship between a muscle's fiber length and its force-generating capacity. Peak force occurs at the optimal fiber length. Modeled as a Gaussian in Hill-type models. See Appendix A.6.3.

**Force-velocity relationship**
: The relationship between a muscle's shortening/lengthening velocity and its force output. Force decreases with shortening velocity (concentric) and increases above isometric during lengthening (eccentric). See Appendix A.6.4.

**Friction cone**
: The set of contact forces that satisfy Coulomb friction: $\|f_t\| \leq \mu f_n$, where $f_t$ is tangential force, $f_n$ is normal force, and $\mu$ is the friction coefficient. Forms a cone in 3D force space.

**Gear effect**
: The phenomenon where off-center impacts on a club head with a bulging face cause the ball to spin in a direction that curves it back toward the center of the face. Toe hits produce draw spin; heel hits produce fade spin.

**Generalized coordinates**
: The minimal set of independent variables $q$ that completely describe the configuration of a mechanical system. For a planar double pendulum: $q = [\theta_1, \theta_2]^T$.

**Gimbal lock**
: A loss of one degree of freedom in Euler angle representation when two rotation axes align ($\theta = \pm \pi/2$ in ZYX convention). Avoided by using quaternions.

**Grasp matrix**
: A matrix that maps contact forces to the resultant wrench on a grasped object. Used in multi-contact analysis of the golf grip.

**Hill-type muscle model**
: A phenomenological muscle model consisting of a contractile element (active force), parallel elastic element (passive stiffness), and series elastic element (tendon). Named after A.V. Hill (1938). See Appendix A.6.

**Impedance control**
: A control paradigm where the controller accepts motion inputs and produces force outputs. The robot behaves as if it has a virtual mass-spring-damper: $\tau = M_d \ddot{e} + D_d \dot{e} + K_d e$ where $e$ is the position error. Complementary to admittance control.

**Invariant**
: In Design by Contract, a condition that must be true before and after every public method call on an object. Example: "mass must always be positive." See `InvariantError` in `contracts.py`.

**Jacobian matrix**
: The matrix of partial derivatives $J(q) = \partial \mathbf{x} / \partial q$ relating joint velocities to task-space velocities: $\dot{\mathbf{x}} = J(q) \dot{q}$. Fundamental for operational space control and inverse kinematics.

**Kinematic chain**
: A series of rigid bodies (links) connected by joints. Open chains have one fixed base and free end-effectors. Closed chains form loops (e.g., both hands on the golf club).

**Kinematic sequence**
: The temporal ordering of peak angular velocities during the golf swing downswing, typically: pelvis, torso, arm, hand/club. Optimal sequencing maximizes energy transfer to the club.

**Kinetic energy**
: The energy of motion. In generalized coordinates: $T = \frac{1}{2} \dot{q}^T M(q) \dot{q}$. Decomposed into rotational and translational components for analysis.

**Knot point**
: A discrete time point in a trajectory optimization problem where decision variables (states and controls) are defined. Dynamics are enforced between consecutive knot points.

**Lag (club lag / wrist lag)**
: The angle between the lead forearm and the club shaft during the downswing. A large lag angle that is maintained until late in the downswing allows greater clubhead speed at impact.

**Moment arm (muscle)**
: The perpendicular distance from a joint's axis of rotation to the muscle's line of action. Determines the torque a muscle produces about a joint per unit force: $\tau = r \cdot F$. Computed as $r_j = -\partial l_{MT} / \partial q_j$.

**Nullspace**
: The set of joint velocities $\dot{q}$ that produce zero task-space velocity: $J(q)\dot{q} = 0$. Used for secondary task execution (posture optimization, singularity avoidance) in redundant systems.

**Operational space**
: The task space in which end-effector motion is controlled. Operational space control computes joint torques that produce desired end-effector forces/accelerations while accounting for the full dynamics.

**Pennation angle**
: The angle between a muscle's fibers and its tendon direction. Muscles with pennation can pack more fibers but transmit force at $\cos\alpha$ efficiency.

**Postcondition**
: In Design by Contract, a condition that must be true after a method completes. Example: "returned array must contain only finite values." See `PostconditionError` in `contracts.py`.

**Precondition**
: In Design by Contract, a condition that must be true before a method executes. Example: "engine must be initialized before calling step()." See `PreconditionError` in `contracts.py`.

**Proximal**
: Closer to the body's center or the base of a kinematic chain. The shoulder is proximal to the elbow. Opposite of distal.

**Sim-to-real transfer**
: The process of deploying policies trained in simulation to physical robots or real-world systems. Challenges include the "reality gap" due to modeling inaccuracies.

**Singularity (kinematic)**
: A configuration where the Jacobian $J(q)$ loses rank, meaning certain task-space velocities become unachievable. Near singularities, required joint velocities become very large.

**Smash factor**
: The ratio of ball speed to clubhead speed at impact. For a driver: typically 1.48-1.51. Determined by COR and the mass ratio $m_{club} / (m_{club} + m_{ball})$.

**Spatial velocity (twist)**
: A 6D vector combining angular velocity $\omega$ and linear velocity $v$ of a rigid body. See Appendix A.3.1.

**Spin axis tilt**
: The orientation of the ball's spin axis relative to vertical. Zero tilt means pure backspin. Tilt causes the ball to curve laterally (draw/fade).

**Spin loft**
: The 3D angle between the club face normal and the club path direction at impact. The primary determinant of spin rate.

**Stimpmeter**
: A device used to measure green speed in golf. A ball is rolled down a standardized ramp, and the distance it travels (in feet) is the Stimp reading. Fast greens: 10-13 ft.

**Stiffness (joint)**
: In impedance control, the virtual spring constant $K_d$ that defines how much force is produced per unit displacement. Higher stiffness means more resistance to perturbation.

**System identification**
: The process of estimating dynamic model parameters (masses, inertias, friction coefficients) from experimental data. Used to calibrate simulation models against real-world measurements.

**Tendon**
: The elastic tissue connecting muscle to bone. In Hill-type models, tendon compliance affects force transmission and energy storage. Characterized by slack length $l_T^{slack}$ and stiffness.

**Trajectory optimization**
: Finding an optimal trajectory (time-history of states and controls) that minimizes a cost function subject to dynamic constraints. Can be solved by direct methods (collocation, shooting) or indirect methods (Pontryagin's principle).

**Twist**
: See "Spatial velocity."

**Wrench**
: See "Spatial force" (Appendix A.3.2). A 6D vector combining moment $n$ and force $f$.

**X-factor**
: The angular difference between shoulder rotation and hip rotation at the top of the backswing. A larger X-factor indicates greater trunk coil and potential energy storage. Related to the "X-factor stretch" during the early downswing.

---

### Model Format Terms

**URDF (Unified Robot Description Format)**
: An XML format for describing robot kinematic trees with links and joints. Supports visual geometry, collision geometry, and inertial properties. Primary format in ROS ecosystem and the main exchange format for UpstreamDrift models. Limitations: no closed-loop kinematic chains, limited actuator/sensor description.

**MJCF (MuJoCo Format)**
: MuJoCo's native XML format. Richer than URDF: supports tendons, actuator models, contact parameters, equality constraints (closed loops), and sensors. UpstreamDrift auto-converts URDF to MJCF when using the MuJoCo engine.

**SDF (Simulation Description Format)**
: XML format from the Gazebo simulator. Can describe complete worlds with multiple models, lights, and terrain. Supports features beyond URDF including closed kinematic chains and model nesting.

**C3D**
: A binary file format for storing 3D motion capture data (marker positions and analog signals like force plates and EMG). UpstreamDrift reads C3D files via the `ezc3d` library for validation against real swing data.

**STL (Stereolithography)**
: A file format for 3D mesh geometry using triangulated surfaces. Used for collision and visual geometry in URDF models. Both ASCII and binary variants are supported.

---

### Software Engineering Terms

**Design by Contract**
: A methodology introduced by Bertrand Meyer where component interfaces are specified using preconditions, postconditions, and invariants. Violations indicate bugs in the caller (precondition) or implementer (postcondition/invariant). See `src/shared/python/contracts.py`.

**DRY (Don't Repeat Yourself)**
: The principle that every piece of knowledge should have a single, authoritative representation. Violations lead to inconsistency bugs. UpstreamDrift enforces DRY through shared constants, base classes, and the `physics_constants.py` module.

**Editable install**
: A Python package installation mode (`pip install -e .`) where the package is installed as a link to the source directory. Changes to source code take effect immediately without reinstalling.

**Orthogonality**
: The principle that components should be independent -- changing one should not affect others. In UpstreamDrift, physics engines, analysis modules, and the API layer are orthogonal: each can be modified independently.

**PhysicalConstant**
: A custom `float` subclass in UpstreamDrift that carries unit, source citation, and description metadata alongside the numeric value. Defined in `src/shared/python/physics_constants.py`.

**Structured logging**
: A logging approach where log entries are machine-parsable (typically JSON) with typed fields rather than free-form text. UpstreamDrift uses `structlog` for all logging.

---

_End of Appendices A through F._

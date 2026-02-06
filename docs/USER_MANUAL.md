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

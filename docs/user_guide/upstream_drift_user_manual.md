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

- **[Chapter 1: Introduction](upstream_drift_user_manual/ch01_introduction.md#chapter-1-introduction)**

  - [1.1 Project Overview](upstream_drift_user_manual/ch01_introduction.md#11-project-overview)
  - [1.2 Mission and Design Philosophy](upstream_drift_user_manual/ch01_introduction.md#12-mission-and-design-philosophy)
  - [1.3 Target Audience](upstream_drift_user_manual/ch01_introduction.md#13-target-audience)
  - [1.4 Version History](upstream_drift_user_manual/ch01_introduction.md#14-version-history)
  - [1.5 Document Conventions](upstream_drift_user_manual/ch01_introduction.md#15-document-conventions)
  - [1.6 How to Read This Manual](upstream_drift_user_manual/ch01_introduction.md#16-how-to-read-this-manual)

- **[Chapter 2: Architecture Overview](upstream_drift_user_manual/ch02_architecture_overview.md#chapter-2-architecture-overview)**

  - [2.1 High-Level System Architecture](upstream_drift_user_manual/ch02_architecture_overview.md#21-high-level-system-architecture)
  - [2.2 Multi-Engine Design](upstream_drift_user_manual/ch02_architecture_overview.md#22-multi-engine-design)
  - [2.3 Module Structure and Package Organization](upstream_drift_user_manual/ch02_architecture_overview.md#23-module-structure-and-package-organization)
  - [2.4 Design by Contract (DbC)](upstream_drift_user_manual/ch02_architecture_overview.md#24-design-by-contract-dbc)
  - [2.5 Dependency Graph](upstream_drift_user_manual/ch02_architecture_overview.md#25-dependency-graph)
  - [2.6 Data Flow and Communication Patterns](upstream_drift_user_manual/ch02_architecture_overview.md#26-data-flow-and-communication-patterns)
  - [2.7 Security Architecture](upstream_drift_user_manual/ch02_architecture_overview.md#27-security-architecture)

- **[Chapter 3: Installation and Setup](upstream_drift_user_manual/ch03_installation_and_setup.md#chapter-3-installation-and-setup)**

  - [3.1 System Prerequisites](upstream_drift_user_manual/ch03_installation_and_setup.md#31-system-prerequisites)
  - [3.2 Installation via Conda (Recommended)](upstream_drift_user_manual/ch03_installation_and_setup.md#32-installation-via-conda-recommended)
  - [3.3 Installation via Pip](upstream_drift_user_manual/ch03_installation_and_setup.md#33-installation-via-pip)
  - [3.4 Docker Installation](upstream_drift_user_manual/ch03_installation_and_setup.md#34-docker-installation)
  - [3.5 Environment Configuration](upstream_drift_user_manual/ch03_installation_and_setup.md#35-environment-configuration)
  - [3.6 Verifying the Installation](upstream_drift_user_manual/ch03_installation_and_setup.md#36-verifying-the-installation)
  - [3.7 Platform-Specific Notes](upstream_drift_user_manual/ch03_installation_and_setup.md#37-platform-specific-notes)
  - [3.8 Development Environment Setup](upstream_drift_user_manual/ch03_installation_and_setup.md#38-development-environment-setup)
  - [3.9 Troubleshooting Installation Issues](upstream_drift_user_manual/ch03_installation_and_setup.md#39-troubleshooting-installation-issues)

- **[Chapter 4: Quick Start Guide](upstream_drift_user_manual/ch04_quick_start_guide.md#chapter-4-quick-start-guide)**
  - [4.1 Running Your First Simulation](upstream_drift_user_manual/ch04_quick_start_guide.md#41-running-your-first-simulation)
  - [4.2 Launching the GUI (PyQt6 Classic Launcher)](upstream_drift_user_manual/ch04_quick_start_guide.md#42-launching-the-gui-pyqt6-classic-launcher)
  - [4.3 Launching the Web UI (Tauri/React)](upstream_drift_user_manual/ch04_quick_start_guide.md#43-launching-the-web-ui-taurireact)
  - [4.4 Starting the API Server](upstream_drift_user_manual/ch04_quick_start_guide.md#44-starting-the-api-server)
  - [4.5 Basic Workflows](upstream_drift_user_manual/ch04_quick_start_guide.md#45-basic-workflows)
  - [4.6 Exploring the Example Scripts](upstream_drift_user_manual/ch04_quick_start_guide.md#46-exploring-the-example-scripts)
  - [4.7 Understanding Output and Results](upstream_drift_user_manual/ch04_quick_start_guide.md#47-understanding-output-and-results)

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
  - [22.2 Tauri/React Web Desktop Application](#222-taurireact-web-desktop-application)
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
  - [24.2 Code Style and Formatting (Ruff)](#242-code-style-and-formatting-ruff)
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

## Manual Layout

This index keeps the front matter and the table of contents. Each chapter
lives in its own file under `docs/user_guide/upstream_drift_user_manual/` so that
every part stays under the documentation size budget:

- [Introduction](upstream_drift_user_manual/ch01_introduction.md)
- [Architecture Overview](upstream_drift_user_manual/ch02_architecture_overview.md)
- [Installation and Setup](upstream_drift_user_manual/ch03_installation_and_setup.md)
- [Quick Start Guide](upstream_drift_user_manual/ch04_quick_start_guide.md)
- [Physics Engine Framework](upstream_drift_user_manual/ch05_physics_engine_framework.md)
- [MuJoCo Physics Engine](upstream_drift_user_manual/ch06_mujoco_physics_engine.md)
- [Rigid Body Dynamics Algorithms](upstream_drift_user_manual/ch07_rigid_body_dynamics_algorithms.md)
- [Drake Physics Engine](upstream_drift_user_manual/ch08_drake_physics_engine.md)
- [Pinocchio Physics Engine](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md)
- [OpenSim Physics Engine](upstream_drift_user_manual/ch10_opensim_physics_engine.md)
- [MyoSuite Musculoskeletal Engine](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md)
- [Biomechanical Analysis Suite](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md)
- [Contact Dynamics](upstream_drift_user_manual/ch13_contact_dynamics.md)
- [Whole-Body Control](upstream_drift_user_manual/ch14_whole_body_control.md)
- [Locomotion](upstream_drift_user_manual/ch15_locomotion.md)
- [Sensing](upstream_drift_user_manual/ch16_sensing.md)
- [Reinforcement Learning](upstream_drift_user_manual/ch17_reinforcement_learning.md)
- [Imitation Learning](upstream_drift_user_manual/ch18_imitation_learning.md)
- [Sim-to-Real Transfer](upstream_drift_user_manual/ch19_sim_to_real_transfer.md)
- [Deployment Infrastructure](upstream_drift_user_manual/ch20_deployment_infrastructure.md)
- [Research Modules](upstream_drift_user_manual/ch21_research_modules.md)
- [Visualization System](upstream_drift_user_manual/ch22_visualization_system.md)
- [REST API Reference](upstream_drift_user_manual/ch23_rest_api_reference.md)
- [Motion Retargeting](upstream_drift_user_manual/ch24_motion_retargeting.md)
- [Tools and Utilities](upstream_drift_user_manual/ch25_tools_and_utilities.md)
- [Mathematical Reference / Model Specifications / Golf Ball Physics](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md)
- [Configuration Reference / Error Codes Reference](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md)
- [Glossary](upstream_drift_user_manual/x_appendix_f_glossary.md)

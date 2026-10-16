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

<!-- manual-toc:start -->

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

- **[Chapter 5: Physics Engine Framework](upstream_drift_user_manual/ch05_physics_engine_framework.md#chapter-5-physics-engine-framework)**

  - [5.1 Overview](upstream_drift_user_manual/ch05_physics_engine_framework.md#51-overview)
  - [5.2 The `PhysicsEngine` Protocol](upstream_drift_user_manual/ch05_physics_engine_framework.md#52-the-physicsengine-protocol)
  - [5.3 Design by Contract -- `contracts.py`](upstream_drift_user_manual/ch05_physics_engine_framework.md#53-design-by-contract----contractspy)
  - [5.4 `BasePhysicsEngine`](upstream_drift_user_manual/ch05_physics_engine_framework.md#54-basephysicsengine)
  - [5.5 Engine Registry](upstream_drift_user_manual/ch05_physics_engine_framework.md#55-engine-registry)
  - [5.6 Engine Manager](upstream_drift_user_manual/ch05_physics_engine_framework.md#56-engine-manager)
  - [5.7 Engine Probes](upstream_drift_user_manual/ch05_physics_engine_framework.md#57-engine-probes)

- **[Chapter 6: MuJoCo Physics Engine](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#chapter-6-mujoco-physics-engine)**

  - [6.1 Overview](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#61-overview)
  - [6.2 `MuJoCoPhysicsEngine`](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#62-mujocophysicsengine)
  - [6.3 Model Hierarchy](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#63-model-hierarchy)
  - [6.4 Simulation and Controllers](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#64-simulation-and-controllers)
  - [6.5 Advanced Kinematics](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#65-advanced-kinematics)
  - [6.6 Advanced Control](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#66-advanced-control)
  - [6.7 Kinematic Forces Analysis](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#67-kinematic-forces-analysis)
  - [6.8 Inverse Dynamics](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#68-inverse-dynamics)
  - [6.9 Motion Capture](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#69-motion-capture)
  - [6.10 Motion Optimization](upstream_drift_user_manual/ch06_mujoco_physics_engine.md#610-motion-optimization)

- **[Chapter 7: Rigid Body Dynamics Algorithms](upstream_drift_user_manual/ch07_rigid_body_dynamics_algorithms.md#chapter-7-rigid-body-dynamics-algorithms)**

  - [7.1 Overview](upstream_drift_user_manual/ch07_rigid_body_dynamics_algorithms.md#71-overview)
  - [7.2 Recursive Newton-Euler Algorithm (RNEA)](upstream_drift_user_manual/ch07_rigid_body_dynamics_algorithms.md#72-recursive-newton-euler-algorithm-rnea)
  - [7.3 Composite Rigid Body Algorithm (CRBA)](upstream_drift_user_manual/ch07_rigid_body_dynamics_algorithms.md#73-composite-rigid-body-algorithm-crba)
  - [7.4 Articulated Body Algorithm (ABA)](upstream_drift_user_manual/ch07_rigid_body_dynamics_algorithms.md#74-articulated-body-algorithm-aba)
  - [7.5 Induced Acceleration Analysis](upstream_drift_user_manual/ch07_rigid_body_dynamics_algorithms.md#75-induced-acceleration-analysis)

- **[Chapter 8: Drake Physics Engine](upstream_drift_user_manual/ch08_drake_physics_engine.md#chapter-8-drake-physics-engine)**

  - [8.1 Overview](upstream_drift_user_manual/ch08_drake_physics_engine.md#81-overview)
  - [8.2 `DrakePhysicsEngine`](upstream_drift_user_manual/ch08_drake_physics_engine.md#82-drakephysicsengine)
  - [8.3 Motion Optimization](upstream_drift_user_manual/ch08_drake_physics_engine.md#83-motion-optimization)
  - [8.4 Swing Plane Integration](upstream_drift_user_manual/ch08_drake_physics_engine.md#84-swing-plane-integration)
  - [8.5 Direct Collocation (Drake Integration Notes)](upstream_drift_user_manual/ch08_drake_physics_engine.md#85-direct-collocation-drake-integration-notes)
  - [8.6 LQR Control in Drake Context](upstream_drift_user_manual/ch08_drake_physics_engine.md#86-lqr-control-in-drake-context)
  - [8.7 Contact Modeling](upstream_drift_user_manual/ch08_drake_physics_engine.md#87-contact-modeling)
  - [8.8 Engine Comparison: MuJoCo vs. Drake](upstream_drift_user_manual/ch08_drake_physics_engine.md#88-engine-comparison-mujoco-vs-drake)

- **[Chapter 9: Pinocchio Physics Engine](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#chapter-9-pinocchio-physics-engine)**

  - [9.1 Overview](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#91-overview)
  - [9.2 Core Algorithms](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#92-core-algorithms)
  - [9.3 Fast Analytical Jacobians and Derivatives](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#93-fast-analytical-jacobians-and-derivatives)
  - [9.4 Drift-Control Decomposition (Section F)](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#94-drift-control-decomposition-section-f)
  - [9.5 Counterfactual Experiments (Section G)](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#95-counterfactual-experiments-section-g)
  - [9.6 URDF Model Loading and Joint Configuration](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#96-urdf-model-loading-and-joint-configuration)
  - [9.7 MuJoCo-Pinocchio Bridge](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#97-mujoco-pinocchio-bridge)
  - [9.8 PinocchioBackend for Standalone Usage](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#98-pinocchiobackend-for-standalone-usage)
  - [9.9 PINK Inverse Kinematics Integration](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#99-pink-inverse-kinematics-integration)
  - [9.10 Performance Comparison With MuJoCo](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#910-performance-comparison-with-mujoco)
  - [9.11 Complete Workflow Example](upstream_drift_user_manual/ch09_pinocchio_physics_engine.md#911-complete-workflow-example)

- **[Chapter 10: OpenSim Physics Engine](upstream_drift_user_manual/ch10_opensim_physics_engine.md#chapter-10-opensim-physics-engine)**

  - [10.1 Overview](upstream_drift_user_manual/ch10_opensim_physics_engine.md#101-overview)
  - [10.2 Engine Initialization and Model Loading](upstream_drift_user_manual/ch10_opensim_physics_engine.md#102-engine-initialization-and-model-loading)
  - [10.3 Simulation and Integration](upstream_drift_user_manual/ch10_opensim_physics_engine.md#103-simulation-and-integration)
  - [10.4 Dynamics Computations](upstream_drift_user_manual/ch10_opensim_physics_engine.md#104-dynamics-computations)
  - [10.5 Drift-Control Decomposition (Section F)](upstream_drift_user_manual/ch10_opensim_physics_engine.md#105-drift-control-decomposition-section-f)
  - [10.6 Counterfactual Experiments (Section G)](upstream_drift_user_manual/ch10_opensim_physics_engine.md#106-counterfactual-experiments-section-g)
  - [10.7 Muscle Model Integration (Section J)](upstream_drift_user_manual/ch10_opensim_physics_engine.md#107-muscle-model-integration-section-j)
  - [10.8 Grip Modeling (Section J1)](upstream_drift_user_manual/ch10_opensim_physics_engine.md#108-grip-modeling-section-j1)
  - [10.9 Musculoskeletal Model Validation Workflows](upstream_drift_user_manual/ch10_opensim_physics_engine.md#109-musculoskeletal-model-validation-workflows)
  - [10.10 Complete Workflow Example](upstream_drift_user_manual/ch10_opensim_physics_engine.md#1010-complete-workflow-example)

- **[Chapter 11: MyoSuite Musculoskeletal Engine](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#chapter-11-myosuite-musculoskeletal-engine)**

  - [11.1 Overview](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#111-overview)
  - [11.2 Model Configurations](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#112-model-configurations)
  - [11.3 Engine Initialization](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#113-engine-initialization)
  - [11.4 Hill-Type Muscle Model](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#114-hill-type-muscle-model)
  - [11.5 Muscle Analysis Module](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#115-muscle-analysis-module)
  - [11.6 Moment Arm Analysis](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#116-moment-arm-analysis)
  - [11.7 Muscle-Generated Joint Torques](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#117-muscle-generated-joint-torques)
  - [11.8 Muscle-Induced Accelerations](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#118-muscle-induced-accelerations)
  - [11.9 Muscle Co-Contraction Analysis](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#119-muscle-co-contraction-analysis)
  - [11.10 Power Analysis](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#1110-power-analysis)
  - [11.11 Muscle Recruitment Optimization](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#1111-muscle-recruitment-optimization)
  - [11.12 Comprehensive Analysis](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#1112-comprehensive-analysis)
  - [11.13 Grip Modeling (Section K1)](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#1113-grip-modeling-section-k1)
  - [11.14 Drift-Control Decomposition and Counterfactuals](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#1114-drift-control-decomposition-and-counterfactuals)
  - [11.15 RL Adapter: MuscleDrivenEnv](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#1115-rl-adapter-muscledrivenenv)
  - [11.16 Cross-Validation With OpenSim](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#1116-cross-validation-with-opensim)
  - [11.17 Complete Workflow Example](upstream_drift_user_manual/ch11_myosuite_musculoskeletal_engine.md#1117-complete-workflow-example)

- **[Chapter 12: Biomechanical Analysis Suite](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#chapter-12-biomechanical-analysis-suite)**

  - [12.1 Overview](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#121-overview)
  - [12.2 Swing Metrics (`swing_metrics.py`)](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#122-swing-metrics-swing_metricspy)
  - [12.3 Energy Metrics (`energy_metrics.py`)](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#123-energy-metrics-energy_metricspy)
  - [12.4 Ground Reaction Force Metrics (`grf_metrics.py`)](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#124-ground-reaction-force-metrics-grf_metricspy)
  - [12.5 Angular Momentum (`angular_momentum.py`)](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#125-angular-momentum-angular_momentumpy)
  - [12.6 Phase Detection (`phase_detection.py`)](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#126-phase-detection-phase_detectionpy)
  - [12.7 Stability Metrics (`stability_metrics.py`)](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#127-stability-metrics-stability_metricspy)
  - [12.8 Shared Dataclasses](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#128-shared-dataclasses)
  - [12.9 Muscle Synergy Analysis](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#129-muscle-synergy-analysis)
  - [12.10 Complete Analysis Workflow](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#1210-complete-analysis-workflow)
  - [12.11 Summary of Equations](upstream_drift_user_manual/ch12_biomechanical_analysis_suite.md#1211-summary-of-equations)

- **[Chapter 13: Contact Dynamics](upstream_drift_user_manual/ch13_contact_dynamics.md#chapter-13-contact-dynamics)**

  - [13.1 Core Types and Protocols](upstream_drift_user_manual/ch13_contact_dynamics.md#131-core-types-and-protocols)
  - [13.2 Friction Cone Analysis](upstream_drift_user_manual/ch13_contact_dynamics.md#132-friction-cone-analysis)
  - [13.3 Contact Management](upstream_drift_user_manual/ch13_contact_dynamics.md#133-contact-management)
  - [13.4 Grasp Analysis](upstream_drift_user_manual/ch13_contact_dynamics.md#134-grasp-analysis)

- **[Chapter 14: Whole-Body Control](upstream_drift_user_manual/ch14_whole_body_control.md#chapter-14-whole-body-control)**

  - [14.1 Task-Based Control](upstream_drift_user_manual/ch14_whole_body_control.md#141-task-based-control)
  - [14.2 QP Solver](upstream_drift_user_manual/ch14_whole_body_control.md#142-qp-solver)
  - [14.3 Control Modes](upstream_drift_user_manual/ch14_whole_body_control.md#143-control-modes)

- **[Chapter 15: Locomotion](upstream_drift_user_manual/ch15_locomotion.md#chapter-15-locomotion)**

  - [15.1 Gait Types and Parameters](upstream_drift_user_manual/ch15_locomotion.md#151-gait-types-and-parameters)
  - [15.2 Gait State Machine](upstream_drift_user_manual/ch15_locomotion.md#152-gait-state-machine)
  - [15.3 Footstep Planning](upstream_drift_user_manual/ch15_locomotion.md#153-footstep-planning)
  - [15.4 Zero Moment Point](upstream_drift_user_manual/ch15_locomotion.md#154-zero-moment-point)

- **[Chapter 16: Sensing](upstream_drift_user_manual/ch16_sensing.md#chapter-16-sensing)**

  - [16.1 Noise Models](upstream_drift_user_manual/ch16_sensing.md#161-noise-models)
  - [16.2 IMU Sensor](upstream_drift_user_manual/ch16_sensing.md#162-imu-sensor)
  - [16.3 Force-Torque Sensor](upstream_drift_user_manual/ch16_sensing.md#163-force-torque-sensor)
  - [Summary of Key Imports](upstream_drift_user_manual/ch16_sensing.md#summary-of-key-imports)

- **[Chapter 17: Reinforcement Learning](upstream_drift_user_manual/ch17_reinforcement_learning.md#chapter-17-reinforcement-learning)**

  - [17.1 Configuration System](upstream_drift_user_manual/ch17_reinforcement_learning.md#171-configuration-system)
  - [17.2 Gymnasium Environment Base](upstream_drift_user_manual/ch17_reinforcement_learning.md#172-gymnasium-environment-base)
  - [17.3 Humanoid Environments](upstream_drift_user_manual/ch17_reinforcement_learning.md#173-humanoid-environments)
  - [17.4 Manipulation Environments](upstream_drift_user_manual/ch17_reinforcement_learning.md#174-manipulation-environments)

- **[Chapter 18: Imitation Learning](upstream_drift_user_manual/ch18_imitation_learning.md#chapter-18-imitation-learning)**

  - [18.1 Demonstration Dataset](upstream_drift_user_manual/ch18_imitation_learning.md#181-demonstration-dataset)
  - [18.2 Behavior Cloning](upstream_drift_user_manual/ch18_imitation_learning.md#182-behavior-cloning)
  - [18.3 DAgger (Dataset Aggregation)](upstream_drift_user_manual/ch18_imitation_learning.md#183-dagger-dataset-aggregation)
  - [18.4 GAIL (Generative Adversarial Imitation Learning)](upstream_drift_user_manual/ch18_imitation_learning.md#184-gail-generative-adversarial-imitation-learning)

- **[Chapter 19: Sim-to-Real Transfer](upstream_drift_user_manual/ch19_sim_to_real_transfer.md#chapter-19-sim-to-real-transfer)**

  - [19.1 Domain Randomization](upstream_drift_user_manual/ch19_sim_to_real_transfer.md#191-domain-randomization)
  - [19.2 System Identification](upstream_drift_user_manual/ch19_sim_to_real_transfer.md#192-system-identification)

- **[Chapter 20: Deployment Infrastructure](upstream_drift_user_manual/ch20_deployment_infrastructure.md#chapter-20-deployment-infrastructure)**

  - [20.1 Real-Time Control](upstream_drift_user_manual/ch20_deployment_infrastructure.md#201-real-time-control)
  - [20.2 Digital Twin](upstream_drift_user_manual/ch20_deployment_infrastructure.md#202-digital-twin)
  - [20.3 Safety System](upstream_drift_user_manual/ch20_deployment_infrastructure.md#203-safety-system)
  - [20.4 Teleoperation](upstream_drift_user_manual/ch20_deployment_infrastructure.md#204-teleoperation)
  - [Summary](upstream_drift_user_manual/ch20_deployment_infrastructure.md#summary)

- **[Chapter 21: Research Modules](upstream_drift_user_manual/ch21_research_modules.md#chapter-21-research-modules)**

  - [21.1 Model Predictive Control (MPC)](upstream_drift_user_manual/ch21_research_modules.md#211-model-predictive-control-mpc)
  - [21.2 Differentiable Physics](upstream_drift_user_manual/ch21_research_modules.md#212-differentiable-physics)
  - [21.3 Deformable Objects](upstream_drift_user_manual/ch21_research_modules.md#213-deformable-objects)
  - [21.4 Multi-Robot Systems](upstream_drift_user_manual/ch21_research_modules.md#214-multi-robot-systems)

- **[Chapter 22: Visualization System](upstream_drift_user_manual/ch22_visualization_system.md#chapter-22-visualization-system)**

  - [22.1 MuJoCo 3-D Visualization](upstream_drift_user_manual/ch22_visualization_system.md#221-mujoco-3-d-visualization)
  - [22.2 Matplotlib Plotting Framework](upstream_drift_user_manual/ch22_visualization_system.md#222-matplotlib-plotting-framework)
  - [22.3 Animation System](upstream_drift_user_manual/ch22_visualization_system.md#223-animation-system)
  - [22.4 Export System](upstream_drift_user_manual/ch22_visualization_system.md#224-export-system)
  - [22.5 Dashboard](upstream_drift_user_manual/ch22_visualization_system.md#225-dashboard)

- **[Chapter 23: REST API Reference](upstream_drift_user_manual/ch23_rest_api_reference.md#chapter-23-rest-api-reference)**

  - [23.1 Server Architecture](upstream_drift_user_manual/ch23_rest_api_reference.md#231-server-architecture)
  - [23.2 Core Endpoints](upstream_drift_user_manual/ch23_rest_api_reference.md#232-core-endpoints)
  - [23.3 Authentication](upstream_drift_user_manual/ch23_rest_api_reference.md#233-authentication)
  - [23.4 Request/Response Models](upstream_drift_user_manual/ch23_rest_api_reference.md#234-requestresponse-models)

- **[Chapter 24: Motion Retargeting](upstream_drift_user_manual/ch24_motion_retargeting.md#chapter-24-motion-retargeting)**

  - [24.1 Skeleton Configuration](upstream_drift_user_manual/ch24_motion_retargeting.md#241-skeleton-configuration)
  - [24.2 Retargeting Methods](upstream_drift_user_manual/ch24_motion_retargeting.md#242-retargeting-methods)
  - [24.3 Motion Capture Integration](upstream_drift_user_manual/ch24_motion_retargeting.md#243-motion-capture-integration)

- **[Chapter 25: Tools and Utilities](upstream_drift_user_manual/ch25_tools_and_utilities.md#chapter-25-tools-and-utilities)**

  - [25.1 Model Generation](upstream_drift_user_manual/ch25_tools_and_utilities.md#251-model-generation)
  - [25.2 Model Explorer](upstream_drift_user_manual/ch25_tools_and_utilities.md#252-model-explorer)
  - [25.3 Video Analyzer](upstream_drift_user_manual/ch25_tools_and_utilities.md#253-video-analyzer)
  - [25.4 Injury Risk Assessment](upstream_drift_user_manual/ch25_tools_and_utilities.md#254-injury-risk-assessment)
  - [25.5 Signal Processing Toolkit](upstream_drift_user_manual/ch25_tools_and_utilities.md#255-signal-processing-toolkit)
  - [25.6 Spatial Algebra](upstream_drift_user_manual/ch25_tools_and_utilities.md#256-spatial-algebra)
  - [Summary of Key File Paths](upstream_drift_user_manual/ch25_tools_and_utilities.md#summary-of-key-file-paths)

- **[Appendix A: Mathematical Reference](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#appendix-a-mathematical-reference)**

  - [A.1 Notation Conventions](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#a1-notation-conventions)
  - [A.2 Equations of Motion](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#a2-equations-of-motion)
  - [A.3 Spatial Algebra](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#a3-spatial-algebra)
  - [A.4 Rotation Representations](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#a4-rotation-representations)
  - [A.5 Optimization Methods](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#a5-optimization-methods)
  - [A.6 Hill-Type Muscle Model -- Complete Derivation](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#a6-hill-type-muscle-model----complete-derivation)
  - [A.7 Numerical Integration Methods](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#a7-numerical-integration-methods)

- **[Appendix B: Model Specifications](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#appendix-b-model-specifications)**

  - [B.1 Double Pendulum (2 DOF)](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#b1-double-pendulum-2-dof)
  - [B.2 Triple Pendulum (3 DOF)](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#b2-triple-pendulum-3-dof)
  - [B.3 Upper Body Model (10 DOF)](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#b3-upper-body-model-10-dof)
  - [B.4 Full Body Model (15 DOF)](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#b4-full-body-model-15-dof)
  - [B.5 Advanced Model (28 DOF)](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#b5-advanced-model-28-dof)
  - [B.6 MyoSuite Full Body (52 DOF, 290 Muscles)](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#b6-myosuite-full-body-52-dof-290-muscles)
  - [B.7 Standard Humanoid (66 DOF)](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#b7-standard-humanoid-66-dof)

- **[Appendix C: Golf Ball Physics](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#appendix-c-golf-ball-physics)**

  - [C.1 USGA Ball Specifications](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#c1-usga-ball-specifications)
  - [C.2 Ball Flight Model](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#c2-ball-flight-model)
  - [C.3 Impact Model](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#c3-impact-model)
  - [C.4 Typical Ball Flight Parameters](upstream_drift_user_manual/x_appendix_a-c_mathematical_reference.md#c4-typical-ball-flight-parameters)

- **[Appendix D: Configuration Reference](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#appendix-d-configuration-reference)**

  - [D.1 `pyproject.toml`](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#d1-pyprojecttoml)
  - [D.2 environment.yml](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#d2-environmentyml)
  - [D.3 Makefile Targets](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#d3-makefile-targets)
  - [D.4 Docker Configuration](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#d4-docker-configuration)
  - [D.5 Logging Configuration](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#d5-logging-configuration)
  - [D.6 Numerical Constants](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#d6-numerical-constants)

- **[Appendix E: Error Codes Reference](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#appendix-e-error-codes-reference)**

  - [E.1 Error Code Format](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#e1-error-code-format)
  - [E.2 Complete Error Code Table](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#e2-complete-error-code-table)
  - [E.3 Error Response Format](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#e3-error-response-format)
  - [E.4 Exception Hierarchy](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#e4-exception-hierarchy)
  - [E.5 Contract Violation Errors](upstream_drift_user_manual/x_appendix_d-e_configuration_reference.md#e5-contract-violation-errors)

- **[Appendix F: Glossary](upstream_drift_user_manual/x_appendix_f_glossary.md#appendix-f-glossary)**

  - [Acronyms](upstream_drift_user_manual/x_appendix_f_glossary.md#acronyms)
  - [Technical Terms](upstream_drift_user_manual/x_appendix_f_glossary.md#technical-terms)
  - [Model Format Terms](upstream_drift_user_manual/x_appendix_f_glossary.md#model-format-terms)
  - [Software Engineering Terms](upstream_drift_user_manual/x_appendix_f_glossary.md#software-engineering-terms)

<!-- manual-toc:end -->

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

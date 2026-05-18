# UpstreamDrift User Manual

Welcome to the UpstreamDrift Biomechanical Golf Simulation and Analysis Suite.
This manual provides actionable instructions for users and AI agents interacting with the suite.

## 1. Introduction

UpstreamDrift offers advanced 3D simulation of golf swings using MuJoCo and OpenSim. It is designed for biomechanical analysis, swing path optimization, and club-face impact dynamics.

## 2. Launching the Suite

Use the integrated launcher to open different components.

- **Model Explorer**: View and edit URDF models of golfers and clubs.
- **Video Analyzer**: Upload videos for automated pose estimation.
- **Data Explorer**: View telemetry from tracked sessions.

## 3. Using the Document Reader

To assist with troubleshooting and learning, the application includes a built-in document reader capable of displaying:

- **Markdown (.md)** files (like this manual)
- **PDF (.pdf)** research papers
- **LaTeX (.tex)** formula specifications

You can open files from the `Help > Document Reader` menu.

## 4. Modules

### 4.1 Biomechanics Simulation

Provides access to MyoSuite-based musculoskeletal simulations. Use the 'Controls' tab to adjust muscle activation parameters.

### 4.2 Physics Engines

Switch between MuJoCo (default), Drake, and Pinocchio depending on the needed fidelity.

### 4.3 Diagnostics

For diagnostics, open the internal logging console located in the bottom dock of the main window.

## 5. Troubleshooting

If you experience a freeze or crash:

1. Open this manual in the in-app Document Reader.
2. Check the `logs/` directory for tracebacks.
3. Verify that all dependencies from `pyproject.toml` are correctly installed in your environment.

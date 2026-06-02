# UpstreamDrift User Manual

Welcome to the UpstreamDrift Biomechanical Golf Simulation and Analysis Suite.
This manual provides detailed, actionable instructions for users and AI agents interacting with the suite. It is designed to be read within the application's built-in Document Reader.

---

## 1. Introduction

UpstreamDrift offers advanced 3D simulation of golf swings using state-of-the-art physics engines including MuJoCo, Drake, and OpenSim. It is designed for biomechanical analysis, swing path optimization, and high-fidelity club-face impact dynamics.

This document serves as the central hub for learning how to use the application. Click any link below to jump to the relevant section or related documentation.

---

## 2. Launching the Suite

Use the integrated Python-based launcher to access different components of the suite:

1. **Launch the application** via `python src/launchers/upstream_drift_launcher.py`.
2. **Select a Module** from the main sidebar.
3. **Use the Sidekick AI** for contextual help by clicking the chat icon in the bottom right.

### Available Modules:

- **[Model Explorer](#3-model-explorer)**: View and edit URDF models of golfers and clubs.
- **[Video Analyzer](#4-video-analyzer)**: Upload videos for automated pose estimation.
- **[Data Explorer](#5-data-explorer)**: View telemetry from tracked sessions.
- **[Biomechanics Simulation](#6-biomechanics-simulation)**: Run forward kinematics and muscle activation models.

---

## 3. Model Explorer

The Model Explorer allows you to visualize and edit the underlying skeletal and physical models used in the simulation.

**Instructions:**

1. Navigate to the `Model Explorer` tab.
2. Click **Load URDF** to select a model (e.g., `human.urdf` or `golf_club.urdf`).
3. Use the **3D Viewport** to rotate (Left Click + Drag) and pan (Right Click + Drag).
4. Use the **Properties Panel** on the right to modify joint limits and mass properties.
5. Click **Save Model** to persist your changes.

_See also: [URDF Cross-Engine Verification Guide](engines/physics_engines/mujoco/MUJOCO_PARITY_SPEC.md)_

---

## 4. Video Analyzer

The Video Analyzer ingests raw 2D video (e.g., from a smartphone) and extracts 3D pose data using MediaPipe or OpenPose.

**Instructions:**

1. Navigate to the `Video Analyzer` tab.
2. Click **Import Video** and select an `.mp4` file of a golf swing.
3. Select your preferred extraction backend (MediaPipe is recommended for speed).
4. Click **Process**. The application will display a progress bar.
5. Once complete, click **Export Pose Data** to save the 3D joint trajectories as a `.json` file for use in the Biomechanics Simulation.

_See also: [Motion Pipeline Readiness](integration/motion_pipeline/adversarial/READINESS.md)_

---

## 5. Data Explorer

The Data Explorer visualizes telemetry data, such as club head speed, club path, and impact forces.

**Instructions:**

1. Navigate to the `Data Explorer` tab.
2. Load a telemetry dataset (either a `.json` pose file or a `.csv` TrackMan/GCQuad export).
3. Select the metrics you wish to plot from the **Metrics List**.
4. Use the **Timeline Scrubber** to move through the swing phases (Address, Top, Impact, Follow-through).
5. The 3D view will synchronize with the scrubber, showing the golfer's position at the selected time.

---

## 6. Biomechanics Simulation

This is the core module for running physics-based simulations of the golf swing.

**Instructions:**

1. Navigate to the `Biomechanics Simulation` tab.
2. Select a **Physics Engine** from the dropdown menu:
   - **MuJoCo**: Best for fast, contact-rich simulations (default).
   - **Drake**: Best for rigid body dynamics and optimization.
   - **Pinocchio**: Best for fast forward/inverse kinematics.
3. In the **Controls Panel**, adjust the target muscle activation signals or joint torques.
4. Click **Run Simulation**.
5. Analyze the resulting trajectory and impact forces in the integrated data viewer.

_See also: [MuJoCo Parity Spec](engines/physics_engines/mujoco/MUJOCO_PARITY_SPEC.md) | [Drake Parity Spec](engines/physics_engines/drake/DRAKE_PARITY_SPEC.md)_

---

## 7. Using the Document Reader

To assist with troubleshooting and learning, the application includes a built-in document reader capable of displaying:

- **Markdown (.md)** files (like this manual)
- **PDF (.pdf)** research papers (e.g., biomechanics papers in `docs/papers/`)
- **LaTeX (.tex)** formula specifications

**Instructions:**

1. Open files from the `Help > Document Reader` menu.
2. Use the **Zoom** controls in the toolbar to adjust text size.
3. For PDFs, use the **Page Navigation** buttons or scroll down.
4. For Markdown and LaTeX, the reader automatically renders equations and formatting.

---

## 8. Troubleshooting & Diagnostics

If you experience issues, follow these steps:

1. **Check the Internal Console**: Open the internal logging console located in the bottom dock of the main window. Look for red `ERROR` messages.
2. **Verify Dependencies**: Ensure that all dependencies from `pyproject.toml` are correctly installed. Specifically, check that `mujoco` is installed if you intend to run MuJoCo simulations.
3. **Reset Configuration**: Go to `File > Preferences` and click **Reset to Defaults**.
4. **Consult the AI Sidekick**: Open the Sidekick panel and type a description of your problem. The Sidekick has access to your recent diagnostic logs and can suggest fixes.

For further assistance, consult the developer documentation in the `docs/` folder or create an issue on GitHub.

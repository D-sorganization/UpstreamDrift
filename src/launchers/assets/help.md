# Golf Modeling Suite - Help & Documentation

## Overview
The Golf Modeling Suite is a comprehensive environment for biomechanical simulation and analysis of golf swings. It leverages industry-standard physics engines to model the human body, club dynamics, and impacts.

## Physics Engines

### 1. MuJoCo (Multi-Joint dynamics with Contact)
*   **Best for:** High-performance biomechanical simulation, contact dynamics, and fast iterations.
*   **Humanoid Model:** Features a full-body humanoid with muscle/actuator models capable of realistic swing mechanics.
*   **Comprehensive:** Advanced analysis tools including phase plots, energy monitoring, and interactive pose adjustment.

### 2. Drake (MIT)
*   **Best for:** Model-based design, trajectory optimization, and verification.
*   **Features:** Robust constraint handling and geometry inspection via Meshcat.

### 3. Pinocchio
*   **Best for:** Rigid body dynamics algorithms, inverse kinematics (Pink), and robotics control.
*   **Features:** Fast recursive algorithms used for motion planning and control loop testing.

## Docker Environment
The suite runs within a unified `robotics_env` Docker container to ensure consistent dependencies across all platforms.

### Key Dependencies
*   **MuJoCo 3.x:** Core physics engine.
*   **dm_control:** DeepMind's control suite for reinforcement learning and simulation.
*   **Pinocchio:** Rigid body dynamics library.
*   **PyQt6:** GUI framework for the launcher and tools.
*   **NumPy/SciPy:** Scientific computing stack.

## Usage Tips
*   **Live Visualization:** Enable this to see the simulation window. Ensure X11/VcXsrv is running on Windows.
*   **GPU Acceleration:** Enable if you have an NVIDIA GPU and the NVIDIA Container Toolkit installed for faster rendering and computation.
*   **Build Docker:** Use the "Manage Environment" menu to rebuild the Docker image if you modify dependencies. You can choose to build specific stages (e.g., only MuJoCo) to save time.

## Troubleshooting
*   **"Docker not found":** Ensure Docker Desktop is running.
*   **"ImportError: libEGL":** The Docker image needs to be rebuilt to include graphics libraries. Click "Rebuild Environment".
*   **Display Issues:** Check that VcXsrv is running with "Disable access control" checked.

# System Architecture

The Golf Modeling Suite is designed as a unified interface over multiple independent physics backends.

## Core Components

1.  **Launchers**: The entry point (Python/PyQt) that manages simulation configuration and lifecycle.
2.  **Engine Interface**: A common abstraction (implicit or explicit) that allows the launcher to invoke different engines.
3.  **Engines**: Independent implementations (MuJoCo, Drake, Pinocchio, Simscape) that perform the actual physics simulation.
4.  **Shared Utilities**: Common code in `shared/` used by Python engines for logging, plotting, and data management.

## Directory Structure

*   `launchers/`: UI and startup logic.
*   `engines/`:
    *   `physics_engines/`: Python-based engines.
    *   `Simscape_Multibody_Models/`: MATLAB models.
*   `shared/`: Python libraries (`common_utils.py`, constants).
*   `tools/`: Development scripts.

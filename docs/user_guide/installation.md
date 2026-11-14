# Installation

## Prerequisites

### General

- **Git** with submodules enabled (for `vendor/ud-tools` and optional engine model packs).

### Python Environment

- **Python 3.11** or **3.12** (`requires-python = ">=3.11"` in `pyproject.toml`).
- We recommend using a dedicated virtual environment (`venv`, `conda`, or `uv`).

### MATLAB Environment (Optional)

- **MATLAB R2025b** (required release for Simscape golf model execution, model saves, and validation; on Windows launch `C:/Program Files/MATLAB/R2025b/bin/matlab.exe` explicitly rather than relying on `PATH` or R2026a).
- **Simulink** and **Simscape Multibody** (for the legacy MATLAB/Simscape models).

## Step-by-Step Setup

1.  **Clone the Repository and Initialize `ud-tools`**

    ```bash
    git clone https://github.com/D-sorganization/UpstreamDrift.git
    cd UpstreamDrift
    git submodule update --init --recursive vendor/ud-tools
    ```

2.  **Create and Activate a Virtual Environment**

    ```bash
    python -m venv .venv
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows (PowerShell):
    .venv\Scripts\Activate.ps1
    ```

3.  **Install UpstreamDrift from `pyproject.toml`**

    ```bash
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    ```

    Optional extras:

    - Extended engines (Drake, Pinocchio): `pip install -e ".[dev,all-engines]"`
    - Experimental biomechanics engines (OpenSim, MyoSuite; prefer a dedicated environment): `pip install -e ".[dev,biomechanics]"`
    - Additional GUI tools: `pip install -e ".[dev,gui-tools]"`

4.  **Verify the Installation**

    ```bash
    python scripts/ci/verify_installation.py
    python -m src.shared.python.feature_registry
    ```

5.  **MATLAB Setup (Optional, If Using Simscape)**
    - Open MATLAB and navigate to `src/shared/matlab/` inside the `UpstreamDrift` repository.
    - Run the setup script:
      ```matlab
      setup_golf_suite()
      ```

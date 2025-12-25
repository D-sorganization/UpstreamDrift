# Installation

## Prerequisites

### General
- **Git** with **Git LFS** (Large File Storage) enabled.
  ```bash
  git lfs install
  ```

### Python Environment
- **Python 3.10** or higher is required.
- We recommend using a virtual environment (venv, conda, etc.).

### MATLAB Environment (Optional)
- **MATLAB R2023a** or newer.
- **Simulink**.
- **Simscape Multibody**.

## Step-by-Step Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
    cd Golf_Modeling_Suite
    ```

2.  **Pull Large Files**
    ```bash
    git lfs pull
    ```

3.  **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs dependencies for MuJoCo, Drake, and Pinocchio engines.*

4.  **MATLAB Setup (If using Simscape)**
    - Open MATLAB.
    - Navigate to the `Golf_Modeling_Suite` directory.
    - Run the setup script:
      ```matlab
      setup_golf_suite()
      ```

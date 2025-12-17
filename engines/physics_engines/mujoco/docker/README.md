# Humanoid Golf Simulation

A DeepMind Control Suite simulation of a humanoid golfer, featuring a custom GUI for scaling (height/weight), appearance customization, and data logging.

## Features

- **Customizable Humanoid**:
  - **Scale Height**: Adjustable from 1.5m to 2.5m (Default: 1.8m).
  - **Scale Weight**: Adjustable volume/thickness (50% - 200%).
  - **Appearance**: Custom colors for Shirt, Pants, Shoes, Skin, and Club.
- **Data Logging**: Captures joint angles (`qpos`) and motor forces (`qfrc_actuator`) to `golf_data.csv`.
- **GUI Control**: User-friendly Windows interface to run Docker simulations.

## Prerequisites

1.  **Docker Desktop** (running with WSL2 backend).
2.  **Python 3** installed on Windows.
3.  **VcXsrv** (Windows X Server) - **REQUIRED for Live Interactive View**.
    - Install from [SourceForge](https://sourceforge.net/projects/vcxsrv/).
    - Run `XLaunch`, perform defaults but ensure **"Disable access control"** is CHECKED.

## How to Run (Recommended)

The easiest way is to run the GUI natively on Windows. It will execute the Docker commands for you.

1.  Open **Command Prompt** or **PowerShell**.
2.  Navigate to the `gui` folder:
    ```cmd
    cd <path_to_repo>\docker\gui
    ```
3.  Run the script:
    ```cmd
    python deepmind_control_suite_MuJoCo_GUI.py
    ```
4.  **In the GUI**:
    - Go to **Appearance Tab** to set Height, Weight, and Colors.
    - Click **Run Simulation** on the Control Tab.
    - Click **Open Video** to watch the result.

## Project Structure

- `gui/`: Contains the Windows GUI application.
- `src/humanoid_golf/`: Python package containing the simulation logic.
  - `sim.py`: Main simulation loop and data logging.
  - `utils.py`: Logic for XML patching, scaling, and customization.
- `simulation_config.json`: Stores your user settings (auto-generated).

# Getting Started

This guide will help you install and set up the Golf Modeling Suite on your machine.

## Prerequisites

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS (12+)
- **Python**: Version 3.10 or higher
- **Git**: For version control
- **Docker** (Optional): For running the full validation suite without local dependencies

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
cd Golf_Modeling_Suite
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the core suite along with development and engine dependencies:

```bash
pip install -e ".[dev,engines,analysis]"
```

This command installs:
- Core utilities (`numpy`, `pandas`)
- Visualization tools (`pyqt6`, `matplotlib`)
- Physics engines (`mujoco`, `pydrake`, `pinocchio`)
- Development tools (`pytest`, `black`, `ruff`)

## Verifying Installation

To verify that the suite is correctly installed, you can run the status check:

```bash
python launch_golf_suite.py --status
```

You should see a report listing the available physics engines and their status (Loaded/Available).

## Next Steps

- Learn how to [Configure the Suite](configuration.md)
- Try [Running Simulations](running_simulations.md)
- Explore the [Examples](../examples/index.rst)

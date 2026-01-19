# Tutorial 1: Getting Started with Golf Modeling Suite

**Estimated Time:** 30 minutes
**Difficulty:** Beginner

## Prerequisites
- Python 3.11+ installed
- Git with Git LFS
- 2 GB free disk space

## Learning Objectives
By the end of this tutorial, you will:
- ✅ Install the Golf Modeling Suite
- ✅ Verify your installation
- ✅ Run your first simulation
- ✅ Visualize simulation results

## Step 1: Installation

### Clone the Repository
\`\`\`bash
git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
cd Golf_Modeling_Suite
git lfs install && git lfs pull
\`\`\`

### Create Conda Environment (Recommended)
\`\`\`bash
conda env create -f environment.yml
conda activate golf-suite
\`\`\`

### Verify Installation
\`\`\`bash
python scripts/verify_installation.py
\`\`\`

Expected output:
\`\`\`
✅ Python version: 3.11.5
✅ MuJoCo installed
✅ Core dependencies available
Installation verified successfully!
\`\`\`

## Step 2: Run Your First Simulation

### Launch the Unified GUI
\`\`\`bash
python launchers/golf_launcher.py
\`\`\`

### Select Engine and Model
1. Choose **MuJoCo** from the engine dropdown
2. Select **2-DOF Pendulum** model
3. Click **Load Model**

### Configure Simulation
- Duration: 2.0 seconds
- Timestep: 0.001 seconds
- Initial angle: 45 degrees

### Run Simulation
Click **Run Simulation** button.

## Step 3: Visualize Results

The GUI will display:
- 3D animation of the swing
- Joint angle plots
- Angular velocity plots
- Energy conservation plot

### Export Results
Click **Export Data** → Choose **CSV** format.

## Next Steps
- [Tutorial 2: Loading C3D Motion Capture Data](02_c3d_data.md)
- [Tutorial 3: Parameter Sweep Analysis](03_parameter_sweeps.md)

## Troubleshooting
See [Installation Troubleshooting Guide](../../troubleshooting/installation.md)

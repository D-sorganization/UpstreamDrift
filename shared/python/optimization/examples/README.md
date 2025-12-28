# Optimization Examples

This directory contains examples for using CasADi with Pinocchio for trajectory optimization.

## 2-Link Arm Optimization (`optimize_arm.py`)

This example demonstrates how to optimize the swing of a 2-link robotic arm (simulating a simplified arm + club system) using Direct Multiple Shooting.

### Objective
Minimize the squared torque effort to swing the arm from a hanging position (0,0) to an upright position (180,0).

### Methods
- **Dynamics**: Pinocchio (ABA algorithm) autodifferentiated via CasADi.
- **Integration**: Semi-implicit Euler.
- **Solver**: IPOPT (Interior Point Optimizer).

### Running the Example

Ensure you have the dependencies installed:
```bash
pip install casadi
conda install pinocchio -c conda-forge
# or if using pip for pinocchio (experimental)
# pip install pin
```

Run the script:
```bash
python shared/python/optimization/examples/optimize_arm.py
```

It should find the optimal trajectory in < 1 second and save `.csv` files.

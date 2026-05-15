# EPIC: Physics-Informed Neural Networks (PINNs) Integration

## Objective
Implement Physics-Informed Neural Networks (PINNs) across the UpstreamDrift suite and related golf models to capture "unmodeled" physics (shaft flexibility, aerodynamic drag, wrist-hinge friction) that purely rigid multi-body dynamics models often miss.

## Background
As we transition toward a Python-centric architecture using **Pinocchio** and **Drake**, modeling the high-velocity, high-impact nature of a golf swing requires more fidelity than purely rigid models can provide. We will use a **Residual Physics** approach to combine the analytical models with neural networks.

## Phase 1: Data Preparation & Hybrid Architecture
Since we already use **Pinocchio**, we won't throw away the analytical model. Instead, we use a hybrid architecture:
1.  **Analytical Base**: Pinocchio/Drake provides the base rigid-body kinematics and dynamics.
2.  **PINN Residual**: The neural network predicts the residuals (differences between the rigid model and real-world data).

## Key Goals
1.  **Define the Hybrid State**: Extend existing state vectors to accommodate residual network inputs.
2.  **Implement PINN Scaffolding**: Build the PyTorch/JAX scaffolding for the PINN alongside the physics engine.
3.  **Data Ingestion Store**: Begin scaffolding the simulation data store to hold high-frequency capture data needed for training.
4.  **Integration**: Combine the outputs of the rigid solver and the PINN in the final integration step.

## Related Issues
- Simulation Data Store Scaffolding (Phase 1)

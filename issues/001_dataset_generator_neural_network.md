# Issue: Dataset Generator for Neural Network Training

## Summary
Create a shared dataset generator module that produces training datasets from physics
engine simulations. The generator should output all kinematics (positions, velocities,
accelerations), kinetics (torques, forces, energies), and model data (inertia matrices,
mass properties) into a structured database suitable for ML/neural network training.

## Motivation
To train neural networks for learning equations of motion and control laws, we need
large-scale, varied simulation datasets. Currently, the recorder captures data for
single runs but lacks the infrastructure for systematic parameter variation, batch
generation, and ML-ready dataset compilation.

## Requirements
- [ ] Vary simulation inputs (initial conditions, control profiles, model parameters)
- [ ] Record all output kinematics: joint positions (q), velocities (v), accelerations (a)
- [ ] Record all kinetics: joint torques (tau), contact forces, energies
- [ ] Record model data: mass matrices, bias forces, gravity vectors, Jacobians
- [ ] Compile into structured database (SQLite + HDF5)
- [ ] Support reproducibility (seed-based, provenance tracking)
- [ ] Design by Contract: clear pre/post conditions on all public methods
- [ ] Integration with existing `GenericPhysicsRecorder`

## Acceptance Criteria
- Dataset generator produces valid, queryable training data
- Data schema documented and validated
- Unit tests with >90% coverage
- Works with all physics engines (MuJoCo, Drake, Pinocchio)

## Labels
`enhancement`, `machine-learning`, `shared`

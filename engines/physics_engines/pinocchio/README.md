# Pinocchio Golf Model

## Vision

This project aims to produce a complete, free, Python-based golf biomechanics platform, capable of:

- Modeling a full 3D golfer with realistic joints, limbs, and club
- Simulating both kinematic and kinetic motion
- Running inverse dynamics (Pinocchio) and inverse kinematics (PINK)
- Exploring counterfactual physics (ZTCF, ZVCF, affine control perspective)
- Visualizing everything in MeshCat, MuJoCo, and Geppetto
- Supporting future ML-based optimization, control, and experiment generation
- Running entirely from a user-friendly GUI, with no code needed
- Embedding a canonical model specification to keep all backends perfectly synchronized

## Architecture Overview

The project is built around one invariant principle: **One canonical model → multiple backends.**

### Core Backend Stack

- **Pinocchio**: Forward/inverse dynamics, Jacobians, constraints
- **PINK**: IK, task-space goals, closed-loop IK for complicated joint structures
- **MuJoCo**: Contact simulation (feet-ground, club-ball), GRF, realistic motion integration
- **MeshCat**: Browser visualization of Pinocchio/PINK (fast debugging)
- **Geppetto Viewer**: Desktop visualization of Pinocchio models; ideal for joint validation
- **Python GUI (PySide6 / Qt)**: End-user interface for interactive exploration

## Installation

```bash
pip install -r python/requirements.txt
```

## Usage

(Coming soon)

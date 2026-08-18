# BunkerShot3D Backend Comparison

This document tracks the divergences between the three granular backends (Project Chrono, LIGGGHTS, MuJoCo MPM) used to simulate the bunker shot.

## Physical Divergences

### 1. Contact Models

- **Project Chrono**: Uses SMC (Smooth-Contact Method) with Hertz-Mindlin contact forces. This resolves true multi-contact interactions with micro-slip.
- **LIGGGHTS**: Uses a linear spring-dashpot model. While computationally faster, it may under-predict peak shear forces compared to Chrono.
- **MuJoCo MPM**: Uses a continuum approximation (Material Point Method). The mapping from grain-scale friction (e.g. $\mu=0.5$) to the continuum Drucker-Prager yield criterion introduces smearing at the boundaries.

### 2. Timestep Selection and Rayleigh Time

- **Chrono**: Safety factor of 0.2 of Rayleigh time is required due to high-stiffness clubhead impact.
- **LIGGGHTS**: Timestep is bounded by the linear spring stiffness; typically allows slightly larger timesteps.
- **MuJoCo MPM**: Bounded by the grid resolution and sound speed of the continuum.

### 3. Clubface Boundary Representation

- **Chrono**: Rigid triangular mesh with face friction.
- **LIGGGHTS**: Uses `fix mesh/surface` with imported STL.
- **MuJoCo MPM**: Uses a signed distance field (SDF) implicitly generated from the mesh, causing slight "padding" around the clubface where continuum points interact before strict geometric intersection.

### 4. Coarse-Graining Factor

To keep the particle count under 200,000 for reasonable compute times, a coarse-graining factor $C_g = 1.0$ is currently used (true scale). If this is scaled up, Chrono and LIGGGHTS preserve bulk invariants differently than MPM.

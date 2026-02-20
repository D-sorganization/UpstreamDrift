# GitHub Issue Draft: Physics Fidelity and Accuracy Issues

**Title:** [CRITICAL] Physics Fidelity Gaps in Ball Flight, Shaft, Impact, and GRF Models
**Labels:** accuracy-issue, physics-fidelity, critical

## Description
A comprehensive review of the physics modules has identified several critical gaps in fidelity and accuracy. These issues affect the realism of the simulation and must be addressed to meet the project's scientific goals.

## Critical Issues

### 1. Ball Flight Physics (`src/shared/python/physics/ball_flight_physics.py`)
- **Hardcoded Coefficients:** Aerodynamic coefficients (`cd0`, `cd1`, `cl0`, etc.) are hardcoded and lack citation or configurability.
- **Missing Environmental Models:** The implementation lacks:
    - Environmental Gradient Modeling (wind shear, temperature gradients).
    - Hydrodynamic Lubrication (wet ball physics).
    - Dimple Geometry Optimization.
    - Turbulence Modeling.
    - Mud Ball Physics.

### 2. Flexible Shaft (`src/shared/python/physics/flexible_shaft.py`)
- **Missing Torsional Dynamics:** The current Euler-Bernoulli beam model ignores torsional twisting, which is critical for clubface orientation at impact.
- **Symmetric Assumption:** The model assumes symmetric cross-sections, preventing the modeling of shaft spine alignment and manufacturing tolerances.

### 3. Impact Model (`src/shared/python/physics/impact_model.py`)
- **Simplified Effective Mass:** The `RigidBodyImpactModel` uses a scalar effective mass formula (`1 / (1/m + r^2/I)`). This ignores the full 3D inertia tensor and the direction of the impact vector, leading to inaccuracies in off-center impacts.

### 4. Ground Reaction Forces (`src/shared/python/physics/ground_reaction_forces.py`)
- **Inaccurate Fallback:** When native contact data is unavailable, `extract_grf_from_contacts` falls back to summing static gravity forces ($W=mg$). This ignores dynamic acceleration forces ($F=m(g+a)$), leading to errors during dynamic movements.

## Acceptance Criteria
- [ ] **Ball Flight:** Implement configurable aerodynamic coefficients and add missing environmental models (or clearly mark as out of scope).
- [ ] **Shaft:** Update beam model to include torsional degrees of freedom and support asymmetric cross-sections.
- [ ] **Impact:** Replace scalar effective mass with full 3D impulse-momentum formulation using the inertia tensor.
- [ ] **GRF:** Update `extract_grf_from_contacts` fallback to include body acceleration (F=ma) in the force calculation.

## References
- `src/shared/python/physics/ball_flight_physics.py`
- `src/shared/python/physics/flexible_shaft.py`
- `src/shared/python/physics/impact_model.py`
- `src/shared/python/physics/ground_reaction_forces.py`
- `docs/assessments/implementation_gaps_report.md`

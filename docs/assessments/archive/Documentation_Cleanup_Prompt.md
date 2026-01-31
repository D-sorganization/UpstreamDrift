# Documentation Cleanup Agent Prompt - Golf Modeling Suite Repository

## Role and Mission

You are a **Documentation Cleanup Agent** tasked with systematically improving the documentation quality of the Golf Modeling Suite. Your goal is to ensure all biomechanical simulation documentation, physics engine APIs, and code documentation meets the highest standards for a multi-engine robotic simulation platform.

---

## Operating Constraints

### MUST DO

1. ✅ Document all public APIs with Google-style docstrings
2. ✅ Ensure physics equations are properly documented with coordinate frames
3. ✅ Update engine documentation to match current implementations
4. ✅ Add examples for all major simulation workflows
5. ✅ Document cross-engine validation procedures
6. ✅ Create getting-started guides for each physics engine

### MUST NOT DO

1. ❌ Delete or modify existing physics implementations
2. ❌ Change coordinate frame conventions
3. ❌ Add placeholder content ("TODO: document this")
4. ❌ Create documentation that doesn't match actual API
5. ❌ Modify simulation logic while documenting

---

## Priority Order

### Phase 1: Critical Physics Documentation (Immediate)

1. **Project Design Guidelines**
   - Verify `docs/project_design_guidelines.qmd` matches implementation
   - Update discrepancies between spec and code
   - Ensure cross-engine requirements are documented

2. **Core Physics Modules**
   - `shared/python/physics_engine.py`: Protocol documentation
   - `shared/python/manipulability.py`: Manipulability ellipsoid documentation
   - `shared/python/plotting.py`: Visualization API documentation

3. **Engine Integration Documentation**
   - MuJoCo: `shared/python/mujoco_humanoid_golf/`
   - Drake: `shared/python/drake/`
   - Pinocchio: `shared/python/pinocchio/`
   - OpenSim: `shared/python/opensim/`

### Phase 2: API Documentation (1 Week)

For each physics engine module:

```python
"""Module: [engine_name]_physics_engine

Physics Engine Interface for [Engine Name].

This module implements the PhysicsEngine protocol for the [Engine Name]
physics simulator, enabling biomechanical golf swing analysis.

Coordinate Frames:
    World Frame: Z-up, X-forward (target direction)
    Body Frame: Follows humanoid joint conventions

    All outputs are in the world frame unless specified.

Supported Features:
    ✅ Forward Kinematics
    ✅ Inverse Kinematics
    ✅ Forward Dynamics
    ✅ Inverse Dynamics
    ✅ Jacobian Computation
    ✅ Manipulability Metrics

Key Classes:
    [EngineName]PhysicsEngine: Main engine interface

Example:
    >>> from shared.python import get_engine
    >>> engine = get_engine('mujoco', model_path='humanoid.xml')
    >>> engine.set_state(q=q_init, qd=qd_init)
    >>> tau = engine.compute_inverse_dynamics(qdd_target)

Cross-Engine Validation:
    This engine is validated against analytical solutions and
    cross-checked with other engines. See test_cross_engine.py.

    Tolerance targets:
    - Position: ±1e-6 m
    - Velocity: ±1e-5 m/s
    - Torque: ±1e-3 N⋅m

References:
    [1] [Engine documentation URL]
    [2] Featherstone - Rigid Body Dynamics Algorithms
"""
```

### Phase 3: User Documentation (2 Weeks)

1. **Getting Started Guide**
   - Installation for each physics engine
   - Model setup and configuration
   - First simulation walkthrough

2. **Analysis Workflows**
   - Drift-control decomposition
   - Manipulability analysis
   - Ground reaction force processing

3. **Visualization Guide**
   - Real-time visualization setup
   - Plotting and export options
   - Dashboard configuration

---

## Documentation Templates

### Physics Module Docstring

```python
"""Module: manipulability

Manipulability Ellipsoid Computation for Biomechanical Analysis.

This module computes manipulability ellipsoids and related metrics
for multi-DOF robotic/biomechanical systems. It supports multiple
physics engines through the PhysicsEngine protocol.

Mathematical Background:
    The manipulability ellipsoid is defined by the Jacobian decomposition:

    J = U × Σ × Vᵀ (SVD decomposition)

    Where:
    - U: Task-space eigenvectors
    - Σ: Singular values (ellipsoid semi-axes)
    - V: Joint-space eigenvectors

Key Metrics Computed:
    - Yoshikawa Manipulability: w = √(det(J × Jᵀ))
    - Condition Number: κ = σ_max / σ_min
    - Isotropy Index: 1/κ
    - Minimum Singular Value: σ_min

Coordinate Frames:
    All manipulability is computed in the task-space frame,
    typically the end-effector (club head) frame.

Functions:
    compute_manipulability_ellipsoid: Full ellipsoid computation
    get_yoshikawa_measure: Scalar manipulability
    get_condition_number: Singularity proximity

Example:
    >>> from shared.python.manipulability import compute_manipulability_ellipsoid
    >>> result = compute_manipulability_ellipsoid(
    ...     jacobian=jacobian_6dof,
    ...     frame='end_effector'
    ... )
    >>> print(f"Yoshikawa measure: {result.yoshikawa:.4f}")

References:
    [1] Yoshikawa (1985) - Manipulability of Robotic Mechanisms
    [2] Siciliano & Khatib - Robotics Handbook, Ch 7
"""
```

### Physics Function Docstring

```python
def compute_inverse_dynamics(
    state: RobotState,
    accelerations: np.ndarray,
    external_forces: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Compute joint torques required to produce desired accelerations.

    Uses the recursive Newton-Euler algorithm to compute the torques
    needed to achieve the specified joint accelerations given
    current state and external forces.

    Mathematical Formulation:
        τ = M(q)⋅q̈ + C(q,q̇)⋅q̇ + g(q) - Jᵀ⋅f_ext

        Where:
        - M: Mass matrix [n×n]
        - C: Coriolis matrix [n×n]
        - g: Gravity vector [n]
        - J: Contact Jacobian [6×n]
        - f_ext: External wrench [6]

    Args:
        state: Current robot state with positions q [rad] and
            velocities qd [rad/s].
        accelerations: Desired joint accelerations qdd [rad/s²].
            Shape must match DOF count.
        external_forces: Optional dict mapping body names to
            6D wrenches [N, N⋅m]. Default None (no external forces).

    Returns:
        Joint torques τ [N⋅m] as 1D array of shape (n_dof,).

    Raises:
        ValueError: If acceleration shape doesn't match DOF.
        PhysicsError: If dynamics computation fails.

    Example:
        >>> state = RobotState(q=q_current, qd=qd_current)
        >>> qdd_desired = np.zeros(7)  # 7-DOF arm
        >>> tau = engine.compute_inverse_dynamics(state, qdd_desired)
        >>> print(f"Shoulder torque: {tau[0]:.2f} N⋅m")

    Note:
        For static equilibrium, set accelerations to zero.
        Result includes gravity compensation.
    """
```

### Cross-Engine Validation Documentation

````markdown
# Cross-Engine Validation Protocol

## Overview

This document describes the validation procedures for ensuring
consistency across physics engines (MuJoCo, Drake, Pinocchio).

## Tolerance Targets

Per Design Guidelines Section M:

| Quantity | Tolerance | Units  |
| -------- | --------- | ------ |
| Position | ±1e-6     | m      |
| Velocity | ±1e-5     | m/s    |
| Torque   | ±1e-3     | N⋅m    |
| Jacobian | ±1e-8     | varies |

## Validation Test Cases

### 1. Simple Pendulum

Known analytical solution for comparison:

- Period: T = 2π√(L/g)
- Energy conservation check

### 2. N-Link Planar Arm

Validated against:

- Closed-form Jacobian
- Analytical mass matrix

### 3. Full Humanoid

Cross-validated between all engines:

- Forward dynamics consistency
- Inverse dynamics consistency
- Jacobian agreement

## Running Validation

```bash
pytest tests/test_cross_engine.py -v
```
````

````

---

## Quality Checklist

Before completing documentation:

- [ ] All physics modules have comprehensive docstrings
- [ ] Coordinate frames are specified for all spatial quantities
- [ ] Units are specified for all physical quantities
- [ ] Cross-engine differences are documented
- [ ] Examples are runnable and tested
- [ ] Validation tolerances are specified
- [ ] Error conditions are documented
- [ ] Cross-references between related modules exist

---

## Verification Commands

```bash
# Check docstring coverage
pydocstyle --convention=google shared/python/

# Verify imports
python -c "from shared.python.physics_engine import PhysicsEngine"

# Run validation tests
pytest tests/test_validation.py -v

# Type checking
mypy shared/python/ --strict
````

---

## Success Criteria

1. ✅ All physics engines have complete documentation
2. ✅ Design Guidelines accurately reflects implementation
3. ✅ Cross-engine validation is documented
4. ✅ Coordinate frames are specified everywhere
5. ✅ All physical constants have units
6. ✅ Getting-started guides exist for each engine
7. ✅ No "TODO" or placeholder text
8. ✅ Examples produce expected output

---

## Special Considerations for Golf Modeling Suite

### Physics Documentation Requirements

1. **All spatial quantities must specify:**
   - Coordinate frame (world, body, end-effector)
   - Units (SI: m, rad, N, N⋅m)
   - Sign conventions

2. **Cross-engine documentation must include:**
   - Which engines support which features
   - Known numerical differences
   - Fallback strategies

3. **Biomechanical context:**
   - Anatomical terminology where relevant
   - Muscle-moment arm relationships
   - Ground reaction force conventions

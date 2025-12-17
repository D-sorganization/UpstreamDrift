# Golf Swing Analysis Workflow Plan

## Overview

This document outlines the step-by-step workflow for analyzing golf swings using Rob Neal data, inverse kinematics (PINK), inverse dynamics (Pinocchio), forward dynamics simulation, and counterfactual analysis (ZVCF, ZTCF).

## Workflow Summary

```
Rob Neal Data → Target Kinematics → IK (PINK) → Joint Angles → 
Inverse Dynamics → Joint Torques → Forward Dynamics → Motion Match → 
Counterfactual Analysis (ZVCF, ZTCF)
```

## Detailed Workflow Steps

### Step 1: Extract Target Kinematics from Rob Neal Data

**Goal**: Convert Rob Neal `.mat` data into target kinematics for the club.

**Inputs**:
- Rob Neal `.mat` files (e.g., `TW_wiffle.mat`, `TW_ProV1.mat`)
- Data structure: `data.midhands_xyz`, `data.clubface_xyz`, `data.time`

**Process**:
1. Load `.mat` file using `RobNealDataViewer` or `matlab_importer`
2. Extract time series:
   - Position: `clubface_xyz(t)` [3D]
   - Orientation: Compute from club shaft direction (midhands → clubface)
   - Velocity: `clubface_xyz_vel(t)` or differentiate position
   - Acceleration: Differentiate velocity or compute from `clubface_xyz_acc(t)` if available
3. Optionally extract hand kinematics: `midhands_xyz(t)`

**Outputs**:
- `target_positions`: `np.ndarray` shape `(N, 3)` - clubface positions over time
- `target_orientations`: `np.ndarray` shape `(N, 4)` - quaternions
- `target_velocities`: `np.ndarray` shape `(N, 3)` - linear velocities
- `target_angular_velocities`: `np.ndarray` shape `(N, 3)` - angular velocities
- `target_accelerations`: `np.ndarray` shape `(N, 3)` - linear accelerations
- `time`: `np.ndarray` shape `(N,)` - time vector

**Implementation**:
- Module: `dtack.data.rob_neal_processor`
- Class: `RobNealKinematicsExtractor`

---

### Step 2: Solve Inverse Kinematics (PINK)

**Goal**: Find feasible joint angle trajectories that achieve the target club kinematics.

**Inputs**:
- Target kinematics from Step 1
- Pinocchio model (URDF) with constraints (hand-to-club)
- Initial joint configuration `q_init`

**Process**:
1. **Task Definition**:
   - Primary task: Clubface position + orientation
   - Constraints: Hand-to-club rigid constraints (6 DOF locked)
2. **IK Solver Setup**:
   - Load model in PINK
   - Define task frames: `clubface_frame`, `right_hand_frame`, `left_hand_frame`, `club_grip_frame`
   - Set joint limits from URDF
3. **Solve IK for each time step**:
   - For `t_i` in time series:
     - Set target: `target_positions[i]`, `target_orientations[i]`
     - Solve IK: `q[i] = PINK.solve_ik(tasks, q[i-1])`
     - Check feasibility: joint limits, constraint satisfaction

**Outputs**:
- `q_trajectory`: `np.ndarray` shape `(N, nq)` - joint angles over time
- `qdot_trajectory`: `np.ndarray` shape `(N, nv)` - joint velocities
- `qddot_trajectory`: `np.ndarray` shape `(N, nv)` - joint accelerations
- `ik_success`: `np.ndarray` shape `(N,)` - boolean array indicating IK convergence

**Implementation**:
- Module: `dtack.ik.pink_ik_solver`
- Class: `PINKIKSolver`

---

### Step 3: Compute Inverse Dynamics (Pinocchio)

**Goal**: Compute joint torques required to achieve the joint angle trajectories.

**Inputs**:
- `q_trajectory`, `qdot_trajectory`, `qddot_trajectory` from Step 2
- Pinocchio model with constraints

**Process**:
1. Load model in Pinocchio with constraints: `RigidConstraintModel` for hand-to-club
2. For each time step:
   - Compute inverse dynamics: `tau[i] = rnea(model, data, q[i], qdot[i], qddot[i])`
   - Account for constraints: Add constraint torques

**Outputs**:
- `tau_trajectory`: `np.ndarray` shape `(N, nv)` - joint torques over time
- `constraint_forces`: `np.ndarray` shape `(N, n_constraints)` - constraint forces

**Implementation**:
- Module: `dtack.dynamics.inverse_dynamics`
- Class: `InverseDynamicsSolver`

---

### Step 4: Forward Dynamics Simulation

**Goal**: Simulate the motion using computed joint torques and verify it matches target kinematics.

**Inputs**:
- `tau_trajectory` from Step 3
- Initial conditions: `q[0]`, `qdot[0]`
- Pinocchio model (or MuJoCo for contact simulation)

**Process**:
1. Choose backend: Pinocchio (fast) or MuJoCo (contact)
2. Simulation loop:
   - For `t_i` in time series:
     - Apply torques: `tau[i]`
     - Integrate forward dynamics: `q[i+1], qdot[i+1] = integrate(q[i], qdot[i], tau[i], dt)`
     - Check constraints: Ensure hand-to-club constraint is satisfied
3. Validation:
   - Compare simulated club kinematics to target
   - Compute error metrics

**Outputs**:
- `q_simulated`: `np.ndarray` shape `(N, nq)` - simulated joint angles
- `club_kinematics_simulated`: Simulated club position/orientation/velocity
- `error_metrics`: Dictionary of position/orientation/velocity errors

**Implementation**:
- Module: `dtack.simulation.forward_dynamics`
- Class: `ForwardDynamicsSimulator`

---

### Step 5: Counterfactual Analysis (ZVCF, ZTCF)

**Goal**: Explore relative contributions of drift (passive dynamics) and control (active torques).

**Theory**:
- **ZVCF (Zero Velocity Counterfactual)**: Set `qdot = 0`, keep `tau` (active control)
- **ZTCF (Zero Torque Counterfactual)**: Set `tau = 0`, keep `qdot` (passive dynamics)

**Process**:
1. **ZVCF**: Simulate with `qdot = 0`, `tau` from Step 3
2. **ZTCF**: Simulate with `tau = 0`, `qdot` from Step 2
3. **Decomposition**:
   - Control contribution: `||q_actual - q_ztcf||`
   - Drift contribution: `||q_actual - q_zvcf||`
   - Coupling: `q_actual - q_zvcf - q_ztcf`

**Outputs**:
- `q_zvcf`: ZVCF trajectory
- `q_ztcf`: ZTCF trajectory
- `control_contribution`: Control contribution over time
- `drift_contribution`: Drift contribution over time

**Implementation**:
- Module: `dtack.analysis.counterfactuals`
- Class: `CounterfactualAnalyzer`

---

## Workflow Evaluation

### Strengths
1. **Systematic**: Clear progression from data → kinematics → dynamics → analysis
2. **Physics-grounded**: Uses proper inverse/forward dynamics
3. **Research-grade**: Counterfactual analysis enables deep insights

### Recommended Enhancements
1. **Step 1**: Also target hand positions to better constrain IK
2. **Step 2**: Use incremental IK solving with smoothness penalty
3. **Step 3**: Include GRF estimation from foot contact
4. **Step 4**: Iterative refinement until convergence
5. **Step 5**: Time-windowed counterfactuals (backswing, downswing)

### Implementation Priority
- **Phase 1**: Complete URDF, Rob Neal loader, basic IK, inverse dynamics
- **Phase 2**: Forward dynamics, constraint handling, validation
- **Phase 3**: Counterfactual analysis, visualization, documentation

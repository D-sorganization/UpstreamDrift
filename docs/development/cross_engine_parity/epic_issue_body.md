## Epic: Cross-Engine Physics Equivalency (Simscape, MuJoCo, Pinocchio, Drake)

### Objective

Establish complete functional and dynamical equivalency across all four physics engines in the fleet:

1. **Simscape Multibody** (`GolfSwing3D_Kinetic.slx`, ground-truth baseline oracle)
2. **MuJoCo** (MJCF / XML high-speed forward dynamics)
3. **Pinocchio** (URDF rigid-body dynamics & analytical Jacobians)
4. **Drake** (MultibodyPlant mathematical programming & verification)

Any continuous polynomial torque profile $\tau(t) = \sum_{k=0}^6 c_k B_{k,6}(t)$ fitted against measured tour motion capture (`data/C3D_TA_Driver.c3d`) in Simscape must be directly executable in MuJoCo, Pinocchio, and Drake, producing identical forward dynamics, joint angle trajectories, and anatomical marker kinematics ($< 5\text{ mm}$ RMSE vs Simscape).

---

### Core Architecture & Technical Requirements

- **Authoritative Specification**: [`docs/development/cross_engine_parity/CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC.md`](docs/development/cross_engine_parity/CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC.md)
- **Canonical Model Definition**: Derived directly from `golfer_canonical.yaml`, synchronizing link lengths, masses, centers of mass, inertia tensors, and joint degree-of-freedom limits.
- **Unified URDF / MJCF Pipeline**:
  - `golfer.urdf` for Pinocchio and Drake.
  - `golfer.xml` for MuJoCo.
- **Common Forward-Dynamics Simulation API**:
  Every engine implements:
  ```python
  def simulate_with_coefficients(
      theta: np.ndarray,            # (27, 7) Bernstein polynomial torque coefficients
      options: SimOptions = ...,    # Solver tolerances & time horizons
      initial_pose: dict | None = None, # (q_0, qd_0)
  ) -> SimOut:
  ```
- **Equivalence Benchmark Gate**: Round-trip fixed benchmark $\theta^*$ to within **$< 5.0\text{ mm}$ grip RMSE** and **$< 10.0\text{ mm}$ clubhead RMSE** vs Simscape reference across the entire swing.

---

### Work Packages & Child Issues

- [ ] **WP1**: URDF / MJCF Exporter Synchronization from `golfer_canonical.yaml`
- [ ] **WP2**: MuJoCo Continuous-Torque Simulation Harness (`src/engines/physics_engines/mujoco`)
- [ ] **WP3**: Pinocchio Articulated-Body Simulation Harness (`src/engines/physics_engines/pinocchio`)
- [ ] **WP4**: Drake MultibodyPlant Simulation Harness (`src/engines/physics_engines/drake`)
- [ ] **WP5**: Automated Cross-Engine Parity Benchmark Suite (`tests/cross_engine/`)

Parent Epic: #9921  
Cross-repo references: `MuJoCo_Models`, `Pinocchio_Models`, `Drake_Models`

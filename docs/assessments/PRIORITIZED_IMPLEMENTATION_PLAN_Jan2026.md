# Golf Modeling Suite - REVISED Prioritized Implementation Plan
**Date:** 2026-01-06 (Updated after user feedback)  
**Based On:** Assessments A, B, C + Project Design Guidelines + User Priorities  
**Top Priority:** Drift-Control Decomposition for All 5 Engines

---

## User-Directed Priorities (Confirmed 2026-01-06)

1. **HIGHEST PRIORITY**: Implement drift-control decomposition for all 5 engines
2. Ensure complete OpenSim integration
3. Add grip modeling to OpenSim and MyoSuite (not just MuJoCo)
4. Utilize Drake's block diagram capabilities
5. Understand and implement indexed acceleration closure

---

## What is "Indexed Acceleration Closure"? (Explained)

**The Physics:**

When you break down (decompose) the total acceleration of a joint into its causes, the sum of those causes **MUST** equal the total. This isn't a "nice to have" - it's fundamental physics.

**Example - Elbow Joint During Golf Swing:**

```
Total elbow acceleration = 15.7 rad/s² (measured via forward dynamics)

Decomposed components:
├─ Gravity pulling arm down:         3.2 rad/s²
├─ Coriolis from shoulder rotation:  2.1 rad/s²
├─ Bicep muscle contraction:          8.9 rad/s²
├─ Tricep muscle (negative):         -1.2 rad/s²
└─ Constraint from wrist grip:        2.7 rad/s²
                                    ─────────
                           Sum:      15.7 rad/s²  ✓ CLOSURE VERIFIED
```

If the sum was 14.3 rad/s² instead, you'd have **1.4 rad/s² missing** - meaning either:
- You forgot a force component
- Your physics engine has a bug
- Your decomposition math is wrong

**Why "Closure"?**
"Closure" means the components "close the loop" back to the total - no gap, no residual.

**Mathematical Requirement:**

```
Residual = Total - (Gravity + Coriolis + Applied + Constraint + External)

PASS: ||Residual|| < 1e-6 rad/s²  (essentially zero, allowing for floating-point error)
FAIL: ||Residual|| > 1e-4 rad/s²  (significant missing physics)
```

**Why It Matters for Golf:**

If you claim "the lead arm contributed 42% of clubhead speed," that percentage is **only valid** if closure holds. Without closure, your percentages are guesses.

---

## PRIORITY 1: Drift-Control Decomposition - All 5 Engines (Week 1-2)

**Goal:** Every engine can separate passive (drift) from active (control) dynamics.

### Implementation Order (Easiest → Hardest)

#### 1.1 Pinocchio (Day 1-2) - EASIEST

**Why easiest:** Pinocchio's ABA (Articulated Body Algorithm) explicitly computes `C(q,v)v + g(q)`.

```python
# engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py

class PinocchioEngine(PhysicsEngine):
    
    def compute_drift_acceleration(self) -> np.ndarray:
        """Section F: Drift = passive dynamics with zero control."""
        import pinocchio as pin
        
        q = self.data.q
        v = self.data.v
        tau_zero = np.zeros(self.model.nv)
        
        # ABA with zero torque = pure drift
        a_drift = pin.aba(self.model, self.data, q, v, tau_zero)
        
        return a_drift
    
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Section F: Control = acceleration from applied torques only."""
        M = pin.crba(self.model, self.data, self.data.q)
        M_inv = np.linalg.inv(M)
        
        a_control = M_inv @ tau
        
        return a_control
    
    def verify_superposition(self, tau: np.ndarray) -> bool:
        """Test: drift + control = full dynamics."""
        a_full = pin.aba(self.model, self.data, self.data.q, self.data.v, tau)
        a_drift = self.compute_drift_acceleration()
        a_control = self.compute_control_acceleration(tau)
        
        residual = a_full - (a_drift + a_control)
        return np.max(np.abs(residual)) < 1e-6
```

**Test:**
```python
def test_pinocchio_drift_control_superposition():
    engine = PinocchioEngine("simple_pendulum.urdf")
    engine.set_state(q=[0.1], v=[0.0])
    
    tau = np.array([0.5])
    assert engine.verify_superposition(tau), "Pinocchio superposition failed"
```

#### 1.2 MuJoCo (Day 3-4)

**Implementation:**

```python
# engines/physics_engines/mujoco/python/mujoco_physics_engine.py
import mujoco

class MuJoCoEngine(PhysicsEngine):
    
    def compute_drift_acceleration(self) -> np.ndarray:
        """Section F: MuJoCo drift via mj_forward with zero control."""
        # Save current control
        ctrl_saved = self.data.ctrl.copy()
        
        # Zero all actuators
        self.data.ctrl[:] = 0.0
        
        # Forward dynamics with zero control = drift only
        mujoco.mj_forward(self.model, self.data)
        a_drift = self.data.qacc.copy()
        
        # Restore control
        self.data.ctrl[:] = ctrl_saved
        mujoco.mj_forward(self.model, self.data)  # Restore state
        
        return a_drift
    
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Section F:Control component via mass matrix inversion."""
        mujoco.mj_forward(self.model, self.data)
        
        # Get mass matrix
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        
        # a_control = M^-1 * tau
        M_inv = np.linalg.inv(M)
        a_control = M_inv @ tau
        
        return a_control
```

#### 1.3 Drake (Day 5-6) - Use Block Diagram

**Implementation:** Create a `DriftControlDecomposerSystem`

```python
# engines/physics_engines/drake/python/drift_control_system.py
from pydrake.systems.framework import LeafSystem, BasicVector

class DriftControlDecomposerSystem(LeafSystem):
    """Section F: Drake system for drift-control decomposition."""
    
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self.plant = plant
        
        # Input ports
        self.q_port = self.DeclareVectorInputPort("q", plant.num_positions())
        self.v_port = self.DeclareVectorInputPort("v", plant.num_velocities())
        self.tau_port = self.DeclareVectorInputPort("tau", plant.num_actuators())
        
        # Output ports
        self.DeclareVectorOutputPort(
            "q_ddot_drift", 
            plant.num_velocities(),
            self._compute_drift_output
        )
        self.DeclareVectorOutputPort(
            "q_ddot_control",
            plant.num_velocities(),
            self._compute_control_output
        )
    
    def _compute_drift_output(self, context, output):
        """Compute drift acceleration (zero control)."""
        q = self.q_port.Eval(context)
        v = self.v_port.Eval(context)
        
        plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(plant_context, q)
        self.plant.SetVelocities(plant_context, v)
        
        # Zero control input
        self.plant.get_actuation_input_port().FixValue(
            plant_context, np.zeros(self.plant.num_actuators())
        )
        
        a_drift = self.plant.EvalTimeDerivatives(plant_context).get_vector().CopyToVector()[self.plant.num_positions():]
        
        output.SetFromVector(a_drift)
    
    def _compute_control_output(self, context, output):
        """Compute control acceleration (M^-1 * tau)."""
        tau = self.tau_port.Eval(context)
        
        # Get mass matrix from plant
        plant_context = self.plant.CreateDefaultContext()
        M = self.plant.CalcMassMatrix(plant_context)
        
        a_control = np.linalg.solve(M, tau)
        output.SetFromVector(a_control)
```

**Wire into diagram:**

```python
# tests/acceptance/test_drake_drift_control.py

def test_drake_drift_control_block_diagram():
    """Section F + B4a: Verify Drake block diagram drift-control decomposition."""
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    
    Parser(plant).AddModelFromFile("simple_pendulum.urdf")
    plant.Finalize()
    
    # Add drift-control decomposer system
    decomposer = builder.AddSystem(DriftControlDecomposerSystem(plant))
    
    # Build diagram
    diagram = builder.Build()
    simulator = Simulator(diagram)
    
    # Test superposition
    context = simulator.get_mutable_context()
    plant.SetPositions(context, [0.1])
    plant.SetVelocities(context, [0.0])
    
    tau = np.array([0.5])
    
    # Get outputs from decomposer
    a_drift = decomposer.GetOutputPort("q_ddot_drift").Eval(context)
    a_control = decomposer.GetOutputPort("q_ddot_control").Eval(context)
    
    # Compare to full dynamics
    plant.get_actuation_input_port().FixValue(context, tau)
    a_full = plant.EvalTimeDerivatives(context).get_vector().CopyToVector()[1]  # Skip position derivative
    
    # Superposition test
    assert abs((a_drift + a_control) - a_full) < 1e-6
```

#### 1.4 OpenSim (Day 7-9)

**Implementation:**

```python
# engines/physics_engines/opensim/python/opensim_physics_engine.py
import opensim

class OpenSimEngine(PhysicsEngine):
    
    def compute_drift_acceleration(self) -> np.ndarray:
        """Section F: OpenSim drift via zero muscle activation."""
        state = self.model.initSystem()
        
        # Zero all muscle activations
        muscles = self.model.getMuscles()
        for i in range(muscles.getSize()):
            muscle = opensim.Muscle.safeDownCast(muscles.get(i))
            muscle.setActivation(state, 0.0)
        
        # Compute acceleration with zero activation = drift
        self.model.realizeDynamics(state)
        a_drift = state.getUDot()  # Generalized accelerations
        
        return np.array([a_drift.get(i) for i in range(a_drift.size())])
    
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Section F: Muscle/torque-driven acceleration."""
        # For muscle-driven: compute M^-1 * (sum of muscle forces)
        # For torque-driven: compute M^-1 * tau
        
        state = self.model.initSystem()
        M = self.model.getMatterSubsystem().calcM(state)
        M_np = np.array([[M.get(i, j) for j in range(M.ncol())] for i in range(M.nrow())])
        
        a_control = np.linalg.solve(M_np, tau)
        
        return a_control
```

#### 1.5 MyoSuite (Day 10-12)

**Implementation:**

```python
# engines/physics_engines/myosuite/python/myosuite_physics_engine.py
from myosuite.envs import gym

class MyoSuiteEngine(PhysicsEngine):
    
    def compute_drift_acceleration(self) -> np.ndarray:
        """Section F: MyoSuite drift via zero action."""
        # Save current state
        qpos_saved = self.env.sim.data.qpos.copy()
        qvel_saved = self.env.sim.data.qvel.copy()
        
        # Step with zero action (no muscle activation)
        zero_action = np.zeros(self.env.action_space.shape)
        self.env.step(zero_action)
        
        a_drift = self.env.sim.data.qacc.copy()
        
        # Restore state
        self.env.sim.data.qpos[:] = qpos_saved
        self.env.sim.data.qvel[:] = qvel_saved
        self.env.sim.forward()
        
        return a_drift
    
    def compute_control_acceleration(self, muscle_activations: np.ndarray) -> np.ndarray:
        """Section F: Muscle-driven acceleration component."""
        # Compute muscle forces from activations
        muscle_forces = self._activations_to_forces(muscle_activations)
        
        # Map muscle forces to joint torques via moment arms
        joint_torques = self._muscle_forces_to_joint_torques(muscle_forces)
        
        # M^-1 * tau
        M = np.zeros((self.env.model.nv, self.env.model.nv))
        mujoco.mj_fullM(self.env.model, M, self.env.sim.data.qM)
        
        a_control = np.linalg.solve(M, joint_torques)
        
        return a_control
```

### Deliverable (End of Week 2)

- [ ] All 5 engines implement `compute_drift_acceleration()`
- [ ] All 5 engines implement `compute_control_acceleration()`
- [ ] All 5 engines pass `test_superposition_drift_plus_control_equals_full()`
- [ ] Drake implementation uses block diagram architecture (Section B4a)
- [ ] Documentation: `docs/features/drift_control_decomposition.md`

---

## PRIORITY 2: Complete OpenSim Integration (Week 3-4)

**Goal:** OpenSim fully operational per Section J requirements.

### 2.1 Component Checklist

| Component | Status | Target |
|-----------|--------|--------|
| Hill-type muscle models | ⚠️ Unknown | Functional with activation dynamics |
| Tendon compliance | ⚠️ Unknown | Togglable in model files |
| Wrapping geometry for grip | ❌ Missing | Cylinder/ellipsoid wraps implemented |
| Activation → force → torque | ⚠️ Unknown | Pipeline tested end-to-end |
| Induced acceleration (muscle) | ❌ Missing | Muscle contributions to joint accel |
| Muscle force/power logging | ⚠️ Unknown | CSV export per muscle |
| Grip modeling (Section J1) | ❌ Missing | Via-points + constraint reactions |

### 2.2 Implementation Tasks

#### Task 2.2a: Verify Hill-Type Muscle Models (Day 1-2)

```python
# tests/integration/test_opensim_muscles.py

def test_hill_muscle_force_length_curve():
    """Section J: Validate Hill-type F-L relationship."""
    model = opensim.Model("arm_with_muscles.osim")
    state = model.initSystem()
    
    biceps = opensim.Muscle.safeDownCast(model.getMuscles().get("biceps"))
    
    # Set activation to 1.0 (fully active)
    biceps.setActivation(state, 1.0)
    
    # Test at optimal length
    biceps.setLength(state, biceps.getOptimalFiberLength())
    model.realizeDynamics(state)
    
    F_optimal = biceps.getActiveFiberForce(state)
    F_max = biceps.getMaxIsometricForce()
    
    # At optimal length with full activation, F ≈ F_max
    assert abs(F_optimal - F_max) / F_max < 0.05  # Within 5%
```

#### Task 2.2b: Implement Grip Wrapping Geometry (Days 3-5)

```xml
<!-- models/opensim/golfer_with_grip.osim -->
<OpenSimDocument Version="40000">
  <Model>
    <BodySet>
      <Body name="club_grip">
        <WrapObjectSet>
          <WrapCylinder name="grip_cylinder">
            <dimensions>0.025 0.15</dimensions> <!-- radius 2.5cm, length 15cm -->
            <xyz_body_rotation>0 1.5708 0</xyz_body_rotation> <!-- 90° rotation -->
          </WrapCylinder>
        </WrapObjectSet>
      </Body>
    </BodySet>
    
    <ForceSet>
      <Muscle name="flexor_digitorum_left">
        <GeometryPath>
          <PathPointSet>
            <PathPoint name="fdp_origin" body="forearm_left" location="-0.02 -0.15 0.01"/>
            <PathPoint name="fdp_grip" body="club_grip" location="0 0.05 0.025"/>
          </PathPointSet>
          <PathWrapSet>
            <PathWrap>
              <wrap_object>grip_cylinder</wrap_object>
              <method>hybrid</method>
            </PathWrap>
          </PathWrapSet>
        </GeometryPath>
        <max_isometric_force>500.0</max_isometric_force>
        <optimal_fiber_length>0.08</optimal_fiber_length>
      </Muscle>
    </ForceSet>
  </Model>
</OpenSimDocument>
```

#### Task 2.2c: Muscle-to-Joint-Torque Pipeline (Day 6-7)

```python
# shared/python/opensim_utils.py

def compute_muscle_induced_accelerations(
    model: opensim.Model, 
    state: opensim.State
) -> dict[str, np.ndarray]:
    """Section J: Induced acceleration from each muscle."""
    
    muscles = model.getMuscles()
    muscle_contributions = {}
    
    for i in range(muscles.getSize()):
        muscle = opensim.Muscle.safeDownCast(muscles.get(i))
        muscle_name = muscle.getName()
        
        # Get muscle force
        F_muscle = muscle.getActuation(state)
        
        # Get moment arms for all joints
        coords = model.getCoordinateSet()
        torques = np.zeros(coords.getSize())
        
        for j in range(coords.getSize()):
            coord = coords.get(j)
            moment_arm = muscle.computeMomentArm(state, coord)
            torques[j] = F_muscle * moment_arm
        
        # M^-1 * tau_muscle = induced acceleration from this muscle
        M = model.getMatterSubsystem().calcM(state)
        M_np = np.array([[M.get(r, c) for c in range(M.ncol())] for r in range(M.nrow())])
        
        a_induced = np.linalg.solve(M_np, torques)
        muscle_contributions[muscle_name] = a_induced
    
    return muscle_contributions
```

### 2.3 Validation Tests

```python
def test_opensim_muscle_contribution_closure():
    """Section H2 + J: Sum of muscle contributions = total acceleration."""
    model, state = setup_arm_model_with_biceps_triceps()
    
    # Set muscle activations
    set_muscle_activation(model, state, "biceps", 0.8)
    set_muscle_activation(model, state, "triceps", 0.2)
    
    # Compute total acceleration
    model.realizeDynamics(state)
    a_total = np.array([state.getUDot().get(i) for i in range(state.getNU())])
    
    # Compute indexed components
    muscle_contribs = compute_muscle_induced_accelerations(model, state)
    a_gravity = compute_gravity_induced_acceleration(model, state)
    a_coriolis = compute_coriolis_induced_acceleration(model, state)
    
    # Closure test
    a_reconstructed = (
        muscle_contribs["biceps"] + muscle_contribs["triceps"]
        + a_gravity + a_coriolis
    )
    
    residual = a_total - a_reconstructed
    assert np.max(np.abs(residual)) < 1e-4, "OpenSim indexed acceleration closure failed"
```

### Deliverable (End of Week 4)

- [ ] Hill-type muscles operational (F-L-V curves validated)
- [ ] Grip wrapping geometry implemented (3+ muscles routing through grip)
- [ ] Muscle → joint torque → induced acceleration pipeline working
- [ ] OpenSim passes indexed acceleration closure test (Section H2)
- [ ] Documentation: `docs/engines/opensim_complete_integration.md`

---

## PRIORITY 3: Grip Modeling - All Engines (Week 5)

**Goal:** Implement grip modeling for OpenSim, MyoSuite, Drake, Pinocchio (MuJoCo already has K2a spec).

Expanded requirements are in `DESIGN_GUIDELINES_ADDENDUM_Drake_Grip_Jan2026.md`.

### Quick Reference:

- **OpenSim** (Section J1): Wrapping geometry + constraint reactions
- **MyoSuite** (Section K1): Fingertip contacts + activation-driven force
- **Drake** (Section K2b): Compliant contact (Hunt-Crossley)
- **Pinocchio** (Section K2c): Bilateral constraints + Lagrange multipliers
- **MuJoCo** (Section K2a): Contact pairs + friction cones (existing spec)

**Cross-Engine Validation (Mandatory):**
- Static test: All engines show club weight supported by grip forces
- Dynamic test: No slip when centrifugal force < μ*N
- Tolerance: Grip force magnitude ±15% between engines
- Impulse test: ∫F_grip dt matches across engines (±10%)

---

## PRIORITY 4: Indexed Acceleration Closure Tests (Week 6)

**Goal:** Automate closure verification for all engines.

### 4.1 Shared Test Infrastructure

```python
# tests/acceptance/test_indexed_acceleration_closure.py

@pytest.mark.parametrize("engine_name", ["mujoco", "drake", "pinocchio", "opensim", "myosuite"])
def test_indexed_acceleration_closure(engine_name):
    """Section H2: Universal closure test for all engines."""
    
    engine = load_engine(engine_name, "simple_pendulum.urdf")
    engine.set_state(q=[0.1], v=[0.0])
    
    tau = np.array([0.5])
    engine.set_control(tau)
    
    # Forward dynamics (ground truth)
    a_total = engine.compute_forward_dynamics()
    
    # Indexed decomposition
    indexed = engine.compute_indexed_acceleration(tau)
    
    # Reconstruct
    a_reconstructed = (
        indexed.gravity + indexed.coriolis + indexed.applied_torque
        + indexed.constraint + indexed.external
    )
    
    # Closure requirement
    residual = a_total - a_reconstructed
    
    assert np.max(np.abs(residual)) < 1e-6, \
        f"{engine_name}: Closure failed with residual {np.max(np.abs(residual)):.2e}"
```

### 4.2 CI Integration

```yaml
# .github/workflows/indexed-acceleration-closure.yml
name: Indexed Acceleration Closure (Section H2)

on: [push, pull_request]

jobs:
  closure-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        engine: [mujoco, drake, pinocchio, opensim, myosuite]
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e .[engines]
      
      - name: Run closure tests
        run: |
          pytest tests/acceptance/test_indexed_acceleration_closure.py \
            -k "test_indexed_acceleration_closure[${{ matrix.engine }}]" \
            -v --tb=short
```

---

## Summary Timeline

| Week | Priority | Deliverable | Status Gate |
|------|----------|-------------|-------------|
| **1-2** | Drift-Control (5 engines) | All engines pass superposition test | BLOCKER |
| **3-4** | Complete OpenSim | Muscle models + grip + closure test | CRITICAL |
| **5** | Grip modeling (all engines) | Cross-engine grip validation | CRITICAL |
| **6** | Closure tests + CI | Automated H2 verification | CRITICAL |

**After 6 weeks:**
- ✅ Drift-control decomposition operational (Section F)
- ✅ OpenSim fully integrated (Section J)
- ✅ Grip modeling across all engines (Sections J1, K1, K2)
- ✅ Drake utilizing block diagrams (Section B4a)
- ✅ Indexed acceleration closure automated (Section H2)
- ✅ CI enforcing closure requirements

**Then proceed to:**
- Flexible shaft (Section B5)
- Swing plane / FSP (Section E4)
- Ground reaction forces (Section E5)
- Modular impact model (Section K3)

---

**Signed:** Senior Principal Engineer  
**Approved by:** User Feedback 2026-01-06  
**Next Checkpoint:** End of Week 2 (drift-control completion)

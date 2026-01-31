# Golf Modeling Suite - Design Guidelines Addendum
## Drake Block Diagram Architecture & Enhanced Requirements
**Date:** 2026-01-06  
**Supersedes:** Sections added to project_design_guidelines.qmd

---

## Section B4a: Drake-Specific Block Diagram Architecture (MANDATORY)

**Drake's systems framework must be fully utilized for modularity and composability.**

### B4a.1 Block Diagram Requirements

Drake's `LeafSystem` and `Diagram` infrastructure enables composable, modular simulation architecture. **All Drake implementations must use this framework:**

- **System Decomposition**:
  - Player model: `MultibodyPlant` (skeleton + club)
  - Controller: `LeafSystem` (PID, LQR, trajectory tracker, or RL policy)
  - Sensors: `LeafSystem` (joint encoders, IMU, force plates)
  - Visualizer: `MeshcatVisualizer` or `DrakeVisualizer`
  - Logger: `VectorLogSink` for state/control history

- **Block Diagram Structure** (mandatory):
  ```
  ┌─────────────────────────────────────────────────────┐
  │ SwingSimulationDiagram                               │
  │                                                       │
  │  ┌────────────┐   τ    ┌──────────────┐   q,v      │
  │  │ Controller │──────> │ MultibodyPlant│───────┐   │
  │  │ (PID/LQR)  │        │ (golfer+club) │       │   │
  │  └────────────┘        └──────────────┘       │   │
  │        ↑                      │                │   │
  │        │                      │ GRF            │   │
  │        │ q_des,v_des          ↓                │   │
  │  ┌────────────┐        ┌──────────────┐       │   │
  │  │ Trajectory │        │ Ground Contact│       │   │
  │  │  Planner   │        │  Forces       │       │   │
  │  └────────────┘        └──────────────┘       │   │
  │                                                 │   │
  │  ┌────────────┐        ┌──────────────┐       │   │
  │  │   Logger   │<───────┤  Visualizer  │<──────┘   │
  │  │(state/ctrl)│        │  (Meshcat)   │           │
  │  └────────────┘        └──────────────┘           │
  └─────────────────────────────────────────────────────┘
  ```

### B4a.2 Required Drake Systems

**Each must be implemented as standalone, reusable `LeafSystem` blocks:**

1. **GolferPlantSystem**:
   - Wraps `MultibodyPlant` with golfer-specific ports
   - Input ports: `u_torques`, `u_muscle_activations` (if hybrid model)
   - Output ports: `q`, `v`, `qacc`, `joint_torques`, `reaction_forces`

2. **TrajectoryControllerSystem**:
   - Input ports: `q_measured`, `v_measured`, `q_desired`, `v_desired`
   - Output ports: `u_control` (torques or activations)
   - Modes: PID, inverse dynamics, LQR, MPC preview

3. **InducedAccelerationAnalyzerSystem**:
   - Input ports: `q`, `v`, `u`
   - Output ports: `indexed_accelerations` (struct with gravity/coriolis/control/constraint components)
   - Implements Section H2 indexed acceleration closure logic

4. **DriftControlDecomposerSystem** (Section F):
   - Input ports: `q`, `v`, `u`
   - Output ports: `q_ddot_drift`, `q_ddot_control`
   - Validates: `q_ddot_drift + q_ddot_control ≈ q_ddot_total` (closure test)

5. **CounterfactualSimulatorSystem** (Section G):
   - Switches between normal/ZTCF/ZVCF modes via discrete state
   - ZTCF mode: Forces `u = 0` at input port
   - ZVCF mode: Resets velocities to zero at every step

### B4a.3 Composability Requirements

- **Plug-and-Play Controllers**: Users must be able to swap PID ↔ LQR ↔ RL policy without changing plant/logger code
- **Sensor Flexibility**: Add/remove sensors by connecting/disconnecting diagram ports
- **Cross-Engine Parity**: Same block diagram structure should be mirrored in MuJoCo (via custom classes) and Pinocchio (via Crocoddyl graphs)

### B4a.4 Validation

```python
def test_drake_block_diagram_modularity():
    """Verify controller replacement without plant changes."""
    
    # Build diagram with PID controller
    builder1 = DiagramBuilder()
    plant1, pid_controller = add_golfer_plant_with_pid(builder1)
    diagram1 = builder1.Build()
    
    # Build diagram with LQR controller
    builder2 = DiagramBuilder()
    plant2, lqr_controller = add_golfer_plant_with_lqr(builder2)
    diagram2 = builder2.Build()   
    # Same plant, different controller - should both simulate
    assert simulate(diagram1, T=1.0) is not None
    assert simulate(diagram2, T=1.0) is not None
    
    # Controllers should expose same interface
    assert pid_controller.get_input_port().size() == lqr_controller.get_input_port().size()
```

---

## Section J1: OpenSim Grip Modeling (EXPANDED)

**Hand-grip interface must utilize OpenSim wrapping geometry and constraint forces:**

### J1.1 Implementation

- **Wrapping Surfaces**:
  - Cylindrical wrap around grip (radius = shaft radius + hand thickness)
  - Ellipsoidal wraps for palm/finger bulges
  - Via-point constraints for thumb opposition point

- **Muscle Routing**:
  - Flexor digitorum profundus (FDP) routed through fingers to grip
  - Extensor digitorum (ED) for grip release modeling
  - Thumb muscles: Flexor/abductor pollicis

- **Constraint Forces**:
  - Bilateral weld constraints at discrete grip contact points (4-6 points per hand)
  - Constraint reaction forces computed via OpenSim `analyzeStates()`

### J1.2 Required Outputs

```python
class OpenSimGripAnalysis:
    def compute_grip_metrics(self, state: osim.State) -> dict:
        """Section J1: OpenSim grip force analysis."""
        return {
            "muscle_forces": {  # [N] per muscle
                "FDP_index": 45.2,
                "FDP_middle": 52.1,
                # ...
            },
            "constraint_reactions": np.ndarray(...),  # [N] at via-points
            "moment_arms": {  # [m] about club long axis
                "FDP_index": 0.035,
                # ...
            },
            "total_grip_force": 187.3,  # [N]
        }
```

### J1.3 Validation

- Grip force magnitudes: 50-200 N per hand (physiological range from literature)
- Moment arm consistency: Within ±20% of cadaver studies
- Cross-validation: OpenSim constraint forces vs. MuJoCo contact forces (±15% tolerance)

---

## Section K1: MyoSuite Grip Modeling (EXPANDED)

**Hand-grip interface must use MyoSuite contact mechanics with muscle activation:**

### K1.1 Implementation

- **Fingertip Contact Elements**:
  - Sphere contacts at fingertip phalanges (5 per hand)
  - Compliant contact model: Spring-damper with K=1000 N/m, B=10 N·s/m
  - Friction coefficient μ=1.0 (dry skin on rubber grip)

- **Intrinsic Hand Muscles**:
  - Flexor digitorum superficialis (FDS) - 4 compartments
  - Flexor digitorum profundus (FDP) - 4 compartments
  - Lumbricals - 4 muscles (fine motor control)
  - Interossei - abduction/adduction

- **Activation Dynamics**:
  - First-order activation: `da/dt = (u - a) / τ_act` with τ_act = 30 ms
  - Force-length-velocity curve: Hill-type muscle model

### K1.2 Required Outputs

```python
class MyoSuiteGripAnalysis:
    def compute_activation_patterns(self, states: np.ndarray) -> dict:
        """Section K1: MyoSuite muscle activation during grip."""
        return {
            "activations": {  # Dimensionless [0, 1]
                "FDP_index": np.array([0.3, 0.4, 0.5, ...]),
                "FDP_middle": np.array([0.35, 0.45, ...]),
                # ...
            },
            "contact_forces": {  # [N] per contact point
                "index_tip": np.array([12.3, 14.7, ...]),
                # ...
            },
            "total_grip_force_bilateral": np.array([180, 195, ...]),  # [N]
        }
```

### K1.3 Validation

- Maximum voluntary contraction (MVC): 200-800 N total grip force
- Activation-force delay: 30-50 ms (physiological latency)
- Cross-validation: MyoSuite muscle forces → joint torques ≈ OpenSim muscle contributions (±20%)

---

## Section O1a: Drake State Isolation & Diagram Safety

**Extension to Section O (Physics Engine Integration Standards)**

### O1a.1 Drake Context Management

Drake uses `Context` objects for state isolation. **Mandatory practices:**

```python
# CORRECT: Each simulator instance has its own context
simulator_a = Simulator(diagram)
simulator_b = Simulator(diagram)  # Fresh context, isolated

context_a = simulator_a.get_mutable_context()
context_b = simulator_b.get_mutable_context()

# Modifying context_a does NOT affect context_b
diagram.get_plant().SetPositions(context_a, q_a)
diagram.get_plant().SetPositions(context_b, q_b)

assert not np.allclose(
    diagram.get_plant().GetPositions(context_a),
    diagram.get_plant().GetPositions(context_b)
)  # PASS: contexts are isolated
```

### O1a.2 Thread Safety for Drake Diagrams

Drake `System` objects are **stateless**. State lives in `Context`. Thread-safe pattern:

```python
from concurrent.futures import ThreadPoolExecutor

def simulate_swing_parallel(diagram: Diagram, initial_conditions: list[np.ndarray]):
    """Thread-safe Drake simulation."""
    
    def run_one_sim(q0: np.ndarray) -> np.ndarray:
        # Each thread gets its own Simulator and Context
        simulator = Simulator(diagram)  # Stateless diagram, unique simulator
        context = simulator.get_mutable_context()
        
        diagram.get_plant().SetPositions(context, q0)
        simulator.AdvanceTo(1.0)
        
        return diagram.get_plant().GetPositions(context)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_one_sim, initial_conditions))
    
    return results
```

**Test:**
```python
def test_drake_thread_safety():
    """Section O1a: Verify parallel simulations don't interfere."""
    diagram = build_golfer_diagram()
    
    q_initial_list = [
        np.array([0.1, 0.0, ...]),  # Swing 1
        np.array([0.2, 0.0, ...]),  # Swing 2
        np.array([0.3, 0.0, ...]),  # Swing 3
    ]
    
    results = simulate_swing_parallel(diagram, q_initial_list)
    
    # Results should differ (each swing had different initial conditions)
    assert not np.allclose(results[0], results[1])
    assert not np.allclose(results[1], results[2])
```

---

## Priority Notes from User Feedback (2026-01-06)

### Top Priority: Drift-Control Decomposition for All 5 Engines

**Implementation order:**
1. **Drake** (use block diagram `DriftControlDecomposerSystem`) - 2 days
2. **MuJoCo** (implement in `mujoco_physics_engine.py`) - 2 days
3. **Pinocchio** (use ABA algorithm with zero tau) - 2 days
4. **OpenSim** (use prescribed motion with zero muscle activation) - 3 days
5. **MyoSuite** (use environment reset with zero action) - 3 days

**Acceptance: All 5 engines pass `test_superposition_drift_plus_control_equals_full()`**

### Complete OpenSim Integration Checklist

- [ ] Hill-type muscle models operational
- [ ] Tendon compliance togglable
- [ ] Wrapping geometry for grip (Section J1)
- [ ] Activation → force → torque pipeline tested
- [ ] Induced acceleration with muscle contributions
- [ ] Muscle-specific logging (force/power/contribution percentages)
- [ ] Cross-validation with MyoSuite muscle forces (±20% tolerance)

---

**Signed:** Senior Principal Engineer  
**Approved by:** User (2026-01-06 verbal confirmation)  
**Next Review:** After drift-control implementation (Week 4)

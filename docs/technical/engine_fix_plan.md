# Physics Engine Fix Plan

## Phase 1: MuJoCo Engine Hardening (95% -> 100%)
- [ ] **Fix `model_name`**: Replace placeholder logic in `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py` with actual model name extraction (from loaded model object if possible, or reliable file parsing).
- [ ] **Fix `set_state`**: Ensure `forward()` is called after setting state to update dependent quantities (acceleration, sensors).
- [ ] **Fix `set_control`**: Implement strict size validation for control inputs to prevent silent failures.
- [ ] **Cleanup**: Remove unused `Any` import.
- [ ] **Verification**: Add a direct protocol test file `tests/integration/test_mujoco_protocol.py` (mimicking the Drake one) to valid the protocol compliance.

## Phase 2: Pinocchio Engine Polish (88% -> 100%)
- [ ] **Deduplicate `compute_bias_forces`**: Consolidate the dual implementations in `engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py`.
- [ ] **Fix Hardcoded Timestep**: Locate the `0.001` hardcoded value and ensure it respects the simulator's `dt` or `step_size`.
- [ ] **Optimization**: Fix generic body lookup optimization in `compute_jacobian`.

## Phase 3: MyoSim Transparency (70% -> 100% Honest)
- [ ] **Documentation correction**: Update `engines/physics_engines/myosim/python/myosim_physics_engine.py` docstrings to accurately reflect that it is currently a MuJoCo rigid-body wrapper, removing false claims about sarcomere dynamics until they are actually implemented.
- [ ] **Metadata update**: Check `config/models.yaml` (if present) to correct advertised features.
- [ ] **Rename/Refactor**: Consider renaming internal class to `MuJoCoRigidBodyWrapper` or similar if `MyoSim` is too misleading, OR add a `NotImplementedWarning` to muscle-specific methods if they exist.

## Phase 4: OpenSim Implementation (35% -> Usable)
- [ ] **Implement `load_from_string`**: Implement temporary file handling to support string loading (OpenSim API usually requires files).
- [ ] **Implement `set_control`**: Map controls to muscle excitations or coordinate actuators.
- [ ] **Implement Dynamics Methods**:
    - `compute_mass_matrix`: Use `MatterSubsystem.calcMassMatrix`.
    - `compute_bias_forces`: Use `System.getRigidBodyForces` or similar inverse dynamics equivalents.
    - `compute_gravity_forces`: Use `MatterSubsystem` gravity routines.
    - `compute_inverse_dynamics`: Implement full ID.
    - `compute_jacobian`: Implement frame Jacobian lookup.

## Phase 5: Pendulum Protocol Adapter (65% -> Integrated)
- [ ] **Create Adapter**: Create `engines/physics_engines/pendulum_adapter/pendulum_adapter.py`.
- [ ] **Implement Protocol**:
    - Map `load_from_path` to ignoring path (standalone).
    - Map `reset` to model constructor/reset.
    - Map `step` to model integration step.
    - Map `get_state`/`set_state`.
- [ ] **Integration**: Register this adapter with the `EngineManager`.

## Phase 6: Verification
- [ ] **Run Linting**: `ruff check .`
- [ ] **Run Formatting**: `black .`
- [ ] **Run Type Checking**: `mypy .`
- [ ] **Run Tests**: Run the new protocol tests.

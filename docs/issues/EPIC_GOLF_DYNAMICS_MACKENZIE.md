# EPIC: Advanced Golf Swing Dynamics (MacKenzie 2012 Integration)

## Objective
Implement physics-informed high-fidelity models for advanced golf swing dynamics, particularly focusing on the "club position relative to the golfer's swing plane" and its meaningful effects on swing dynamics. This incorporates the research from MacKenzie (2012) into the UpstreamDrift physics pipeline.

## Background
Currently, the `UpstreamDrift` golf swing models rely primarily on rigid-body physics engines (Pinocchio/Drake) and basic biomechanical heuristics. However, pure rigid-body representations fail to capture the subtle complexities of a high-velocity golf swing, such as:
1. Shaft flex, droop, and twist during the downswing.
2. The dynamic shift of the club's center of mass relative to the swing plane.
3. The coupling between wrist hinge forces, shaft stiffness, and aerodynamic drag.

By integrating the MacKenzie (2012) equations and models, we can upgrade the simulator to offer professional-grade predictive analytics for clubhead delivery.

## Key Goals
1. **Develop the MacKenzie Dynamics Module**:
   - Implement the mathematical models from *MacKenzie (2012): Club position relative to the golfer's swing plane meaningfully affects swing dynamics*.
   - Create a specialized computational module in `UpstreamDrift` to handle the calculations.
2. **Integrate with Existing Physics Engines**:
   - Bridge the new calculations with the existing rigid-body Pinocchio/Drake architecture via a **Residual Physics / PINN (Physics-Informed Neural Network)** approach.
   - Use the base rigid model as the analytical foundation, then apply the MacKenzie effects as residuals (delta forces/torques).
3. **Data Visualization**:
   - Update the UI to visually display the swing plane, the club's deviation from the plane, and the resulting aerodynamic/shaft-flex effects.
4. **Agentic Summary Hook**:
   - Expose the calculated differentials (e.g., plane deviation angles, torque impacts) to the `SidekickReportingWidget` so the AI assistant can provide coaching insights to the user.

## Implementation Steps

### Phase 1: Mathematical Foundation & Architecture
- [ ] Research and transcribe the core equations from MacKenzie 2012 into Python (using `numpy`/`scipy` or `Drake` symbolic math).
- [ ] Create `src/physics/advanced_dynamics/mackenzie_model.py`.
- [ ] Write unit tests to validate the calculations against published data points in the paper.

### Phase 2: Hybrid Integration (Pinocchio + MacKenzie)
- [ ] Create a physics wrapper/adapter that intercepts the Pinocchio rigid-body state at each timestep.
- [ ] Calculate the out-of-plane forces and apply them as external perturbations to the solver.
- [ ] Ensure strict typing (`mypy`) and performance bounds (potentially porting the inner loop to Rust later if Python becomes a bottleneck).

### Phase 3: Physics-Informed Neural Network (PINN) Scaffolding
- [ ] Prepare data schemas for logging swing trajectories.
- [ ] Scaffold the PyTorch/JAX models to train on the delta between the rigid Pinocchio state and the MacKenzie-adjusted state, allowing real-time inference without solving the full complex differential equations.

### Phase 4: UI & Reporting
- [ ] Implement advanced plots for "Club position relative to swing plane".
- [ ] Hook the new metrics into the `ProcessSummaryTab` / `ReportGenerator` for AI analysis.

## Related PRs & Issues
- [ ] TBD

## References
- MacKenzie, S. J. (2012). *Club position relative to the golfer's swing plane meaningfully affects swing dynamics*. Sports Engineering. http://www.sashomackenzie.com/publications/MacKenzie%202012%20Club%20position%20relative%20to%20the%20golfer's%20swing%20plane%20meaningfully%20affects%20swing%20dynamics.pdf

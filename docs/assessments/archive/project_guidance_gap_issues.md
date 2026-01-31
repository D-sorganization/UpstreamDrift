# Project Guidance Gap Issues

This document lists proposed GitHub issues derived from the gap analysis between
`docs/assessments/project_design_guidelines.qmd` and the current implementation.
It is intended as a ready-to-copy template for issue creation by a human or
automation workflow.

## Issue 1: Implement A3 Parameter Identification and Sensitivity Reporting

**Summary**
Implement model fitting and parameter identification (segment lengths, masses,
inertias) with sensitivity reporting.

**Context**
Guidance Section A3 requires parameter estimation and sensitivity analysis.
The current video pose pipeline advertises URDF parameter fitting but returns a
non-functional registration result.

**Acceptance Criteria**
- Implement parameter identification for segment lengths and inertias.
- Report sensitivity metrics for each fitted parameter.
- Add unit tests on synthetic data with known parameters.
- Document parameter estimation workflow in user-facing docs.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (A3)
- Implementation: `shared/python/video_pose_pipeline.py`

---

## Issue 2: Embed MuJoCo Visualization in URDF Generator

**Summary**
Add embedded MuJoCo visualization in the URDF generator with required overlays.

**Context**
Guidance Section B3 mandates embedded MuJoCo visualization (collision display,
axes, joint limits, contact visualization). The URDF generator currently lists
3D preview as in-progress.

**Acceptance Criteria**
- Live MuJoCo render of the generated model.
- Toggle controls for collisions, frames/axes, joint limits, contacts.
- Basic headless smoke test for launching the preview widget.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (B3)
- Implementation: `tools/urdf_generator/README.md`

---

## Issue 3: Integrate Flexible Shaft Model Across Engines with Validation

**Summary**
Wire the flexible shaft framework into engine implementations and add
cross-engine validation tests.

**Context**
Guidance Section B5 requires flexible shaft modeling across engines with static
deflection and dynamic response validation. The current module notes that engine
integration is separate.

**Acceptance Criteria**
- Engine adapters implement shaft configuration hooks.
- Static deflection tests and modal response comparisons added.
- Cross-engine deformation comparison tests in `tests/`.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (B5)
- Implementation: `shared/python/flexible_shaft.py`

---

## Issue 4: Replace MuJoCo Hand-Club Weld with Contact-Based Grip Model

**Summary**
Replace the left-hand weld constraint with contact-based grip mechanics in
MuJoCo.

**Context**
Guidance Section K2 mandates contact-based grip modeling. The MuJoCo model uses
a weld constraint between the left hand and club.

**Acceptance Criteria**
- Replace weld with contact pairs and friction settings.
- Provide slip detection and contact force reporting.
- Add static and dynamic validation tests for grip stability.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (K2)
- Implementation: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/models.py`

---

## Issue 5: Integrate Modular Impact Model into MuJoCo Runtime

**Summary**
Connect the modular impact model to MuJoCo runtime and add validation hooks.

**Context**
Guidance Section K3 requires a modular impact model with MuJoCo integration.
The current module is standalone without runtime wiring.

**Acceptance Criteria**
- Add impact hooks for MuJoCo pre/post impact states.
- Expose post-impact metrics and energy balance outputs.
- Add regression tests for energy and spin behavior.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (K3)
- Implementation: `shared/python/impact_model.py`

---

## Issue 6: Add Cross-Engine Constraint Diagnostics (Rank + Nullspace)

**Summary**
Implement standardized constraint Jacobian diagnostics across engines.

**Context**
Guidance Section C2 requires rank diagnostics and null-space tracking.
Current diagnostics are partial and engine-specific.

**Acceptance Criteria**
- Compute constraint Jacobian rank and condition numbers per engine.
- Expose null-space basis for closed-chain constraints.
- Add diagnostics reporting and tests.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (C2)
- Implementation: `engines/physics_engines/drake/python/src/manipulability.py`

---

## Issue 7: Add AIP JSON-RPC Server and Gemini Adapter

**Summary**
Implement the AIP JSON-RPC server and add a Gemini provider adapter.

**Context**
Guidance Section T1 requires multi-provider support including Google Gemini and
an AIP server. Current adapters exclude Gemini and no server implementation is
present.

**Acceptance Criteria**
- Implement the JSON-RPC AIP server with provider negotiation.
- Add Gemini adapter with opt-in dependencies.
- Unit tests for server endpoints and adapter handshake.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (T1)
- Implementation: `shared/python/ai/adapters/__init__.py`

---

## Issue 8: Add Required AI Workflows and Replace Placeholder Tools

**Summary**
Implement required workflows and replace placeholder validation tools.

**Context**
Guidance Section T2 mandates built-in workflows beyond `first_analysis`.
Current validation tools are placeholders and required workflows are missing.

**Acceptance Criteria**
- Implement workflows: `c3d_import`, `inverse_dynamics`,
  `cross_engine_validation`, `drift_control_decomposition`.
- Replace placeholder tool outputs with real engine integrations.
- Add workflow tests for success and failure paths.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (T2)
- Implementation: `shared/python/ai/workflow_engine.py`,
  `shared/python/ai/sample_tools.py`

---

## Issue 9: Integrate Swing Plane Frame Decomposition into Reporting

**Summary**
Wire swing-plane frame decomposition into force/torque outputs.

**Context**
Guidance Section E4 requires swing-plane frame reporting for forces and torques.
A swing plane frame module exists but is not integrated into reporting outputs.

**Acceptance Criteria**
- Add swing-plane decomposition to force/torque outputs.
- Include functional swing plane comparison outputs.
- Add tests validating decomposition values.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (E4)
- Implementation: `shared/python/reference_frames.py`

---

## Issue 10: Formalize U1/U2/U3 Metrics with Method Citations and Validation

**Summary**
Add method citation metadata and validation hooks for kinematic sequence,
X-Factor, and biomechanical load metrics.

**Context**
Guidance Section U requires explicit methodology citations and cross-engine
validation. Implementations exist but lack citation metadata and cross-engine
validation hooks.

**Acceptance Criteria**
- Add method citation metadata to metric outputs.
- Add cross-engine validation checks for timing and angle metrics.
- Update docs with methodology references for each metric.

**References**
- Guidance: `docs/assessments/project_design_guidelines.qmd` (U1-U3)
- Implementation: `shared/python/kinematic_sequence.py`,
  `shared/python/analysis/swing_metrics.py`,
  `shared/python/injury/spinal_load_analysis.py`

# Grip Wrench and Loop Null-Space Exploration Tool

The **Grip Wrench Plots** tool (`src/tools/grip_wrench_plots/`) provides the desktop PyQt6 dock (`GripWrenchPlotWidget`) for inspecting hand-on-club loading (`GCV-10`, #11716) and **Closed-Loop Null-Space Exploration (NSX)** (`NSX-43`, #12575), paired with the web React surface `ui/src/components/analysis/LoopNullSpacePanel.tsx` (`NSX-44`, #12576).

---

## Overview and Tabs

`GripWrenchPlotWidget` hosts a `QTabWidget` combining offline grip-wrench inspection with four interactive NSX plot-model tabs:

1. **Grip Wrench (Offline JSON):**
   - Loads `GET /api/analysis/grip-wrench` payloads or `*_grip_wrench.json` exports into `GripPlotSeries` and renders per-hand force magnitude, net force at the grip midpoint, equivalent couple (World or Club frame), and contact-force moment vs applied free torque.
2. **NSX Invariants (`invariants`):**
   - Renders allocation-invariant series (such as net bare-club wrench $W_{\text{net}}^{\text{bare}}$, summed loop actuator power $P_{\text{net}}$, and constrained ZTCF/ZVCF accelerations) accompanied by their `MODEL_CONDITIONED_INVARIANT` badge from `display_decision`.
3. **NSX Envelopes (`envelopes`):**
   - Renders band-first `[lower, upper]` feasible joint-torque envelopes segmented into `DETERMINED_SIGN`, `INDETERMINATE` (shaded), and `UNBOUNDED` (hatched) regions with active `binding_hypotheses` (`H-*`) annotations and optional overlaid `allocation_representative` curves.
4. **NSX Hand Split (`hand_split`):**
   - Renders the six canonical grip-frame internal split coordinates $s_1 \dots s_6 \in \ker G$ (`s1_axial_internal_force_n` through `s6_bending_w_diff_moment_nm`) in band-first form with optional overlaid representative trajectories.
5. **NSX Diagnostics (`diagnostics`):**
   - Renders closure Jacobian rank ($\operatorname{rank} J_c$), condition number ($\kappa(J_c)$), structural dimension counts ($\nu_{\text{eq}}$, $\nu_{\text{gauge}}$, $\nu_\tau$), Chebyshev feasibility radius, and kinematic admission residuals ($\Phi$, $\dot{\Phi}$, $\ddot{\Phi}$).

---

## Controls and Background Worker

- **Preset Selector (`QComboBox`):**
  - Backed by [`src/config/loop_nullspace_registry.json`](../../config/loop_nullspace_registry.json) (`envelope_only`, `capacity_min_effort`, `smooth_capacity`).
- **Cancellable `QThreadPool` Worker (`NSXAnalysisWorker`):**
  - Runs `loop_nullspace` analysis off the GUI thread, emits partial invariants first via its progress signal so the `Invariants` tab updates immediately, and supports cooperative cancellation via `cancel()`.
- **Evidence Badges and Help Link:**
  - Every NSX tab displays the `display_decision` badge (`badge_text`, `render_mode`, active `hypotheses`, and `conventions`) and links directly to the [Loop Null-Space Exploration User Guide](../../../docs/user_guide/loop_nullspace_exploration.md).

---

## Interpretation Summary

- **`INVARIANT` vs `FAMILY_DEPENDENT`:** Invariant quantities are fixed by the equations of motion $A x = b$ on an admitted trajectory; family-dependent quantities vary across the closed-loop null space and are always rendered as `[lower, upper]` feasible bands first.
- **Why the Hand Split Is Not Measured:** The $6 \times 12$ grasp map $G$ at the grip midpoint has a 6-DOF kernel ($\ker G$), so kinematics and inverse dynamics fix $W_{\text{net}}^{\text{bare}}$ (`0.313 kg` bare club separated from `1.1082 kg` welded hands) but cannot uniquely measure individual hand wrenches without priors or dual-handle force sensors.
- **Presets and Certificates:** Presets select verified hypothesis bundles (`H-TORQUE-CAPACITY-BOX`, `H-GRIP-FRICTION-CONE`, `H-EFFORT-CAPACITY-QUADRATIC`, `H-TORQUE-RATE-SMOOTHNESS`). Certified affine probes report `HOLDS`, `FAILS` (with witness), or `UNDECIDED`.
- **Reaction-Restriction Sensitivity:** Comparing `WELD6` $\to$ `SHAFT_FREE5` $\to$ `BALL3` measures how restricting grip reaction wrench rows shrinks the feasible torque set on a fixed trajectory; it is not a simulation of a physical hinge or ball-joint grip.

For complete mathematical details and reading guidance, see:

- [Loop Null-Space Exploration User Guide](../../../docs/user_guide/loop_nullspace_exploration.md)
- [Closed-Loop Null-Space Exploration Research Reference](../../../docs/research/loop_nullspace/README.md)
- [ADR-0054: Closed-Loop Null-Space Exploration Architecture and Output-Policy Contract](../../../docs/adr/0054-closed-loop-null-space-exploration.md)

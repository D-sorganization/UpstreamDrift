# ADR-0054: Closed-Loop Null-Space Exploration Architecture and Output-Policy Contract

- Status: Accepted
- Date: 2026-10-14
- Decision Makers: repository owner (recorded owner decision on output-policy integration); consolidated in the #12593 v2 review session
- Related Issues/PRs: epic #12593 and children #12549–#12592 (`NSX-00` … `NSX-74`); ADR-0008, ADR-0026, ADR-0041, ADR-0052; GCV-7 #11713, GCV-8 #11714

## Context

In a two-handed golf swing, both hands grip the club, forming a closed kinematic
loop across the torso, lead arm, trail arm, and club handle. Motion capture and
inverse dynamics determine the generalized load vector
$b = M(q)a + h(q, v)$ and the net bare-club wrench $W_{\text{net}}^{\text{bare}}$,
but leave an affine family of internal closure multipliers $\lambda$, joint
torques $\tau$, and ground-reaction forces $f_g$ satisfying $A x = b$.

Prior to this decision, UpstreamDrift had:

1. **Fragmented, mathematically inconsistent allocators:**
   `ContactForceAllocator` and `MultiEngineTorqueAllocator` used `scipy.optimize.minimize(..., method="SLSQP")`, treated `"HARD_ZERO_TRAIL"` as a silently relaxed least-squares objective on overdetermined 9-DOF arms, and equated null-space dimension with $\operatorname{rank}(J_c)$ rather than accounting for unactuated base/club rows.
2. **Composite club-body mass contamination (`#12550`):**
   `Clubface Vector` in the full-body specification welds `LHand` ($0.5491\,\mathrm{kg}$), `LHandStandoff` ($0.010\,\mathrm{kg}$), and `RHand` ($0.5491\,\mathrm{kg}$) into the club body ($1.4212\,\mathrm{kg}$ total vs $0.313\,\mathrm{kg}$ bare club), contaminating `holding_hand_wrench` across MuJoCo, Drake, and Pinocchio.
3. **Incompatible engine Jacobian conventions (`#12551`):**
   Pinocchio (`LOCAL [lin;ang]`), Drake (`LOCAL [ang;lin]`), MuJoCo (`WORLD [lin;ang]`, $a - b$), and OpenSim (`WORLD [lin;ang]`, $b - a$) returned different row orders, frames, and signs from `compute_grip_jacobian`.
4. **Unconstrained counterfactuals (`#12570`):**
   `ztcf_zvcf.py` evaluated unconstrained $M^{-1}$ accelerations without re-solving loop reactions $\lambda$, and `motion_matching/counterfactual.py` contained an $m_{\text{eff}} = 1.0$ mock integrator.
5. **Unresolved output-policy interaction (`#12573`):**
   `output_policy.py` rightly withholds uncalibrated joint loads from user-facing swing reports, leaving undefined how interactive research views may display allocation-invariant quantities and feasible envelopes without misrepresenting them as validated golfer measurements.

## Decision

Establish one engine-agnostic **Closed-Loop Null-Space Exploration (NSX)**
architecture in `src/shared/python/loop_nullspace/` governed by Rules R1–R14 of
epic #12593 and the mathematical specification in
`docs/research/loop_nullspace/loop_nullspace.tex`.

### 1. Single Engine Seam Over Extended `BaseEngineForceAdapter`

Rather than introducing a parallel engine protocol, `BaseEngineForceAdapter`
(`src/shared/python/motion_matching/multi_engine_torque_allocator.py`) is
extended with four methods/properties gated by `Capability.CLOSED_LOOP_SNAPSHOT`:

- `native_closure_convention -> NativeClosureConvention`
- `compute_closure_drift(q, v) -> np.ndarray` ($\dot{J}_c v$ in native convention)
- `closure_wrench_frame(q) -> tuple[np.ndarray, np.ndarray]` (world-from-grip rotation $R_G \in SO(3)$ and midpoint $r_M \in \mathbb{R}^3$)
- `club_inertia_split() -> tuple[SpatialInertia, tuple[SpatialInertia, ...]]` (bare club vs welded hands per `NSX-01`)

`snapshot_from_adapter(adapter, q, v, a, *, input_provenance)` validates
`Capability.CLOSED_LOOP_SNAPSHOT == CapabilityLevel.FULL`, transforms the closure
Jacobian and drift into the canonical convention via `to_canonical_closure`, and
returns a frozen `ClosedLoopSnapshot` value object. Synthetic fixtures report
`CapabilityLevel.NONE` and are rejected by `snapshot_from_adapter`.

### 2. Canonical Closure Convention (Rule R8)

Every `ClosedLoopSnapshot` stores $J_c \in \mathbb{R}^{m_c \times n_v}$ and
$\dot{J}_c v \in \mathbb{R}^{m_c}$ in the **canonical grip-midpoint frame**
(`C-FRAME-GRIP`):

- Origin at $r_M = \tfrac{1}{2}(r_L + r_R)$.
- Axes $(\hat{u}, \hat{v}, \hat{w})$ with $\hat{u}$ along the shaft toward the
  clubhead, $\hat{v}$ along the clubface normal projection orthogonal to
  $\hat{u}$, and $\hat{w} = \hat{u} \times \hat{v}$.
- Row order `[fx, fy, fz, tx, ty, tz]` (for `WELD6`).
- Sign convention: $+J_c^\top \lambda$ acts on the human kinematic tree in
  $M(q)a + h(q,v) = S^\top \tau + J_c^\top \lambda + J_g^\top f_g$, and the
  closing hand exerts wrench $-\lambda$ on the club.

### 3. Six Independent Evidence Axes (Rule R1)

Every user-visible quantity produced by NSX is wrapped in a frozen
`QuantityRecord` carrying:

1. `allocation_dependence`: `INVARIANT`, `FAMILY_DEPENDENT`, or `UNAVAILABLE`.
2. `model_provenance`: `ModelProvenance` (spec SHA-256, inertial source, bare-club partition ID, engine and `kkt_regularization`).
3. `input_provenance`: `InputProvenance` (kinematic source, derivative method, noise floor, time-sync flag).
4. `conventions`: tuple of convention IDs (`C-FRAME-GRIP`, `C-METRIC-*`, etc.).
5. `hypotheses`: tuple of physiological hypothesis IDs (`H-TORQUE-CAPACITY-BOX`, `H-GRIP-FRICTION-CONE`, etc.). By Design-by-Contract (`contracts.py`), `FAMILY_DEPENDENT` requires `len(hypotheses) >= 1`, whereas `INVARIANT` requires `len(hypotheses) == 0`.
6. `validation_status`: `UNVALIDATED` (default until `DV-12554` external validation is completed).

### 4. Convex Solver Routing and Allocator Consolidation (Rules R11, NSX-20, NSX-24)

- **Polyhedral LPs** (static envelopes, Chebyshev center, affine probes, linearized cones) route through `highspy` with persistent model construction, objective-vector swaps, and simplex basis warm starts across frames. `scipy.optimize.linprog` is not used because it lacks warm-start basis reuse.
- **Second-order cone programs (SOCPs)** (exact circular friction cones, quadratic capacity effort, rate smoothness, and $c_+ / c_-$ gross-power epigraphs) route through `clarabel`.
- **Missing solver extra:** If `highspy` or `clarabel` is absent at runtime, the solver raises a clear `CapabilityUnavailable` / returns `UNAVAILABLE(reason="solver_unavailable")`; it never falls back silently to `SLSQP`.
- **Legacy allocator consolidation (`NSX-24`):** `ContactForceAllocator` and `MultiEngineTorqueAllocator` become thin compatibility shims over `FeasibleSet` and `allocate_trajectory`. `HARD_ZERO_TRAIL` delegates to `zero_side_solve` (`NSX-15`) and explicitly reports `INCONSISTENT(residual)` rather than hiding nonzero trail torques.

### 5. Owner Decision on Output-Policy Integration (`NSX-41`)

The repository owner establishes the following binding rules governing how
`output_policy.py` interacts with NSX outputs across product surfaces:

1. **Customer/Golfer Swing Reports (`Surface.SWING_REPORT`):**
   `output_policy.py` remains strictly **fail-closed**. No joint torque, hand
   split, or internal wrench from NSX is admitted into a golfer-facing swing
   report unless subject-specific calibration, external measurement provenance,
   and validation gates are satisfied.
2. **Interactive Research and Analysis Surfaces (`Surface.RESEARCH_VIEW` in PyQt6 and React):**
   - `INVARIANT` quantities (net bare-club wrench, summed loop power within $\dot{\Phi}$ gate, base/centroidal wrench, $N_c^\top \tau$, constrained ZTCF/ZVCF total accelerations) are **permitted for display** accompanied by a mandatory `MODEL_CONDITIONED_INVARIANT` badge displaying `model_provenance` and `conventions`.
   - `FAMILY_DEPENDENT` quantities (individual loop joint torques, lead/trail hand wrenches, grip-split coordinates $s_1 \dots s_6$, per-joint counterfactual attributions) are **permitted for display only in band-first form**:
     - Feasible min/max envelopes (`STATIC` or `TRAJECTORY_COUPLED`) with sign-indeterminate regions shaded and binding hypothesis IDs annotated.
     - Prior-conditioned representative trajectories (`allocate_trajectory`) may **only** be rendered overlaid atop their feasible envelope band, never as a standalone curve, and must display the active preset name and `hypotheses` list.
   - `UNAVAILABLE` quantities render as empty/hatched regions with their explicit `reason` string, never as zero.

## Consequences

- **Positive:** Eliminates duplicated SLSQP allocators, fixes the $1.1082\,\mathrm{kg}$ welded-hand mass bug on the club body, guarantees cross-engine Jacobian parity by construction, and enforces a clear boundary between model-determined invariants and hypothesis-conditioned allocations.
- **Negative / Trade-offs:** Adds `highspy` and `clarabel` to the `loop-nullspace` and `dev` extras (`requirements-dev.lock`), and requires all downstream `tier:cli` children (`NSX-10` through `NSX-62`) to construct full `QuantityRecord` metadata.

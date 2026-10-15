# Closed-Loop Null-Space Exploration Per-Child Contract Appendix

This appendix pins the exact module paths, public Python/TypeScript signatures, Design-by-Contract (`src.shared.python.core.contracts.require` / `ensure`) preconditions and postconditions, and golden test fixtures from [`derivations/golden_fixtures.json`](derivations/golden_fixtures.json) for the children of Epic #12593 (`NSX-01` through `NSX-62`).

---

## Phase 0: Pre-Existing Defects and Specification Children

### NSX-01 (#12550): Bare-Club Solid Partition and Net Hand-on-Club Wrench

- **Module:** `src/shared/python/biomechanics/club_solids_partition.py` (exported via `src/shared/python/biomechanics/__init__.py`)
- **Signatures:**

  ```python
  @dataclass(frozen=True)
  class SpatialInertiaPartition:
      mass_kg: float
      com_local_m: np.ndarray          # shape (3,), read-only
      inertia_com_kgm2: np.ndarray     # shape (3, 3), SPD, read-only
      solid_names: tuple[str, ...]

  def club_solids_partition(
      spec: Mapping[str, Any],
  ) -> tuple[SpatialInertiaPartition, SpatialInertiaPartition]:
      """Return (bare_club_inertia, welded_hand_inertias) from a full-body spec."""
  ```

- **Preconditions:** `spec` contains `Clubface Vector` solids and `club.total_mass_kg`.
- **Postconditions:** `abs(bare_club_inertia.mass_kg - spec["club"]["total_mass_kg"]) <= 1e-9`; `bare_club_inertia.mass_kg == 0.313` and `welded_hand_inertias.mass_kg == 1.1082` on `full_body_spec_anthro_driver.json`; `holding_hand_wrench` records `interface="hand_on_bare_club"` (preserving `"wrist_on_hands_plus_club"` when requested explicitly).
- **Golden Fixture:** `fixtures["bare_club_partition"]`

### NSX-02 (#12551): Canonical Grip-Closure Jacobian Convention Converter

- **Module:** `src/shared/python/loop_nullspace/closure_convention.py`
- **Signatures:**

  ```python
  @dataclass(frozen=True)
  class NativeClosureConvention:
      row_order: Literal["lin_ang", "ang_lin"]
      frame: Literal["local", "world"]
      point: Literal["joint1_placement", "site_xpos", "body_origin"]
      relative_sign: Literal[1, -1]

  def to_canonical_closure(
      jacobian: np.ndarray,
      drift: np.ndarray,
      *,
      native: NativeClosureConvention,
      grip_frame_rotation: np.ndarray,
      grip_origin: np.ndarray,
      reference_point: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
      """Transform native (J_native, Jdot_v_native) to canonical C-FRAME-GRIP [fx,fy,fz,tx,ty,tz]."""
  ```

- **Preconditions:** `jacobian.shape == (6, n_v)`, `drift.shape == (6,)`, `grip_frame_rotation` orthonormal in $SO(3)$ (`||R^T R - I||_F <= 1e-9`).
- **Postconditions:** Output `(J_c, Jdot_v)` is in `C-FRAME-GRIP` with row order `[fx, fy, fz, tx, ty, tz]` and $+J_c^\top \lambda$ acting on the human tree; virtual power $\lambda^\top (J_c v)$ is invariant under conversion to `1e-12`.
- **Golden Fixture:** `fixtures["grasp_map_6x6"]`

### NSX-03 (#12552): Split `force_nullspace.py` Package With Re-Export Shims

- **Modules:** `src/shared/python/motion_matching/force_nullspace/{__init__.py,space.py,constraints.py,redistribution.py,tradeoffs.py}`
- **Contract:** Pure structural refactor of `force_nullspace.py` into four submodules under 600 lines each; `__init__.py` re-exports identical `__all__` symbols with zero numerical or behavioral change.

### NSX-04 (#12553): Joint Torque-Capacity Table and Loader

- **Module:** `src/shared/python/loop_nullspace/capacity.py` and `src/config/loop_nullspace_registry.json` (`torque_capacity[]`)
- **Signatures:**

  ```python
  @dataclass(frozen=True)
  class CapacityVector:
      coordinate_order: tuple[str, ...]
      tau_min_nm: np.ndarray
      tau_max_nm: np.ndarray
      verification_status: tuple[Literal["VERIFIED", "ASSUMED", "UNVERIFIED"], ...]
      sources: tuple[str, ...]

  def capacity_for(
      snapshot_coordinate_order: Sequence[str],
      subject_mass_kg: float = 80.0,
      subject_height_m: float = 1.80,
  ) -> CapacityVector: ...
  ```

- **Preconditions:** `subject_mass_kg > 0`, `subject_height_m > 0`, every actuated coordinate in `snapshot_coordinate_order` exists in the registry table (raises `ValueError` naming missing coordinates otherwise).
- **Postconditions:** `np.all(tau_max_nm > 0)` and `np.all(tau_min_nm < 0)`; any `ASSUMED` slot carries `verification_status == "UNVERIFIED"` and triggers an explicit provenance flag.

---

## Phase 1: Pinocchio Reference Core (`NSX-10` Through `NSX-17`)

### NSX-10 (#12555): Snapshot, Topology, and Evidence-Axis Contracts

- **Module:** `src/shared/python/loop_nullspace/contracts.py`
- **Signatures:**

  ```python
  class ClosureKind(str, Enum):
      WELD6 = "WELD6"
      SHAFT_FREE5 = "SHAFT_FREE5"
      BALL3 = "BALL3"
      PLANAR_POINT2 = "PLANAR_POINT2"
      PLANAR_WELD3 = "PLANAR_WELD3"

  class AllocationDependence(str, Enum):
      INVARIANT = "INVARIANT"
      FAMILY_DEPENDENT = "FAMILY_DEPENDENT"
      UNAVAILABLE = "UNAVAILABLE"

  class ValidationStatus(str, Enum):
      UNVALIDATED = "UNVALIDATED"
      DEFERRED_EXTERNAL = "DEFERRED_EXTERNAL"

  @dataclass(frozen=True)
  class LoopTopology:
      closure_kind: ClosureKind
      closing_side: Literal["left", "right"]
      holding_side: Literal["left", "right"]
      actuated_indices: tuple[int, ...]
      unactuated_indices: tuple[int, ...]
      loop_support_indices: tuple[int, ...]
      coordinate_order: tuple[str, ...]

  @dataclass(frozen=True)
  class ClosedLoopSnapshot:
      q: np.ndarray
      v: np.ndarray
      a: np.ndarray
      M: np.ndarray
      h: np.ndarray
      S: np.ndarray
      J_c: np.ndarray
      Jdot_v: np.ndarray
      grip_frame_rotation: np.ndarray
      grip_origin: np.ndarray
      bare_club: SpatialInertiaPartition
      welded_hand_inertias: SpatialInertiaPartition
      model_provenance: ModelProvenance
      input_provenance: InputProvenance
      topology: LoopTopology
      J_g: np.ndarray | None = None
      foot_contacts: tuple[FootContact, ...] = ()
      grf_observed: np.ndarray | None = None

      def generalized_load(self) -> np.ndarray: ...
      def closure_rows(self) -> int: ...
      def actuated_rows(self) -> np.ndarray: ...

  @dataclass(frozen=True)
  class QuantityRecord:
      values: np.ndarray
      units: str
      allocation_dependence: AllocationDependence
      conventions: tuple[str, ...]
      hypotheses: tuple[str, ...]
      model_provenance: ModelProvenance
      input_provenance: InputProvenance
      validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
      reason: str | None = None
  ```

- **Preconditions & Postconditions:**
  - `M` is finite, symmetric (`||M - M^T||_F <= 1e-10`), and positive definite (`eigvalsh(M).min() > 0`).
  - `J_c.shape[0]` matches `closure_kind` (`6, 5, 3, 2, 3`).
  - All NumPy arrays have `flags.writeable = False`.
  - `QuantityRecord` enforces: `FAMILY_DEPENDENT` requires `len(hypotheses) >= 1`; `INVARIANT` requires `len(hypotheses) == 0`; `UNAVAILABLE` requires non-empty `reason` and all-`NaN` `values`.

### NSX-11 (#12556): Hypothesis, Convention, and Preset Registry Loader

- **Module:** `src/shared/python/loop_nullspace/registry.py`
- **Signatures:**
  ```python
  def load_registry(path: Path | None = None) -> RegistryCatalog: ...
  def hypothesis(hypothesis_id: str) -> HypothesisRecord: ...
  def preset(name: str) -> HypothesisSet: ...
  def require_verified_for_display(hypothesis_ids: Iterable[str]) -> None: ...
  ```
- **Preconditions & Postconditions:** Rejects duplicate IDs, dangling preset references, or any `UNVERIFIED`/`REMOVED` source token in user-facing `statement` or `label` fields.

### NSX-12 (#12557): Analytic Closed-Loop Oracles

- **Module:** `src/shared/python/loop_nullspace/fixtures.py`
- **Signatures:**
  ```python
  def planar_golfer_snapshot(
      *, actuation: Literal["full", "club_unactuated"] = "full"
  ) -> tuple[ClosedLoopSnapshot, dict[str, Any]]: ...
  def straight_arm_snapshot() -> tuple[ClosedLoopSnapshot, dict[str, Any]]: ...
  def spatial_two_arm_club_snapshot(
      *, floating_base: bool = False
  ) -> tuple[ClosedLoopSnapshot, dict[str, Any]]: ...
  def analytic_envelope_case() -> tuple[ClosedLoopSnapshot, dict[str, Any]]: ...
  ```
- **Preconditions & Postconditions:** Loads golden expectations directly from `docs/research/loop_nullspace/derivations/golden_fixtures.json`; verifies `golfer_constraints.py` independent row rank (rank 2).
- **Golden Fixtures:** `planar_two_arm_fixed_base`, `planar_two_arm_underactuated`, `floating_base_two_arm`, `straight_arm_rank_deficient`, `analytic_envelope_case`.

### NSX-13 (#12558): Extend `BaseEngineForceAdapter` and `snapshot_from_adapter`

- **Modules:** `src/shared/python/motion_matching/multi_engine_torque_allocator.py`, `src/engines/physics_engines/pinocchio/python/force_adapter.py`, `src/shared/python/loop_nullspace/snapshot.py`
- **Signatures:**
  ```python
  def snapshot_from_adapter(
      adapter: BaseEngineForceAdapter,
      q: np.ndarray,
      v: np.ndarray,
      a: np.ndarray,
      *,
      input_provenance: InputProvenance,
  ) -> ClosedLoopSnapshot: ...
  ```
- **Preconditions:** `adapter` advertises `Capability.CLOSED_LOOP_SNAPSHOT == CapabilityLevel.FULL` (raises `CapabilityUnavailable` on synthetic or unsupported adapters).
- **Postconditions:** Calls `to_canonical_closure` (`NSX-02`) and `club_solids_partition` (`NSX-01`); `PinocchioForceAdapter.verify_acceleration_parity` uses `pinocchio.constraintDynamics` to `1e-9` scaled.

### NSX-14 (#12559): Kinematic and Contact Admission Gate

- **Module:** `src/shared/python/loop_nullspace/admission.py`
- **Signatures:**

  ```python
  @dataclass(frozen=True)
  class AdmissionGates:
      max_phi_scaled: float = 1e-6
      max_phidot_scaled: float = 1e-5
      max_phiddot_scaled: float = 1e-4
      max_stance_foot_vel_mps: float = 1e-2

  @dataclass(frozen=True)
  class AdmittedTrajectory:
      snapshots: tuple[ClosedLoopSnapshot, ...]
      report: AdmissionReport

  def admit(
      trajectory_snapshots: Sequence[ClosedLoopSnapshot],
      *,
      gates: AdmissionGates = AdmissionGates(),
      repair: Literal["none", "project_to_closure"] = "none",
  ) -> AdmittedTrajectory | AdmissionReport: ...
  ```

- **Preconditions & Postconditions:** Computes scaled position $\Phi$, velocity $\dot{\Phi} = J_c v$, and acceleration $\ddot{\Phi} = J_c a + \dot{J}_c v$ residuals per frame; returns `AdmittedTrajectory` only when all frames pass (or after explicit `repair="project_to_closure"` with repair deltas recorded).

### NSX-15 (#12560): Closed-Loop Decomposition and Zero-Side Three-Case Solve

- **Module:** `src/shared/python/loop_nullspace/decomposition.py`
- **Signatures:**

  ```python
  def decompose(
      snapshot: ClosedLoopSnapshot,
      *,
      metric_convention: str = "C-METRIC-IDENTITY",
  ) -> Decomposition: ...

  def zero_side_solve(
      snapshot: ClosedLoopSnapshot,
      side: Literal["left", "right"],
  ) -> ZeroSideResult: ...

  def min_metric_torque(
      snapshot: ClosedLoopSnapshot,
      W: np.ndarray,
      *,
      convention_id: str,
  ) -> QuantityRecord: ...
  ```

- **Preconditions & Postconditions:** Reuses `ForceNullSpace.from_balance` on $A = [S^\top \;\; J_c^\top \;\; J_g^\top]$; reports `equality_nullity`, `multiplier_gauge`, and `attainable_torque_variation` matching Table 1 of `loop_nullspace.tex`; `zero_side_solve` returns `CONSISTENT_UNIQUE`, `CONSISTENT_NONUNIQUE`, or `INCONSISTENT(residual_scaled)`.
- **Golden Fixtures:** `planar_two_arm_fixed_base`, `planar_two_arm_underactuated`, `floating_base_two_arm`, `straight_arm_rank_deficient`, `zero_side_cases`.

### NSX-16 (#12561): Grip-Frame Hand-Split Coordinates ($6\times 6$ Kernel)

- **Module:** `src/shared/python/loop_nullspace/grip_split.py`
- **Signatures:**
  ```python
  def grasp_map(grip_frame_rotation: np.ndarray, r_L: np.ndarray, r_R: np.ndarray) -> np.ndarray: ...
  def split_basis(G: np.ndarray, half_separation_m: float) -> np.ndarray: ...
  def net_club_wrench(snapshot: ClosedLoopSnapshot) -> QuantityRecord: ...
  def hand_wrenches(net_wrench: np.ndarray, s: np.ndarray, half_separation_m: float) -> tuple[np.ndarray, np.ndarray]: ...
  def split_coordinates(W_L: np.ndarray, W_R: np.ndarray, half_separation_m: float) -> np.ndarray: ...
  ```
- **Preconditions & Postconditions:** `G @ K == 0` to `1e-12`; `rank(K) == 6`; `split_coordinates(*hand_wrenches(W_net, s, h), h) == s` to `1e-12`; `net_club_wrench` is tagged `AllocationDependence.INVARIANT`.
- **Golden Fixture:** `fixtures["grasp_map_6x6"]`

### NSX-17 (#12562): Invariant Quantities With Validity Gates

- **Module:** `src/shared/python/loop_nullspace/invariants.py`
- **Signatures:**
  ```python
  def compute_invariants(
      admitted: AdmittedTrajectory,
      decompositions: Sequence[Decomposition],
  ) -> InvariantReport: ...
  ```
- **Preconditions & Postconditions:** Computes R5(a)–(f); any invariant whose frame residual exceeds its gate is returned as `QuantityRecord(allocation_dependence=UNAVAILABLE, reason=..., values=NaN)`.

---

## Phase 2: Feasible Set, Envelopes, and Conic Allocation (`NSX-20` Through `NSX-26`)

### NSX-20 (#12563): Convex Solver Backends (`highspy` LP and `Clarabel` SOCP)

- **Module:** `src/shared/python/loop_nullspace/solvers.py`
- **Signatures:**

  ```python
  def solve_lp(
      c: np.ndarray,
      A_eq: np.ndarray | None,
      b_eq: np.ndarray | None,
      A_ub: np.ndarray,
      b_ub: np.ndarray,
      bounds: tuple[np.ndarray, np.ndarray] | None = None,
      *,
      warm: HighsBasis | None = None,
  ) -> LPResult: ...

  def solve_socp(
      P: sp.csc_matrix | np.ndarray,
      q: np.ndarray,
      A: sp.csc_matrix | np.ndarray,
      b: np.ndarray,
      cones: Sequence[ConeSpec],
  ) -> ConicResult: ...
  ```

- **Postconditions:** Status is mapped to `OPTIMAL | INFEASIBLE(certificate) | UNBOUNDED(ray) | NUMERICAL(reason)`; lazy imports raise `CapabilityUnavailable` when `loop-nullspace` extra is absent.

### NSX-21 (#12564): `FeasibleSet` in Null-Space Coordinates

- **Module:** `src/shared/python/loop_nullspace/feasible_set.py`
- **Signatures:**
  ```python
  class FeasibleSet:
      @classmethod
      def from_hypotheses(
          cls,
          decomposition: Decomposition,
          hypothesis_set: HypothesisSet,
          capacity: CapacityVector,
      ) -> FeasibleSet: ...
      def is_empty(self) -> bool: ...
      def conflict_set(self) -> tuple[str, ...]: ...
      def to_lp(self) -> LPProblemData: ...
      def to_socp(self) -> SOCPProblemData: ...
  ```
- **Postconditions:** Every constraint row maps to a registry hypothesis ID; `conflict_set()` returns an inclusion-minimal IIS of hypothesis IDs via elastic LP filtering.

### NSX-22 (#12565): Static Torque and Hand-Split Envelopes With Chebyshev Margin

- **Module:** `src/shared/python/loop_nullspace/envelopes.py`
- **Signatures:**
  ```python
  def static_envelopes(
      feasible_sets: Sequence[FeasibleSet],
      quantities: Sequence[LinearQuantitySpec],
  ) -> EnvelopeSeries: ...
  ```
- **Postconditions:** Warm-starts `highspy` basis across frames; classifies each frame as `POSITIVE`, `NEGATIVE`, `INDETERMINATE`, or `ZERO`; computes Chebyshev center and radius per Eq. (17) of `loop_nullspace.tex`; labels series `STATIC`.
- **Golden Fixture:** `fixtures["analytic_envelope_case"]`

### NSX-23 (#12566): Trajectory Conic Allocation With Curvature Guarantee

- **Module:** `src/shared/python/loop_nullspace/allocation.py`
- **Signatures:**
  ```python
  def allocate_trajectory(
      feasible_sets: Sequence[FeasibleSet],
      objective: ObjectiveSpec,
      *,
      reference: np.ndarray | None = None,
  ) -> AllocationResult: ...
  ```
- **Preconditions:** Reduced Hessian satisfies $\lambda_{\min}(N^\top H_{\text{quad}} N) \ge 10^{-8}$ (raises `ValueError` if non-strictly convex); detects uniform $c_+ = c_-$ or uniform $c_- / c_+$ ratio per Proposition 2 of `loop_nullspace.tex`.
- **Postconditions:** Returns `QuantityRecord(allocation_dependence=FAMILY_DEPENDENT, hypotheses=...)` with duals and KKT residuals.
- **Golden Fixture:** `fixtures["c_plus_minus_identity"]`

### NSX-24 (#12567): Consolidate Legacy Allocators Onto Conic Core

- **Modules:** `src/shared/python/motion_matching/contact_force_allocator.py`, `src/shared/python/motion_matching/multi_engine_torque_allocator.py`
- **Contract:** Shims both legacy classes onto `FeasibleSet` and `allocate_trajectory`, emits `DeprecationWarning`, removes `SLSQP` from convex paths, and routes `HARD_ZERO_TRAIL` through `zero_side_solve`.

### NSX-25 (#12568): Pareto Fronts ($\epsilon$-Constraint) and Certified Affine Probes

- **Module:** `src/shared/python/loop_nullspace/probes.py`
- **Signatures:**

  ```python
  def pareto_front(
      feasible_sets: Sequence[FeasibleSet],
      objectives: tuple[ObjectiveSpec, ObjectiveSpec],
      eps_grid: np.ndarray,
  ) -> ParetoFront: ...

  def certify(
      feasible_sets: Sequence[FeasibleSet],
      predicate: AffinePredicate,
      window: slice,
  ) -> Certificate: ...
  ```

- **Preconditions:** Rejects non-affine predicates at construction.
- **Postconditions:** Returns `HOLDS`, `FAILS(witness)`, or `UNDECIDED(reason)` over the trajectory-coupled feasible set.

### NSX-26 (#12569): Reaction-Restriction Sensitivity (`WELD6` $\to$ `SHAFT_FREE5` $\to$ `BALL3`)

- **Module:** `src/shared/python/loop_nullspace/restriction.py`
- **Signatures:**
  ```python
  def restrict(snapshot: ClosedLoopSnapshot, kind: ClosureKind) -> ClosedLoopSnapshot: ...
  def restriction_sensitivity(
      trajectory: AdmittedTrajectory,
      kinds: Sequence[ClosureKind],
      quantities: Sequence[LinearQuantitySpec],
  ) -> RestrictionReport: ...
  ```
- **Postconditions:** Enforces $\mathcal{F}(\text{BALL3}) \subseteq \mathcal{F}(\text{SHAFT\_FREE5}) \subseteq \mathcal{F}(\text{WELD6})$ on actuated torques and attaches label `REACTION_RESTRICTION_SENSITIVITY`.

---

## Phase 3: Counterfactuals and Replay (`NSX-30` Through `NSX-31`)

### NSX-30 (#12570): Constrained `EquationsOfMotionProvider` and Constrained ZTCF/ZVCF

- **Modules:** `src/shared/python/loop_nullspace/constrained_dynamics.py`, `src/shared/python/simulation_backends/ztcf_zvcf.py`, `src/shared/python/motion_matching/counterfactual.py`
- **Signatures:**
  ```python
  def constrained_acceleration(
      M: np.ndarray,
      h: np.ndarray,
      tau: np.ndarray,
      J_c: np.ndarray,
      Jdot_v: np.ndarray,
      *,
      S_T: np.ndarray | None = None,
  ) -> tuple[np.ndarray, np.ndarray]: ...
  ```
- **Postconditions:** Computes KKT projector $P_{\text{KKT}} = M^{-1}(I - J_c^\top \Lambda_c J_c M^{-1})$; removes `m_eff = 1.0` and `5 * sin(t)` mocks from `motion_matching/counterfactual.py`; tags total constrained ZTCF/ZVCF accelerations `INVARIANT` and per-joint attributions `FAMILY_DEPENDENT`.
- **Golden Fixture:** `fixtures["planar_two_arm_fixed_base"]["a_ztcf"]`

### NSX-31 (#12571): Open-Loop Versus Feedback-Assisted Replay With Separate Budget

- **Module:** `src/shared/python/loop_nullspace/replay.py`
- **Signatures:**
  ```python
  def replay(
      allocation: AllocationResult,
      provider: ConstrainedEquationsOfMotionProvider,
      *,
      policy: ReplayPolicy,
  ) -> ReplayReport: ...
  ```
- **Postconditions:** Reports open-loop drift separately from feedback-assisted tracking and records the feedback torque norm budget $\lVert\tau_{\text{fb}}\rVert / \lVert\tau_{\text{alloc}}\rVert$ as a distinct channel.

---

## Phase 4: Product Surfaces (`NSX-40` Through `NSX-44`)

### NSX-40 (#12572): NSX Report Schema, Facade, Service, and Async Cancellable Job Routes

- **Modules:** `schemas/loop-nullspace-report-v1.json`, `src/shared/python/loop_nullspace/facade.py`, `src/api/services/loop_nullspace_service.py`, `src/api/routes/analysis.py`
- **Signatures:**
  ```python
  def analyze(
      trajectory: Sequence[ClosedLoopSnapshot],
      *,
      preset_name: str = "envelope_only",
      options: AnalysisOptions | None = None,
  ) -> NSXReport: ...
  ```
- **Routes:** `POST /analysis/loop-nullspace/jobs`, `GET /analysis/loop-nullspace/jobs/{id}`, `DELETE /analysis/loop-nullspace/jobs/{id}` with content-addressed result cache.

### NSX-41 (#12573): Output-Policy Integration (`display_decision`)

- **Module:** `src/shared/python/loop_nullspace/output_policy_bridge.py`
- **Signatures:**
  ```python
  def display_decision(
      record: QuantityRecord,
      surface: Literal["swing_report", "research_view"],
  ) -> DisplayDecision: ...
  ```
- **Postconditions:** Enforces ADR-0054 §5: `swing_report` withholds uncalibrated loads (`ALLOWED = False`); `research_view` admits `INVARIANT` with provenance badge and `FAMILY_DEPENDENT` only as band-first series with hypothesis badges.

### NSX-42 (#12574): Framework-Free NSX Plot Model

- **Module:** `src/shared/python/loop_nullspace/plot_model.py`
- **Signatures:**
  ```python
  def build_plot_model(
      report: NSXReport,
      decisions: Mapping[str, DisplayDecision],
  ) -> NSXPlotModel: ...
  ```
- **Postconditions:** Emits serializable panels (`invariants`, `envelopes`, `hand_split`, `allocation_overlay`, `diagnostics`) with `DETERMINED_SIGN`, `INDETERMINATE`, and `UNBOUNDED` segment tags and binding hypothesis annotations; forbids standalone representative curves without their envelope band.

### NSX-43 (#12575) and NSX-44 (#12576): PyQt6 Tabs and React Panel With Cross-UI Parity

- **Modules:** `src/tools/grip_wrench_plots/gui.py` (PyQt6) and `ui/src/components/analysis/LoopNullSpacePanel.tsx` (React)
- **Contract:** Both UIs render `NSXPlotModel` directly without client-side math, backed by `QThreadPool` / async job polling and preset selector, and register `loop_nullspace_exploration` in `src/config/feature_parity.json`.

---

## Phase 5: Engine Breadth and Conformance (`NSX-50` Through `NSX-56`)

### NSX-50 (#12578): `pose_interchange` Dynamic State and Dual Force/Wrench Transforms

- **Module:** `src/shared/python/pose_interchange/dynamic_transforms.py`
- **Signatures:**
  ```python
  def transform_velocity(adapter: PoseInterchangeAdapter, q: np.ndarray, v: np.ndarray) -> np.ndarray: ...
  def transform_acceleration(adapter: PoseInterchangeAdapter, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray: ...
  def transform_generalized_force(adapter: PoseInterchangeAdapter, q: np.ndarray, tau: np.ndarray) -> np.ndarray: ...
  ```
- **Postconditions:** Preserves virtual power $\tau^\top v$ across engine coordinate conventions to `1e-12`.

### NSX-51 (#12579), NSX-52 (#12580), NSX-53 (#12581), NSX-54 (#12582), NSX-55 (#12583): Per-Engine Closed-Loop Snapshots

- **Modules:**
  - `NSX-51` (MuJoCo hard-KKT): `src/engines/physics_engines/mujoco/python/force_adapter.py` (`kkt_regularization = 0.0` for `FULL` capability; stock soft stepping separated in `divergence_registry.yaml`).
  - `NSX-52` (MyoSuite): `src/engines/physics_engines/myosuite/python/full_body_parity.py` (delegates to MuJoCo hard-KKT adapter and records `kkt_regularization`).
  - `NSX-53` (Drake): `src/engines/physics_engines/drake/python/force_adapter.py` (`CalcMassMatrix`, `CalcBiasTerm`, `CalcGravityGeneralizedForces`, `CalcBiasSpatialAcceleration`, replacing quarantined synthetic adapter).
  - `NSX-54` (OpenSim): `src/engines/physics_engines/opensim/python/force_adapter.py` (delegates to `OpenSimFullBodyParityAdapter`).
  - `NSX-55` (Simscape R2025b): `src/engines/simscape/loop_nullspace_snapshot.py` (loads MATLAB R2025b exported snapshot or records deferred validation).

### NSX-56 (#12584): Two-Track Cross-Engine Conformance

- **Module:** `tests/integration/cross_engine/test_loop_nullspace_conformance.py`
- **Contract:** Projects states onto closure via `project_to_closure` first; Track N asserts canonical $(M, h, J_c, \dot{J}_c v)$ agreement to `1e-6` scaled against Pinocchio; Track S asserts invariant and static-envelope agreement within propagated numerical tolerance.

---

## Phase 6: Lower-Body Loop and Uncertainty Ensembles (`NSX-60` Through `NSX-62`)

### NSX-60 (#12585): Lower-Body Two-Foot Unilateral Loop With Optional GRF Observations

- **Module:** `src/shared/python/loop_nullspace/lower_body_loop.py`
- **Signatures:**
  ```python
  def include_ground_loop(
      snapshot: ClosedLoopSnapshot,
      *,
      use_observed_grf: bool = False,
  ) -> Decomposition: ...
  ```
- **Postconditions:** Incorporates $J_g$ into $A = [S^\top \;\; J_c^\top \;\; J_g^\top]$; when `use_observed_grf=True` and `grf_observed` is present, substitutes observed GRF with declared measurement covariance and reduces nullity by $\operatorname{rank}(J_{g,u}^\top)$.
- **Golden Fixture:** `fixtures["floating_base_two_arm"]`

### NSX-61 (#12586): Kinematic Ensemble Over Admitted Trajectory Samples

- **Module:** `src/shared/python/loop_nullspace/ensemble.py`
- **Signatures:**
  ```python
  def kinematic_ensemble(
      source: TrajectorySampleSource,
      n_samples: int,
      seed: int,
  ) -> KinematicEnsembleReport: ...
  ```
- **Postconditions:** Reprojects samples via `project_to_closure`, gates each through `admit` (`NSX-14`), counts rejected samples, and computes the union and intersection of feasible envelopes across admitted samples.

### NSX-62 (#12587): Parameter Ensemble Over Identifiability-Gated Inertial Parameters

- **Module:** `src/shared/python/loop_nullspace/parameter_ensemble.py`
- **Signatures:**
  ```python
  def parameter_ensemble(
      base_snapshot_sequence: Sequence[ClosedLoopSnapshot],
      identifiability: IdentifiabilitySubspace,
      n_samples: int,
      seed: int,
  ) -> ParameterEnsembleReport: ...
  ```
- **Postconditions:** Perturbs inertial parameters strictly within $\ker(Y_{\text{ident}})$ subject to the pseudo-inertia positive-definite LMI (`estimation/mosaic/inertial.py`), preserving MOSAIC-identified base combinations to `1e-10` and reporting sampled envelope coverage.

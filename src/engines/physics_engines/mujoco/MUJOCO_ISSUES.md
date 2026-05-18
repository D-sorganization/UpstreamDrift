# MuJoCo Parity — Implementation Issues

These six issues constitute the work to bring MuJoCo to parity with the
Simscape Multibody motion-matching pipeline. Bulk-create with `gh issue create`.
See `MUJOCO_PARITY_SPEC.md` for the architectural backdrop. All work targets
`main` (per the GAAI fleet policy). Branch names follow `feat/mujoco-...`.

---

## Issue: Fix gravity-constant interpolation bug in MuJoCo MJCF generators

**Labels:** mujoco, motion-matching, priority:high, bug
**Depends on:** —

**Goal.** Make the three `_golf_swing_*_xml.py` model variants compile under
`mujoco.MjModel.from_xml_string`. Today **all three are broken** because
`GRAVITY_M_S2` (and friends) are imported as `PhysicalConstant` instances
from `src.shared.python.core.constants` and the f-string interpolates
`repr(GRAVITY_M_S2)` literally — so the MJCF gravity attribute renders as
`gravity="0 0 -PhysicalConstant(9.807, unit='m/s^2')"`, which fails XML
parsing with `XML Error: bad format in attribute 'gravity'`.

**Spec reference.** `MUJOCO_PARITY_SPEC.md` §1.3 (broken/stubbed) and §3
(model recommendation).

**Deliverable.**

- Cast `GRAVITY_M_S2`, `GOLF_BALL_MASS_KG`, `GOLF_BALL_RADIUS_M`,
  `DEFAULT_TIME_STEP` to `float()` at module load (already done for some;
  apply consistently to _every_ f-string interpolation site).
- Equivalent fix in `_golf_swing_advanced_xml.py`,
  `_golf_swing_full_body_xml.py`, `_golf_swing_upper_body_xml.py`.

**Acceptance.**

- New test `tests/motion_matching/mujoco/test_model_builder.py::test_all_three_variants_compile`
  loads each XML and constructs an `MjModel` without raising.
- Test asserts `model.opt.gravity[2] ≈ -9.807` for each variant.

**Out of scope.** Adding actuators (ISSUE-MUJOCO-3), YAML-driven build
(ISSUE-MUJOCO-2 + PARITY-DIMENSIONS).

**Size.** XS (≤ 50 LoC, single-file edit per variant + 1 test).

---

## Issue: Add `_model_builder.py` to load + cache compiled MuJoCo MJCF

**Labels:** mujoco, motion-matching, priority:high
**Depends on:** ISSUE-MUJOCO-1

**Goal.** Provide a single entry point for the rest of the parity package
to obtain a compiled `MjModel`, without each caller re-running `from_xml_string`.
Also defines `<actuator><motor joint="..."/></actuator>` blocks that the
polynomial-torque driver will write into.

**Spec reference.** `MUJOCO_PARITY_SPEC.md` §2.1 (file-by-file deliverables)
and §3.2 (club attachment).

**Deliverable.**

- `src/engines/physics_engines/mujoco/motion_matching/_model_builder.py`:
  ```python
  @lru_cache(maxsize=4)
  def build_model(variant: Literal["upper","full","advanced"]) -> CompiledModel:
      ...
  ```
  where `CompiledModel` is a frozen dataclass holding `model`, `data` factory,
  joint-id list, club-grip body id, club-head body id.
- Add `<motor>` actuators to each MJCF, one per controlled joint, with `gear=1`.
- Document the joint-id ordering convention; expose it as
  `CompiledModel.joint_names: list[str]`.

**Acceptance.**

- `test_model_builder.py::test_build_model_returns_consistent_id_order`:
  asserts joint name list is deterministic across calls.
- `test_model_builder.py::test_actuators_present`: asserts `model.nu == len(joint_names)`.
- `test_model_builder.py::test_club_grip_body_id_resolves`.

**Out of scope.** YAML-driven generation (gated on PARITY-DIMENSIONS landing).

**Size.** S (~150 LoC including XML edits across 3 variants and tests).

---

## Issue: `simulate_with_coefficients.py` + polynomial-torque driver

**Labels:** mujoco, motion-matching, priority:high
**Depends on:** ISSUE-MUJOCO-2

**Goal.** Provide the canonical forward-sim wrapper required by
`CROSS_ENGINE_PARITY_SPEC.md §2.2`. Given a coefficient vector
`θ ∈ ℝ^(n_joints*7)`, run the MuJoCo model for `T` seconds applying
`τ_j(t) = Σ_{k=0..6} θ[j,k] · t^k` to each motor, and return a `SimOutput`
with `time, grip, grip_quat, clubhead, club_quat, tau, omega`.

**Spec reference.** `MUJOCO_PARITY_SPEC.md` §2.2 (signatures), §2.3 (driver).
Mirrors the Simscape `simulate_with_coefficients` callback (shared/COST_FUNCTION_SPEC.md).

**Deliverable.**

- `src/engines/physics_engines/mujoco/motion_matching/simulate_with_coefficients.py`
- `src/engines/physics_engines/mujoco/motion_matching/_torque_driver.py`
  (`PolynomialTorqueDriver` class wired via `mujoco.set_mjcb_control`).
- Re-uses the shared `SimOutput` dataclass from
  `src/shared/python/motion_matching/cost.py` — do not redefine.
- `SimOptions` frozen dataclass: `variant`, `T_s=0.3`, `dt=None` (use model's
  own timestep), `t0=0.0`, `output_rate_hz=1000`.

**Acceptance.**

- `test_simulate_with_coefficients.py::test_zero_torque_falls_under_gravity`:
  with θ=0, the grip Z drops monotonically.
- `test_simulate_with_coefficients.py::test_output_shapes`: every array has
  `N = round(T_s * output_rate_hz) + 1` rows.
- `test_simulate_with_coefficients.py::test_quaternion_unit_norm`:
  `‖grip_quat[k]‖ ≈ 1` to within `1e-6` for all k.
- `test_torque_driver.py::test_callback_uninstalls_cleanly`: after running
  `simulate_with_coefficients` twice with different θ, the second run uses
  the new θ (no global-state leakage between calls).

**Out of scope.** The fit driver itself (ISSUE-MUJOCO-4).

**Size.** M (~300 LoC + 4 tests).

---

## Issue: `fit_swing_mujoco.py` canonical fit driver

**Labels:** mujoco, motion-matching, priority:high
**Depends on:** ISSUE-MUJOCO-3, PARITY-COST

**Goal.** Mirror the Simscape `fit_swing_fmincon` driver in MuJoCo. Consumes
a canonical `ClubTarget` from `CLUB_IK_SPEC.md`, runs scipy SLSQP (or L-BFGS-B
behind a flag) with `compute_cost` from
`src/shared/python/motion_matching/cost.py`, and returns a provenance-rich
`FitResult`.

**Spec reference.** `MUJOCO_PARITY_SPEC.md` §2.2 (fit signature),
§5.1 (DbC pattern), §6.2 (perf targets);
`shared/COST_FUNCTION_SPEC.md`; `shared/CODING_STANDARDS.md` §provenance.

**Deliverable.**

- `src/engines/physics_engines/mujoco/motion_matching/fit_swing_mujoco.py`:
  ```python
  def fit_swing_mujoco(target: ClubTarget, opts: FitOptions) -> FitResult:
  ```
- `FitOptions` frozen dataclass holding `CostOptions`, `SimOptions`,
  `MinimizerOptions`, `rng_seed`.
- `FitResult` mirrors Simscape's: `coefficients, final_rmse_m,
final_total_work_J, solver, solver_options, target_hash, git_commit,
mujoco_version, duration_s, timestamp_utc`.
- Bounds via the shared `build_coefficient_bounds(n_joints)` helper.

**Acceptance.**

- `test_fit_swing_mujoco.py::test_synth_then_fit_recovers_theta`:
  fit a synthesized target back to its θ_truth, RMSE < 1 mm,
  `‖θ_fit - θ_truth‖∞ < 1e-2`.
- `test_fit_swing_mujoco.py::test_result_struct_complete`: every field
  in the CODING_STANDARDS provenance block is populated.
- `test_fit_swing_mujoco.py::test_target_hash_is_sha256`: 64 hex chars.
- Performance smoke test (marked `@pytest.mark.benchmark`): one full fit
  completes in < 2.0 s wall-clock on CI hardware (target < 0.5 s on
  developer hardware; CI gets a more lenient bound).

**Out of scope.** Multi-start / multistart bookkeeping (follow-up issue).
GPU / MJX path (deferred per spec §6.3).

**Size.** M (~400 LoC including bounds wiring + 5 tests).

---

## Issue: `synthesize_target_from_coefficients` engine implementation

**Labels:** mujoco, motion-matching, priority:medium
**Depends on:** ISSUE-MUJOCO-3, PARITY-LOADERS

**Goal.** Implement the TDD oracle: run the MuJoCo forward model with a
known θ, build the resulting trajectory into a canonical `ClubTarget`.
Replace the `NotImplementedError` stub at
`src/shared/python/motion_matching/loaders/synthetic.py` with a dispatcher
that delegates to this engine when `opts.engine == "mujoco"`.

**Spec reference.** `MUJOCO_PARITY_SPEC.md` §2.2; `CLUB_IK_SPEC.md`
§"Synthetic source"; `loaders/synthetic.py` line 36 (`NotImplementedError`).

**Deliverable.**

- `src/engines/physics_engines/mujoco/motion_matching/synthesize_target.py`:
  ```python
  def synthesize_target_from_coefficients(
      theta: np.ndarray, opts: AlignOptions
  ) -> ClubTarget:
  ```
- Update `src/shared/python/motion_matching/loaders/synthetic.py` to dispatch
  on `opts.engine` (add the field if missing) — import is **lazy** to keep
  the GUI from pulling mujoco transitively.
- Provenance: `SourceProvenance(filename="synthetic", format="synthetic",
subject_id="theta_seed_<n>", trial_id="...", sha256=sha256(theta.tobytes()))`.

**Acceptance.**

- `test_synthesize_target.py::test_validates_against_clubtarget_schema`:
  the result construct passes the `_validate_clubtarget` postcondition
  block in `club_target.py`.
- `test_synthesize_target.py::test_provenance_sha_reproducible`: same θ
  → same sha256.
- `test_synthesize_target.py::test_dispatcher_routes_to_mujoco`: shared
  loader picks the right backend.
- `test_synthesize_target.py::test_round_trip_with_simulate`:
  asserts `synthesize_target_from_coefficients(θ).clubhead` matches
  `simulate_with_coefficients(θ).clubhead` exactly (within 1e-12).

**Out of scope.** Excel and C3D loaders (PARITY-LOADERS).

**Size.** S (~200 LoC + 4 tests).

---

## Issue: `viz/render_swing.py` — thin renderer over `mujoco.viewer`

**Labels:** mujoco, motion-matching, priority:low
**Depends on:** ISSUE-MUJOCO-3

**Goal.** Provide a CLI / programmatic visualizer that replays a `SimOutput`
in `mujoco.viewer` (or `mujoco.Renderer` for headless PNG / MP4 output).
Mirrors the Simscape `animate_trajectory_overlay.m` thin renderer at
`shared/animate_trajectory_overlay.m`.

**Spec reference.** `MUJOCO_PARITY_SPEC.md` §2.1 (visualisation layout);
`shared/VISUALIZATION_SPEC.md`.

**Deliverable.**

- `src/engines/physics_engines/mujoco/motion_matching/viz/render_swing.py`:
  ```python
  def render_swing(
      sim_out: SimOutput,
      target: ClubTarget | None = None,
      *,
      mode: Literal["live", "headless"] = "live",
      output_path: Path | None = None,
      fps: int = 60,
  ) -> Path | None:
  ```
- Live mode: opens `mujoco.viewer.launch_passive` and replays the trajectory
  in real time; if `target` is given, draws a ghost trace of the measured
  clubhead path.
- Headless mode: `mujoco.Renderer` to write an MP4 to `output_path`.

**Acceptance.**

- `test_render_swing.py::test_headless_mode_writes_file`
  (skip on `requires_gl` if the test runner can't render — gate with
  the existing `headless_safe` marker).
- `test_render_swing.py::test_live_mode_smoke` (skipped under
  `requires_gl=False`): instantiates the viewer, steps once, closes.
- No GUI launches when imported in headless CI.

**Out of scope.** Side-by-side multi-engine playback (cross-engine spec).
Interactive scrubbing UI (`golf_gui_docker.py` already exists; that's
separate).

**Size.** S (~200 LoC + 2 tests, both `headless_safe` / `requires_gl`-gated).

---

## Cross-cutting acceptance for the parity work as a whole

Once all six issues are merged:

1. `python3 -m pytest tests/motion_matching/mujoco -n auto --timeout=60` passes.
2. `python3 -c "from src.engines.physics_engines.mujoco.motion_matching import \
simulate_with_coefficients, fit_swing_mujoco, synthesize_target_from_coefficients"`
   succeeds with no transitive heavy GUI imports.
3. Synth-then-fit acceptance gate (`MUJOCO_PARITY_SPEC.md` §4 closing line)
   holds on developer hardware.
4. CI passes: ruff check, ruff format --check, file-size budget,
   coverage threshold per `pyproject.toml`.

After that, the cross-engine harness (sibling spec) can plug MuJoCo into the
multi-engine `compare_fits.py` runner without further engine-side changes.

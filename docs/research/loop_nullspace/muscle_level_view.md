# Muscle-Level View: Scoping a Muscle-Redundancy Layer Above the NSX Envelopes

Research report for NSX-73 (issue #12591, parent epic #12593). Status:
exploratory scoping. No product code is added; no numerical result about any
golfer is claimed.

## Question and Review Finding A4

Review finding A4 says joint torques cannot represent co-contraction or
stiffness. The NSX envelopes (NSX-22) bound the joint torques and hand-split
coordinates that are consistent with the closed-loop dynamics. A bound on a
torque is silent about how many muscles produce it and how hard they are
activated against each other. This report scopes a layer above the envelopes
that adds that muscle-level freedom, states what the repository already
provides, and lists what is missing.

## How Every Claim Was Checked

Every file, function and number below was read or measured in this worktree
(branch `feat/12591-nsx-muscle-level-view`, based on NSX-52). Measurements are
labelled "measured" and carry the command that reproduces them in
[Reproduction](#reproduction). Items taken from a document rather than
re-measured are labelled "cited".

An important finding about the base: the envelope modules named in the issue
(`envelopes.py`, `feasible_set.py`, `decomposition.py`) are **not present in the
NSX-52 base tree**. `git ls-files src/shared/python/loop_nullspace` here lists
only `closure_convention.py`, `contracts.py`, `evidence.py`, `fixtures.py`,
`planar_model.py`, `snapshot.py`, `spatial_fixture.py` and `__init__.py`. They
exist on sibling stack branches (`origin/feat/12564-12565-12569-nsx-feasible-set-envelopes-restriction`
for `feasible_set.py` and `envelopes.py`; `origin/feat/12560-nsx-decomposition`
for `decomposition.py`), which are not ancestors of this branch. Statements about
those modules below are read from those branch tips and may change before merge.

## Muscle Forces and Joint Torques

### The Moment-Arm Map

For `m` muscles and `n` coordinates, the repository's convention is that muscle
`k` contributes `R[k, c] * F_k` to coordinate `c`, where `R` is the `(m, n)`
moment-arm matrix with `R = -dL/dq` (muscle-tendon length against coordinate).
This is stated in `src/shared/python/biomechanics/muscle_allocation.py`
(module docstring) and implemented for MyoFullBody in
`src/shared/python/myofullbody/redundancy.py`:
`MuscleBasis.evaluate` returns a `FrameBasis` whose `moment` is
`-actuator_moment @ phi[:, columns]` (MuJoCo `actuator_moment` mapped to spec
coordinates by the rate map `phi`).

The muscle contribution to generalised force is

    tau_mus = R^T f,        f = f_p + diag(f_a) a,        0 <= a <= 1,

where `f_p` is the force at zero activation and `f_a` the force per unit
activation. `FrameBasis` documents exactly this split (`active`, `passive`), and
`MuscleBasis.evaluate` states the postcondition that actuation is linear in
activation at a fixed state; `redundancy.solve_frame` and
`musculoskeletal_static_opt.solve_frame` both use
`reserve = tau - moment.T @ (passive + a * active)`. Linearity holds only at a
fixed length, velocity and (for the Hill model) fixed pennation. Over a
trajectory the pair `(f_p, f_a, R)` is state dependent, so the pullback is a
per-frame object. This matches the `STATIC` label the NSX envelope series uses.

The single-joint Hill-muscle fixture in `multi_muscle.py` is the same relation
with a constant moment arm: `MuscleGroup.compute_net_torque` sums
`moment_arm * force` (`tau = r x F`), and `create_elbow_muscle_system` builds
two flexors and one extensor. Measured: with all three at optimal fiber length
and zero velocity the forces at unit activation are 1000, 800 and 1200 N, and
the moment-arm row is `[0.04, 0.03, -0.035]` m. Test
`tests/unit/loop_nullspace/test_muscle_redundancy_example.py` verifies linearity
in activation against `HillMuscleModel.compute_force`.

### Pulling a Torque Envelope Back to Activations

The NSX allocation equation fixture (`docs/research/loop_nullspace/derivations/envelope_case.py`,
base problem E1) writes joint torque as `tau(lambda) = tau_ID - J_c^T lambda`
with box capacity `|tau_i| <= c_i`. The NSX feasible set lives in `lambda` (or,
in `feasible_set.py` on the stack branch, in equality-reduced null-space
coordinates `x(z) = x_0 + N z` with rows `A_ub z <= b_ub` traced to hypothesis
ids).

A muscle layer replaces the capacity box by the muscle-reachable set. Per
frame, with a reserve `rho` for coordinates the muscles do not cover:

    R^T (f_p + F_a a) + rho = tau_ID - J_c^T lambda,
    0 <= a <= 1,      rho = 0 (strict) or |rho| <= rho_max (declared).

This is a polytope in `(a, lambda, rho)`. Its three projections are:

1. onto `lambda`: the hand-split envelope restricted to what the muscles can
   realise at that pose. It is a subset of the NSX envelope built from the
   capacity box only if the box was not narrower than the muscle set;
   otherwise the two are different hypotheses, not a nested pair.
2. onto `tau`: a zonotope `{R^T (f_p + F_a a) : a in [0,1]^m}` (image of the unit
   box under a linear map, with `m` generators). It is generally not a box:
   biarticular muscles couple joints, and because `a >= 0` and forces are
   one-sided, the set is not symmetric about zero. The scalar-per-joint capacity
   table of NSX-04 (`capacity.py`) cannot represent that coupling.
3. onto `a`: the fiber of activation vectors, the object that carries
   co-contraction and stiffness (next section).

The LP machinery to compute min/max of any linear quantity over this polytope
exists in design: `static_envelopes` (stack branch) takes `FeasibleSet`
objects and `LinearQuantitySpec` quantities and solves two LPs per quantity per
frame. The muscle layer is the same call with a different `FeasibleSet`, built
from an augmented variable `x = [tau; lambda; a]` and one extra equality, so no
new solver is needed for linear quantities.

## What Co-Contraction and Stiffness Add

### The Null Space of the Moment-Arm Map

Joint torque depends on `a` only through `G a` with `G = R^T F_a` of shape
`(n, m)`. When `m > rank(G)`, every torque has a whole polytope of activation
vectors that produce it; the affine hull of that polytope is parallel to
`null(G)`, of dimension `m - rank(G)`. Moving along it leaves every net joint
torque unchanged and changes only which muscles pull against which. That motion
is co-contraction. A torque envelope, however tight, quotients it away; this is
exactly review item A4.

Measured on the repository's elbow fixture: `R` has shape `(1, 3)`, rank 1,
null-space dimension 2. Moving the activations from `(0.5, 0.5, 0.2)` by a null
vector scaled so the largest component changes by 0.1 leaves
`AntagonistPair.compute_net_torque` unchanged (agreement below 1e-9) while the
triceps activation increases. This is the mechanism in one joint; the test
asserts it.

Measured on the real model: the MyoFullBody asset at pinned commit
`93b0ca8f4ec90c9899ee7f05fee561e9911da91b` (416 muscle actuators, `nq = 129`,
`nv = 128`, cited in the committed receipt
`docs/development/full_body_models/evidence/myofullbody/receipt_driver.json`)
gives, at the pose in which the asset loads, an `actuator_moment` matrix of
shape `(416, 128)` with numerical rank 110 (a gap from singular value 2.4e-3 to
3e-16, stable for relative tolerances 1e-12 to 1e-6). So the muscle-to-generalised-
force map in that pose has a nullity of 416 - 110 = 306 in muscle space, and 18 of
the 128 degrees of freedom are outside the range of any muscle (6 have no moment
arm at all). With activation bounds the fiber is a bounded polytope of up to
306 dimensions, not a subspace. This is the muscle-count redundancy, one pose,
default posture; it is not a swing result, and the rank of `G` (which also scales
by `F_a`) was not measured.

Be careful with interpretation: nullity here counts redundancy before
activation bounds, before a smoothness prior and before any objective. Bounds
shrink it: the fiber over a torque near capacity collapses toward a point.

### Stiffness

Joint stiffness has a part that depends only on geometry and force and a part
that depends on muscle stiffness. For muscle forces `f` with fiber stiffness
`k_k = dF_k/dl` and moment arms `R`, the joint stiffness matrix is

    K = R^T diag(k) R + sum_k f_k d^2 L_k/dq^2   (sign convention to be fixed).

At a fixed state, `k` is affine in `a` for a Hill-type model (active part scales
with `a`), so `K` is affine in `a`. Two consequences follow, both for the
fiber over a fixed torque:

- co-contraction raises `K` without changing `tau` (Hogan 1984, cited in the
  `multi_muscle.py` module docstring; not independently verified in this work);
- the extremes of any linear functional of `K` (for example its trace) over the
  fiber are LPs, so a stiffness envelope for trace is computable with the same
  engine as the torque envelope. A bound on the smallest eigenvalue of `K` is
  concave in `a` and needs an LMI or SDP. `highspy` and `clarabel` are installed
  in the venv, but no code in this report's scope was checked for use of a
  positive-semidefinite cone, so that path is unproven here.

The current Hill model gives no usable stiffness. Reading
`hill_muscle.py`: `compute_force` returns `(F_active + F_passive + F_damping) *
cos(alpha)` with a rigid tendon (`tendon_force` exists but `compute_force` never
calls it), a constant pennation angle, and a Gaussian active force-length
curve. Measured by central differences on the three fixture muscles at `a = 1`,
`dF/dl_CE` is -3376 N/m at 1.1 `l_opt` (identical for all three because
`F_max / l_opt` is 6667 N/m for each; a coincidence of the fixture) and about
249 N/m at `l_opt`, where the value is an average of one-sided slopes 0 and
about 497 N/m across the passive-curve kink. Two points matter: the active
stiffness on the descending limb of the Gaussian is negative, and the passive
curve is not smooth at `l_opt`. A fiber-only slope is not the musculotendon
stiffness the joint feels; that requires a series tendon, which this model does
not have. Any stiffness number from this model alone must not be reported.

## What Existing Code Supports

| Need | Verified location | Status |
| --- | --- | --- |
| Hill force, force-length, force-velocity, passive | `src/shared/python/biomechanics/hill_muscle.py`: `HillMuscleModel.compute_force`, `force_length_active`, `force_velocity` | Works; rigid tendon; no stiffness API |
| Antagonist pairs, constant moment arms, elbow fixture | `src/shared/python/biomechanics/multi_muscle.py`: `MuscleGroup`, `AntagonistPair`, `create_elbow_muscle_system` | Works; constant `r`; Rust path optional (`upstream_muscle`) |
| Tendon equilibrium, activation dynamics | `biomechanics/muscle_equilibrium.py` (`EquilibriumSolver`), `activation_dynamics.py`, `muscle_allocation.py` (`allocate_muscle_excitations`) | One-step bounded recruitment; module states it is not OpenSim parity |
| Per-frame static optimisation with reserves | `src/shared/python/myofullbody/redundancy.py`: `MuscleBasis.evaluate`, `solve_frame` (projected Newton), `group_reserve_metrics`, `qualification`; reference solver `opensim/python/musculoskeletal_static_opt.py::solve_frame` (BVLS) | Picks one point of the fiber (`min sum a^2`); does not expose the fiber |
| Moment arms from MuJoCo | `redundancy.py::MuscleBasis.evaluate` (`actuator_moment`); `myosuite/python/muscle_analysis.py::compute_moment_arms` (finite differences) | Two independent routes; the second perturbs `qpos[dof_idx]` for every `dof_idx < nv`, which equals the velocity index only if `nq == nv` (MyoFullBody has `nq = 129`, `nv = 128`) |
| Generic equality null space | `src/shared/python/motion_matching/force_nullspace.py`: `ForceNullSpace`, `NullSpaceAnalysis` (rank, nullity, singular values) | Reusable for `null(G)` |
| NSX null-space envelopes | `feasible_set.py`, `envelopes.py`, `decomposition.py` | Present on stack branches only (see above) |
| NSX MyoSuite snapshot | `src/engines/physics_engines/myosuite/python/force_adapter.py`; `tests/unit/loop_nullspace/test_myosuite_snapshot.py` | Reuses the MuJoCo path; the test substitutes plain MuJoCo for MyoSuite because MyoSuite is not importable |
| Antagonist and co-contraction concept | `multi_muscle.py` docstring and `__main__` stiffness proxy `K = (1000*0.5 + ...) * 0.04` | A hand-typed proxy, not a model quantity |
| Prior study of the same structure | `docs/research/musculoskeletal_swing/` (80 leg muscles, Rajagopal) and `docs/research/myofullbody_swing/` (416 muscles) | Status `NOT_QUALIFIED` |

MyoFullBody swing receipts (`receipt_driver.json`, `receipt_iron.json`, cited,
frames 366, stride 5) report reserve RMS over effort RMS of 0.65 (arms), 0.32
(legs) and 0.53 (trunk) for the driver, with iron comparable (arms 0.65, legs 0.33,
trunk 0.56), against a 0.10 limit. Both swings are `NOT_QUALIFIED`. The
neck is torque-actuated by six `neck_torque_*` actuators, so its reserve is
nominally zero by construction. This matters directly: a muscle layer built on
this basis would put most of its torque demand into reserves. A co-contraction
view of a basis that cannot yet carry the net torque would be uninformative.
Muscle saturation also appears in the receipts: five muscles stay above 90
percent of the window (driver) and two (iron).

## What Is Missing

1. **The MyoSuite runtime.** `import myosuite` fails in the venv
   (`ModuleNotFoundError: No module named 'myosuite'`, measured). NSX-52 tests run
   by substituting plain MuJoCo; the MyoFullBody asset itself is a MuJoCo model
   and loads without MyoSuite, as the measurement above did. A MyoSuite-specific
   muscle layer would be untested in this environment.
2. **Muscle data for the golfer.** No muscle parameters for the individual are
   available or invented here. The MyoFullBody geometry and the Rajagopal leg
   model are generic. Maximum isometric forces, optimal fiber lengths and
   tendon slack lengths for the subject are unmeasured. Without them the
   activation bounds map to unknown forces.
3. **A tendon-compliant muscle model with a smooth stiffness.** Required by the
   stiffness section. The MyoFullBody MuJoCo actuators are `gaintype` 2
   (MuJoCo muscle model, measured); their stiffness would have to be taken from
   MuJoCo's own muscle curves, which has not been examined.
4. **Pose-dependent stiffness terms.** The second term in `K` needs
   `d^2 L / dq^2`; no repository function computes it.
5. **A basis that closes the torque.** Reserves exceed the 10 percent limit
   (above). Upper-limb and trunk muscle sets are the limiting items; the
   `musculoskeletal_swing` notes record that no surveyed OpenSim model offers a
   3D shoulder or trunk set.
6. **Registry hypotheses.** `src/config/loop_nullspace_registry.json` lists
   `H-TORQUE-CAPACITY`, `H-EFFORT-CAPACITY`, `H-SMOOTH-TORQUE-RATE`,
   `H-GROSS-POWER-COST`, `H-GRIP-PRELOAD` and `H-FRICTION-CONE`. There is no
   muscle hypothesis (activation box, reserve bound). A new hypothesis id and a
   registered `torque_capacity`-like table for muscle parameters would be needed.
7. **A conic or LMI path** for eigenvalue-based stiffness bounds (above).
8. **Validation data.** EMG co-contraction ratios or a measured joint
   impedance. Both are external and are deferred in the NSX ledger; they are not
   obtainable by computation.

## Proposed Staged Plan

Each stage is a narrow PR with a ledger entry, and each acceptance is
software-correctness only.

**Stage 0: Fixture and Contract (this report, plus the elbow example).**
Acceptance: the report is merged; the elbow example test passes (3 tests, run
here).

**Stage 1: Muscle Basis Value Object.** A frozen dataclass wrapping a
`FrameBasis`-like `(f_p, f_a, R)` with shape and sign validation and the
`tau = R^T(f_p + a f_a)` map. Reuse `redundancy.FrameBasis`, do not duplicate it.
Acceptance tests: round trip against `solve_frame` residual at the returned
activations; the Hill elbow example gives `rank(G) = 1`, nullity 2; shape and
non-finite inputs raise `ValueError`.

**Stage 2: Pullback of a Torque Envelope.** Build a `FeasibleSet` over
`[tau; lambda; a]` with the muscle equality and the activation box, a new
hypothesis id for the box, and a reserve variable. Acceptance tests: on the E1
analytic fixture extended with a constructed `R` of full row rank, the lambda
envelope equals the exact rational vertex enumeration to 1e-9, mirroring the
NSX-22 test; the tau projection equals the zonotope support function
(`h(u) = sum_k max(0, u . G_k)` plus the passive term) for random `u`; an
activation box that cannot realise `tau_ID` reports `INFEASIBLE`.

**Stage 3: Co-Contraction Fiber.** Report, per frame and torque, the fiber
dimension (`m - rank G`, with rank from `ForceNullSpace.analyze`), the
min and max of a declared co-contraction index (for example total activation
`sum a`) over the fiber, and the Chebyshev margin inside the activation box.
Acceptance tests: elbow fixture min/max of `sum a` at a fixed torque equals the
closed-form LP vertices; unbounded directions never appear (box is bounded);
fiber collapses to a point at a torque on the zonotope boundary.

**Stage 4: Stiffness Envelope for Linear Functionals.** Only after a stiffness
model exists (Missing items 3 and 4). Acceptance tests: finite-difference check
of `K` against the force change under a small coordinate perturbation to a
stated tolerance; trace envelope equals LP vertices on the fixture; eigenvalue
bounds, if added, are checked against dense sampling of the fiber.

**Stage 5: MyoFullBody Frame Application.** Apply Stages 1 to 3 to one
MyoFullBody frame from the existing swing bundles with the existing reserves.
Acceptance: the receipt reports reserve fraction, fiber dimension and the
activation-bound margin with their provenance; status stays `NOT_QUALIFIED`
unless the reserve criterion passes in the same run.

**Stage 6: External Validation (deferred).** EMG or measured impedance. Record
in the deferred-validation catalog; do not close with computation.

## Non-Claims

- No stiffness, co-contraction level or muscle activation is inferred from
  net joint torques; this report shows only that the fiber exists, not where on
  it a golfer sits.
- No muscle forces, activations or stiffness values are reported for any golfer.
- The rank and nullity numbers are for one MuJoCo model at its loaded pose; they
  are not a statement about the swing, a tour average or the subject.
- The elbow example uses the fixture's invented parameters; it demonstrates the
  mechanism, not physiology.
- Muscle parameters, tendon compliance and EMG are unavailable; nothing here
  supplies them.
- The stack-branch modules are described as read, not as merged.
- Hogan 1984 is cited through a source docstring and has not been checked
  against the paper; Stanev and Moustakas 2019 is marked VERIFIED in the NSX
  registry for its abstract only.
- No scientific qualification: both underlying MyoFullBody studies are
  `NOT_QUALIFIED`.

## Reproduction

```bash
# Elbow fixture tests (rank 1, nullity 2, linearity, null-space step)
python3 -m pytest tests/unit/loop_nullspace/test_muscle_redundancy_example.py -n 0 -q

# MyoFullBody moment-arm rank at the as-loaded pose (needs the cached asset,
# fetched by scripts/fetch_myofullbody.py)
python3 - <<'PY'
import numpy as np, mujoco
from src.shared.python.myofullbody import assets
tree = assets.cache_dir(assets.load_manifest())
model, data = assets.load_myofullbody(tree)
mujoco.mj_forward(model, data)
m = np.zeros((model.nu, model.nv))
mujoco.mju_sparse2dense(m, data.actuator_moment, data.moment_rownnz,
                        data.moment_rowadr, data.moment_colind)
s = np.linalg.svd(m, compute_uv=False)
rank = int((s > 1e-9 * s[0]).sum())
print(model.nu, model.nv, rank, model.nu - rank)
PY
```

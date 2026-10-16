# ADR-0056: Canonical Source for Golfer Kinematics and the Closure Jacobian

- Status: Accepted
- Date: 2026-10-15
- Decision Makers: Repository owner (directed implementation of #12671 on 2026-10-15)
- Related Issues/PRs: #12671; UpstreamDrift #12608, #12668, #12669, #12792; Tools #5626, #5627, #5628

## Context

UpstreamDrift and Tools both implemented the planar golfer upper-body model:

- UpstreamDrift: `src/shared/python/pendulum_simulator/` (`golfer_kinematics`,
  `golfer_constraints`, `golfer_dynamics`, `physics_golfer_jax`,
  `native_backend`, `constraint_solver`);
- Tools, vendored at `vendor/ud-tools`:
  `src/pendulum_simulator/src/double_pendulum_golf/` (the same module names)
  and the Rust `pendulum-core/src/golfer.rs`.

The club y-sign error had to be fixed twice: in UpstreamDrift (#12608, PRs
#12668 and #12669) and in Tools (#5626, PRs #5627 and #5628). PR #12792 moved
`vendor/ud-tools` to the Tools commit that holds the fix. After that bump the
two copies of the forward kinematics and the closure agree numerically. They
differ only in formatting, contract spelling and three UpstreamDrift-only
additions:

1. `golfer_kinematics.club_shaft_offset`, `club_shaft_offset_derivative` and
   `point_on_club`. These are the club formula in a backend-agnostic form,
   used by the JAX twin and the native backend.
2. `golfer_constraints.INDEPENDENT_CLOSURE_ROWS = (0, 1)`.
3. A least-squares KKT on those rows in `constraint_solver`.

Repository policy (`CLAUDE.md`, "Where to edit shared code") makes Tools the
source of truth for code vendored as `vendor/ud-tools`.

## Decision

### 1. Tools Is Canonical for the Golfer Kinematics and Closure

The vendored Tools package `double_pendulum_golf` is the single
implementation of:

- the forward kinematics: `golfer_kinematics.forward_kinematics` and
  `_hub_position`;
- the closure: `golfer_constraints.constraint_vector`,
  `analytical_constraint_jacobian` and `numerical_constraint_jacobian`;
- the quantities derived from them in the same module:
  `linear_accelerations`, `net_joint_forces` and `friction_torque_vector`.

A change to any of these is made in Tools first and reaches UpstreamDrift
through a `vendor/ud-tools` bump.

### 2. UpstreamDrift Keeps Its Import Paths as Thin Adapters

`golfer_kinematics` and `golfer_constraints` keep their public names. Each
delegated name is built by `pendulum_simulator._canonical_golfer.delegate`.
That function:

- loads the Tools package under a private module name through the existing
  seam loader `_seam_redirect.load_pinned_tools_package`, so the package is
  never grafted onto UpstreamDrift's own `pendulum_simulator` namespace;
- converts each UpstreamDrift `GolferParams` argument to the Tools dataclass
  field by field. The Tools functions check `isinstance(p, GolferParams)`
  against their own class. The conversion is cached per value, because the
  dataclass is frozen;
- exposes the Tools function as `__wrapped__`.

The adapters emit **no `DeprecationWarning`**. The relocation shim in
`AGENTS.md` section C applies to a path that is being retired. These paths are
not being retired: they are UpstreamDrift's stable surface for about ten
internal consumers (`constraint_solver`, `golfer_dynamics`, `club_forces`,
`hub_options`, the motion-matching golfer adapter and others). A warning would
fire on every internal import and ask callers to move to a private module
name.

`GolferParams` stays an UpstreamDrift class. Aliasing it to the Tools class
would give every UpstreamDrift parameter object a private module name, which
breaks pickling by reference and `repr`-based diagnostics. It would also widen
this change to every consumer of `physics_golfer`. The parity test pins the
two dataclasses to the same field names instead, and the adapter copies every
field explicitly, so the conversion is lossless. The defaults are not
identical: the default `g` is standard gravity (9.80665 m/s²) in UpstreamDrift
and 9.81 m/s² in Tools. Because no field is left to its default, a converted
object keeps UpstreamDrift's value.

### 3. The Club Helpers Stay in UpstreamDrift, Pinned to Tools

`club_shaft_offset`, `club_shaft_offset_derivative` and `point_on_club` stay in
`golfer_kinematics`. The JAX twin (`physics_golfer_jax`) needs a traceable
form of the club formula, and the NumPy native-backend path needs a
contract-checked form. The Tools copy exposes neither. The parity test checks
both forms against the Tools club points and the Tools club-tip Jacobian
column. Moving the helpers to Tools is optional: it would let these lines be
deleted here, but it is not needed for a single source of truth.

### 4. The Independent Closure Rows and the Least-Squares KKT Are UpstreamDrift Solver Policy

`INDEPENDENT_CLOSURE_ROWS` and `constraint_solver._solve_independent_kkt` stay
in UpstreamDrift. They consume the Tools Jacobian; they do not define it.

- The Tools closure is complete without them. `constraint_vector` and its
  Jacobian are correct and fully specified with all four rows, and rows 2-3
  are an orthogonal rotation of rows 0-1.
- Choosing which rows to invert, and solving the singular KKT (the massless
  wrist DOFs leave zero columns) by minimum-norm least squares, is a decision
  of the constrained-dynamics consumer.
- No Tools change is therefore needed for this issue, and the UpstreamDrift
  PR does not depend on a Tools PR.

The risk is that a future Tools change reorders the rows or alters their rank.
The parity test guards against this on every vendor bump. It checks that the
independent rows have full row rank and span every Tools row on the closure
manifold. It also checks that the KKT solved on those rows drives the
acceleration-level residual of all four Tools rows to zero, with zero
multipliers on the redundant rows.

Tools' own `constraint_solver` still assembles the full four-row KKT and uses
`numpy.linalg.solve`, falling back to least squares only on `LinAlgError`.
That is the conditioning issue #12608 removed here. Whether Tools adopts the
independent-row solve is a Tools decision (see Follow-ups). If it does, the
constant moves to Tools, and UpstreamDrift re-exports it and deletes its copy.

## Alternatives Considered

1. **UpstreamDrift canonical, Tools re-exports.** Rejected. It inverts the
   vendoring direction (UpstreamDrift vendors Tools, never the reverse) and
   contradicts the repository policy.
2. **Re-export the whole `pendulum_simulator` package through
   `TOP_LEVEL_REDIRECTS`.** Rejected for now. This is how `movement_optimizer`
   was retired (#12267). Here, however, the copies genuinely differ elsewhere:
   `physics_golfer_jax` and the `constraint_solver` KKT, and `native_backend`
   (the double-pendulum `hub` key). Replacing the whole package would change
   behaviour beyond this issue.
3. **Identity re-export (`forward_kinematics = tools.forward_kinematics`).**
   Rejected. The Tools functions reject UpstreamDrift `GolferParams` instances
   (`isinstance` against the Tools class), so a converting adapter is needed.
4. **Move `INDEPENDENT_CLOSURE_ROWS` and the least-squares KKT to Tools first.**
   Deferred. It is not required for one canonical Jacobian. It would also put
   a Tools PR and a vendor bump on the critical path of a refactor that
   changes no numbers.

## Consequences

- Positive: one implementation of the golfer kinematics and closure. A sign or
  convention fix now lands once, in Tools, and the parity test fails
  UpstreamDrift CI if a vendor bump breaks a pinned assumption.
- Positive: UpstreamDrift callers and tests are unchanged.
- Negative: each call converts parameters once per distinct value through a
  small cache, which adds a dictionary lookup per call.
- Negative: `golfer_dynamics` (mass matrix, Coriolis, gravity, FK Jacobians),
  `physics_golfer_jax` and `native_backend` remain duplicated. The FK
  Jacobians are pinned numerically by the parity test, and the JAX twin is
  pinned by `test_physics_golfer_jax_parity` against the now-canonical NumPy
  path.
- Follow-ups:
  - Tools: decide whether `double_pendulum_golf.constraint_solver` should
    solve on the independent rows (the #12608 conditioning fix).
  - Owner: choose one default gravity for `GolferParams.g` (UpstreamDrift
    9.80665, Tools 9.81 m/s²). The adapters are unaffected, but code that
    builds the Tools dataclass directly gets a different default.
  - UpstreamDrift: retire the remaining duplicates (`golfer_dynamics`, then
    `physics_golfer_jax`) the same way, once their Tools counterparts match.

## Validation

`tests/unit/pendulum_simulator/test_golfer_canonical_parity.py` runs in the
standard unit lane. It loads the pinned Tools package independently of the
adapters and asserts:

- each adapter's `__wrapped__` is the Tools function;
- the two `GolferParams` dataclasses have the same field names, and the
  conversion copies every field (gravity included);
- the forward kinematics and the closure match Tools exactly on
  representative configurations, including the scapula links;
- the UpstreamDrift club helpers and `golfer_dynamics.analytical_fk_jacobians`
  match Tools to rounding;
- the independent closure rows and the least-squares KKT are valid for the
  Tools Jacobian on the closure manifold.

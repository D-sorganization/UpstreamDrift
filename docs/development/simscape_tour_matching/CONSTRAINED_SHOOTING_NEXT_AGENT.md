# Constrained Shooting Implementation Assignment

## Current Implementation Checkpoint

Backend77eb87cca plus type fix5c6765de4 is implemented and pushed;33 relevant
tests and normal push checks pass. Do not repeat the implementation plan below
as if it were unstarted. The current next action is the native projected-equality
audit recorded at the TOP of NATIVE_PORT_CHECKPOINT_20260911.md. Qualify its
actual results before starting optimization. Historical design steps below
remain useful acceptance and review requirements, not instructions to duplicate
the new equality_least_squares.py backend.

## Evidence and Scope

Read the top of NATIVE_PORT_CHECKPOINT_20260911.md and the raw run13 and
conditioning receipts first. Run13 is terminal, unaccepted, through0.85 s.
The column-normalized Jacobian at its initial point has singular values from
7.53552 to3.09930e-10. The210 strongest directions retained at relative1e-4
barely reduce the linearized cost; this count equals the five42-dimensional
shooting charts. This is consistent with penalty dominance, not proof that
all210 directions contain only continuity information. Larger linearized gains
require weak directions and steps exceeding current bounds. Linear predictions
are not forward simulations or evidence that a bounded optimum is impossible.

Implement one explicitly constrained shooting backend using existing native
forward windows and analytic sensitivities. Preserve the least-squares backend
as the default and preserve all previous runtime directories. Do not start
another unchanged penalty/budget trial. No inverse-dynamics assumption is needed.

## First Bounded Implementation

1. Read repo guidance, check/renew issue9967 ownership, discover existing
   constrained optimizers and reuse public modules. Refactor shared shooting
   evaluation into marker residual/Jacobian and physical continuity
   residual/Jacobian components with shared caching. Do not duplicate native
   simulation, global polynomial conversion or marker masks in a second fitter.
2. Write failing tests before adding the backend: a toy forward system with
   analytically known constrained solution; exact continuity despite a competing
   marker objective; non-square transformed node variables; independent central
   checks of objective gradient and constraint Jacobian; bounded variables;
   nonfinite/rank-deficient projections; failed convergence never accepted;
   missing observations; existing callback isolation and checkpoint semantics.
3. Expose the constrained backend explicitly, with documented iteration and
   function-evaluation budgets. Do not relabel a solver's iteration limit as
   max_nfev. Choose the supported local solver after inspecting its actual
   interface. Supply analytic objective and constraint derivatives. Retain
   source-hashed immutable run configuration and complete evaluation snapshots.
4. Avoid redundant equality rows from the closed loop. Native physical nodes
   have54 entries but only42 local degrees of freedom. A candidate formulation
   is fixed-chart projection N_next.T @ ((x_end-x_next)/state_scales), where
   N_next is the existing orthonormal scaled tangent basis. Qualify its rank,
   derivative and local equivalence to physical continuity; require both states
   to remain on the closure manifold and inside the chart's valid neighborhood.
   Do not silently discard physical defect components and call them zero.
5. Keep FULL physical scaled-defect norms, continuous marker replay, closure,
   terminal pointwise replay gap and application acceptance as final diagnostics.
   A projected equality solver's success is insufficient. Retain one global
   degree-six profile and the independent0.8 s basis/0.85 s coverage contract.

## Native Qualification Before Optimization

Use a new runtime based on13; never edit13. Start from the run12 duration-only
extension with continuous chart references at0.2,0.4,0.6,0.7,0.8 s and final0.85 s.
This is a known feasible shooting initialization. Check actual capture samples,
candidate/model/capture hashes, original q0/qd0, and original effort bounds.
Do not silently widen bounds or change geometry to make this experiment pass.

Audit projected constraints and their analytic Jacobians against centered
finite differences at zero and nonzero chart points. Report full physical
defects as well as projected ones. Audit the feasible objective directions;
the previous combined penalty Jacobian test alone does not qualify this new
formulation. Record source, solver, numerical tolerances and column scaling.

Only then run one bounded native trial and independently replay the returned
candidate. Compare all metrics with run13 and its actual0.85 s starting point,
including the0.8 s endpoint. Inspect proximity to each effort bound; zero active
bounds in SciPy's report does not mean every coordinate is far from a bound.
If the constrained solve fails, preserve the exact failure and diagnose its
rank/feasibility/derivative evidence before changing another parameter.

## Handoff and Remaining Goal

Every commit/checkpoint must identify terminal versus live processes, current
candidate, raw package, tested source and the next exact command. Update the
owned Gemini checkpoint and issue9964 with material changes. Do not overwrite
Gemini source or interrupt MATLAB jobs. Native MATLAB qualification is R2025b.

The actual outcome remains the full1.8138888889 s capture driven continuously
by global sextic inputs, plus final-candidate R2025b and other-engine native
qualification. OpenSim epic10003 remains a separate staged implementation lane.
This assignment fixes an evidenced numerical bottleneck; it does not redefine
the goal as a short-horizon or segmented-only match.

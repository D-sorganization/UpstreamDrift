# ADR-0054: Closed-Loop Null-Space Exploration

- Status: Accepted for D1 to D6 (owner-directed implementation of epic #12593, 2026-10-14). D7 (output policy) is Proposed and awaits an explicit owner decision; NSX-41 (#12573) must not start before it.
- Date: 2026-10-14
- Decision Makers: repository owner (acceptance pending); proposed in the #12593 planning session
- Related Issues/PRs: epic #12593, NSX-00 #12549, NSX-01 #12550, NSX-02 #12551, NSX-13 #12558, NSX-24 #12567, NSX-41 #12573; ADR-0052; ADR-0041; #10382 (output policy)

## Context

A two-handed golf swing closes a kinematic loop through the club. The
motion fixes the generalized load `b = M(q) a + h(q, v)`, yet many torque and
hand-load combinations produce the same motion. Epic #12593 explores that
family and reports what is invariant across it. The specification lives in
`docs/research/loop_nullspace/loop_nullspace.tex`; its binding rules are R1 to
R14 in the epic.

Five questions needed one answer each before any `tier:cli` child can start.

1. **Where does the snapshot come from?** `BaseEngineForceAdapter` already has
   native Pinocchio and MuJoCo implementations. A parallel provider protocol
   would create a second engine seam.
2. **How is evidence typed?** A torque that depends on a prior must never look
   like a measurement.
3. **Which closure convention is canonical?** Today Pinocchio is LOCAL
   `[lin; ang]`, Drake LOCAL `[ang; lin]`, MuJoCo WORLD `[lin; ang]` and
   OpenSim WORLD `[lin; ang]`, with differing relative signs.
4. **Which solver handles which problem?** `scipy.optimize.linprog` has no warm
   start and SLSQP is not a convex solver.
5. **What happens to the two existing allocators?** `ContactForceAllocator`
   and `MultiEngineTorqueAllocator` both solve versions of the same problem.

A sixth question is a policy question and is not an engineering one. The
fail-closed `output_policy.py` withholds joint loads from swing reports
unless the receipt supplies both calibration and evidence (owner ruling,
2026-10-11). NSX research views display envelopes and invariants that carry
neither. Whether and how they may be shown is the owner's decision (D7).

## Decision

**D1. One engine seam.** The snapshot value object, `ClosedLoopSnapshot`, is
built from the existing `BaseEngineForceAdapter` through
`snapshot_from_adapter()`. The adapter gains optional methods
`compute_closure_drift`, `closure_wrench_frame` and `club_inertia_split`, gated
by `Capability.CLOSED_LOOP_SNAPSHOT`. Pinocchio is the reference and reports
`FULL`. Synthetic fixtures report `NONE`. No second provider protocol is
created.

**D2. Evidence axes.** Every user-visible number is a `QuantityRecord` with
`allocation_dependence` (`INVARIANT`, `FAMILY_DEPENDENT`, `UNAVAILABLE`),
`model_provenance`, `input_provenance`, `conventions`, `hypotheses` and
`validation_status`. `INVARIANT` forbids hypotheses; `FAMILY_DEPENDENT`
requires at least one; `UNAVAILABLE` requires a reason and NaN values.
`INVARIANT` means "identical for every admissible allocation, given model X";
it never means "measured".

**D3. Canonical closure convention (R8).** `J_c` and `J̇_c v` are stored in
the grip frame with rows `[fx fy fz tx ty tz]`, the linear rows being the
velocity of the closing-hand grip point, and `+J_cᵀλ` acting on the human tree.
The closing hand therefore exerts `−λ` on the club. Each engine adapter declares
a `NativeClosureConvention` constant and one converter,
`to_canonical_closure`, does the rest. The conversion matrix, its power
invariance and a four-engine golden are in the specification
(`tests/fixtures/loop_nullspace/goldens/closure_convention.json`).

**D4. Solver routing (R11).** Polyhedral problems use `highspy` with a
persistent model and basis reuse. Second-order cones use Clarabel. A
polyhedral approximation of a circular cone declares inner or outer
approximation, the facet count and the envelope error. A missing backend gives
`UNAVAILABLE` with the reason. No silent fallback may change a result.

**D5. Legacy allocators become shims.** `ContactForceAllocator` and
`MultiEngineTorqueAllocator` keep their signatures and become adapters over the
conic allocation core (NSX-24). `HARD_ZERO_TRAIL` maps to the three-case
zero-side solve, so an inconsistent frame is reported and not hidden. Both emit a
`DeprecationWarning`. No SLSQP remains on a convex allocation path.

**D6. No output without a stated gate.** Admission (R7) precedes allocation.
Every invariant carries its validity gate. Replay is labelled open-loop or
feedback-assisted (R10). Hypotheses cite sources with a verification status
and an unverified source cannot appear in a user-facing string (R13).

**D7. Output policy. Proposed, awaiting owner decision.** The question is how
NSX research views may display `INVARIANT` quantities, envelopes and
representatives while `output_policy.py` continues to withhold joint loads from
swing reports without calibration and evidence.

*Option A (recommended): two surfaces, evidence-axis mapping.*
`display_decision(record, surface)` returns a decision per surface.
- Research surface (the NSX tabs and panel): `INVARIANT` quantities are shown
  with a model-provenance badge. `FAMILY_DEPENDENT` quantities are shown only
  as bands, or as a labelled representative overlaid on its band under a named
  preset. `UNAVAILABLE` shows its reason.
- Swing-report surface: nothing from NSX enters unless `output_policy.py`
  already admits that output name with calibration and evidence. The default
  for an unknown combination is to withhold.

*Option B: research views only.* NSX is shown in the research tabs. It is
never exposed to the swing report or any export path, even when the policy
would admit the output.

*Option C: policy extension.* `output_policy.py` gains a new advertised class
for `INVARIANT` quantities with uncertainty, so they can enter swing reports
as well.

Option A is recommended because it keeps the owner's 2026-10-11 ruling intact
for reports and still lets a user see what is invariant. Option B is the most
conservative and is a strict subset of A. Option C changes the policy itself
and needs the owner to say which invariants count as advertised outputs.
NSX-41 (#12573) implements whichever option is accepted and must not start
before this ADR changes status.

## Alternatives Considered

- **A new provider protocol for closed-loop snapshots.** Rejected: it
  duplicates `BaseEngineForceAdapter` and its capability levels.
- **Storing a frame tag instead of converting.** Rejected: the row order, the
  relative sign and the reference point all differ, so a tag cannot make a
  Jacobian comparable across engines.
- **Keep SLSQP for allocation.** Rejected: the problems are convex, SLSQP
  gives no certificate of infeasibility and no warm start.
- **Hide the allocators' duplication.** Rejected: a swing would have two
  different "allocation" numbers.

## Consequences

- Children implement transcription of the specification and the contract
  appendix (`docs/research/loop_nullspace/contracts.md`). Goldens come from the
  sympy derivations and never from the code under test.
- Two new runtime dependencies (`highspy`, `clarabel`) enter through a
  `loop-nullspace` extra and `dev` (NSX-20).
- The legacy allocators gain deprecation warnings and their documented
  outputs may change where they were mathematically wrong.
- NSX makes no physiological claim. External validation stays deferred
  (NSX-05).

## Validation

- `python3 -m pytest tests/unit/loop_nullspace` regenerates every golden from
  the sympy derivations and compares.
- `python3 -m scripts.check_design_manual_governance` keeps the blocker
  `UP-NSX-loop-nullspace-exploration` open until a calculation is registered.
- `src/config/loop_nullspace_registry.json` is checked by the NSX-11 citation
  gate when it lands.

# ADR-0044: Out-of-Plane Fidelity for BunkerShot3D

- **Status:** Proposed
- **Date:** 2026-08-29
- **Deciders:** UpstreamDrift maintainers
- **Related Issues/PRs:** #9247 (F0 inverts the primary design variable, open),
  epic #8699, #8733 (F1 remainder), PR #9184 (`SliceFidelity.EXTRUDED`,
  merged) — amends the fidelity-tier posture of
  [ADR-0032](0032-bunkershot3d-club-design-architecture.md) and builds on
  [ADR-0033](0033-bunkershot3d-sand-field-tier.md)

## Context

BunkerShot3D exists to answer "given two wedge sole geometries, which
performs better, in what conditions, and how confident are we" (ADR-0032).
`WedgeGeometry` carries `heel_relief_fraction`, `toe_relief_fraction`,
`heel_rocker_radius_m` and `toe_rocker_radius_m` as first-class design
parameters — these, plus camber, are exactly the geometry a wedge designer
varies heel-to-toe, and they are exactly what the tool currently cannot speak
to for the fidelity tier ADR-0033 chose to answer sand-field questions with.

**F1 is plane strain and cannot see them by construction, not by omission.**
It solves one 2-D section (`src/bunkershot3d/solvers/mpm/`). Its own envelope
module refuses the quantity outright:

```python
# src/bunkershot3d/solvers/mpm/envelope.py
OUT_OF_PLANE = "out_of_plane"
"""Any heel-toe or lateral distribution. The quantity does not exist
in plane strain, so this is refused rather than approximated."""
```

and its ball is documented as "an infinite cylinder, not a sphere"
(`solvers/mpm/ball.py`, `solvers/mpm/ballreach.py`) — flux onto it is per
unit out-of-plane width, and `sphere_mass_kg` and the per-width force helpers
raise rather than fabricate a number. PR #9184 (merged) renders a 3-D sand
volume by drawing the solved plane as `SliceFidelity.EXTRUDED` — repeated,
bit-identical sheets, deliberately never blended into a continuum, because
blending would draw across-width structure the model does not have. None of
that is a bug; it is the honest shape of a plane-strain solve, and this ADR
does not propose changing any of it.

**F0 is not geometrically blind to heel/toe geometry, only reporting-blind.**
This is worth stating precisely, because it changes the cost of one of the
options below. `WedgeGeometry`'s relief and rocker fields already drive the
lofted 3-D mesh (`geometry/lofting.py::_relief_fractions`,
`rocker_offsets_m`, both indexed by spanwise station across the full blade
length), and `SurfaceElements` (`solvers/elements.py`) is built directly from
that `TriangleMesh` — centroid, normal and area per triangle, heel to toe.
`DRFTSolver` integrates a per-element resistive response over that whole
surface (`solvers/drft.py`) and sums it into one `Wrench`. The geometry is
there in every per-element computation; **`ShotResult` only ever exposes the
aggregate `forces_n_m` / `torques_n_m` time series** (`solvers/shot.py`), so
nothing downstream can currently ask "how much of that came from the toe."

**F0's other problem is not a gap but an open, unresolved inversion of its
main design lever.** Issue #9247, reproduced on `main`, holds attack angle,
sand and swing fixed and varies only marketed bounce:

| marketed bounce | sole depth | divot mass | peak force |
| --------------: | ---------: | ---------: | ---------: |
|              8° |   19.69 mm |    109.3 g |   1188.0 N |
|             14° |   21.40 mm |    122.3 g |   1241.1 N |
|             20° |   22.99 mm |    135.3 g |   1152.3 N |
|             26° |   24.57 mm |    149.2 g |   1135.1 N |

More bounce digs _deeper_ — monotone across the whole range, the opposite of
what bounce is for. Two candidate causes (a bounce-dependent sole reference
point; a mismatch between the drop pose and the depth reference) have been
measured and ruled out; the root cause is still open. Bounce and the
heel/toe relief fractions are computed by the same lofting and integrated by
the same per-element sum, so any near-term work on F0's sole-geometry
reporting inherits whichever bug is producing this until #9247 is closed.

**F2 remains the GPU-only reference tier**, unchanged from ADR-0032: Newton's
`SolverImplicitMPM` requires Maxwell+ with driver 545+, and the primary
development machine has no NVIDIA GPU (Intel Iris Xe). At 30–90 min/shot it
was already scoped as "reference truth; calibrates F0; optional, fleet-only,"
never as the design-loop tier.

**Measured costs, for grounding rather than speculation:**

- F1 (the plane-strain MPM solver) costs on the order of **24–38 ms/step at a
  few thousand particles**, i.e. roughly seconds to minutes per shot,
  consistent with ADR-0032's original F1 estimate.
- ADR-0033's MuJoCo grain-proxy measurement — **0 of 1000 grains** placed in
  the clubhead's swept path, 0.125 grains per sampling cell on the coarsest
  grid anyone would call a cross-section — is not re-litigated here; it is
  why F3 is not one of the options below.
- Both existing tiers sit at **NASA-STD-7009B Validation 0 of 4**
  (`vandv/credibility.py`; F0's own credibility statement: "Use it to rank
  two sole geometries against each other; do not quote an absolute force
  from it"). No published measurement exists anywhere for ball launch, spin,
  head deceleration in sand, energy split, or ejecta mass from a bunker shot
  (`docs/bunkershot3d/upgrade/STATE.md`). Adding a dimension to either tier
  does not change that number.
- A design-space comparison already run on this tool found **attack angle
  dominates sole geometry roughly 9×**: 13.3 mm of sole-depth sensitivity to
  attack angle against 1.5 mm for the _full_ bounce range — the single
  largest scalar sole parameter the tool already models in-plane. Heel/toe
  relief and rocker have not been measured against that axis at all; that
  gap is exactly what the decision below turns on.

**RFT's own theoretical basis is superposition of independent elements.**
The envelope module documents this directly: "RFT is _exact_ for a
frictional-plastic medium because superposition reproduces
`F = rho_c g L^3 Psi(beta, gamma)`" and carries a standing `SHADOWING`
caveat — "there is no wake model, so leading-edge elements sheltered behind
other parts of the body are counted at full strength." F0 already computes
every element's response as if no other element existed. That matters below:
a heel/toe force split built from F0's existing per-element sum would not be
a new physical claim about lateral coupling — it would be the same
zero-coupling assumption the tool already makes for the total wrench, sliced
thinner by position.

## Decision

**BunkerShot3D remains an in-plane design tool for the foreseeable future.**
No new out-of-plane tier — GPU-rented F2, quasi-3-D strip coupling, or an F0
heel/toe reporting layer — is built now. `RefusedQuantity.OUT_OF_PLANE`, the
`EXTRUDED` labelling, and the ball-as-cylinder caveats are not stopgaps
awaiting near-term replacement; they are recorded here as the tool's durable
architectural position, to be revisited only on the explicit trigger below.

This is not "accept forever." The 9× dominance finding was measured for one
in-plane parameter (bounce) against attack angle; it says nothing directly
about heel/toe relief, which acts through a different mechanism (shifting
where the sole engages the sand across the blade, not how deep a
symmetric sole digs) and has never been measured. Treating "one in-plane
parameter turned out to be secondary" as proof that "the whole out-of-plane
axis is secondary" would be exactly the kind of unmeasured extrapolation this
codebase's other ADRs (ADR-0033's whole premise) have been built to refuse.
So the decision carries a concrete, cheap, and immediately actionable
reopening condition instead of a permanent door-close:

1. **#9247 must be fixed first**, with the physical-ordering regression test
   it calls for (more bounce must not produce a deeper sole or more
   displaced mass in the non-burying regime). Bounce and heel/toe relief are
   computed by the same lofting and summed by the same per-element
   integration; no sole-geometry axis, in-plane or out, is trustworthy while
   the tool's main lever runs backward.
2. **Run the existing Sobol'/Morris sensitivity machinery
   (`bunkershot3d.study.sensitivity`, `bunkershot3d.study.morris`) over the
   full `WedgeGeometry` parameter set — including `heel_relief_fraction`,
   `toe_relief_fraction`, `heel_rocker_radius_m` and `toe_rocker_radius_m` —
   against the quantities F0 _already_ reports** (total dig depth, divot
   mass, peak force), not a quantity that has to be built first. This costs
   nothing new: F0 already varies the full 3-D mesh per shot at millisecond
   cost, and the sensitivity study only needs the total wrench it already
   returns. It directly extends the comparison that produced the 9× finding
   to the parameters this ADR is about, instead of assuming the answer.
3. **The trigger fires only if that study shows at least one out-of-plane
   parameter's effect size is not dwarfed by attack angle the way bounce
   already is.** If it is dwarfed, out-of-plane fidelity is confirmed to be
   precision the model cannot cash, on this tool's own terms, and this ADR's
   position holds without further action.

If the trigger fires, the fallback order is stated here so it is decided
once rather than re-litigated per issue:

- **Quasi-3-D strip theory first**, not F2. It reuses F1's calibrated MPM
  constitutive model (SPEC 1.0.655 fitted friction angle), needs no new
  dependency, and its cost scales linearly with strip count off F1's
  existing per-shot cost — plausibly tens of seconds to tens of minutes per
  shot for a handful of strips, still orders of magnitude below F2.
- **F2 GPU rental stays reserved for calibration**, exactly the role
  ADR-0032 already gave it, never promoted to a design-loop tier by this
  decision.

**Make the refusal prominent, not incidental.** Today `RefusedQuantity`
raises correctly but only when a caller reaches for the quantity. A designer
who never asks does not learn the tool has no opinion on heel/toe relief at
all. A follow-up (tracked, not implemented here — this ADR changes no
production code) should surface a standing "what this tool cannot tell you"
statement — heel-toe distribution, out-of-plane camber effects, the
ball-as-cylinder caveat — at the same top level the credibility statement
already occupies (`docs/bunkershot3d/credibility.md`), so the boundary is
read once rather than discovered per query.

## Alternatives Considered

1. **F2 3-D MPM on rented/remote GPU (the ADR-0032 reference tier).** The
   hardware blocker ADR-0032 recorded — no NVIDIA GPU on the primary
   machine — is not actually a blocker for _rental_: a cloud GPU meeting
   Newton's driver-545+/Maxwell+ requirement is reachable from any client.
   At ADR-0032's 30–90 min/shot, rounding to roughly one GPU-hour per shot,
   and at the order-of-magnitude market range for a suitable GPU instance
   (very roughly $0.50–$4/GPU-hour, stated as a planning range, not a
   quote), a handful of shots to characterize one design change costs on the
   order of a few dollars to a few tens of dollars. **Raw compute is not the
   obstacle.** Three things are:

   - It inverts the tool's whole economy. The DOE/sensitivity machinery in
     `bunkershot3d.study` is built to run hundreds to thousands of cheap
     evaluations; F2 at ~1 GPU-hour/shot is roughly five orders of magnitude
     more expensive than F0's millisecond cost. It can validate a handful of
     points a sweep already chose; it cannot run the sweep.
   - Reproducibility "off a machine nobody here owns" means a pinned
     container image, an exact Newton/Warp/CUDA-driver version pin, and a
     run manifest extended to cover GPU architecture — MPM contact behavior
     is sensitive to parallel-reduction order, so results are not guaranteed
     portable across GPU generations the way F0/F1's CPU results are. None
     of that infrastructure exists today, and ADR-0032 already records that
     whatever depends on it "must be optional and CI-skippable" — it stays
     outside the fleet's gates permanently, not until someone gets around to
     wiring it in.
   - **It does not buy validation.** No published measurement exists for any
     quantity a bunker-shot solver produces. F2 can be cross-checked against
     F1 the way ADR-0033's B5 gate already cross-checks F1 against F0, but
     agreement between two uncalibrated models is consistency, not
     validation — NASA-STD-7009B stays at 0 of 4 regardless. F2 buys
     reference truth relative to F0/F1's approximations, not truth relative
     to a real bunker shot.

   Rejected as a near-term investment: real, but not proportionate to what
   it would let the tool claim.

2. **Quasi-3-D / strip theory** — solve several F1 plane-strain sections
   across the blade span and couple them weakly (a declared effective width
   or blend, not a solved coupling). Cheap relative to F2, and the only
   option here that would let a design comparison speak to heel/toe relief
   by name using the already-calibrated F1 material model. **State plainly
   what it gets wrong:** there is no lateral momentum transfer between
   strips — sand pushed toward the toe by heel relief cannot be represented
   moving into the toe strip's solve, because each strip is an independent
   2-D solve with no channel between them. That is a modelling assumption
   the coupling width encodes, not a result, in exactly the sense ADR-0033
   already states for F1's declared effective width. It is also, notably,
   _the same class of assumption_ F0 already makes at finer granularity: RFT
   sums independent per-element responses with no wake model (the
   `SHADOWING` caveat); strip theory sums independent per-strip continuum
   solves with no lateral flux. Neither adds coupling physics the tool
   lacks today — they trade constitutive fidelity (a calibrated
   Drucker-Prager continuum per strip vs. a borrowed RFT polynomial per
   element) for cost, at different length scales, not a qualitatively new
   claim about lateral flow.

   Rejected for now, not on merit but on trigger: the cheapest of the "do
   something" options, and the one to build first if the sensitivity study
   above shows it is warranted.

3. **Extend F0's resistive model to resolve out-of-plane distribution.**
   Costed accurately, this is smaller than it looks: F0's mesh and
   per-element loop already carry the heel/toe geometry (see Context); the
   gap is that `ShotResult` only exposes the summed wrench, not a spanwise
   breakdown. The work is an aggregation layer over `SurfaceElements`'
   existing `centroids_m` span coordinate, not a new physics engine — the
   cheapest of the three "do something" options in engineering terms. It
   inherits two problems that are not implementation defects: it must wait
   on #9247 (reporting a spanwise breakdown of a depth axis the tool is
   currently proven to get the sign of wrong on an adjacent sole parameter
   would compound a known-wrong result), and it shares strip theory's
   fundamental limitation — a per-region sum of independently-computed,
   uncoupled elements is not a model of lateral sand flow between heel and
   toe, only a finer-grained view of the same zero-coupling total. Shipping
   it before the sensitivity trigger fires would be the specific risk this
   ADR is written to avoid: a number that looks like new capability while
   carrying the same unmodelled interaction the total wrench already
   carries, sliced thin enough to look more authoritative than it is.

   Rejected for now on the same trigger as strip theory, not ruled out.

4. **Accept the limitation permanently, with no reopening condition.**
   Considered and rejected as too strong. The available evidence — 9× and
   0/4 — supports declining to invest _now_; it does not support declaring
   the question closed, because it was measured for a different parameter
   (bounce) than the one this ADR is about (heel/toe relief), and a
   permanent refusal removes the incentive to ever run the one cheap,
   already-available check (the sensitivity study in the Decision) that
   would tell the difference. A legitimate "accept" needs a falsifiable
   condition or it is not a decision, it is an assumption wearing one.

## Consequences

**Positive.** No new dependency, no new CI/GPU liability, no infrastructure
this repo does not already operate. The decision matches both existing
tiers' actual validation state (0/4) rather than treating dimensionality as
a proxy for credibility. `RefusedQuantity.OUT_OF_PLANE`, `SliceFidelity.
EXTRUDED`, and the ball-as-cylinder caveats are confirmed correct as
written and need no code change from this ADR. The reopening condition is
cheap enough (an existing sensitivity study, run once #9247 is fixed) that
"accept the limitation" costs the team nothing beyond running a check it can
already run.

**Negative.** Heel/toe relief, camber-driven engagement, and blade-spanning
rocker — parameters a real wedge designer varies — remain unaddressable by
name in the tool's stated purpose indefinitely, i.e. until the trigger
fires. A user asking "does more toe relief help this shot" gets a refusal,
not an answer, and today that refusal is easy to miss unless the caller
reaches for the specific quantity. #9247 blocks even the in-plane bounce
question in the meantime, which this ADR does not resolve — it only refuses
to let any out-of-plane work proceed ahead of it.

**Accepted debt.** The MuJoCo/F3 proxy stays recorded as non-functional per
ADR-0033; this ADR adds nothing there. The gap between "the tool's stated
purpose" and "what F1 can answer" stays open and documented rather than
closed, which is a deliberate scope decision, not an oversight.

**Follow-ups (tracked separately; none implemented by this ADR):**

- Land #9247's fix and its physical-ordering regression test.
- Once #9247 is closed, run `bunkershot3d.study.sensitivity` /
  `bunkershot3d.study.morris` over the full `WedgeGeometry` relief and
  rocker parameters against F0's existing dig-depth, divot-mass and
  peak-force outputs, and record the result next to the 9× finding it
  extends.
- Surface a standing "what this tool cannot answer" statement at the top
  level of `docs/bunkershot3d/credibility.md` (or equivalent), covering
  out-of-plane distribution, the ball-as-cylinder caveat, and F0's
  reporting gap, so the refusal is read once rather than discovered per
  query.
- If and when the sensitivity trigger fires, open the quasi-3-D strip-theory
  work first, per the fallback order in the Decision; do not open F2 rental
  work as a design-loop feature.

## Validation

This ADR changes no production code, so there is nothing in `src/` for CI to
gate today. What it commits the project to verifying, going forward:

- The existing tests asserting `RefusedQuantity.OUT_OF_PLANE`,
  `RefusedQuantity.BALL_LAUNCH` and `RefusedQuantity.CLUB_FORCE` raise
  `OutOfEnvelopeError`, and that `SliceFidelity.EXTRUDED` frames stay
  bit-identical across heel-to-toe stations, continue to pass unchanged —
  this ADR is a statement that those tests encode the _right_ long-term
  behaviour, not a stopgap to be relaxed.
- The sensitivity study named above, once run, is committed as data (not
  prose) the same way the 9× finding and the #9247 table are: numbers a
  reader can check, with the design-space bounds and RNG seed recorded per
  `bunkershot3d.study.manifest.StudyManifest`.
- Any future PR that adds a new out-of-plane capability (strip coupling, an
  F0 spanwise reporting layer, or F2 rental tooling) must cite the
  sensitivity result that triggered it in its description; a PR that adds
  such capability without that citation should be read as reopening this
  ADR without the evidence it requires, and reviewed accordingly.

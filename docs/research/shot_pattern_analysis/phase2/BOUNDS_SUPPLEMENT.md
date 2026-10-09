# Nominal Matching Bounds Supplement

## Scope and Scientific Question

This is a separate calculation-level reference for the phase-two matched-shot
study and [issue #12000](https://github.com/D-sorganization/UpstreamDrift/issues/12000).
It records an adapted search protocol after the original 324-cell and 72-cell
plans were frozen. Those results retain their original bounds and source
receipts. A bounded optimizer returning no match does not prove that a shot is
physically impossible. No supplemental Monte Carlo result is reported here.

The original nominal matching search gave each pattern a face-angle interval
centered on its reference face, with half-width $3^\circ$, and club speed
between $0.75v_0$ and $1.25v_0$. The match solved raw landing, without
rotating a trajectory after flight. Strong pitching-wedge curves at zero lean
were outside that face interval even though a wider bounded search found
matches. This supplement keeps the original model, seed, target definitions,
noise profiles, and $0.05\,\mathrm{m}$ nominal-landing tolerance.

## Geometry, Units, and Calculation

The positive face and path directions are right of the target for a
right-handed player; forward landing is $x$ and right lateral landing is $y$.
The draw has face-to-path $-3^\circ$ at curve scale two, so
$p=f+3^\circ$; the fade mirrors that relation. For each candidate, the solver
chooses face $f$ and club speed $v$ such that the physical first-ground-contact
endpoint $(x(f,p,v),y(f,p,v))$ is within $0.05\,\mathrm{m}$ of the stated
$(x_\mathrm{target},0)$. Static loft, shaft lean, attack, head properties,
contact law, flight integration, and scoring remain those in the parent
[matched-study design](DESIGN.md) and the exact source-bound receipt.

The absolute face interval for all three supplemental protocols is
$[-6^\circ,+6^\circ]$. The pitching-wedge protocols retain speed factors
$[0.75,1.25]$, which yield $[24,40]\,\mathrm{m\,s^{-1}}$ for the illustrative
$32\,\mathrm{m\,s^{-1}}$ preset. The separate driver speed sensitivity uses
$[0.75,1.30]$, or $[33.75,58.5]\,\mathrm{m\,s^{-1}}$ for the
$45\,\mathrm{m\,s^{-1}}$ preset. These are *optimizer bounds*, not observations
of a golfer's achievable delivery. A Gaussian perturbation outside a physical
simulation's domain remains an unsupported sample rather than being clipped or
resampled.

## Predeclared Supplemental Protocols

| Protocol ID | Cells | Target Forward Distance | Face Bounds | Speed Factors | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `pw-primary-face-bounds/1` | 18 | Within-lean straight nominal target | $[-6,+6]^\circ$ | $[0.75,1.25]$ | Strong PW, zero lean; nine noise profiles at shaft-twist SD $1^\circ$ and $2^\circ$ |
| `pw-control-face-bounds/1` | 4 | $77.40247724866299\,\mathrm{m}$ | $[-6,+6]^\circ$ | $[0.75,1.25]$ | Strong PW, $5^\circ$ lean; shaft-only and combined2 profiles at both SDs |
| `driver-control-speed-bounds/1` | 4 | $178.94875782337786\,\mathrm{m}$ | $[-6,+6]^\circ$ | $[0.75,1.30]$ | Strong driver, $5^\circ$ lean; same two profiles and SDs, separately labeled speed sensitivity |

Each cell has three physical patterns and, if accepted for the final stage,
10,000 requested shots per pattern: 26 distinct cells and 780,000 attempted
shot records. The four PW zero-lean shaft-only/combined2 cells that a
cross-lean control would use are **candidate references** to four of the 18
primary cells. They are not extra simulations. An independent audit must prove
equality of target, delivery, noise stream, nominals, scoring inputs, source
snapshot, and cell hashes before treating them as aliases in a comparison.

These protocols were specified **after** the feasibility audit and **before**
any supplemental Monte Carlo. They are sensitivity analyses, not part of the
original preregistration. The exact proposed 26-cell names and four references
are in `review_followups/SUPPLEMENTAL_EXECUTION_INVENTORY.json` in the main
study branch; execution plans will embed the selected names and numerical
bounds in immutable `matrix_plan.json` files under separate output roots.

## Probe Evidence and Numerical Method

Read-only native nominal probes found a PW zero-lean strong draw at face
$-0.260459639^\circ$, path $+2.739540361^\circ$, and speed
$32.045914261\,\mathrm{m\,s^{-1}}$ for a $77.402477249\,\mathrm{m}$ target.
The mirrored fade also matched; fine-step replay residual was
$0.000211396\,\mathrm{m}$. At $5^\circ$ lean, PW strong draw matched at face
$-0.076707602^\circ$, path $+2.923292398^\circ$, speed
$29.161308145\,\mathrm{m\,s^{-1}}$, fine residual
$0.000289371\,\mathrm{m}$. The driver $5^\circ$ lean strong draw required
$56.345074274\,\mathrm{m\,s^{-1}}$, only
$0.095074274\,\mathrm{m\,s^{-1}}$ above the original $56.25$ upper bound;
its fine residual was $0.000943089\,\mathrm{m}$. These are deterministic
model probes, not observed outcomes or Monte Carlo results. Detailed inputs,
solver starts, and source hashes are retained in the three probe receipts in
the main study's `review_followups` directory.

The implementation uses the existing native contact and ball-flight solver,
SciPy nominal optimizer, `ExperimentConfig`, `matched_matrix_cells`, and
`run_matched_matrix`. The supplement selector supplies only a reviewed cell
set and bounds. The existing runner preserves common random samples, full
unsupported rows, immutable cell manifests, source snapshots, and a separate
reviewed final-stage acceptance gate. No numerical tolerance or contact
parameter changed. A native integration test checks a zero-lean PW raw match
and exact default-bound parity with the original explicit bounds.

## Acceptance and Reproduction

Before any 10,000-shot supplemental run, a 32-shot source-bound pilot must
establish all selected cells' status, numerical refinement, raw-landing and
figure checks, independent review, and an acceptance receipt for this source
snapshot. The supplemental auditor must check its own 26-cell allowlist and
four alias proofs; it must not relax the original 324/72-cell auditor. A
nominal matching exception means only that the solver did not find a match
under its stated finite search box and starts. Runtime physics errors remain
errors and are never silently reclassified as bounded unavailability.

The following commands describe the pilot interface; they have **not** been
executed to create a supplemental Monte Carlo result in this change:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
  python -m src.tools.shot_pattern_analysis.matched_supplement \
  pw_primary_face /tmp/shot-pattern-pw-face-pilot --stage pilot --shots 32
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
  python -m src.tools.shot_pattern_analysis.matched_supplement \
  pw_control_face /tmp/shot-pattern-pw-control-pilot --stage pilot --shots 32
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
  python -m src.tools.shot_pattern_analysis.matched_supplement \
  driver_control_speed /tmp/shot-pattern-driver-speed-pilot --stage pilot --shots 32
```

The acceptance condition is narrower than a successful process exit: the
reviewed native solver and numerical refinement must agree, every saved cell
must pass manifest/source checks, all planned statuses must be retained, and
the effect estimates must use explicitly paired supported populations.

## Limitations and Failed Search

The original strong PW zero-lean bounded search failed to find a nominal draw;
the widened face search found one. The original strong driver $5^\circ$ lean
fixed-target search remained about $0.635\,\mathrm{m}$ short at its speed cap,
and the separately labeled speed extension found a deterministic match. These
changes reveal sensitivity to arbitrary optimizer boxes, not evidence of
real-world shot controllability. The entire study remains an illustrative
rigid-body, no-wind, first-ground-contact simulation with assumed club
geometry, contact properties, and Gaussian delivery variation. It does not
include roll or measured golfer covariance, and conditional strokes gained
is scenario scoring rather than empirical player performance.

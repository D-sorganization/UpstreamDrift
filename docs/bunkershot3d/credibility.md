# BunkerShot3D Credibility Statement

**Issue #8616 (W9). Framing: NASA-STD-7009B (2024-03-05; 7009A is superseded).
Metric: ASME V&V 20-2009. Solution verification: Celik et al. (2008).**

This document answers four questions — what is verified, what is validated, against
what, over what domain of applicability, and with what uncertainty — and reports the
**gap to the threshold** for every credibility factor rather than only the level
achieved. Reporting the achieved level alone is how a model that is nowhere near fit for
purpose reads as respectable.

Everything below that is a number is generated from the code by
`bunkershot3d.vandv` and checked for freshness by
[`tests/bunkershot3d/vandv/test_credibility.py`](../../tests/bunkershot3d/vandv/test_credibility.py),
so this document cannot drift away from the solver.

## The One-Paragraph Version

The F0 solver is **verified but not validated.** Its arithmetic is checked against
closed-form answers and exact discrete identities, and its discretisation error is
quantified. Nothing about its _physics_ has been confirmed against a measurement,
because for the quantities this tool exists to predict **no measurement exists
anywhere.** It is run about 63 times outside 3D-RFT's own stated Froude limit and about
17 times beyond the fastest intrusion in the published RFT/DRFT corpus. Its two most
influential constants, `lambda` and `delta_h`, have no wedge-specific value at all. Use
it to rank two sole geometries against each other; do not quote an absolute force from
it.

## How Far Outside the Envelope We Are

<!-- generated:envelope -->

At 25 m/s on the 100 mm clubhead scale, Fr = 25.2 against 3D-RFT's stated limit of 0.4: about 63x outside the stated envelope, and about 17x beyond the fastest intrusion (1.44 m/s) anywhere in the published RFT/DRFT validation corpus.

<!-- end:envelope -->

The 100 mm clubhead is the **most flattering** of the three scales the envelope judges.
The 30 mm sole width is at Fr ≈ 46 and micro-inertial `I` ≈ 0.77; the 5 mm leading edge
is at Fr ≈ 113 and `I` ≈ 11.3, against a limit of 0.1. The solver reports all three with
every result and lets the smallest one govern, because that is where the physics fails
first.

This does not invalidate the architecture — DRFT remains the only per-geometry method
cheap enough for a design loop — but it does mean **published RFT coefficients are an
initial guess, not a validated model.**

### The Two Uncalibrated Constants

- **`lambda`** carries roughly 90 % of the load at greenside delivery speed, and no wedge
  value exists. The published spread across motion types is 1.0 (grousered wheel) to 2.8
  (2-D plane-strain vertical plate); the package defaults to 1.1, the oblique horizontal
  plate, because a planing sole is the closest published motion. **The single most
  influential constant in the model is known to within a factor of nearly three.**
- **`delta_h`**, the dynamic structural correction, has no published wedge form either.
  The only closed form is for a wheel and evaluates to tens of metres at clubhead speed.
  The shipped default is a documented _convention_ chosen so the model is well behaved —
  it vanishes in the quasi-static plate limit, saturates below the element depth so the
  depth term cannot invert, and preserves monotonicity — and it reports
  `is_calibrated_for_wedge = False` on every result. Omitting `delta_h` entirely is worse:
  the source paper found that applying the inertial term without it gave the **wrong sign**
  of sinkage at every `lambda` from 1 to 100.

## Credibility Assessment

<!-- generated:credibility-table -->

| Factor                | Achieved     | Threshold | Gap              |
| --------------------- | ------------ | --------- | ---------------- |
| Verification          | 2 / 4        | 3 / 4     | 1 level(s) short |
| Validation            | 0 / 4        | 3 / 4     | 3 level(s) short |
| Input Pedigree        | 2 / 4        | 3 / 4     | 1 level(s) short |
| Results Uncertainty   | 2 / 4        | 3 / 4     | 1 level(s) short |
| Results Robustness    | 1 / 4        | 2 / 4     | 1 level(s) short |
| Use History           | 0 / 4        | 2 / 4     | 2 level(s) short |
| M&S Management        | 3 / 4        | 3 / 4     | met              |
| People Qualifications | not assessed | 2 / 4     | n/a              |

<!-- end:credibility-table -->

The threshold column is the level the **intended use** demands: choosing between two
wedge sole geometries and believing the answer. One factor of eight meets it.

`People Qualifications` is deliberately **not** self-scored. A team rating its own
competence is not evidence, and a number there would dilute the seven factors that rest
on artefacts. Assess it externally or leave it blank; do not fill it in to make the table
look complete.

The per-factor evidence and gap statements live in
`bunkershot3d.vandv.credibility.CREDIBILITY_ASSESSMENT` and are reproduced in full by
`python -c "from bunkershot3d.vandv import CREDIBILITY_ASSESSMENT; ..."`.

## What Is Verified

Code verification uses **no experimental data**, by construction.

| Check                                                    | Class      | Result                                                         |
| -------------------------------------------------------- | ---------- | -------------------------------------------------------------- |
| Linear-impulse identity `m dv = sum F dt`                | round-off  | relative residual ~1e-16, fixed 1e-12 tolerance, no order test |
| Moment transfer `tau(p2) = tau(p1) + (p1-p2) x F`        | round-off  | relative residual ~1e-16                                       |
| Element moment against a naive per-element oracle        | round-off  | relative residual ~1e-16                                       |
| Work-energy residual under semi-implicit Euler           | truncation | observed order **1.00**, as derived                            |
| Surface-quadrature order of accuracy vs a closed form    | -          | observed order **2.00**, monotone, spread < 0.01               |
| Quasi-static flat-plate limit vs `xi_n alpha_z abs(z) A` | -          | exact to 1e-12                                                 |
| Zero-speed limit: inertial force proportional to `v^2`   | -          | exact to 1e-12                                                 |
| Depth/inertia crossover vs `crossover_speed_m_s`         | -          | exact to 1e-12                                                 |

Two of these deserve naming.

**The conservation classes are split, and the split is enforced.** Round-off quantities
(mass, linear and angular momentum) get a fixed ~1e-12 tolerance and **no order test** —
refining the step does not shrink floating-point noise, so an order fitted to it describes
the hardware. Truncation quantities (energy under a non-symplectic scheme) get an order
test on the residual and **no fixed tolerance** — a truncation residual can always be made
small by shrinking `dt`, which says nothing about the scheme. Asking for the wrong test
raises `ConservationClassError`.

**Angular momentum is the check that finds this class of bug.** The solver replaces
`np.cross` with a hand-written component form for speed. That is sound, and it is also
exactly where an index or sign error survives review, because _the resultant force never
touches the cross product_. The suite injects an axis swap into that function and
demonstrates that the resultant force does not move by a single bit while both
angular-momentum residuals blow up. Both checks refuse to run on a configuration whose
torque is too nearly axis-aligned to expose a swap, so neither can pass vacuously.

### What Verification Does Not Cover

No method of manufactured solutions for the coupled shot. No verification of the F1, F2
or F3 tiers at all. And a trap worth stating: refining the surface mesh **raises**
Askari & Kamrin's `I_G = v²d²/(gλ²)`, so a mesh fine enough to converge the quadrature is
a mesh further outside RFT's superposition argument. The refinement study is verification
and can never be validation. (The MPM analogue is worse still: at ~3.5 particles per cell
a linear-basis MPM's error _increases_ with refinement beyond ~20 cells, so a grid study
run on one reports a number that means nothing.)

## What Is Validated

**Nothing.**

That is the finding, not a placeholder. Of the two comparisons that can be constructed
at all, one is _indeterminate_ and one is _noise-limited_; neither carries any information
about model error.

| Comparison                                              | E            | u_val       | Verdict                                                                 |
| ------------------------------------------------------- | ------------ | ----------- | ----------------------------------------------------------------------- |
| Plate response `α_z` vs Quikrete analogue, as published | +0.108 N/cm³ | —           | **indeterminate**: the source reports no uncertainty on the measurement |
| Same, with the measurement granted `u_exp = 0`          | +0.108 N/cm³ | 0.278 N/cm³ | **noise-limited**: \|E\| ≤ u_val                                        |

The second row is the stronger statement. Granting the experiment _zero_ uncertainty is
the most favourable assumption available to the model, and the comparison is still
noise-limited. The reason is leverage: the material-scaling cubic moves **12.6 % per
degree of friction angle**, and the friction angle is published as "34°" with no error
bar. A 5 % agreement cannot confirm a model whose answer moves 12.6 % per degree of an
input known to the nearest degree.

Note what this comparison is even in principle: the scaling cubic against a **laboratory
analogue sand**, quasi-static, at a flat plate. It says nothing about golf bunker sand,
nothing about a wedge, and nothing about any speed above quasi-static.

The solver's own docstring calls this gap "a 4.6 % independent cross-check". Under a
proper V&V 20 treatment it is not a cross-check at all.

### Against What — The Reference Data That Exists

- **Wivou, Udawatta & Pathirana (2016), ISBS 2016, pp. 1147–1150.** The only primary
  greenside-bunker dataset located. Entry distance 80–280 mm, divot depth 25–52 mm, carry
  1–12 m; carry correlations r = −0.98 (entry, other variables held constant) and −0.91
  (divot depth). It contains **no** clubhead speed, launch angle, ball speed or spin, and
  the dataset object raises if asked for them.
- **The granular-intrusion benchmark** behind the DRFT constants: wheel-in-sand sinkage
  MAE of 2.7 mm (RFT), 3.2 mm (MPM) and 26.1 mm (Bekker/Wong-Reece). This is where the
  real leverage is, because reproducing it validates the _solver_ independently of golf.
  It has not been reproduced here. Note that the whole corpus tops out at **1.44 m/s**.

### Against What — The Data That Does Not Exist

An exhaustive enumeration of ISEA / _Procedia Engineering_ / _The Engineering of Sport_
volumes 2, 13, 32, 34, 60, 72, 112 and 147, of _Sports Engineering_ and of the _Journal
of Sports Sciences_ found **no paper on bunkers, sand, wedges, club-turf interaction or
divot mechanics at all.** This is a real gap in the field, not a search failure; it cannot
be closed by reading more.

There is therefore **no published value anywhere** for:

- ball launch angle, ball speed or ball spin from a splash shot
- clubhead deceleration in sand
- the energy split between club, ball and sand
- ejecta mass
- the coefficient of restitution through a sand layer

Every one of these is a quantity this model produces, which is exactly what makes them
dangerous. `bunkershot3d.vandv.require_measurable` raises `NoReferenceDataError` for each,
so a `ValidationComparison` against them **cannot be constructed** — the refusal is in the
constructor, not in a review comment.

Two further things are _not_ validation and are labelled as such in the code:

- The addendum's "≈ 1550 N on a 20 × 80 mm sole at 25 m/s" is an analytic estimate from
  `C_d ≈ 2`, not a measurement. The solver agrees with it to within a factor of two. That
  is a smoke test of the arithmetic.
- The plate-drag law `F = K|z| + λρAv²` with `K = 580 N/m` cannot be used at all, for a
  duller reason than the rest: `K` scales with plate area and the area it was measured on
  is not recorded. Comparing against it would be comparing against an unknown scale factor.

### The Validation That Is Ready but Has Not Been Run

`carry_correlation_comparison` implements the Wivou comparison, including a Fisher-z
uncertainty on the published `r` (n = 55, two controlled variables **assumed**, since the
paper states neither the degrees of freedom nor an interval). It is unused because
**nothing in this package computes a model carry correlation**. When the W7 carry pipeline
produces one, this comparison becomes the first real validation the tool has.

## Over What Domain of Applicability

<!-- generated:domain-table -->

| Factor                         | Swept           | Measured (Wivou 2016) | Inside measured domain |
| ------------------------------ | --------------- | --------------------- | ---------------------- |
| `entry_distance_behind_ball_m` | 0.025 to 0.15 m | 0.08 to 0.28 m        | 56%                    |
| `divot_depth_m`                | 0.02 to 0.06 m  | 0.025 to 0.052 m      | 68%                    |

<!-- end:domain-table -->

Only two of the nine declared sweep factors have any published bunker measurement at all,
and **neither sweep sits inside it.** The entry-distance sweep starts at 25 mm, where the
players Wivou measured never actually entered — the prescribed 25–50 mm was the _target_
and the delivered range was 80–280 mm — so 44 % of that sweep is outside any measurement.
The remaining seven factors (attack angle, face opening, shaft lean, strike location, lie
deviation, sand firmness, sand depth) have **no measured domain of applicability
whatsoever.**

Two further domain limits travel with every result:

- **Speed.** Validated to 1.44 m/s; used at 20–27 m/s.
- **Sand.** Every fitted constant is borrowed from an analogue. One real bunker sand is
  characterised (Covia Signature 500, Turf and Soil Diagnostics #22040060, ASTM F1632
  Method B and F1815) and seeds the presets, but it is one commercial product in one lab
  report, not a population.

## With What Uncertainty

`u_num = u_h + u_it + u_ro` by **simple addition**, not root-sum-square: the three are
correlated faces of one discrete solve and V&V 20 treats them as epistemic. `u_val` then
combines `u_num`, `u_input` and `u_exp` **in quadrature**, because those three are
independent. `U = k·u_val` with `k = 2`. The asymmetry between the two combinations is
the substance of the standard, and the suite pins it with a test.

For the F0 tier:

- **`u_h`** comes from a Celik GCI over the surface discretisation, with the apparent
  order observed rather than assumed (`Fs = 1.25`; `Fs = 3.0` on two grids, where the
  order has to be assumed and there is no evidence of an asymptotic range at all).
  Observed `p ≈ 2.00`, `r₂₁ = 2`, monotonic, 0 % oscillatory. GCI ≈ 0.001 % of the fine
  solution — and because that case has a closed form, the suite checks the thing a GCI
  normally has to be taken on trust: the estimate really does bound the error it
  estimates, by a margin of about 1.6.
- **`u_it` and `u_ro` are zero**, legitimately: F0 solves no linear system and iterates
  nothing.
- **`u_input` is not propagated** through a shot. It is estimated only for the one
  plate-response comparison above.
- **`u_model` does not exist.** It is exactly what validation would have produced, and
  validation is at level 0.

**So the reported `u_num` covers the numerics only. It must not be read as an error bar on
the physics.** The physics error bar is unknown, and its dominant term — a `lambda`
uncertain by a factor of nearly three, carrying 90 % of the load — is larger than every
numerical term in this document combined.

A GCI is a ~95 % band, so feeding it into `u_val` unchanged and then multiplying by
`k = 2` would expand a 95 % band twice. This package takes `u_h = GCI·|φ₁|/2` and says so;
the conservative alternative is available as `expanded_numerical_uncertainty`.

## The Five Standing Caveats

Every F0 result carries these unconditionally, because every one bites a bunker shot and
none is modelled (Zhang & Goldman 2014, _Phys. Fluids_ 26:101308):

1. **Transient response.** RFT assumes steady state; the transient caused a ~30 %
   over-prediction in sand-swimming speed. A bunker shot is _all_ transient.
2. **Disturbed-ground memory.** Sand does not heal, drag drops near disturbed ground, and
   a divot _is_ disturbed ground.
3. **Inclines.** A 20° tilt drops drag ~50 %, and the authors state it is unclear how to
   incorporate inclines into RFT. Bunker faces are inclined.
4. **Shadowing.** No wake model, so sheltered leading-edge elements are counted at full
   strength.
5. **Sharp corners.** Reduced accuracy along sharply varying surfaces — which is exactly
   the leading edge and bounce surface being designed.

Plus a sixth that is this package's own: **every fitted coefficient is borrowed**, none is
measured on golf bunker sand.

## What Would Move the Needle

In order of leverage:

1. **One drag test at wedge-representative speed (20–27 m/s) to fit `lambda`.** It carries
   ~90 % of the load and is currently known to a factor of three. Nothing else in this list
   comes close.
2. **Reproduce the granular-intrusion benchmark.** It validates the solver independently
   of golf, and the data already exists.
3. **A model carry correlation**, to turn the Wivou comparison from implemented into run.
4. **`delta_h` for a wedge**, from an F1/F2 tier or from PIV.
5. Plate penetration at three plate areas (discard widths under 5 cm) and a 6 × 6 cm
   direct shear box for `Phi`. **Not** angle of repose: three AoR methods on the same
   powder produced rolling friction spanning 300×.

## How to Read a Result From This Tool

Ranking two sole geometries against each other under identical conditions is the use this
model supports: the borrowed constants and the uncalibrated `delta_h` shift both designs in
the same direction, so the _difference_ is more trustworthy than either absolute. Quoting
an absolute force, an absolute carry, or any ball-outcome quantity is not supported by
anything in this document.

---

_Generated blocks in this file are checked against `bunkershot3d.vandv` by
`tests/bunkershot3d/vandv/test_credibility.py`. Edit the module, not the block._

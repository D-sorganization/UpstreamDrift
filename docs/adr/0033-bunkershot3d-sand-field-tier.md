# ADR-0033: Sand-Field Visualization Tier for BunkerShot3D

- Status: Proposed
- Date: 2026-08-16
- Decision Makers: UpstreamDrift maintainers
- Related Issues/PRs: #8709 (B1), epic #8699, amends ADR-0032, blocks #8710
  (B2), #8711 (B3), #8712 (B4), #8713 (B5)
- Amends: [ADR-0032](0032-bunkershot3d-club-design-architecture.md) — fidelity
  tier table and the F3 tier's scope

## Context

Epic #8699 splits BunkerShot3D visualization in two. Track A paints quantities
F0 already computes onto the sole and needs no new physics. Track B — sand
velocity fields, cross-sections through the impact zone, and what reaches the
ball — needs a tier that solves the sand's motion.

**F0 does not solve it.** 3-D Dynamic Resistive Force Theory integrates an
empirical resistive stress over the intruder's swept surface. It never forms a
sand velocity, at any resolution, so no amount of post-processing yields one.
This is a property of the constitutive shortcut, not a gap in the
implementation.

ADR-0032 named two candidate tiers below the GPU-only F2 reference:

- **F3**, a MuJoCo discrete-grain proxy, already present at
  `src/bunkershot3d/backends/mpm/driver.py` (327 lines, no
  `NotImplementedError`), with `mujoco 3.9.0` installed in the repo venv.
- **F1**, a 2-D plane-strain continuum, not implemented, costed by ADR-0032 at
  seconds-to-minutes per shot and described as predictive in-plane.

Issue #8709 and epic #8699 both describe the MuJoCo proxy as _available now_
and _producing genuine per-grain velocities today_. **That description is
false, and the rest of this ADR rests on measurements that establish it.**

### The Epic's Premise Does Not Hold

This ADR does not merely choose between two options — it removes the premise
epic #8699 was built on, so that premise is stated and withdrawn explicitly
rather than quietly worked around.

The epic inferred that a grain-resolved run was "reachable now" from two
observations: that `backends/mpm/driver.py` is 327 lines with **zero
`NotImplementedError`**, and that `mujoco 3.9.0` is installed. Neither
observation is evidence of capability, and both were wrong about this file.
Measured below: the model does not build at all; the shipped grain population
places **0 of 1000 grains** anywhere near the clubhead, forming a
single-grain-thick line **150 mm off the club's path**; and the implied bed is
**0.00116 grain diameters** deep. The club swings through empty space.

**Line count and the absence of `NotImplementedError` were used as a proxy for
"works", and they are not one.** A backend that raises is honest about being
unfinished; this one would have run to completion and written a result file
full of zeros with an `OUT_OF_ENVELOPE` verdict that reads as a fidelity
caveat rather than as "nothing happened". That failure mode is the reason this
decision had to be made against a running probe rather than against a reading
of the source.

## Measured Evidence

All measurements were taken on the primary development machine (Windows 11,
Intel Iris Xe, **no NVIDIA GPU**) against `mujoco 3.9.0`, `numpy 2.2.6`,
CPython 3.13.3, using the repo's own canonical configuration
(`src/bunkershot3d/calibration/configs/canonical.yaml`) and the repo's own
`MPMDriver`. The probe script was kept outside the repository.

### 1. The Driver Does Not Run

`MPMDriver.setup()` raises before a single step is integrated:

```
ValueError: Error: mass and inertia of moving bodies must be larger than mjMINVAL
Element name 'g0', id 2, line 19
```

The cause is a defect in the driver's generated MJCF. The grain geoms are
emitted without a `density` attribute, so MuJoCo applies its **default
1000 kg/m³** rather than the configured 2650 kg/m³ silica. For a sphere,

```
I = (2/5) m r²,  m = ρ (4/3) π r³   ⇒   I = (8/15) π ρ r⁵
```

so the inertia falls as the _fifth power_ of the radius, and at the canonical
0.4 mm grain the model is just under MuJoCo's floor:

| density                         | min. representable radius | I at configured r = 0.200 mm   |
| ------------------------------- | ------------------------- | ------------------------------ |
| 1000 kg/m³ (MuJoCo default)     | 0.2266 mm (measured)      | 5.36e-16 — **below mjMINVAL**  |
| 2650 kg/m³ (silica, configured) | 0.1864 mm (measured)      | 1.42e-15 — passes, 1.4x margin |

Both floors were confirmed by bisection against `MjModel.from_xml_string`, not
computed only. Two distinct defects are stacked here: grains that are
water-density (a physics error that would have gone unnoticed had the model
built), and a model that cannot be built at all. USGA bunker sand is specified
at 0.25–1.0 mm diameter, with the bulk at 0.25–0.5 mm, so **real bunker sand
sits at or below MuJoCo's minimum-inertia floor** unless density is set
explicitly.

### 2. There Is No Bed, and the Club Never Touches It

With `MAX_SPHERES = 1000` against a configured population of 50,000, over the
canonical 0.4 × 0.3 × 0.1 m domain:

| quantity                                    | measured                                  |
| ------------------------------------------- | ----------------------------------------- |
| lattice capacity                            | 909 × 681 × 227 = 140,519,583 sites       |
| grains in one complete layer                | 619,029                                   |
| fraction of a single layer filled by 1000   | **0.0016**                                |
| grain x extent                              | −0.19976 … 0.19977 m                      |
| grain y extent                              | −0.14961 … **−0.14915 m**                 |
| grain z extent                              | 0.000200 … 0.000212 m                     |
| implied bed depth at φ = 0.6                | 0.000465 mm = **0.00116 grain diameters** |
| clubhead y-band (half-width 25 mm at y = 0) | −0.025 … +0.025 m                         |
| **grains inside the clubhead's y-band**     | **0 of 1000**                             |

The cap does not thin the bed — it destroys it. The placed grains form a
**single-grain-thick line along the far y wall**, 150 mm from the club's path.
ADR-0032 already found the 50,000-grain configuration to be a bed 1/17 of one
grain diameter deep; the driver's own `MAX_SPHERES` cap takes that to **1/860**.
Repairing the build failure alone would therefore yield a run in which the
clubhead sweeps empty space and returns an identically zero contact wrench.

### 3. The Step Budget Is Dominated by Empty Travel

| quantity                        | measured           |
| ------------------------------- | ------------------ |
| dt (Courant-limited)            | 8.781e-7 s         |
| main-loop steps                 | 113,882            |
| settle steps                    | 500                |
| **total steps per shot**        | **114,382**        |
| clubhead x span over trajectory | −1.2407 … 1.2407 m |
| domain x extent                 | ±0.2 m             |

The reference swing traverses 2.48 m at ~24.8 m/s while the domain is 0.4 m
wide, so only ~16 ms of the 100 ms trajectory places the club inside the box:
roughly **86 % of the integration budget is spent with the club outside the
domain entirely**.

### 4. Cost and Hard Ceiling of a _Repaired_ Proxy

To cost the tier fairly rather than costing its defects, a best-case proxy was
built directly: correct silica density, a compact bed placed under the club
path, the club driven through it at 25 m/s, and the repo's own Courant rule
(`dt = 0.1 d / v`). Timings are per `mj_step` over 300 steps after settling,
extrapolated to a 20 ms shot.

| spheres | d (mm) | nv     | dt (s) | ms/step | max contacts | s per 20 ms shot                                            |
| ------- | ------ | ------ | ------ | ------- | ------------ | ----------------------------------------------------------- |
| 180     | 2.0    | 1,086  | 8.0e-6 | 0.1300  | 40           | 0.3                                                         |
| 1,000   | 2.0    | 6,006  | 8.0e-6 | 0.6133  | 104          | 1.5                                                         |
| 1,000   | 1.0    | 6,006  | 4.0e-6 | 0.6019  | 104          | 3.0                                                         |
| 3,840   | 1.0    | 23,046 | 4.0e-6 | 3.6914  | 260          | 18.5                                                        |
| 10,000  | 1.0    | —      | —      | —       | —            | **build failed: "engine error: Could not allocate memory"** |

**The binding constraint is memory, not time.** Wall-clock at 3,840 spheres is
18.5 s per shot, which would be affordable; but 10,000 spheres cannot be
allocated at all on this machine. The tractable ceiling is **roughly 4,000
spheres**.

### 5. What 4,000 Spheres Represent

| quantity                                                      | value              |
| ------------------------------------------------------------- | ------------------ |
| true-scale grains, 0.4 × 0.3 × 0.1 m bed, φ = 0.6, d = 0.4 mm | 2.149e8            |
| largest tractable proxy                                       | 3,840 @ d = 1.0 mm |
| **fraction of true grain count**                              | **1.79e-5**        |
| grain volume represented                                      | 2.011e-6 m³        |
| bed volume to be represented                                  | 7.200e-3 m³        |
| **fraction of the sand present**                              | **0.028 %**        |

The proxy grain is also **2.5x oversize** against the canonical 0.4 mm sand,
which shifts packing fraction, inertial number and dilatancy together — the
discrepancy is not a single scalar that can be divided out.

### 6. Per-Grain State Is Extractable; a _Field_ Is Not

Per-grain position and velocity are directly available from `MjData`
(`data.xpos`, `data.qvel`), and a 1,000-sphere repaired run gives physically
plausible per-grain motion:

| quantity                     | measured                     |
| ---------------------------- | ---------------------------- |
| grain speed min / mean / max | 0.6062 / 1.1328 / 3.1643 m/s |
| grains moving > 0.5 m/s      | 1000 of 1000                 |
| contacts at final step       | 390                          |

But Track B consumes **fields**, and binning those grains onto a grid is where
the tier fails:

| grid | cells  | grains per cell |
| ---- | ------ | --------------- |
| 20³  | 8,000  | **0.125**       |
| 40³  | 64,000 | **0.0156**      |

At the coarsest grid anyone would call a cross-section, **seven of every eight
cells are empty**. A velocity field, a density field, and a shear overlay — the
substance of #8710, #8711 and #8712 — cannot be formed from this sample. Grain
glyphs can be rendered; a field cannot be honestly interpolated from them.

### 7. Grain State Never Reaches Disk

`BunkerShotResultWriter.write_grain_state(time, positions, velocities)` exists
in `src/bunkershot3d/io/schema.py`, but `MPMDriver.run()` calls only
`write_clubhead_state` and `write_contact_wrench`. **The driver never writes a
single grain.** Even a repaired, well-populated run would produce a result file
containing no sand state at all, so the B2 extraction path does not exist even
in outline.

### 8. The Envelope the Pictures Would Be Drawn Inside

Recomputed from the package's own constants rather than quoted:

| scale                  | Froude number | exceedance of 3D-RFT's stated limit (0.4) |
| ---------------------- | ------------- | ----------------------------------------- |
| clubhead, 100 mm       | 25.24         | 63.1x                                     |
| sole width, 30 mm      | 46.08         | 115.2x                                    |
| **leading edge, 5 mm** | **112.88**    | **282.2x**                                |

Delivery speed is also **17.4x** the fastest intrusion anywhere in the
published RFT/DRFT corpus (1.44 m/s). The NASA-STD-7009B self-assessment in
`vandv/credibility.py` records:

| factor              | achieved | threshold |
| ------------------- | -------- | --------- |
| Verification        | 2        | 3         |
| **Validation**      | **0**    | 3         |
| Input Pedigree      | 2        | 3         |
| Results Uncertainty | 2        | 3         |
| Results Robustness  | 1        | 2         |
| Use History         | 0        | 2         |
| M&S Management      | 3        | 3         |

Validation is at level **0 of 4**. Any frame this epic renders is a picture
drawn by an unvalidated model far outside its stated envelope, and a picture
persuades far more readily than a table.

## Decision

**Field visualization is backed by F1, a 2-D plane-strain continuum, and F1 is
specified as a material-point (MPM) solver. The MuJoCo grain proxy is not
adopted as the field tier.**

This reverses the sequencing epic #8699 assumed. The epic chose the proxy
because it was "reachable now" and F1 "is not implemented". The first half is
measurably untrue — the driver does not build a model — and once both tiers
are un-implemented work, the comparison is decided on what each can produce
rather than on which is nearer to hand.

### Why Not the Grain Proxy

The disqualifying fact is not that the proxy is coarse. It is that **Track B
consumes fields, and a 4,000-sphere sample cannot carry one**.

- #8710 (B2) asks for sand velocity and density **on a grid**.
- #8711 (B3) asks for **cross-sections** with velocity, density and shear
  overlays.
- #8712 (B4) asks for velocity and mass flux **by region** of the ball.

All three are field quantities, and all three need a population per sampling
cell large enough for the cell average to mean something. The measured value is
**0.125 grains per cell on a 20³ grid** — seven of every eight cells empty —
against the ~10²–10³ per cell a granular average is normally formed over. A
field interpolated from that sample is not a coarse field; it is a picture of
the interpolator. Raising the count does not rescue it: the tractable ceiling
on this machine is **~4,000 spheres**, set by an allocation failure at 10,000,
and 3,840 spheres is **1.79e-5 of the true grain count** and **0.028 % of the
sand by volume**.

Three further defects compound it, each measured above: the model does not
build; the shipped grain population places **zero grains in the clubhead's
path**; and `MPMDriver.run()` never calls `write_grain_state`, so no grain
would reach disk even from a repaired run. Adopting this tier means fixing all
four before the first frame, and still arriving at 0.125 grains per cell.

There is also a constitutive objection independent of count. MuJoCo resolves
contact with a regularised soft constraint solver — no yield surface, no flow
rule, no dilatancy, friction imposed at the velocity level. ADR-0032 excluded
PBD/XPBD for materially this reason ("there is no parameter a measured friction
angle maps onto"). The MuJoCo contact model is nearer to that excluded class
than to a Hertzian DEM, so a proxy built on it cannot be calibrated toward the
F2 reference even in principle.

### Why F1, and Why MPM Specifically

A continuum solve produces a field **by construction**: every cell carries a
velocity, a density and a stress, because those are the solution variables.
There is no sampling-population question, and cross-sections are exact slices
rather than reconstructions.

ADR-0032 listed F1 as "reduced-order / 2-D plane-strain continuum" and, in its
alternatives, retained SPH as the plane-strain option. **This ADR narrows that
to MPM.** Two reasons:

1. **The blade-thickness floor follows SPH into 2-D.** ADR-0032 rejected SPH as
   the default because boundary handling needs blade thickness ≥ 4h ≈ 5.2·dx,
   which the ~0.5 mm leading edge cannot satisfy, and the repulsive-BC
   workaround turns the club force into a tuning parameter. Plane strain does
   not remove that constraint, it only makes a fine dx cheaper. ADR-0032 states
   in the same passage that grid-based MPM has no such floor.
2. **MPM shares its constitutive model with F2.** The F2 reference tier is
   Newton's `SolverImplicitMPM`. If F1 is also MPM, the material model and its
   parameters are calibrated once and carry between the cheap in-plane tier and
   the GPU reference; F1 becomes a genuine stand-in for F2 rather than a third
   unrelated rheology. An SPH F1 would need its own calibration against
   measurements that, per #8616, do not exist.

### What F1 Resolves, and What It Deliberately Does Not

**F1's job in this epic is bulk-scale fields, not club force.** The fields
Track B renders live at the 10–100 mm scale: the divot section, the flow ahead
of and along the face, what reaches the ball. Club force lives at the
leading-edge scale and is **F0's job**, where the per-element decomposition
already exists.

That split is load-bearing for cost. Resolving a 0.5 mm leading edge demands
dx ≲ 0.1 mm and drives the cell count and the CFL step together into the same
intractability trap ADR-0032 documented for DEM. Resolving a bulk flow field
needs dx ~ 1–2 mm. **F1 is specified at bulk resolution, with the leading edge
deliberately under-resolved**, and it is therefore barred from being quoted for
club force (see the quotable/qualitative table below). ADR-0032's
"seconds–minutes per shot" estimate is retained only at bulk resolution; it
does not survive a leading-edge-resolving grid, and no one should later read
that row as licence for one.

The other limitation is structural and permanent: **plane strain has no
out-of-plane flow.** Sand moving along the face heel-to-toe does not exist in
this model. That is not a resolution setting and no refinement removes it.

### How an Illustrative Frame Is Marked

This is the crux of the issue, and discipline is not a mechanism. A figure
outlives its caption: it is cropped, pasted into a deck, screenshotted into a
message, and shown to someone who never saw this ADR. The marking must survive
all of that, so it is placed **in the pixels and in the API**, not in prose.

1. **Provenance is composited into the raster.** Every exported field frame
   carries, burned into the image itself, the tier, the validity verdict, the
   Froude exceedance at the governing feature scale, and the words that no
   measured comparison exists. Captions are lost on the first crop; pixels are
   not.
2. **Non-quotable colourbars carry no numeric ticks.** They are ordinal —
   low → high — with the quantity and its units named but unscaled. This is the
   mechanism that makes "qualitative only" enforceable instead of aspirational:
   a reader cannot lift a number off a picture that does not present one. A
   numeric colourbar is available **only** for quantities the table below marks
   quotable.
3. **Illustrative frames have a distinct visual identity.** A fixed
   non-photorealistic palette plus a persistent diagonal watermark, chosen so a
   frame is recognisable as illustrative at thumbnail size and out of context.
   A predictive frame and an illustrative frame must never be distinguishable
   only by reading their labels.
4. **The export path cannot default.** The frame-export function takes the
   validity verdict as a required argument with no default value; a frame whose
   verdict is absent raises rather than rendering. This mirrors the rule
   ADR-0032 already sets for solver results, where every result carries its
   tier and verdict.
5. **Animations stamp every frame.** Not a title card — title cards are trimmed
   on export and skipped on loop.

### Which Quantities May Be Quoted

Validation stands at level 0 of 4 for this package, so **no F1 output is a
physical prediction**, and the table divides model outputs by how far they may
travel rather than by confidence.

| Quantity                                                 | Treatment                                                                 |
| -------------------------------------------------------- | ------------------------------------------------------------------------- |
| Grid-convergence metrics (GCI, residuals)                | **Quotable** — properties of the numerics, not of sand                    |
| In-plane divot section geometry (depth, length)          | **Quotable with tier + GCI band**, as model output, never as measurement  |
| Timing of peak load within a shot                        | **Quotable with tier**                                                    |
| Ordinal ranking between designs at identical settings    | **Quotable as a ranking only** — never the margin                         |
| Sand velocity magnitude and direction fields             | **Qualitative only** — ordinal colourbar, no ticks                        |
| Density / packing-fraction fields, shear-band location   | **Qualitative only**                                                      |
| Anything at the ball surface (velocity, flux, by region) | **Qualitative only**                                                      |
| Absolute club force / wrench from F1                     | **Refused** — F0 owns this; F1 is under-resolved at the edge              |
| Ball speed, launch angle, spin from F1                   | **Refused** — #8657's F0 path remains the only route, itself uncalibrated |
| Any heel–toe or out-of-plane distribution                | **Refused** — the quantity does not exist in plane strain                 |

"Refused" means the API raises, in the same shape as the existing
out-of-envelope refusals in `solvers/envelope.py`. It does not mean "discouraged
by documentation".

### How F1 Is Cross-Checked Against F0

B5 (#8713) compares the tiers on the quantities both produce: **contact wrench,
maximum sole depth, and divot geometry.** Two things must be stated plainly or
the comparison will be over-read.

**F1's wrench is per unit width.** Converting it to a force requires a declared
effective width, which is a modelling assumption and not a result. Therefore:

- **Shape and timing are compared unconditionally** — the rise, the location of
  the peak in time, the decay.
- **Magnitude is compared only after the chosen width is recorded in the run
  manifest**, alongside the config hash and tier, so no magnitude comparison
  can be reproduced without its assumption.

**The comparison is a consistency check between two uncalibrated models, not a
validation.** Agreement raises neither tier's NASA-STD-7009B validation level
above 0, because neither is being compared to a measurement. What the check can
do is falsify: disagreement beyond a declared band means at least one tier is
wrong, and that is worth knowing. It is therefore wired as a **gate**: F1 may
not export field frames for a configuration whose wrench peak-time and divot
depth fall outside the declared band against F0 on the canonical shot.

### Where the Ball Lives

**The ball becomes a body inside F1**, as a rigid circular section in the
plane-strain plane, coupled to the sand by the same traction integration used
for the sole. This is a modelling change and it is decided here rather than
left for #8712 to discover.

The consequences are specific and must travel with it:

- In plane strain the ball is an **infinite cylinder, not a sphere**. Flux onto
  it is per unit width and its geometry is wrong in the third dimension.
- The **below-equator versus face-side** split #8712 asks for is an in-plane
  distinction, so F1 can address it — qualitatively, per the table above.
- Any **heel–toe or lateral** distribution over the ball is out-of-plane and is
  **refused**, not approximated.
- **Ball launch remains F0's** momentum-transfer path (#8657). F1's ball exists
  to _show what reaches the ball_; it may not be used to compute ball speed,
  launch angle or spin.
- Consequently, **F1 as scoped here does not fix the lie-dependent ball-speed
  artefact** (#8704). That defect lives in the F0 transfer model and its
  resolution is not a benefit this ADR may claim.

## Alternatives Considered

1. **MuJoCo grain proxy as the field tier**, as epic #8699 proposed — rejected
   on measurement: the model does not build, the shipped configuration puts
   zero grains under the club, grain state is never written, and a repaired
   best case reaches 0.125 grains per sampling cell at a hard ~4,000-sphere
   memory ceiling.
2. **Repair the proxy and raise `MAX_SPHERES`** — rejected: 10,000 spheres
   fails to allocate on this machine, so the ceiling is a hardware limit, not a
   constant. Even at the ceiling the sample is 1.79e-5 of true grain count.
3. **F2 (Newton `SolverImplicitMPM`) on GPU** — unavailable: no NVIDIA GPU on
   the primary machine, and at 30–90 min/shot it cannot back an interactive
   view. It remains the reference tier per ADR-0032 and the eventual
   calibration target for F1.
4. **Derive a field from F0 by post-processing** — rejected as fabrication. F0
   never forms a sand velocity; a plausible-looking field synthesised from
   surface tractions would be the single most misleading artefact this epic
   could produce.
5. **Ship Track A only and close Track B** — a legitimate fallback, and
   explicitly the right outcome if F1 does not land. Track A answers the
   question a wedge designer actually asks (which part of the grind works) and
   needs no new physics. Track B's questions are real, but no honest answer to
   them exists today.
6. **3-D SPH or 3-D continuum** — out of scope for the same reasons ADR-0032
   gave: cost, and the blade-thickness floor for SPH.

## Consequences

**Positive.** Fields exist by construction, so cross-sections are exact slices
and every cell carries a value. F1 is predictive in-plane rather than merely
illustrative, which is a materially stronger basis for the whole track. Sharing
a constitutive model with F2 means the material calibration is done once. The
marking mechanism removes the affordance to misread a picture rather than
relying on a caption being read.

**Negative.** Track B is now gated on implementing a solver, not on wiring up
an existing one — this is real work and the epic's sequencing must change to
reflect it. Plane strain permanently excludes out-of-plane flow, so the
heel–toe questions are refused rather than answered. #8704 is not fixed here.
F1 at bulk resolution cannot be quoted for club force, so the tier that draws
the sand is not the tier that reports the load.

**Accepted debt.** The MuJoCo driver stays in-tree but is now recorded as
**non-functional**, not merely low-fidelity. It must either be repaired behind
a test that actually builds a model and puts grains under the club, or removed;
leaving 327 lines that raise on first use, described in two issues as working,
is worse than either. This is filed as a follow-up rather than fixed here,
because this ADR is a decision and the repair is a change to
`backends/mpm/`.

**Follow-ups.**

- Repair or remove `backends/mpm/driver.py`; at minimum add a test that
  `setup()` builds, since none currently does.
- Amend the epic #8699 child list: B2/B3/B4 are gated on an F1 implementation
  issue that does not yet exist.
- Record the effective-width assumption in the result manifest schema before
  B5 lands.

## Validation

How CI verifies this decision, in the repo's existing idiom:

- **A test asserts the MuJoCo build defect is closed before the tier is ever
  used** — `MPMDriver.setup()` must construct a model, and a test must assert
  that the placed grain population intersects the clubhead's swept y-band. Both
  are currently false and both are silent.
- **A test asserts F1's refusals raise**, in the same shape as
  `solvers/envelope.py`: club force, ball launch quantities and any
  out-of-plane query must raise rather than return.
- **A test asserts the export path has no default verdict** — calling frame
  export without a validity verdict is a `TypeError`, and a frame carrying a
  non-predictive verdict must render an ordinal colourbar with no numeric
  ticks.
- **The F0↔F1 consistency gate runs on the canonical shot** and blocks field
  export outside the declared band on wrench peak-time and divot depth.
- **Grid convergence is reported through the existing machinery** —
  `vandv/convergence.py` and `vandv/gci.py` already exist and F1 must produce a
  GCI study rather than a single-resolution result.
- **The credibility statement is regenerated** so that adding a tier cannot
  quietly raise a reported validation level; validation stays at 0 until a
  measurement exists.

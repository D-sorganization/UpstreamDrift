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

TODO — recorded in the next commit.

## Alternatives Considered

TODO — recorded in the next commit.

## Consequences

TODO — recorded in the next commit.

## Validation

TODO — recorded in the next commit.

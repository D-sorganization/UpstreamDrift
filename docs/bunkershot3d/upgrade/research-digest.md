# BunkerShot3D Upgrade — Research Digest

Shared context for all implementation agents on epic
[#8607](https://github.com/D-sorganization/UpstreamDrift/issues/8607).
Read this plus `baseline-findings.md` and `0032-bunkershot3d-club-design-architecture.md`
before writing code.

---

## 1. The Solver: DRFT, and Why the Inertial Term Is the Leading Term

Element stress (Agarwal, Karsai, Goldman & Kamrin, _Science Advances_ 2021,
arXiv:2005.10976):

```
t = alpha(beta, gamma) * H(-z_eff) * |z_eff|  -  n_hat * lambda * rho * v_n^2
```

`z_eff` is an **effective** depth accounting for free-surface depression.

Verified in-session with alpha_z = 2.02 N/cm^3, 40 mm divot, lambda = 1.1, rho = 1600:

| v (m/s) | depth term | inertial term | ratio    |
| ------- | ---------- | ------------- | -------- |
| 0.5     | 0.081 MPa  | 0.0004 MPa    | 0.01     |
| 5       | 0.081 MPa  | 0.044 MPa     | 0.54     |
| 25      | 0.081 MPa  | 1.100 MPa     | **13.6** |

Crossover **6.8 m/s**; bunker delivery is 20–27 m/s. **Calibrate lambda before alpha.**

**Constants** (Agarwal et al., _J. Terramechanics_ 2019, arXiv:1901.10667) — Quikrete
medium sand 0.3–0.8 mm, the band overlapping the USGA bunker window:

| quantity                         | value       |
| -------------------------------- | ----------- |
| `alpha_z(0, pi/2)`               | 2.02 N/cm^3 |
| Phi (internal friction)          | 34 deg      |
| packing fraction                 | 0.6         |
| rho_grain                        | 2600 kg/m^3 |
| MPM mu_internal (F2 cross-check) | 0.53        |

lambda: ~1.0 wheels, **1.1 horizontal plate drag** (closest to a planing sole),
1.4–2.8 vertical intrusion, ~4 flap runner. Plate drag followed `F = K|z| + lambda*rho*A*v^2`,
K = 580 N/m.

alpha(beta, gamma) is a 2-D response surface, nine Fourier terms — Li, Zhang & Goldman,
_Science_ 339(6126):1408 (2013), arXiv:1303.7065.

**Accuracy**: same wheel-in-sand experiments, sinkage MAE RFT **2.7 mm** vs MPM 3.2 mm vs
Bekker/Wong-Reece **26.1 mm**. RFT is not a fidelity compromise. Known bias: over-predicts
~35 % on natural sand (grain angularity) — budget for it.

**Do not use Bekker/Wong-Reece.** ~10 bevameter parameters, worse than RFT with one, and
its shear displacement modulus K from a 6x6 cm shear box (~0.6 mm) is 20–50x smaller than
literature values (10–30 mm), producing garbage.

---

## 2. Wedge Geometry: The Acushnet Schema (US10143900B2 / US10661131B2)

All measured in a vertical plane perpendicular to the leading edge.

| Parameter                   | Sym          | Range (broad / pref / most pref)       |
| --------------------------- | ------------ | -------------------------------------- |
| Sole width                  | d1           | 5–22 / 10–22 / 15–22 mm                |
| Offset datum                | d2           | exactly 1.2 mm rearward of LE point    |
| Sole entry height           | d3           | 2.0–8.0 / 2.5–8.0 / 3.0–8.0 mm         |
| Sole entry angle            | Phi          | >60 / >65 / >67.5 deg                  |
| Leading-edge sole radius    | rho1         | <10 / <9 / <8 mm                       |
| Trailing-edge sole radius   | rho2         | >40 / >41 / >42 mm                     |
| Bounce angle                | theta        | >20 deg (examples 15.99, 18.42, 20.78) |
| Sole camber area            | —            | >42 / >45 / >48 mm^2                   |
| Sole Contour Ratio          | rho1/rho2    | <0.25 / <0.21 / <0.19                  |
| Camber-to-Bounce Area Ratio | camber/theta | >2.00 / >2.50 / >3.00 mm^2/deg         |

Heel–toe rocker: toe radius <40 mm, heel <30 mm, centre >70 mm. That asymmetry **is**
heel/toe relief.

> **CORRECTION (found while implementing #8609, verified independently).** These are
> **local curvature measurements at a point, not radii spanning the blade**, and they
> cannot be applied across one. A 75 mm centre radius integrated across a 78 mm blade
> lifts the heel and toe by **10.9 mm** — no wedge does that. Blade-scale radii are
> roughly 250 / 90 / 130 mm (centre / heel / toe), giving ~3.1 mm of lift. Treat the
> patent numbers as a curvature _field_ to be integrated, never as a blade-spanning arc.
> `rocker_offsets_m` integrates the field and states this in its docstring.
> **Open decision:** the blade-scale radii above are inferred, not published — do not
> quote them as patent-conformant without an OEM source or a measured head.

**Two bounce conventions — never mix.** Patent `theta` is geometric (to the true trailing
contact point, >20 deg); marketed bounce is to the ground-contact plane (4–14 deg). Make
the convention part of the type.

### Effective Bounce — Exact, and a Unit Test

Opening the face rotates the rigid head about the **shaft axis** by Omega:

```
L_eff = arcsin[ sin L cos(Om) + cos L cos(lam) sin(Om) + sin^2(lam) sin L (1 - cos Om) ]
delta_loft ~ delta_bounce ~ Om * cos(lam)
delta_aim                 ~ Om * sin(lam)
```

**Test case:** 56 deg loft, 64 deg lie, Omega = 20 deg -> **`L_eff = 64.591 deg`**, a gain
of **8.591 deg**, not 20. First-order 20\*cos(64) = 8.767. Aim opens 20\*sin(64) = 18 deg.
Limiting checks: Om=0 -> L; lam=90 -> L; lam=0 -> L+Om.

> **CORRECTION:** earlier revisions of this digest quoted `64.5` / gain `8.5`. That was a
> rounded restatement, not the value the formula gives. Verified independently: the exact
> closed form evaluates to **64.591 deg**. Pin tests to 64.591, not 64.5.

Shaft lean S: `L_eff = L - S`, `B_eff = B - S`, degree for degree (tour 4–14 deg).
Presentation to velocity: `beta_eff ~ B_static - S + Om cos(lam) + AoA`, AoA -2 to -12 deg,
the largest single term.

**Mass properties:** head 290–310 g; volume 1.8–2.7 in^3; CG height above LE
0.380–0.670 in (9.65–17.02 mm); CG depth ~1.9 mm. MOI axis conventions in patents are
frequently unstated — resolve against a measured head. Gear effect negligible (CG depth
~2 mm vs 35–40 mm driver). What matters is **MOI about the shaft axis**.

---

## 3. Sand

USGA / lab PSD: gravel (2 mm) <=2 %; very coarse (1 mm) <=15 %; **coarse+medium
(0.25–1 mm) 78–100 %**; very fine <=5 %; silt+clay <=3 %; **Cu 2.0–5.0**.
USGA GSR 58(11) 2020 by volume is tighter: coarse+medium **>=65 %**, very coarse <=7 %.

**Firmness axis (penetrometer, golf ball on tip, kg/cm^2):** <1.8 undesirable /
1.8–2.2 acceptable / 2.2–2.4 acceptable / >2.4 desirable. Sweep 1.6 / 2.0 / 2.4 / 2.8.

Depth: **100–150 mm floors, 50–75 mm faces**. Angular + no crusting = desirable.

**Gap to declare honestly:** no published bulk density, friction angle or angle of repose
specific to _golf bunker_ sand was found. Phi = 34 deg is borrowed from the Quikrete
analogue. Any preset must record that it is borrowed, not measured — this package already
had one honesty failure of exactly this kind (#7999).

**Moisture is two regimes**, ~20x apart: damp/capillary (apparent cohesion ~1–10 kPa,
approx `2*sigma/r`) vs saturated/cavitating.

**Drainage — corrects a common assumption.** A 10 ms impact in USGA sand is globally
**drained**, not undrained: `k ~ 3e-4 m/s`, `E_oed ~ 20 MPa` -> `c_v = 0.61 m^2/s`, so over
a 20 mm zone `T = c_v t / L^2 ~ 15`. The real effect is a **local shear-band** dilation whose
suction is **capped by cavitation at ~ -100 kPa gauge** (~65 kPa extra shear strength,
order 130 N against a 200–600 N peak). **Implement the cap** — without it a poroelastic
model invents multi-MPa suction and overpredicts severalfold.

---

## 4. Why Not the Other Methods

- **PBD/XPBD — categorically excluded.** Friction limit is proportional to _numerical
  penetration depth_ (Unified Particle Physics 2014 Eq. 23), not normal stress. No yield
  surface, no flow rule. XPBD explicitly does **not** apply compliance to contact, so no
  contact-force estimate exists. Stack stability comes from scaling particle mass by stack
  height. There is no parameter a measured friction angle maps onto.
- **SPH — not default.** Boundary handling needs blade thickness >= 4h ~ 5.2\*dx; the
  leading edge is ~0.5 mm. Repulsive BCs make the measured club force a tuning knob.
  Artificial stress fires exactly in the cohesive regime you want to calibrate, and its
  epsilon knob is numerically indistinguishable from cohesion. Viable in 2-D plane strain.
- **DEM — infeasible at true scale.** 0.3x0.3x0.1 m at d50 = 0.3 mm, phi = 0.6 is
  **3.8e8 particles**; full-history frictional state is ~300–600 B/particle = 115–230 GB,
  ~2x over a single 80 GB GPU. True quartz stiffness gives 644k steps. Coarse-graining to
  10x puts particles at 3 mm — comparable to the **leading-edge radius**, so bounce
  behaviour becomes discretisation, not physics. **Use grain-diameter-to-leading-edge-radius
  as an explicit validity metric** and run a CG convergence study before any design claim.
- **MuJoCo has no granular capability, structurally**: convex soft-constraint solver, cost
  bound by constraint rows (a packed bed is one giant connected island), heightfield geom
  pairs keep only the **first 64 contact points**, MJX requires preallocated `naconmax`.
- **LIGGGHTS is dead.** Last substantive commit 2023-04-24; the repo description now points
  at commercial Aspherix as its replacement. **Taichi dormant** (no release since Jul 2024,
  no 2026 commits, largest consumer forked it). **CFDEM frozen**, OpenFOAM-6-pinned, GPL-3.
- **Newton `SolverImplicitMPM`** (Apache-2.0, `pip install newton`) ships
  `example_mujoco_mpm_coupled_solver.py` — exactly our architecture. Exposes friction,
  yield_pressure, yield_stress, tensile_yield_ratio, **dilatancy**, hardening, viscosity per
  particle. **Requires an NVIDIA GPU, which this machine does not have** — keep optional and
  CI-skippable. Default `voxel_size` 0.1 m; a wedge needs 1–2 mm.

### Kratos MPM Is the Better F1/F2 Tier _for This Machine_ — It Is CPU

`pip install KratosMPMApplication`. **This is the strongest higher-fidelity option we can
actually run locally**, and it was found late, so it supersedes "Newton or nothing":

- **Maintained far better than anything else surveyed**: ~3,287 commits in 52 weeks.
- **Python-first**, not an input-deck wrapper: 28 Python driver modules, wheels for
  Linux/Windows/macOS, `requires_python >=3.8`.
- **Real granular plasticity**: Mohr-Coulomb (finite strain, associative and non-associative),
  **Mohr-Coulomb with strain softening**, Modified Cam-Clay. Mixed **u–p** formulation with
  stabilisation — necessary, because plain MPM locks badly in near-incompressible dense sand.
- **Purpose-built rigid-tool coupling**: material-point non-conforming Dirichlet conditions via
  penalty, Lagrange multiplier and perturbed Lagrangian, plus MPM–FEM, MPM–DEM and
  MPM–Rigid-Body coupling. This is exactly the machinery for pushing a rigid clubhead through
  material points and **reading reaction force off the interface**. It was built for a thesis on
  granular mass flows loading protective structures — the same "granular flow hits a solid
  object, what force does it feel" problem.
- ⚠️ **Licence is BSD-4-Clause, not BSD-3** — it carries the **advertising clause**. Fine for
  internal use; a commercial product must reproduce the acknowledgement. Flag before shipping.
- Gaps: **no GPU** (OpenMP/MPI only) and **no adjoint/AD for MPM**, so design optimisation stays
  derivative-free (which our DOE/surrogate plan already assumes).
- Note the rename: `ParticleMechanicsApplication` → **`MPMApplication`**. Older tutorials are stale.

### Do Not Chase an ML Surrogate Yet

The GNS literature scores **kinematics, not forces**. Choi & Kumar's barrier-interaction cases
report runout distance and upstream depth and **no reaction force on the barrier at all**. The
only published force error bar for a learned granular surrogate is Haeri, Holz & Skonieczny
(2024), _Eng. Appl. Artif. Intell._ 135:108765 — subspace GNS on an excavation blade and rover
wheel — at **7 % (excavation) and 17 % (wheel) mean percentage force error**, on smooth
quasi-steady low-speed sweeps, trained on the same code and geometry family. Our design deltas
between two grinds are almost certainly smaller than that. The failure is structural: GNS decodes
per-particle **acceleration**, while tool force is a **boundary integral of contact tractions**,
so small spatially-correlated errors near the tool cancel in a position metric and accumulate in
the traction sum. Plus every published GNS result is gravity-driven at ~1 m/s; we are at 20–27 m/s.
If a surrogate is ever trained, put **tool reaction force in the loss** and hold out a geometry
family to measure force error honestly.

### Citation Warning — Do Not Propagate

**"Penner, A. R. (2002), 'The physics of sand wedges', _Am. J. Phys._" appears not to exist.**
Two independent checks failed to find it: a Crossref journal query over AJP 2002 returns 279
works (260 in vol. 70) with no `sand`/`golf`/`wedge`/`granular` title; AJP vol. 70 pp. 100–200 is
**page-contiguous** with no gap at the commonly cited "70, 134"; and **Penner's own review's
reference list does not contain it**. Verified Penner papers: _The physics of golf_,
_Rep. Prog. Phys._ 66(2):131–171, doi:10.1088/0034-4885/66/2/202; optimum loft of a driver,
doi:10.1119/1.1344164; convex face of a driver, doi:10.1119/1.1380380; putting,
doi:10.1139/p01-137; run of a golf ball, doi:10.1139/p02-035. **Cite none of them for sand.**

Also verified dead ends: **SPlisHSPlasH has no granular model whatsoever** (no Drucker-Prager,
no μ(I), no Mohr-Coulomb anywhere in the source — its "elasticity" modules are for rubber-like
solids). **CB-Geo MPM is unmaintained** (0 commits in 52 weeks; two-phase lives on unmerged
branches). **DualSPHysics is stalled** and its non-Newtonian module is frozen at a v5.0 fork
while mainline is v5.4. **NVIDIA FleX was never open source** — it carries the NVIDIA Source
Code License (1-Way Commercial), not BSD/MIT.

---

## 5. V&V (ASME / NASA)

Current standards: **VVUQ 1-2022** (terminology), **V&V 10-2019 (R2025)** (solid mechanics),
**V&V 20-2009** (CFD/heat), **VVUQ 20.1-2024** (multivariate validation metric),
**V&V 40-2018** (credibility vs risk), **NASA-STD-7009B (2024-03-05)** — note **7009A is
superseded**.

**Separate three things, in order.** Code verification (MMS, order-of-accuracy — _never_
uses experimental data) -> solution verification (GCI, gives `u_num`) -> validation
(vs experiment, gives model-form error).

**V&V 20 core equations:**

```
E     = S - D                                   (simulation minus experiment)
u_val = sqrt(u_num^2 + u_input^2 + u_exp^2)     (quadrature; independent aleatory sources)
u_num = u_h + u_it + u_ro                       (SIMPLE ADDITION, not RMS - epistemic)
U_%   = k * u_val,  k = 2 for ~95 %
```

**If |E| <= u_val you have learned nothing about model error** — the comparison is
noise-limited. Say so explicitly rather than plotting and calling it agreement.

**GCI (Celik et al., _J. Fluids Eng._ 130(7):078001, 2008):**

```
p        = (1/ln r21) * | ln|eps32/eps21| + q(p) |     (3a)
q(p)     = ln[ (r21^p - s)/(r32^p - s) ],  s = sign(eps32/eps21)
phi_ext  = (r21^p * phi1 - phi2)/(r21^p - 1)
GCI_fine = Fs * e_a21 / (r21^p - 1),  e_a21 = |(phi1-phi2)/phi1|
Fs = 1.25 (three or more grids), 3.0 (two grids)
```

`r` desirably > 1.3. `eps32/eps21 < 0` means oscillatory convergence — report the
percentage. Grid-to-grid difference **overestimates fine-grid error by 3x** at r=2, p=2;
that is why GCI divides by `(r^p - 1)`.

**Conservation: two classes needing two different tests.** Round-off class (mass, linear
and angular momentum) — fixed absolute tolerance ~1e-12, **do not run an order test**.
Truncation class (energy under a non-symplectic integrator) — the residual **should** scale
as `dt^p`, so the order test _is_ the test.

**Angular momentum is the test that finds DEM bugs.** It is conserved only if tangential
forces are applied as an equal-and-opposite pair **at the contact point** with matching
torques on both bodies. The classic bug — friction torque on one body only, or using the
particle centre instead of the contact point — is **invisible to a linear-momentum test**.
This is exactly finding B5b.

Symplectic drift theorem: symplectic -> `H(y_n) = H(y_0) + O(h^r)`, **bounded**;
non-symplectic -> `O(t*h^r)`, **linear secular drift**. So the test is (a) fit a linear
trend, assert slope ~ 0, and (b) assert the oscillation amplitude scales as `h^r` — halving
dt must shrink the energy swing ~4x for velocity Verlet. **Velocity Verlet's symplecticity
is destroyed by velocity-dependent forces** (damping, Coulomb friction, drag) — run the
bound test on a conservative sub-configuration only.

---

## 6. Testing

**Metamorphic relations** — the only practical oracle for a full bunker shot. Write the
**bit-exact** transforms first (power-of-two translation, 90 deg rotation, sign flip,
permutation) so you can assert to 1e-14:

| Relation                 | Transform                      | Expected                               | Catches                      |
| ------------------------ | ------------------------------ | -------------------------------------- | ---------------------------- |
| Translation              | `pos += c`                     | positions shift by c; forces identical | absolute-coordinate leakage  |
| Rotation (about gravity) | `pos,vel <- R pos, R vel`      | outputs rotate by R                    | axis-swapped indices         |
| Reflection               | mirror geometry                | outputs mirror, **spin flips sign**    | cross-product sign errors    |
| Permutation              | shuffle particle order         | identical to round-off                 | order-dependent accumulation |
| Time reversal            | negate v, damping off          | returns to start within O(h^r)         | integrator asymmetry         |
| Dimensional scaling      | lengths *lam, times *sqrt(lam) | Froude-similar                         | wrong units, hidden scale    |
| Monotonicity             | increase club speed            | ejecta mass, carry do not decrease     | sign errors, instability     |
| Refinement               | halve dt                       | answer changes by O(dt^p)              | order-degrading defects      |

Compose them with Hypothesis `@given` for thousands of instances free.

**Hypothesis settings for solver tests:** `@settings(deadline=None)` is mandatory;
`allow_subnormal=False` (subnormals diverge across platforms and flake);
`derandomize=True` in CI; cache `.hypothesis/`; use `target()` to hill-climb toward worst-case
energy drift; pin historical bugs with `@example` on the same test.

**Regression: regress on derived features, never raw per-particle state.** Lyapunov growth
turns a 1e-16 perturbation into an O(1) difference in a granular bed within ~1e3 collisions,
so a per-particle golden is guaranteed to fail and will be deleted, taking real coverage
with it. Tier tolerances: round-off invariants 1e-12, truncation quantities 1e-6, chaotic
outputs get MRs and statistical moments instead.

`assert_allclose` checks `|actual-desired| <= atol + rtol*|desired|` — **asymmetric**, and
default `atol=0` demands exact equality at zero. Use `strict=True` to catch dtype drift.

**`python -O` strips `assert`.** Anything safety-critical (NaN detection especially) must be
an explicit `raise`, never an `assert`.

---

## 7. Architecture

**The LoD fix is structure-of-arrays, not more accessors.** `sim.state.particles[i].pos.x`
is a Demeter violation _because `Particle` exists at all_. Going SoA
(`state.pos: (n,3)`) makes the chain one dot and is also 10–100x faster in NumPy. This is
the answer to finding B19 and to `check_lod.py`.

**Pass the leaf, not the root.** `hertz_force(gap, rel_vel, p: ContactParams)`, never
`hertz_force(sim)` — a function taking `sim` can reach anything.

**Where DRY should lose:** keep the deliberately naive O(N^2) reference implementation.
`neighbors/brute_force.py` is not dead code, it is **the oracle** every optimised backend is
tested against. Ginkgo does exactly this with a sequential reference executor. Duplication of
_implementation_ is safe when the _knowledge_ still has one authoritative representation.

**Ports/adapters:** the port must be **array-granular, not particle-granular** —
`compute_contact_forces(pairs, state, params) -> (forces, torques)`, never
`force_on(particle_i, particle_j)`, because it is called ~1e6 times per step.

**`@runtime_checkable` checks only method _presence_, not signatures** — never use
`isinstance` as a validation gate. The real contract is the conformance suite, parametrised
over `Registry.names()` so adding a backend without tests is impossible.

**Units:** SI float64 in the core, unit-suffixed names (`dt_s`, `v_mps`, `theta_rad`).
A units library in the hot loop costs 10–100x on small arrays. Pint at the config boundary
only, if at all.

**RNG:** `secrets.randbits(128)` -> `SeedSequence` -> `Generator(PCG64DXSM(ss))`; record
`ss.entropy` **and `numpy.__version__`** (NEP 19 permits stream changes on X.Y releases).
Never `np.random.seed()`; never `root_seed + worker_id` (unsafe, overlapping streams) —
use `parent.spawn(n)`.

**Config hashing:** RFC 8785 JCS, `allow_nan=False`, emit both `config_hash` (everything)
and `physics_hash` (excluding output paths, log level, thread count, seed) so you can ask
"which runs are the same experiment at different seeds?".

**Libraries — verified status Aug 2026.** scikit-optimize is **ARCHIVED** (2024-02-28,
real commits stopped Oct 2021) and Dragonfly is **dead** — these are the two most common
stale recommendations. EasyVVUQ is LGPL+GPL-3.0. OpenTURNS is LGPL-3. Alive and permissive:
SALib (MIT), SMT (BSD-3), BoTorch/Ax/Optuna/GPyTorch (MIT), PyMC (Apache-2.0),
Hypothesis (MPL-2.0), pytest-regressions (MIT), icontract (MIT).
**`scipy.stats.qmc` and `scipy.stats.sobol_indices` (SciPy >= 1.11) may remove the need for
any new dependency** — prefer that, per ADR-0032's no-new-hard-deps rule.

# Research Digest — ADDENDUM (Read Together With `research-digest.md`)

Where this contradicts the main digest, **this file wins.**

---

## 1. The Warning: We Are Far Outside RFT's Validated Envelope

**3D-RFT's own stated limit is Fr = v/sqrt(gL) < 0.4, and the entire RFT/DRFT validation
corpus tops out at 1.44 m/s.** At v = 25 m/s with L = 0.1 m we are at **Fr = 25** — about
60x outside the stated envelope and ~20x beyond any published validation.

| Feature scale L              | Fr (limit 0.4) | micro-inertial I (limit 0.1) | d/L   |
| ---------------------------- | -------------- | ---------------------------- | ----- |
| 100 mm (clubhead)            | 25.2           | 0.126                        | 0.005 |
| 30 mm (sole width)           | 46.1           | 0.768                        | 0.017 |
| 5 mm (bounce / leading edge) | 112.9          | 11.3                         | 0.100 |

This does **not** invalidate the architecture — DRFT is still the only per-geometry method
cheap enough for a design loop. It makes ADR-0032's **validity-envelope requirement the single
most important feature of the solver, not a nicety.** Published RFT coefficients are an initial
guess, not a validated model. The tool must report Fr, I, D/d and delta/d alongside every force,
and must refuse to report forces at Fr > ~1 unless the dynamic terms are active.

Askari and Kamrin's formal failure criterion: RFT is _exact_ for a frictional-plastic medium
because superposition reproduces `F = rho_c * g * L^3 * Psi(beta, gamma)` exactly. It breaks
when rate/size effects let the element size `lambda` survive in the dimensionless groups —
whenever `I_G = v^2 d^2 / (g lambda^2)` or `d/lambda` moves Psi. **For lambda = 2 mm,
d = 0.5 mm, v = 25 m/s: I_G ~ 4.0.** Not small. That is the formal statement of the risk.

Documented RFT failure modes (Zhang and Goldman 2014, _Phys. Fluids_ 26:101308), every one of
which bites a bunker shot:

- **Transient response.** RFT assumes steady state; DEM needs ~1/5 of a stroke to reach it.
  Caused a ~30% over-prediction in sand-swimming speed. A bunker shot is _all_ transient.
- **Disturbed-ground memory.** Sand does not heal; drag drops substantially near previously
  disturbed ground. A divot _is_ disturbed ground.
- **Inclines.** A 20 deg tilt drops drag ~50%, and the authors state it is unclear how to
  incorporate this into RFT. Bunker faces are inclined.
- **Shadowing.** No wake model — leading-edge elements behind other parts are over-counted.
- **Sharp corners.** "Reduced accuracy along surfaces that sharply vary" — i.e. exactly the
  leading edge and bounce surface we are trying to design.

## 2. DRFT Has TWO Corrections, Not One

```
t = alpha(beta, gamma) * H(-z_tilde) * |z_tilde|  -  n_hat * lambda * rho * v_n^2
z_tilde = z + delta_h
```

1. **Inertial** `lambda * rho * v_n^2`. Measured lambda: **1.1** oblique horizontal plates,
   **1.4** sphere vertical impact, **2.8** 2-D plane-strain vertical plate, **1.0** grousered
   wheel.
2. **Dynamic structural correction `delta_h`** — bulk inertia lowers the _effective free
   surface_, which feeds back through the depth-linearity. **The source paper's central finding
   is that the inertial term alone is insufficient: applying `lambda*rho*v^2` without `delta_h`
   gave the wrong SIGN of sinkage for wheels, at every lambda from 1 to 100.** For a wheel,
   `delta_h = r * (r * omega^2 / g)`. The form is geometry-specific — **measure it for a wedge**
   from the F1/F2 tier or from PIV. Vertical/horizontal plate intrusions are the cases where
   delta_h ~ 0, which helps the leading edge but not the sole.

## 3. 3D-RFT, Implementation-Grade

Agarwal, Goldman and Kamrin, _PNAS_ 120 (2023), doi:10.1073/pnas.2214017120.

Local cylindrical frame: `z_hat` up, `r_hat` = horizontal component of `v_hat`,
`theta_hat = z_hat x r_hat`. Angles beta (surface tilt), gamma (attack), psi (twist), all on
[-pi/2, pi/2]. Applied only to leading edges (`v_hat . n_hat >= 0`) and only below the surface.

```
alpha_r     =  f1*sin(beta)*cos(psi) + f2*cos(gamma)
alpha_theta =  f1*sin(beta)*sin(psi)
alpha_z     = -f1*cos(beta) - f2*sin(gamma) - f3

x1 = sin(gamma)
x2 = cos(beta)
x3 = cos(psi)*cos(gamma)*sin(beta) + sin(gamma)*cos(beta)

f_i = sum_k c_i[k] * T_k
```

| k   | T_k      | c1       | c2       | c3       |
| --- | -------- | -------- | -------- | -------- |
| 1   | 1        | 0.00212  | -0.06796 | -0.02634 |
| 2   | x1       | -0.02320 | -0.10941 | -0.03436 |
| 3   | x2       | -0.20890 | 0.04725  | 0.45256  |
| 4   | x3       | -0.43083 | -0.06914 | 0.00835  |
| 5   | x1^2     | -0.00259 | -0.05835 | 0.02553  |
| 6   | x2^2     | 0.48872  | -0.65880 | -1.31290 |
| 7   | x3^2     | -0.00415 | -0.11985 | -0.05532 |
| 8   | x1\*x2   | 0.07204  | -0.25739 | 0.06790  |
| 9   | x2\*x3   | -0.02750 | -0.26834 | -0.16404 |
| 10  | x3\*x1   | -0.08772 | 0.02692  | 0.02287  |
| 11  | x1^3     | 0.01992  | -0.00736 | 0.02927  |
| 12  | x2^3     | -0.45961 | 0.63758  | 0.95406  |
| 13  | x3^3     | 0.40799  | 0.08997  | -0.00131 |
| 14  | x1\*x2^2 | -0.10107 | 0.21069  | -0.11028 |
| 15  | x2\*x1^2 | -0.06576 | 0.04748  | 0.01487  |
| 16  | x2\*x3^2 | 0.05664  | 0.20406  | -0.02730 |
| 17  | x3\*x2^2 | -0.09269 | 0.18519  | 0.10911  |
| 18  | x3\*x1^2 | 0.01892  | 0.04934  | -0.04097 |
| 19  | x1\*x3^2 | 0.01033  | 0.13527  | 0.07881  |
| 20  | x1*x2*x3 | 0.15120  | -0.33207 | -0.27519 |

**Material scaling — recalibrate to bunker sand from two measurable properties:**

```
xi_n = rho_c * g * f_hat(mu_int)
f_hat = 894*mu^3 - 386*mu^2 + 89*mu
```

| rho_c (kg/m^3)                  | mu=0.6 (phi=31 deg) | mu=0.7 (35 deg) | mu=0.84 (40 deg) |
| ------------------------------- | ------------------- | --------------- | ---------------- |
| 1450 (raked loose)              | 1.53e6              | 2.56e6          | 4.73e6           |
| **1550 (measured bunker sand)** | 1.64e6              | **2.73e6**      | 5.05e6           |
| 1700 (compacted)                | 1.79e6              | 3.00e6          | 5.54e6           |

**Surface-friction cutoff** (normal force is nearly independent of mu_surf for mu_int 0.3-0.9):

```
alpha = xi_n * [ |alpha_n|*(-n_hat) + min(mu_surf*|alpha_n|/|alpha_t|, 1) * alpha_t ]
```

**One-shot calibration**: a single vertical flat-plate intrusion gives
`xi_n = F_vertical / (alpha_z_gen(beta=0, gamma=pi/2, psi=0) * ds * |z|)`.

2-D RFT (Li, Zhang and Goldman, _Science_ 339:1408, 2013) uses nine nonzero Fourier
coefficients: A00 0.206, A10 0.169, B11 0.212, B01 0.358, B(-1)1 0.055, C11 -0.124,
C01 0.253, C(-1)1 0.007, D10 0.088.

## 4. Corrections That Supersede the Main Digest

### Wet Sand — Viscous Dissipation Dominates Capillary Cohesion at Clubhead Speed

`F_visc / F_cap = (3/4) * Ca * (R/h)`. For d50 = 0.33 mm, water, v = 25 m/s: **Ca = 0.34**,
F_cap max = 76 uN, F_visc at h = 1 um = 3210 uN = **42x F_cap**. The crossover gap
h_c = 42 um **exceeds the 33 um bridge rupture distance**, so viscous dominates over the
entire bridge lifetime. **The whole quasi-static wet-granular cohesion literature is calibrated
in the wrong regime for impact.** Grain Stokes number ~2400, so lubrication does not prevent
contact — model it as rate-dependent dissipation, not extra static cohesion.

### Measured Cohesion Is Far Lower Than the Main Digest States

Richefeu, El Youssoufi and Radjai (2006), _Phys. Rev. E_ 73:051304: phi is independent of water
content (sand ~33 deg) and **c saturates at w ~ 1-3%**, because cohesion tracks the _number_ of
bonds, not the water volume. Angular sand 0.1-0.4 mm: **c = 600 Pa**; 1 mm beads: 150 Pa.
**Predicted for our d50 = 0.33 mm: c ~ 0.33 kPa** — not the 1-10 kPa quoted in the main digest,
which comes from finer/siltier soils. **Model wet-vs-dry as essentially binary**, not a smooth
function of moisture. Dynamic stress rho\*v^2 ~ 970 kPa is ~1400x the cohesive stress scale, so
moisture enters through the splash layer, the viscous rate term, and packing/crust state — not
through bulk shear resistance.

### Do NOT Soften Contact Stiffness to 1E7-1E8 Pa

Verified in-session: max Hertzian overlap `delta_max/d` is **independent of grain size**, so
coarse-graining does not help — only stiffness does. At v = 25 m/s:

| E (Pa)        | delta_max/d |
| ------------- | ----------- |
| 1e7           | **47%**     |
| 1e8           | 19%         |
| 1e10          | 3.0%        |
| 7e10 (quartz) | 1.4%        |

**`calibration/configs/canonical.yaml` currently sets `youngs_modulus: 1.0e7`** — grains would
pass halfway through each other. The "soften to 1e7-1e8" DEM folklore is calibrated for
quasi-static flow and does not survive a 25 m/s impact. Keep delta/d < 1-2%, i.e. E >= ~1e10.

Related: coarse-graining scaling gives **E proportional to h** (coarse particles must be made
_softer_), which fights the overlap constraint directly. Cohesion has three mutually
incompatible coarse-graining rules (cohesive stress, Bond number, Cohesion number); no single
criterion suffices.

### Inertial vs Frictional Crossover, and a Force Smoke Test

Katsuragi and Durian: inertia dominates when `v^2 > 25*mu*g*z`, i.e. `Fr_z > 3.4`. At 25 m/s,
z = 5 cm the ratio is **113x** — the rate-independent Coulomb term is ~1% of the load.
Anchors: `C_d ~ 2` on rho_bulk; `d1 = 3.4 * D_body`, depth-independent;
`k = 20 * mu * rho_g * g * D^2`.

**Smoke test: a 20x80 mm sole at 25 m/s gives ~1550 N = 527 g on a 0.30 kg head**, stopping it
in ~5 ms of submerged travel. That is the right order for a real bunker shot — use it as an
end-to-end sanity check.

### Mu(I) Is Ill-Posed at Both Ends of Our Range

Barker, Schaeffer, Bohorquez and Gray (2015), _JFM_ 779:794. With `chi = (I/mu)*dmu/dI`, the
system is ill-posed (Hadamard) when `C = 4*chi^2 - 4*chi + mu^2*(1 - chi/2)^2 > 0`. For the
standard parameter set it is well-posed only for **0.019 < I < 0.863**. A bunker shot spans the
ejecta plume (P -> 0, I >> 1) _and_ the static bed ahead of the club (I -> 0) simultaneously.
**Drucker-Prager / Coulomb is _always_ ill-posed**, as is critical-state soil mechanics.
If an F1/F2 continuum tier is built, the **Barker and Gray (2017) regularization
(alpha = 1.9, kappa = 0.05) is mandatory** — it asymptotes to Bagnold/collisional dissipation
as I grows, which is exactly our regime.

### Dunatunga and Kamrin

The 2017 paper is **_JMPS_ 100:45-60** (projectile impact and penetration in dry granular
media) — the high-rate intrusion paper — not a CMAME implicit version. Their return map is a
**closed-form quadratic, not a Newton loop**, with a density-based free-surface criterion
(`rho < rho_c => sigma = 0`). Their own verdict: results depend far more strongly on `mu_s`
than on `mu_2` — **calibrate mu_s carefully, mu_2 loosely.**

### MPM Convergence Trap (For W9)

At ~3.5 particles per cell, standard piecewise-linear MPM **fails to converge beyond ~20 grid
cells — the error _increases_ with refinement** (Steffen, Kirby and Berzins 2008, _IJNME_
76:922). Cubic B-spline converges out to 2560 cells at rates near 2. **Do not report a
grid-convergence study using a linear-basis MPM.**

### Angle of Repose Is a Near-Worthless Identifier, Quantified

Three AoR measurement methods on the same powder produced sliding friction spanning 4x, rolling
friction spanning **300x**, and surface energy spanning 13x — all matching AoR (Gaboriault et
al. 2026, arXiv:2605.09371). AoR is also a zero-shear-rate measurement, and we are at
I ~ 0.1-10. **Do not calibrate on angle of repose.** This is independent confirmation of
baseline finding B14.

### A Fully Characterised Real Bunker Sand Exists — Use It as the Seed Preset

Covia Signature 500, Turf and Soil Diagnostics file #22040060 (Apr 2022), ASTM F1632 Method B
and ASTM F1815:

| Property         | Value                                                 |
| ---------------- | ----------------------------------------------------- |
| D15 / D50 / D85  | **0.19 / 0.33 / 0.57 mm**                             |
| Cu               | 2.2                                                   |
| Bulk density     | **1550 kg/m^3** (e = 0.71, n = 0.415, phi = 0.585)    |
| Penetrometer     | 2.0 kg/cm^2 (196 kPa)                                 |
| Angle of repose  | **30 deg** (straight pile)                            |
| Infiltration     | 36.2 in/hr                                            |
| Shape / crusting | angular to sub-rounded, medium-high sphericity / none |

Sieve (% retained): 1.0 mm 4.5 | 0.5 mm 12.0 | **0.25 mm 53.6** | 0.15 mm 25.1 | 0.10 mm 2.7 |
0.05 mm 0.6. Ottawa F-65 reference: Gs = 2.65, e_min = 0.51, e_max = 0.78. phi_crit for quartz
~33 deg. Penetrometer bands: <1.8 undesirable, 1.8-2.2 acceptable, 2.2-2.4 acceptable,

> 2.4 desirable (kg/cm^2; 1.8 = 177 kPa, 2.4 = 235 kPa).

This partly closes the "no measured bunker-sand properties" gap flagged in the main digest —
but it is **one commercial sand from one lab report**, not a population. Record it as such.

### Bolton Dilatancy Is at Its Cap in a Bunker

`I_R = I_D*(Q - ln p') - R`, Q = 10, R = 1 for quartz, capped at `0 <= I_R <= 4`.
A bunker shears at **p' ~ 0.1-2 kPa**, but Bolton was calibrated at ~150 kPa and above. At
p' = 1 kPa, `I_R = 10*I_D - 1`, so **any I_D > 0.5 saturates the cap**:
phi_peak - phi_crit = 12 deg (triaxial) / 20 deg (plane strain), psi_max = 25 deg.

Compacted sand is therefore at maximum possible dilatancy — sharp peak then softening, strong
volume expansion (the sand "explodes"). Freshly raked sand (I_D -> 0) contracts monotonically:
the club digs and the ball comes out short. **This is the largest dry-side lever in the model.**
Flag it explicitly as an extrapolation outside Bolton's calibrated stress range; the I_R <= 4
cap exists precisely to stop it running away.

### Citation Corrections

- **Third independent confirmation**: the Penner "physics of sand wedges" paper does not exist.
  Penner's complete 74-work publication list was pulled from OpenAlex and AJP vol. 70 searched
  directly. Do not cite it.
- Soulie et al. (2006) capillary-bridge model is **doi:10.1002/nag.476**, not nag.512.
- Column-collapse scaling: 2-D/channel is `1.2a` (a < 1.8) then `1.9*a^(2/3)` (a > 2.8);
  **axisymmetric** is `r_inf = r_i*(1 + 1.24a)` (a < 1.7) then `r_i*(1 + 1.6*sqrt(a))`.
  Do not splice the 2-D prefactor onto the axisymmetric exponent, and always state whether
  a = H/R or H/half-width.

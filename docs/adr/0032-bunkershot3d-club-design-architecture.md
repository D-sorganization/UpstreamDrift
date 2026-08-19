# ADR-0032: BunkerShot3D as a multi-fidelity club-design tool

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** UpstreamDrift maintainers
- **Supersedes:** the implicit architecture of epic #5398 (3-D granular bunker-shot simulation)

## Context

`src/bunkershot3d/` was built to answer "can we simulate a bunker shot with a
granular solver?" The question we now need it to answer is different:

> Given two wedge sole geometries, which one performs better, in what
> conditions, and how confident are we?

That is a design tool, not a demo. An audit of the existing package (recorded
in the epic) found 30 defects. Three of them are not gaps but **arithmetic
impossibilities** that invalidate the original architecture:

1. **The canonical configuration contains no sand bed.** 50,000 grains at
   d = 0.4 mm in a 0.4 × 0.3 × 0.1 m domain is a solid volume fraction of
   1.4 × 10⁻⁴ — a settled bed 0.023 mm deep, about 1/17 of one grain
   diameter.
2. **Resolved DEM cannot reach the real scale.** A 100 mm USGA bunker base at
   φ = 0.60 requires 2.1 × 10⁸ grains. At the Rayleigh-limited timestep
   (t_R = 2.1 × 10⁻⁷ s for 0.2 mm quartz, so Δt ≈ 4.2 × 10⁻⁸ s), a 20 ms shot
   is ~5 × 10¹³ contact updates — days per shot on a GPU.
3. **The Chrono backend integrates at ~11,900× the stability limit**, having
   conflated the output sampling rate with the integration timestep.

A design tool needs to run hundreds to thousands of shots for a sweep. Any
architecture whose base unit of work is a resolved granular simulation cannot
deliver that, no matter how well implemented.

Two further constraints are binding:

- **No NVIDIA GPU on the primary development machine** (Intel Iris Xe only).
  Newton's `SolverImplicitMPM` — otherwise the strongest high-fidelity
  candidate, Apache-2.0 and experimentally validated against granular
  collapses — requires Maxwell+ with driver 545+. It cannot be the default
  path, and anything that depends on it must be optional and CI-skippable.
- **The thin leading edge (~0.5 mm) is the geometric crux.** SPH boundary
  handling needs blade thickness ≥ 4h ≈ 5.2·dx, which a 0.5 mm edge cannot
  satisfy; repulsive boundary forces would make the club force — the primary
  output — a tuning parameter. Grid-based (MPM) or analytic contact does not
  have this floor.

## Decision

**Restructure BunkerShot3D as a multi-fidelity model with a fast, analytic,
CPU-native solver as the default**, and treat high-fidelity granular solvers
as optional reference backends used to calibrate and validate it.

### Fidelity tiers

| Tier   | Solver                                                 | Cost/shot                 | Role                                                             |
| ------ | ------------------------------------------------------ | ------------------------- | ---------------------------------------------------------------- |
| **F0** | **Dynamic** Resistive Force Theory (3D-DRFT), analytic | ~ms                       | **Default.** Design iteration, DOE, optimisation, interactive UI |
| **F1** | Reduced-order / 2-D plane-strain continuum             | ~seconds–minutes          | Cross-check on F0; cheap sensitivity                             |
| **F2** | MPM (Newton `SolverImplicitMPM`), GPU                  | ~30–90 min                | Reference truth; calibrates F0; optional, fleet-only             |
| **F3** | DEM (MuJoCo proxy / Chrono / LIGGGHTS)                 | intractable at true scale | Grain-scale studies only, explicitly scoped and labelled         |

RFT is the right default because it is the only method that is genuinely
_per-geometry_: the force on an intruder is an integral of a local stress
response over the swept surface, so changing the sole shape changes the answer
without re-running a granular solve. That is exactly the design question. It is
also not a fidelity compromise — benchmarked against the same wheel-in-sand
experiments, RFT's sinkage error (2.7 mm MAE) is comparable to MPM's (3.2 mm)
and an order of magnitude better than classical Bekker/Wong-Reece
terramechanics (26.1 mm), at roughly 10⁶ times lower cost.

**F0 must be the _dynamic_ formulation (DRFT), not quasi-static RFT.** The
stress on an element is

```
t = α(β,γ)·H(−z̃)·|z̃|  −  n̂·λ·ρ·v_n²
```

and at bunker-shot speeds the second term dominates. With α_z ≈ 2.02 N/cm³
(medium sand, 0.3–0.8 mm — the size band overlapping the USGA bunker window),
a 40 mm divot, λ ≈ 1.1 and ρ ≈ 1600 kg/m³:

| v (m/s) | depth term | inertial term | ratio |
| ------- | ---------- | ------------- | ----- |
| 0.5     | 0.081 MPa  | 0.0004 MPa    | 0.01  |
| 5       | 0.081 MPa  | 0.044 MPa     | 0.54  |
| 25      | 0.081 MPa  | 1.100 MPa     | 13.6  |

The two terms cross at **6.8 m/s**. A greenside bunker shot is delivered at
20–27 m/s, so the inertial term carries ~90 % of the load: it is the leading
term, not a correction. Any quasi-static solver — baseline RFT, Bekker,
Wong-Reece — is wrong here by an order of magnitude. **λ is therefore the
primary calibration target, ahead of α.**

Note in passing that the existing code's hard-coded 5.0 m/s "impact velocity"
sits below the crossover, in a regime with the wrong dominant physics.

**Every tier implements one `GranularSolver` Protocol** and every result
carries its fidelity tier plus a validity verdict. A solver used outside its
calibrated envelope must say so rather than return a plausible number.

### Structural decisions

1. **Domain objects, not a god config.** `BunkerShotConfig`'s 15 flat
   delegating accessors are replaced by narrow value objects (`SandState`,
   `WedgeGeometry`, `SwingCondition`, `SolverSettings`) passed directly to the
   code that needs them. This satisfies the Law of Demeter by _reducing
   coupling_ rather than by adding forwarding methods.
2. **The wedge is a first-class parametric model**, not a mesh. Sole geometry
   (bounce, leading-edge radius, sole width, camber radius, rocker, heel/toe
   relief, grind) is the design vector. Meshes and mass properties are
   _derived_ from it and are verified watertight before use. The parameter set
   and its measurement procedures follow the Acushnet sole-geometry patent
   family (US10143900B2 / US10661131B2), which defines sole width d₁, sole
   entry height d₃ and angle Φ at a 1.2 mm datum, leading- and trailing-edge
   sole radii ρ₁/ρ₂, camber area, and the derived Sole Contour Ratio ρ₁/ρ₂ and
   camber-to-bounce area ratio — a consistent, machine-checkable schema.
   **Two bounce conventions must not be mixed:** the patent's geometric bounce
   (measured to the true trailing contact point, >20°) is not marketed bounce
   (measured to the ground-contact plane, 4–14°).
3. **Effective bounce is a computed field over the sole, not a scalar input.**
   Opening the face is a rotation about the shaft axis, so at wedge lie angles
   λ ≈ 64° it yields only `Δloft ≈ Δbounce ≈ Ω·cos λ` (≈0.44°/°) while costing
   `Ω·sin λ` (≈0.9°/°) of aim. Shaft lean subtracts degree-for-degree, and
   attack angle (−2° to −12°) is the largest term of all. What the solver needs
   per surface element is the local attack angle β and intrusion angle γ
   relative to the velocity vector — which is exactly DRFT's input, so the
   geometry and solver parameterisations coincide.
4. **Mass properties are computed natively.** Volume, centroid and inertia
   tensor come from exact divergence-theorem integration over the triangle
   mesh (numpy only), with `trimesh` used as an independent cross-check in
   tests when installed. An OEM tool must be able to verify its own numbers,
   and `trimesh` is not a hard dependency of this repo.
5. **The ball exists.** The package computes ball launch conditions and hands
   off to the existing `SwingBallFlightPipeline` / `ImpactSolverAPI` rather
   than reimplementing flight.
6. **Results are reproducible artifacts.** A versioned result schema with
   contiguous arrays (replacing one HDF5 group per timestep), carrying a run
   manifest: config hash, RNG seeds, library versions, git SHA, solver tier,
   and validity verdict.
7. **Design-by-Contract uses the existing platform toolkit**
   (`shared.python.core.contracts`: `@precondition`, `@postcondition`,
   `require`, `ensure`, `check_finite`). No new contract framework.
8. **No new hard dependencies.** DOE, sensitivity and surrogate modelling are
   built on numpy/scipy (`scipy.stats.qmc` provides Sobol/LHS/Halton).
   `mujoco`, `newton`, `warp`, `trimesh`, `pychrono` are all optional.

### Physics decisions

- **Wet sand is two distinct regimes, not one.** Damp/capillary
  (apparent cohesion ~1–10 kPa) and saturated/cavitating are ~20× apart in
  force contribution and are modelled separately.
- **A 10 ms bunker impact in USGA-spec sand is globally _drained_, not
  undrained.** With k ≈ 3 × 10⁻⁴ m/s and E_oed ≈ 20 MPa, c_v ≈ 0.61 m²/s and
  the time factor over a 20 mm disturbed zone is T ≈ 15. The real effect is
  **local shear-band dilation whose suction is capped by cavitation at about
  −100 kPa gauge**, worth roughly 65 kPa of extra shear strength — order 130 N
  on the club against a 200–600 N peak. The cavitation cap is mandatory:
  without it a poroelastic model invents multi-MPa suction and overpredicts
  force severalfold.
- **Position-based dynamics (PBD/XPBD) is excluded.** Its friction limit is
  proportional to numerical penetration depth rather than normal stress, it
  has no yield surface or flow rule, contact compliance is explicitly disabled
  so no contact force estimate exists, and stack stability is achieved by
  scaling particle mass with stack height. There is no parameter a measured
  friction angle maps onto.
- **CFDEMcoupling is excluded**: upstream states it will not be updated, it is
  pinned to OpenFOAM-6, and it is GPL-3.
- **Taichi is excluded**: no release since v1.7.4 (Jul 2024), no commits in
  2026, and its largest downstream consumer forked it rather than contribute.

## Consequences

**Positive.** Design iteration becomes interactive. Every result is
reproducible and carries a validity verdict. The wedge parameter set is the
actual OEM vocabulary, so the tool answers the question a designer asks. No
new hard dependencies, so CI stays green on machines without a GPU.

**Negative.** RFT has a real validity envelope (it degrades near free
surfaces, at very shallow depth, and outside its calibrated speed range) and
enforcing that envelope honestly means the tool will sometimes refuse to
answer. Calibrating F0 against F2 requires GPU access the primary dev machine
does not have, so that calibration is fleet-scheduled rather than local.

**Accepted debt.** The Chrono and LIGGGHTS backends stay in-tree but are
explicitly labelled non-viable at true grain scale and are not part of the
supported path. Neither `pychrono` nor a LIGGGHTS binary is a declared
dependency; both are only ever exercised against mocks.

## Alternatives considered

- **Fix the DEM backends and keep them primary** — rejected: 2.1 × 10⁸ grains
  and days-per-shot make sweeps impossible regardless of code quality.
- **SPH as the default solver** — rejected: the blade-thickness floor
  (t ≥ 5.2·dx) cannot represent a 0.5 mm leading edge, and the repulsive-BC
  workaround turns the measured club force into a tuning knob. Viable in 2-D
  plane strain, which is retained as an F1 option.
- **MPM as the default** — rejected as _default_ only because it requires a
  GPU absent from the development machine and costs 30–90 min/shot; retained
  as the F2 reference tier.

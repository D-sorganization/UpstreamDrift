# BunkerShot3D Backend Comparison

> **This document was rewritten under issue #8616 (W9, Verification & Validation).**
> The previous version asserted eight physical divergences between the three granular
> backends. Seven of them were **contradicted by the code in this repository**, and the
> eighth — a Chrono-specific Rayleigh safety factor — described something that did not
> exist until
> issue #8612 implemented it. Nothing was tested. Those claims have been deleted rather
> than softened.
>
> **Every statement below is asserted by a test in
> [`tests/bunkershot3d/vandv/test_backend_claims.py`](../../tests/bunkershot3d/vandv/test_backend_claims.py).**
> If you want to add a claim here, add its test first. A claim about backend behaviour
> that nothing exercises is how this document went wrong the first time.

## What These Backends Are

Per [ADR-0032](../adr/0032-bunkershot3d-club-design-architecture.md) all three are **F3
(grain-scale DEM)** — explicitly _not_ the supported design path. The default solver is
the F0 dynamic-RFT tier in `src/bunkershot3d/solvers/`. A USGA bunker base at true scale
needs 2.1 × 10⁸ grains and days per shot, so any tractable run on any of these three is a
coarse-grained proxy, not the bunker.

Neither `pychrono` nor a LIGGGHTS binary is a declared dependency. Both are exercised
only against mocks, which is why so few behavioural claims can be made about them.

## What Is Actually Different

|                         | Project Chrono                            | LIGGGHTS                                                  | MuJoCo                                       |
| ----------------------- | ----------------------------------------- | --------------------------------------------------------- | -------------------------------------------- |
| Contact resolution      | soft-sphere Hertz–Mindlin (`ChSystemSMC`) | soft-sphere Hertz–Mindlin (`pair_style gran model hertz`) | soft-constraint solver at the velocity level |
| Clubhead present?       | yes, a **box** collision shape            | **no clubhead at all**                                    | yes, a **box** geom                          |
| Rayleigh timestep limit | enforced                                  | enforced                                                  | **deliberately not enforced**                |
| Courant traversal limit | enforced                                  | enforced                                                  | enforced                                     |
| Particle representation | grains                                    | grains                                                    | spheres, capped at 1000                      |

Three things follow, and they are the whole of the honest comparison:

1. **Chrono and LIGGGHTS use the same contact model.** Both are soft-sphere
   Hertz–Mindlin. There is no contact-model divergence between them to report, and the linear
   contact law the previous version attributed to LIGGGHTS appears nowhere in the
   generated deck.
2. **The LIGGGHTS deck contains no intruder.** It is a sand box with walls, gravity,
   particle insertion and a run command. The generated deck says so in its own header,
   and the driver refuses to execute it. No club force can come out of it, so no
   comparison of club forces against it is possible.
3. **MuJoCo is not a continuum and has no grid.** It resolves contacts through a convex
   soft-constraint solver rather than a Hertzian contact law, so the grain wave-speed
   (Rayleigh) limit does not govern and is switched off on purpose; the Courant traversal
   limit still applies. Its sphere count is capped at 1000 against a true-scale
   requirement of 2.1 × 10⁸.

## Timestep Stability

The 0.2 fraction of the Rayleigh time is real, but it is **shared, not a Chrono
property**. It
lives in `src/bunkershot3d/backends/stability.py` as `RAYLEIGH_SAFETY_FACTOR` and is
applied by `require_stable_timestep` to every soft-sphere backend. Quoting it as a
divergence gets the architecture backwards.

Two limits are applied together:

- **Rayleigh**: `dt <= 0.2 t_R`, where `t_R` is the surface-wave transit time across a
  grain. Enforced for Chrono and LIGGGHTS; disabled for MuJoCo, whose contacts are not
  soft-sphere Hertzian.
- **Courant**: the body may not traverse more than a fixed number of grain diameters per
  step, or contacts are never detected. Enforced everywhere.

## Coarse-Graining

`GrainPopulation.coarse_graining_factor` defaults to **1.0** (no coarse-graining) and
cannot be set below 1: one simulated grain cannot stand for less than one real grain.

What happens above 1.0 is **not documented here, because it has not been tested.** The
research digest records that grain-diameter-to-leading-edge-radius is the metric that
matters — coarse-graining to 10× puts particles at 3 mm, comparable with the leading-edge
radius, at which point bounce behaviour is discretisation rather than physics — and that a
coarse-graining convergence study must precede any design claim. No such study has been
run. The previous version of this document asserted that "Chrono and LIGGGHTS preserve
bulk invariants differently than MPM"; that assertion had no test and no source, and it
has been removed.

## What This Document Does Not Claim

- No claim about which backend predicts higher or lower peak shear forces. Nothing
  compares them, because two of the three cannot be executed here at all.
- No claim about boundary "padding", micro-slip, or smearing at the clubface. Those
  described a signed-distance-field and a triangular-mesh clubface that do not exist; both
  backends that have a clubhead represent it as a box.
- No claim about a particle-count budget. The 200,000-particle rationale the previous
  version gave for `C_g = 1.0` does not match the code: MuJoCo caps at 1000 spheres and
  the configured grain count is routinely 50,000, which ADR-0032 shows is not a bed at
  all.

For what _is_ verified and validated across the package, see
[`credibility.md`](credibility.md).

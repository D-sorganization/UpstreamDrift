# EPIC: MOSAIC — Model-Aware, Multi-Trial Forward-Dynamics Matching

**Status:** implementation started (reference kernels merged on the planar
analytic fixture); no human capture qualified. Supersedes the estimator design
of DIME (#11421) while reusing its contracts, reporting and provenance work.
Methods reference: `docs/research/model_aware_matching/model_aware_matching.tex`.
Code: `src/shared/python/estimation/mosaic/`.

## Purpose and User Outcome

Given many swings of one player, produce a forward-dynamics model of that
player — segment geometry, physically consistent inertial parameters, the
joint inputs of every swing and a shared torque template — such that
**uninterrupted open-loop forward replay reproduces each measured swing within
a declared tolerance**, with every identifiability limit reported rather than
hidden. The same kernels must serve humanoid motion retargeting and
manipulation (known payload / object identification).

## What Changed Since DIME

The 2026-10-05 audit (recorded in the methods reference, §12) found the DIME
children delivered contracts and reporting but not an estimator: no engine
provider, control-only single shooting with zero defects, an MHE whose arrival
factor is never applied, placeholder learned initialisers and benchmark
numbers, and no inertial regressor anywhere. MOSAIC replaces the estimator
core with a method that exploits two exact structural facts:

1. **Inverse dynamics is linear in the inertial parameters**:
   `Y(q,v,a) π = B u + J_cᵀ λ`. With a kinematic spline, the inner block
   (inputs, torque template, wrenches) is a sparse *linear* least-squares
   problem solved exactly — variable projection (Golub–Pereyra/Kaufman) — and
   the outer Gauss–Newton step over kinematics, geometry and inertia uses the
   exact Schur-reduced Hessian. No forward integration inside the optimiser:
   no exponential sensitivity, every node independent, fully batchable.
2. **The unactuated rows carry all the torque-free information** about π
   (floating base, passive pivot). ZTCF/ZVCF are exact row/column projections
   of the same regressor; the DIME "torque-independent feasibility criterion"
   is this projection, not an extra likelihood.

Inertial parameters live on the physically consistent manifold (log-Cholesky
of the pseudo-inertia / planar `(log m, h, log I_c)`), so every iterate is a
realisable body; the dynamics Jacobian in the inertial direction is exact.

Identifiability is treated as a first-class output: global mass gauge,
base-parameter lumping, and the **massless-body degeneracy** of minimum-effort
fitting (Proposition 3 in the reference) are stated, tested, and closed by
anchors — in golf, the **club as a known-inertia force sensor** — and by a
credible anthropometric prior.

"Locally learned inputs" are two concrete objects: the inner linear solve for
`u` (exact, not a linearisation) and the time-varying Riccati gains `K_t`
around the fit (the only linearisation), delivered together as the
reference-plus-gains schedule a whole-body controller consumes.

## Measured Reference Results (Planar 3-Link Fixture, NumPy, Single Core)

Two trials × 150 nodes at 200 Hz, 0.5 mm marker noise, +5 % geometry error,
±10–15 % inertial prior, total mass and club inertia anchored:
geometry < 0.1 mm; kinematics 0.5 mrad; torques 20 % RMS; observable inertial
parameters 1–3 % (prior 4–10 %); structural rank 9/12 with `m_0` correctly
unobservable; open-loop replay 0.030 rad RMS over 0.75 s with no divergence;
closed-loop 0.0005 rad at 5 % feedback effort; 37 outer iterations, ~6 s
total. Negative results (massless collapse with an under-resolved basis;
acceleration-noise torques without input smoothness; scale collapse without an
anchor) are retained as tests and gates.

## Qualification Gates (Frozen Before Human Experiments)

1. Kinematic whiteness at γ = 0 (basis resolution) before dynamics weighting.
2. Declared dynamics discrepancy σ_τ ≥ basis acceleration error × mass scale.
3. Physical consistency by construction; sanitised-prior violations logged.
4. Observability rank and per-parameter observable fraction published.
5. Open-loop replay configuration RMS and divergence time within tolerance.
6. Closed-loop feedback effort relative to nominal inputs.
7. Leave-one-trial-out spread of shared parameters; template explained variance.

Promotion target (to ratify in MOSAIC-01): open-loop replay tolerance and
torque RMS on synthetic truth, and time-to-accepted-match versus the
IK+smoothing baseline — all measured, never estimated.

## Child Issues and Dependency Order

| Task | Title | Depends on |
|---|---|---|
| MOSAIC-01 | Freeze benchmark protocol, tolerances and promotion metrics; wire the planar fixture as the conformance oracle | Ready |
| MOSAIC-02 | `RegressorModel` provider for Pinocchio (`computeJointTorqueRegressor`, batched) with conformance against the planar oracle | 01 |
| MOSAIC-03 | MuJoCo/MJX regressor provider (autodiff over body inertias, `vmap` batching) | 01 |
| MOSAIC-04 | Floating-base manifold spline: local-coordinate B-spline with retraction; outer solve on the manifold | 02 |
| MOSAIC-05 | Contact wrenches in the inner solve: unilateral/friction-cone SOCP, measured GRF as observations | 04 |
| MOSAIC-06 | Observation factors: 2-D keypoint projection, robust kernels, marker attachment offsets as geometry | 02 |
| MOSAIC-07 | Kinematic jerk prior and whiteness gate; declared-discrepancy calculator | 01 |
| MOSAIC-08 | Analytic dynamics derivatives (Singh–Russell–Wensing) replacing vectorised finite differences | 02 |
| MOSAIC-09 | Capture-A / capture-O fits through all gates; leave-one-trial-out study | 04, 05, 06, 07 |
| MOSAIC-10 | Multi-swing acquisition and registration (phase alignment, event anchoring) — the repository has one trial per capture | 01 |
| MOSAIC-11 | Actuation layer: declared joint-level activation model with bounds and first-order dynamics | 09 |
| MOSAIC-12 | Humanoid/manipulation transfer: known-payload anchor, reference-plus-gains export to WBC format | 04, 08 |
| MOSAIC-13 | Remediate DIME defects (list in reference §12); retire or re-scope stub modules; unify ZVCF semantics | Ready |
| MOSAIC-14 | Vectorised multi-trial batching over trials (block-arrow Schur, time-parallel Riccati) and runtime receipts | 08 |

## Non-Goals

No new solver framework, no engine abstraction beyond `RegressorModel`, no
claim of real-time performance or global optimality, no muscle-force
identification, no qualification of the GS3DX IK videos by this epic.

## Execution Policy

Per `AGENTS.md`: TDD (failing behavioural test first, with analytic or
synthetic truth), DbC preconditions via `core.contracts.require`, Law of
Demeter, DRY (reuse `estimation/mosaic` kernels — do not re-implement
regressors, bases, or solvers per engine), file-size budget, change fragments,
calculation-level LaTeX update in the same PR, headless tests, unavailable
engines reported as unqualified.

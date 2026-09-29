# 3D Golf Model — Project Specification

This is the canonical "what we are trying to build, and why" document for
the 3D Golf Model project. Every implementation decision should ladder up
to one of the goals here. Every PR should cite the section it advances or
the issue it closes.

> **Audience:** every contributor (human or agent) working on the
> [3D_Golf_Model](.) tree. Pair this with
> [matlab/MATLAB_GOLF_MODEL_GUIDE.md](matlab/MATLAB_GOLF_MODEL_GUIDE.md)
> (architecture & how-to-run) and
> [matlab/motion_matching/shared/GRIP_FIT_PLAYBOOK.md](matlab/motion_matching/shared/GRIP_FIT_PLAYBOOK.md)
> (the practical fit recipe).

---

## 1. Vision

> Given a measured golf swing — for now, club kinematics; eventually full
> body markers — produce a Simscape simulation that **reproduces the
> motion** with **physiologically efficient torque profiles**, in a way
> that lets us study the inverse-dynamics question (which muscle/joint
> contributions produced this swing?) and forward-predict counterfactuals
> (what if the player had X cm more wrist supination at the top?).

The simulation already exists ([GolfSwing3D_Kinetic.slx](matlab/src/model/GolfSwing3D_Kinetic.slx)).
The **fitting problem** — inverse-engineering the polynomial torque
coefficients and starting pose that drive the model to match observed data —
is the active research surface.

## 2. Concrete success criteria

A swing is "matched" when, on the canonical Wiffle ProV1 test trial:

| Metric                                                | Target                                                            | Status (2026-05-06)                                                            |
| ----------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Grip-position RMSE across the 0.30 s impact window    | **< 5 mm**                                                        | starting-position alignment hits 0.0 mm at impact; full-window fit not yet run |
| Grip-orientation RMSE (geodesic)                      | **< 1°**                                                          | not yet measured                                                               |
| Clubhead-position RMSE                                | < (model−measured shaft length) + 5 mm                            | currently ~27 mm at alignment, equal to club-length difference                 |
| Total simulated work                                  | within 30 % of physiological estimate (~280 J for a driver swing) | not yet measured                                                               |
| Wall-clock per fit (single-start fmincon)             | **< 15 minutes**                                                  | predicted ~10 minutes with FastRestart                                         |
| Wall-clock per fit (NN surrogate inference, Option 2) | **< 5 seconds**                                                   | not implemented                                                                |

Production-grade is reached when:

1. The end-to-end pipeline runs unattended on a held-out swing.
2. All four options produce comparable answers (or the divergences are
   documented and intentional).
3. Quantitative results land in the
   [shared/leaderboard.m](matlab/motion_matching/shared/leaderboard.m) format
   so we can compare options on the same trials.
4. CI on every PR exercises the cost contract and the loaders against a
   small real dataset.

## 3. Project surface — what exists, what's missing

### 3.1 Layered architecture

```
APPS         src/apps/golf_gui/                     ← Skeleton viewer, signal plots
                                                     (mostly working)
─────────────────────────────────────────────────────
MOTION       motion_matching/                       ← Where the inverse problem lives
MATCHING       option1_direct_optimization/         ← fmincon / multistart / hybrid (✅)
               option2_nn_surrogate/                ← PyTorch surrogate (🟡 stub)
               option3_inverse_nn/                  ← Inverse cVAE       (🟡 stub)
               option4_python_bridge/               ← scipy/JAX over MATLAB Engine (🔴 spec only)
               shared/                              ← Specs, loaders, cost, viz (✅)
─────────────────────────────────────────────────────
DATASET      src/scripts/dataset_generator/         ← Generates (θ, kinematics) pairs (✅)
GENERATOR    src/functions/dataset_generator/         at ~1856 columns/trial; 10k-trial
                                                     parquet exists on another box
─────────────────────────────────────────────────────
POSTPROC &   src/scripts/post_processing/            ← ZTCF/ZVCF analysis, plotting (✅)
ANALYSIS     src/scripts/plotting/
─────────────────────────────────────────────────────
SIMSCAPE     src/model/                              ← The physics. Don't change without
MODEL          GolfSwing3D_Kinetic.slx                 a documented reason.
               Kinetically_Driven_*_Joint.slx
               inputs/3DModelInputs_*.mat            ← Per-pose tunable inputs
```

### 3.2 What "production-grade" means per layer

| Layer             | Production-grade definition                                                                                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Simscape model    | All persisted settings stable; structural review documents every workspace parameter; new contributors can `sim()` it in < 5 min after `setup_matlab_environment`                             |
| Dataset generator | Reproducible; runs on parpool with checkpointing; produces a versioned parquet schema (DATASET_SCHEMA.md); exercises every joint coefficient                                                  |
| Motion matching   | All four options have a working `fit_swing_*` entry point that consumes the canonical `target` struct and produces the canonical `result` struct; cross-option comparison via `leaderboard.m` |
| Apps              | Skeleton + signal plotters consume the same canonical schemas as motion_matching; offset-tuning UI saves to the format motion_matching reads                                                  |
| CI                | Every PR runs `runtests motion_matching/shared/tests` + a smoke fit; reproducibility test on a fixed-seed trial                                                                               |

## 4. The fitting strategy (canonical workflow)

Per [GRIP_FIT_PLAYBOOK.md](matlab/motion_matching/shared/GRIP_FIT_PLAYBOOK.md), every full fit is a 2-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Starting Pose (seconds)                                        │
│   Goal: place the body so its grip lands at the measured grip pose at    │
│   the address frame, before any swing motion.                            │
│                                                                          │
│   Decision variables: a small number of *StartPosition* / *StartVelocity*│
│   model-workspace overrides (3–10 vars, e.g. hip translation,            │
│   shoulder/elbow rest angles).                                           │
│                                                                          │
│   Cost: ‖ model.grip(t=0) − target.grip(address_frame) ‖²                │
│                                                                          │
│   Output: input_overrides struct, applied via opts.input_overrides.      │
├─────────────────────────────────────────────────────────────────────────┤
│ Stage 2 — Torque Polynomial Coefficients (~10 min/swing)                 │
│   Goal: with starting pose fixed, find theta that drives the grip to     │
│   track the measured grip path while minimising total work.              │
│                                                                          │
│   Decision variables: theta ∈ ℝ^{n_joints × 7} = ℝ^{161} for n_joints=23 │
│                                                                          │
│   Cost: w_pg · ‖grip_sim − grip_meas‖² + w_og · d_geo² + w_a · anchor    │
│         + λ · W_total(θ)                                                  │
│                                                                          │
│   Solver: fmincon (Option 1) | NN surrogate (Option 2) | inverse NN     │
│           (Option 3) | external Python (Option 4)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

Why grip-primary? See [CLUB_IK_SPEC.md](matlab/motion_matching/shared/CLUB_IK_SPEC.md):
the grip is the rigid body→club interface. Matching it directly is independent
of the player's actual club length and robust to shaft flex (neither of
which the model simulates). Clubhead alignment then follows deterministically
from the modeled club geometry.

### 4.1 Coefficient bounds — spec vs. dataset reality

The torque polynomial is `τ_j(t; θ) = A_j·t^6 + B_j·t^5 + … + G_j` and the
**nominal** per-letter bounds are `|A,B|≤1000, |C,D|≤500, |E,F|≤100, |G|≤25`
(see [COMPACT_DATASET_SCHEMA.md](matlab/motion_matching/shared/COMPACT_DATASET_SCHEMA.md)).

**These bounds are nominal targets, not what the existing 10k-trial parquet
contains.** The random-sweep generator that produced the current dataset drove
all seven letters across a uniform `±1000` range — every `G` coefficient in
the dataset reaches `±1000`, not `±25`. Future generator runs may tighten to
the per-letter limits, but **motion-matching code (loaders, decoders,
clamping layers) must accept the current dataset as-is**. Decoders that hard-
clamp to the nominal per-letter bounds will silently truncate real targets
(e.g. PR #4240's cVAE clipped letter-G targets to 1/40 of their true
magnitude); use a uniform scale (e.g. `coefficient_scale_factor=50` per
PR #4267) when consuming this parquet.

## 5. Which option to use when

| Situation                                                 | Pick                             | Status                                                         |
| --------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------- |
| First time fitting a swing; want a baseline you can trust | **Option 1 — fmincon**           | ✅ Production                                                  |
| Already have a trained surrogate, want sub-second fits    | Option 2 — NN surrogate          | ✅ Shipped (see Option 2 notes below)                          |
| Full-trial inverse `(trajectory → 189 coefficients)`      | Option 3a — full-trial inverse   | 🟡 Shipped but data-limited (see Option 3 notes below)         |
| Per-timestep inverse-dynamics on filtered realistic data  | Option 3b — per-timestep inverse | 🟢 New production inverse path (`feat/timestep-inverse-model`) |
| Want JAX / scipy.optimize over the MATLAB sim             | Option 4 — Python bridge         | 🔴 Needs implementation                                        |

Every option must consume the same `target` schema and emit the same
`result` schema. Mixing options is then a one-line code change (see the
leaderboard helper).

### 5.1 Option 2 — forward NN surrogate (shipped)

- Production checkpoint: `data/training/surrogate_v4/checkpoint_production.pt`
  (~541 k parameters; clubhead-speed MAE 0.04 mph; grip RMSE ~43 mm).
  > **Note (codex review feedback, issue #4312):** the
  > `data/training/surrogate_v4/` directory is **not committed to this
  > repository** — checkpoints are produced by a local training run and live
  > outside version control. The shipped MATLAB entry point
  > (`matlab/motion_matching/shared/fit_swing_surrogate.m`,
  > `local_resolve_checkpoint`) currently defaults to
  > `<repo_root>/output/surrogate/checkpoint_best.pt`; pass
  > `opts.surrogate_checkpoint` (or stage your trained
  > `checkpoint_production.pt` at that path) to consume the production
  > artifact described below. A future PR will align the MATLAB default
  > with this guidance and provide a download/cache helper.
- **Always load `checkpoint_production.pt`, not `checkpoint_best.pt`.** The
  current early-stop criterion is single-objective on grip RMSE and selects
  an unconverged epoch (e.g. epoch 2) before the model has learned to predict
  clubhead speed; `checkpoint_production.pt` is the last-epoch checkpoint and
  is the only one that ships a balanced model. A multi-objective stop
  criterion is on the backlog; until it lands, treat `checkpoint_best.pt` as
  diagnostic only.

### 5.2 Option 3 — inverse pathways

- **Option 3a — full-trial inverse (data-limited, not architecture-limited).**
  Both the cVAE (PR #4240) and the deterministic regressor (PR #4267) plateau
  at ~5 % below the mean-prediction baseline regardless of aggregation
  strategy. This is consistent with the `(31 × 12)`-dim hand-path trajectory
  not uniquely identifying the 189-dim coefficient vector on the random-sweep
  dataset (multiple coefficient vectors produce nearly-identical hand paths).
  A future generator run constrained to physical-swing distributions would
  likely fix this; meanwhile the full-trial inverse is best used as a
  warm-start for Option 1/4 rather than a standalone fitter.
- **Option 3b — per-timestep inverse-dynamics (new production path,
  incoming PR `feat/timestep-inverse-model`).** Reformulates the inverse as
  a per-timestep `(q, q̇, q̈) → τ` regression on the realistic-speed slice
  of the dataset (the user's "each timestep is an independent sample"
  framing). Avoids the global-coefficient under-determination of Option 3a.

## 6. Data assets

### 6.1 Canonical training parquet (10k trials)

The user is migrating a parquet file containing 10,000 dataset_generator
trials from a separate machine. Once landed:

- Lives at `src/engines/Simscape_Multibody_Models/3D_Golf_Model/data/sweep_10k.parquet`
  (or similar — see Issue #DATASET).
- Schema documented in [DATASET_SCHEMA.md](matlab/motion_matching/shared/DATASET_SCHEMA.md).
- Loader: [load_sweep_dataset.py](matlab/motion_matching/shared/load_sweep_dataset.py).
- This is the substrate for Options 2 and 3 (NN surrogate + inverse).
  Option 1 doesn't need it; Option 4 may use it for warm-starts.

### 6.1.1 Random-sweep dataset characterisation

Empirical observations from the existing 10k-trial parquet (relevant for
anyone training Option 2 / Option 3 models on it):

- **Shape.** 10 000 trials × 31 timesteps spanning a 0.30 s impact window at
  100 Hz.
- **Coefficient distribution.** Uniform `±1000` for all seven letters A–G
  (see §4.1). Inputs were random — do not over-interpret marginal coefficient
  histograms as physical priors.
- **Speed saturation.** Every trial saturates at the Simscape integration
  ceiling (clubhead speed peak ≈ 1334 m/s ≈ 2984 mph) because the random
  bounds are aggressive relative to physical swing torques. The single
  global max-speed mode is a generator artefact, not a model property.
- **Realistic-speed slice.** Roughly 29 % of timesteps fall in the realistic
  50–150 mph clubhead-speed range. **Production training pipelines (Option 3b
  per-timestep inverse-dynamics) should filter to this slice**; full-trial
  pipelines (Options 2/3a) currently ingest all timesteps but inherit the
  unphysical-tail caveat.
- **Inverse identifiability.** The full-trial inverse mapping
  `(trajectory → 189 coefficients)` is **under-determined on this dataset** —
  both PR #4240's cVAE and PR #4267's deterministic regressor plateau at
  ~5 % below mean-prediction baseline. This is a data property, not an
  architecture bug; a generator run with physical-swing-distribution
  coefficient priors would likely resolve it.

### 6.2 Canonical test target

- Wiffle ProV1: [Wiffle_ProV1_club_3D_data.xlsx](matlab/src/apps/golf_gui/Motion%20Capture%20Plotter/Wiffle_ProV1_club_3D_data.xlsx).
- Documented event markers (A=240, T=418, I=525, F=725, CHS=114.5 mph at 240 Hz).
- This is the trial every CI smoke test should run.
- Eventually we also want a held-out trial with body-marker mocap; not yet available.

## 7. Milestones (in priority order)

The roadmap below is the same set of items tracked as GitHub issues; this
is the **why**, the issues are the **what** and **how**.

### M1 — Pipeline production readiness (active)

- Stage-1 starting-pose solver (`solve_starting_pose.m`) — the playbook recipe today is on paper only.
- End-to-end Wiffle ProV1 fit hitting grip RMSE < 5 mm in CI, run on every PR that touches `motion_matching/`.
- FK chain calibration (`compute_skeleton_fk` currently has 1.4 m wrist residual; it's a debug helper, not a primary path, but the residual obscures genuine model issues when used as a validator).

### M2 — Performance unblocking

- `MaxStep` loosening experiment (predicted 2–3× extra speedup on top of FastRestart).
- Hot-path warmer for parpool workers so each batch starts FastRestart-ready.
- Validate `accelerator` simulation mode and document the trade-off.

### M3 — NN options online (largely landed)

- ✅ Compact dataset format derived from the 10k parquet
  (`COMPACT_DATASET_SCHEMA.md`), validated by the loader.
- ✅ Option 2 surrogate trained and shipped
  (`data/training/surrogate_v4/checkpoint_production.pt`); MATLAB-side
  `fit_swing_surrogate.m` wrapper still TODO.
- 🟡 Option 3a full-trial inverse (cVAE PR #4240, regressor PR #4267) shipped
  but data-limited; see §5.2.
- 🟢 Option 3b per-timestep inverse-dynamics is the new production inverse
  path (`feat/timestep-inverse-model`); ship `fit_swing_inverse.m` against
  it once the model lands.

### M4 — Option 4 bridge

- Python-side `SimscapeAdapter` already drafted in #4006 — verify in current code.
- Wire scipy.optimize.minimize and (optionally) a JAX gradient path.

### M5 — Robustness / generalisation

- Multiple test trials (currently CI uses one). User-supplied additional swings.
- Cross-option comparison run across all available trials; populate the leaderboard.
- Sensitivity study: how much does total-work `λ` move the answer?

### M6 — Body-marker IK (post-MVP)

- When body mocap data lands, the IK stage in `motion_matching/shared` becomes a real solver, not a stub.
- Spec extension: `target.body_markers`, `target.joint_angles_meas`.
- Cost function gains body-tracking terms.

## 8. Cross-cutting non-goals

To keep scope honest, this project is **not**:

- A real-time golf swing analyser. Fits take minutes, not milliseconds.
- A muscle-level model. We solve for joint torques, not individual muscles.
- A shaft-flex model. The shaft is rigid; clubhead is a deterministic offset of the grip.
- A ball-flight predictor. We stop at impact; ball flight is downstream.
- A coaching tool. Until validated against many golfers and clubs, the answers are research output, not advice.

## 9. Working agreements

For everyone (humans, agents) touching this tree:

- Read this doc + [MATLAB_GOLF_MODEL_GUIDE.md](matlab/MATLAB_GOLF_MODEL_GUIDE.md) before opening a PR.
- Cite the SPEC section or issue number in your commit message.
- New code goes through [CODING_STANDARDS.md](matlab/motion_matching/shared/CODING_STANDARDS.md): TDD, DbC, DRY, LOD ≤ 2.
- Update this SPEC when scope changes — staleness here is the worst kind of staleness.
- Small PRs preferred. The grip-primary refactor (PR #4071) was big because of architecture-level naming changes; that should be the exception, not the rule.

---

_Last updated 2026-05-06._

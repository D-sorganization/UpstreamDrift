# MATLAB Golf Model — Architecture & User Guide

This document is the entry point for working with the Simscape Multibody
golf-swing model, the dataset generator that exercises it at scale, and the
motion-matching pipeline that fits it to measured swing data.

> **Audience:** anyone (human or agent) about to touch the MATLAB side of this
> repo. Pair this with the spec docs under `motion_matching/shared/` for the
> data and cost contracts.

---

## 1. What's actually here

The MATLAB tree under [src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/](.)
contains five layers stacked from bottom to top:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Apps         │ src/apps/golf_gui/   (MATLAB & Python visualisers)       │
├─────────────────────────────────────────────────────────────────────────┤
│ Motion-      │ motion_matching/                                          │
│ matching     │   ├── shared/    cost, loaders, sim wrapper, viz         │
│              │   ├── option1_direct_optimization/   fmincon-sqp         │
│              │   ├── option2_nn_surrogate/          PyTorch surrogate   │
│              │   ├── option3_inverse_nn/            inverse cVAE        │
│              │   └── option4_python_bridge/         MATLAB-to-Python    │
├─────────────────────────────────────────────────────────────────────────┤
│ Dataset      │ src/scripts/dataset_generator/  +                        │
│ generator    │ src/functions/dataset_generator/  (~1856 column extraction)│
├─────────────────────────────────────────────────────────────────────────┤
│ Postproc &   │ src/scripts/post_processing/, src/scripts/plotting/      │
│ analysis     │ src/functions/                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ Simscape     │ src/model/GolfSwing3D_Kinetic.slx       (top-level)      │
│ model        │ src/model/Kinetically_Driven_*_Joint.slx (3 sub-blocks)  │
│              │ src/model/inputs/3DModelInputs_*.mat     (per-pose params)│
│              │ src/model/mdl_reference/  (text-format snapshots)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Simscape model

### 2.1 Topology (top-level `GolfSwing3D_Kinetic.slx`)

The model is a kinetic chain rooted at the hip with the club extending from the
hands. Body chain (each arrow is a Simscape Multibody joint subsystem):

```
World ─[6-DOF Hip]─ Pelvis ─[Spine universal]─ Spine
       Spine ─[Torso revolute]─ Torso ─[L/R Scapula universal]─ L/R Scapula
       Scapula ─[L/R Shoulder gimbal (3-DOF)]─ Upper arm
       Upper arm ─[L/R Elbow revolute]─ Forearm
       Forearm ─[L/R Wrist universal]─ Hand
       Both hands ─ Mid-grip ─ Shaft ─ Clubhead
```

Each rotational joint subsystem is one of three reusable types:

| Subsystem                                | DOFs | Use                   |
| ---------------------------------------- | ---- | --------------------- |
| `Kinetically_Driven_Revolute_Joint.slx`  | 1    | Elbow, Torso          |
| `Kinetically_Driven_Universal_Joint.slx` | 2    | Wrist, Spine, Scapula |
| `Kinetically_Driven_Gimbal_Joint.slx`    | 3    | Shoulder              |

Each subsystem outputs a **`SignalBus`** containing every kinematic and
dynamic signal Simscape will give you for that joint:
`AngularPosition`, `AngularVelocity`, `AngularAcceleration`,
`Rotation_Transform`, `GlobalPosition`, `GlobalVelocity`,
`GlobalAcceleration`, `GlobalAngularVelocity`, `ConstraintForceLocal`,
`ConstraintTorqueLocal`, `ForceLocal`, `TorqueLocal`. All of these flow into
the top-level `CombinedSignalBus`.

Text-format snapshots of the four `.slx` files are in
[src/model/mdl_reference/](src/model/mdl_reference/) so anyone (including LLMs
without MATLAB) can grep block names, parameter expressions, and the Stateflow
torque equations.

### 2.2 Driver: polynomial torques

A Stateflow chart inside the top-level model computes the torque for every
joint as a polynomial of time:

```
τ_j(t; θ) = A_j·t^6 + B_j·t^5 + C_j·t^4 + D_j·t^3 + E_j·t^2 + F_j·t + G_j
```

with seven coefficients per joint. Coefficient bounds (from
`generateRandomCoefficients.m`):
`|A,B| ≤ 1000`, `|C,D| ≤ 500`, `|E,F| ≤ 100`, `|G| ≤ 25`.

For an N-joint model this is `N × 7` decision variables — currently `N=23`
(see [getPolynomialParameterInfo.m](src/functions/dataset_generator/getPolynomialParameterInfo.m)).
The decision vector is called `theta` in motion-matching code.

### 2.3 Damping selector — `LocalDampeningEnable`

A model-workspace parameter (with a Stateflow PARAMETER_DATA mirror) that toggles
between two damping modes:

```
% Joint torque sources include (in Stateflow chart equations):
T_j = ... - DampeningGlobalGain * Damp_j * (1 - LocalDampeningEnable) * ω_j

% Joint subsystem damping coefficients:
Damping_j = Damp_j * LocalDampeningEnable * DampeningGlobalGain
```

So `LocalDampeningEnable=1` puts damping inside the joint blocks (Simscape
default behaviour); `LocalDampeningEnable=0` moves it into the Stateflow
torque computation. Either way the global gain `DampeningGlobalGain` scales it.
**Never overwrite this from the per-trial input MAT** — it lives in the model
workspace and is structural, not a tuning input.

### 2.4 Input MATs (`src/model/inputs/`)

| File                               | Contents                                                                                                                 | Use                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| `3DModelInputs.mat`                | 669 vars: tunable inputs (gains, set-points, polynomial coefficients, damping tuning, etc.) for a generic mid-swing pose | reference / starting point                          |
| `3DModelInputs_Impact.mat`         | 595 vars: same shape, tuned for an impact-pose initial condition                                                         | the "starting position" the user typically asks for |
| `3DModelInputs_TopofBackswing.mat` | 668 vars: top-of-backswing pose                                                                                          | alternate IC                                        |
| `ImpactVelocityOptimization.mat`   | 1 var (a `Simulink.SimulationInput` object)                                                                              | legacy fixture                                      |

These are flat structs with scalar parameters; `setVariable` overlays them onto
a `Simulink.SimulationInput` in our wrappers. They do **not** include
structural model-workspace constants (segment lengths, `LocalDampeningEnable`,
inertia tensors), only tunable inputs.

### 2.5 Persistent logging configuration

Set inside the .slx (verified line 4256 of the MDL):

```
SimscapeLogType    = 'all'    ← home-licence workaround: log every block
SignalLogging      = 'on'
SignalLoggingName  = 'logsout'
SimscapeLogName    = 'simlog'
```

The `SimscapeLogType='all'` setting is what lets us extract ~1856 columns
without manually adding signal markers (which would hit the home-licence
virtual-block / virtual-signal limit). This is the single most important
"don't touch" line in the model configuration.

---

## 3. Data flow at simulation time

When you call `sim(simIn)` you get back a `Simulink.SimulationOutput` with
four populated fields:

| Field               | What's in it                                                                          | Used by                                                                     |
| ------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `tout`              | scalar time vector                                                                    | everyone                                                                    |
| `simlog`            | full Simscape per-block log (`SimscapeLogType='all'`)                                 | dataset generator (`extractSimscapeDataRecursive`)                          |
| `CombinedSignalBus` | structured bus with named groups (LSLogs, LFLogs, ClubLogs, AngularKinematicsLogs, …) | motion-matching loaders, `extract_sim_out`, `load_impact_starting_position` |
| `xout`              | continuous-state log                                                                  | rarely used directly                                                        |

Note: **SaveOutput defaults to `off`** for `.tout` in this model; the dataset
generator and motion-matching tooling explicitly turn it on via
`setModelParameter('SaveOutput','on')`.

### Key signal locations in the bus (cheat sheet)

| What                              | Path                                                                                                                                                                                                                                                                             |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hip angles, vel, accel            | `CombinedSignalBus.AngularKinematicsLogs.HipAngularPosition{X,Y,Z}` etc.                                                                                                                                                                                                         |
| Hip world position                | `…HipPosition{X,Y,Z}`                                                                                                                                                                                                                                                            |
| Spine, torso, scapula angles      | `…{Spine,Torso,LScap,RScap}AngularPosition{X,Y}`                                                                                                                                                                                                                                 |
| Shoulder angles                   | `…{LS,RS}AngularPosition{X,Y,Z}` (gimbal = 3-DOF)                                                                                                                                                                                                                                |
| Elbow angles                      | `…{LE,RE}AngularPosition` (revolute = 1-DOF)                                                                                                                                                                                                                                     |
| Wrist angles                      | `…{LW,RW}AngularPosition{X,Y}`                                                                                                                                                                                                                                                   |
| **Body landmarks (world)**        | `LSLogs.GlobalPosition`, `RSLogs.GlobalPosition`, `LFLogs.GlobalPosition` (= L elbow), `RFLogs.GlobalPosition` (= R elbow), `LWLogs.LHGlobalPosition` (= L hand), `RWLogs.RHGlobalPosition`, `HipLogs.HUBGlobalPosition`, `TorsoLogs.GlobalPosition`, `SpineLogs.GlobalPosition` |
| **Mid-grip world position**       | `MidpointCalcsLogs.MPGlobalPosition`                                                                                                                                                                                                                                             |
| **Mid-grip orientation**          | `MomentandCoupleLogs.RotationTransformMP`                                                                                                                                                                                                                                        |
| **Clubhead world position**       | `ClubLogs.CHGlobalPosition`                                                                                                                                                                                                                                                      |
| Clubhead velocity                 | `ClubLogs.CHGlobalVelocity`                                                                                                                                                                                                                                                      |
| Hand forces / torques on club     | `LWLogs.LHonClubFGlobal`, `RWLogs.RHonClubFGlobal`, etc.                                                                                                                                                                                                                         |
| Total hand force / torque on club | `CalculatedSignalsLogs.TotalHandForceGlobal`, `…TotalHandTorqueGlobal`                                                                                                                                                                                                           |
| Equivalent midpoint couple        | `MomentandCoupleLogs.EquivalentMidpointCoupleGlobal`                                                                                                                                                                                                                             |
| Butt of club world position       | `LHCalcsLogs.ButtPosition`                                                                                                                                                                                                                                                       |

### What the dataset generator extracts (~1856 columns)

[runSingleTrial.m](src/functions/dataset_generator/runSingleTrial.m) →
[processSimulationOutput.m](src/functions/dataset_generator/processSimulationOutput.m)
→ [extractSignalsFromSimOut.m](src/functions/dataset_generator/extractSignalsFromSimOut.m)
calls three readers in order:

1. [extractFromCombinedSignalBus.m](src/functions/dataset_generator/extractFromCombinedSignalBus.m) — recursive walker that flattens `CombinedSignalBus` to `<group>_<signal>_{x,y,z}` columns.
2. [extractLogsoutDataFixed.m](src/functions/dataset_generator/extractLogsoutDataFixed.m) — the few signals that are explicitly piped into `logsout` (sparse).
3. [extractSimscapeDataRecursive.m](src/functions/dataset_generator/extractSimscapeDataRecursive.m) calling [traverseSimlogNode.m](src/functions/dataset_generator/traverseSimlogNode.m) — walks the per-block simlog and emits `<path>_<leaf>` columns (`q`, `w`, `t`, `b`, `f`, etc.).

Then [combineDataSources.m](src/functions/dataset_generator/combineDataSources.m)
outerjoins them on `Time` and [addModelWorkspaceData.m](src/functions/dataset_generator/addModelWorkspaceData.m)
appends model-workspace constants (segment lengths, masses, inertias) as
repeated columns. The result is one CSV per trial; `compileDataset` then
merges all trials into `master_dataset.csv`.

---

## 4. Performance — what's worth optimising

Empirical measurement on this codebase (R2025b, Home licence, 0.30 s sim
window, default Impact MAT inputs; full data in
[motion_matching/shared/scripts/probe_perf.m](motion_matching/shared/scripts/probe_perf.m)):

| Configuration                                    | Mean wall-clock | vs. baseline |
| ------------------------------------------------ | --------------- | ------------ |
| Cold sim, model defaults                         | **14.9 s**      | 1.00×        |
| `SimscapeLogType='all'`, `SignalLogging='off'`   | 14.8 s          | 0.99×        |
| `SimscapeLogType='local'`, `SignalLogging='on'`  | 14.9 s          | 1.00×        |
| `SimscapeLogType='local'`, `SignalLogging='off'` | 14.9 s          | 1.00×        |
| `SimscapeLogType='none'`, `SignalLogging='off'`  | 14.8 s          | 0.99×        |
| **Warm sim, FastRestart=on**                     | **6.9 s**       | **0.46×**    |

**Conclusion: the solver dominates wall-clock; logging is essentially free.**
Don't bother stripping the `CombinedSignalBus` or `simlog` plumbing.

What actually moves the needle:

1. **`FastRestart='on'`** — saves ~50% per call after the first. The single
   most important knob for any optimization workflow that runs many sims
   with the same model topology and only changes inputs/coefficients.
   Use [`prepare_fast_sim_input.m`](motion_matching/shared/prepare_fast_sim_input.m) which sets it for you.
2. **Solver tolerances** — the model uses `ode23t` with `RelTol=1e-3`, `AbsTol=1e-5`. Loosening to `1e-2`/`1e-4` may save another 20-30% at some accuracy cost. Not yet measured on this codebase.
3. **Shorter sim windows** — the cost-function uses 0.3 s by default. Don't extend it unless you need to.
4. **Parallel pool** (`parsim`) — for population methods (multistart, NN training set generation). Already wired in `runParallelSimulations`.
5. **Sample-rate downscaling** — `default_align_options.sample_rate=1000`. The motion-matching cost only needs the simulation frequency to match the resampled target; you can drop both to 240 Hz (the source's native rate) for a small win.

What does **NOT** help:

- Disabling the bus or simlog (negligible save; loader can't function without bus output anyway).
- Manually trimming the number of CSB groups (the bus is built once per sim regardless of how many signals you read).

### Why the solver is so expensive

The model is set to `Solver=ode23t`, `RelTol=1e-3`, `AbsTol=1e-5`, **`MaxStep=0.001`**, with `SaveOutput=on`. That `MaxStep` is the biggest single cost driver — at 1 ms max-step over a 0.30 s window the solver is doing at least 300 steps per simulation regardless of how stiff the dynamics actually are. Loosening to e.g. `MaxStep=0.005` should give another 2–3× warm-restart speedup at the cost of a small accuracy hit; it has not been validated yet so we don't change it by default. Try it on your branch when you want to fit a lot of swings:

```matlab
in = prepare_fast_sim_input(theta);
in = in.setModelParameter('MaxStep', '0.005');     % 5× looser; verify accuracy
sim_out = sim(in);
```

Other deferred experiments worth running:

- `SignalLoggingSaveFormat='ModelDataLogs'` (the alternative format) sometimes serialises faster than `Dataset`.
- Setting `SimulationMode='accelerator'` builds a C-MEX accelerator on first call; can give 2–10× speedup on stiff continuous models. Cost is a longer first-run compile.

---

## 5. The four motion-matching options

All four consume the same `target` schema (CLUB*IK_SPEC.md) and should produce
the same `result` shape (CODING_STANDARDS.md). They differ in \_how* they get
from a target to a coefficient vector.

| Option                                                                                              | Status                                                       | Approach                                                                                                                                | When to reach for it                                             |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **1 — Direct fmincon** ([option1_direct_optimization](motion_matching/option1_direct_optimization)) | ✅ Implemented (`fit_swing_fmincon.m`, `fit_swing_hybrid.m`) | SQP on `compute_cost`; ~10 wall-minutes per fit; deterministic                                                                          | Default. Use first to get a baseline; needs no offline training. |
| **2 — Forward NN surrogate** ([option2_nn_surrogate](motion_matching/option2_nn_surrogate))         | 🟡 Scaffolded (`surrogate.py`)                               | PyTorch model trained on dataset_generator output; cheap forward eval; gradient-based fit on the surrogate then refined on the true sim | Use when you have ≥5k labelled trials and want sub-second fits   |
| **3 — Inverse NN (cVAE)** ([option3_inverse_nn](motion_matching/option3_inverse_nn))                | 🟡 Scaffolded (`inverse_cvae.py`)                            | Train a conditional VAE that emits `theta` from a target; one-shot inference                                                            | Use when you need to fit many swings interactively               |
| **4 — MATLAB↔Python bridge** ([option4_python_bridge](motion_matching/option4_python_bridge))      | 🟡 Spec-only                                                 | Forward sim still runs in MATLAB; optimizer runs in Python (scipy/JAX)                                                                  | Use when you want Python-native tooling around the optimizer     |

For all four the **target** is now grip-primary (see the new
[CLUB_IK_SPEC.md](motion_matching/shared/CLUB_IK_SPEC.md) — `target.grip` and
`target.grip_quat` are the canonical fields; `target.butt` is a backward-compat
alias). The cost function in
[compute_cost.m](motion_matching/shared/compute_cost.m) already does the right
thing automatically.

---

## 6. How to actually run things

### 6.1 Set up the environment (once per session)

```matlab
cd 'C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab'
setup_matlab_environment        % adds src/ subtree + motion_matching/shared/
```

### 6.2 Visualise the starting-position match

Loads the Impact pose, loads the measured Wiffle ProV1 swing, aligns
grip-pose, renders a four-panel comparison + 90-frame animation:

```matlab
addpath(genpath('motion_matching/shared'))
res = demo_starting_position_match;             % grip_pose alignment + animation
% Artefacts in matlab/output/starting_position_match/<timestamp>/
```

### 6.3 Generate a dataset (for NN training)

```matlab
addpath(genpath('src/scripts/dataset_generator'))
addpath(genpath('src/functions/dataset_generator'))
config = createSimulationConfig('num_simulations', 1000, ...
                                'output_folder', 'output/dataset_run_001', ...
                                'execution_mode', 'parallel', ...
                                'num_workers', feature('numcores'));
[trials, dataset_path, meta] = runSimulation(config);
% Produces 1000 trial_*.csv files + master_dataset.csv (~1856 columns)
```

### 6.4 Fit one swing with fmincon (Option 1)

```matlab
addpath(genpath('motion_matching/shared'))
addpath(genpath('motion_matching/option1_direct_optimization'))

target = load_club_target_excel( ...
    "src/apps/golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx", ...
    "TW_ProV1");
opts = default_option1_options();
opts.cost = default_cost_options();   % grip-primary by default
result = fit_swing_fmincon(target, opts);
plot_trajectory_overlay(result, target);
```

### 6.5 Probe the perf yourself

```matlab
addpath(genpath('motion_matching/shared'))
probe_perf;                              % prints the timing table for your machine
```

---

## 7. Two-stage fit recipe (recommended starting workflow)

See [GRIP_FIT_PLAYBOOK.md](motion_matching/shared/GRIP_FIT_PLAYBOOK.md) for the
full recipe. In short:

> **Stage 1 — initial pose**: anchor `Hub` at the centre of the measured
> swing arc; solve a small inverse-kinematics problem so the hands lie at the
> measured `target.grip(impact_idx,:)` (or wherever you pick as anchor).
> This is a few-DOF problem over starting-position parameters in the input MAT.
>
> **Stage 2 — torque coefficients**: with the starting pose fixed, run
> `fit_swing_fmincon` (Option 1) on the grip-primary cost. The
> total-work regularizer is already wired in (`opts.cost.regularizer = "total_work"`,
> `opts.cost.lambda = 1e-4`).

---

## 8. Known issues / caveats

| Issue                                                                                      | Where                                  | Impact                                             | Mitigation                                                                                                        |
| ------------------------------------------------------------------------------------------ | -------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Wiffle xlsx Definitions tab claims "inches" but data is **centimetres**                    | `Wiffle_ProV1_club_3D_data.xlsx`       | Misreading would 2.54×-inflate everything          | Loader applies `0.01` (cm→m); regression test guards plausibility                                                 |
| Speed-argmax impact detection latches on wrong frame                                       | `private/detect_clubhead_impact.m`     | Was returning idx 254 instead of doc'd 525         | Loader now reads documented `I_sample` from row-1 header and passes it as `known_impact_s`                        |
| Measured shaft length differs from modeled club length                                     | xlsx vs `LowerArmLength`/`ShaftLength` | ~7 cm gap on TW_ProV1                              | `grip_pose` alignment matches grip+shaft direction; clubhead residual is the legitimate club-length difference    |
| `ScaleFactor` of CSB output struct is ~40 MB                                               | model intrinsic                        | Memory pressure for large datasets                 | Don't aggregate sim_out across many trials in memory; persist per-trial CSV (dataset generator already does this) |
| LE/RE/LS/RS world positions came online only after we noticed `LFLogs.GlobalPosition` etc. | model                                  | Earlier code computed FK from joint angles instead | Fixed: extractor now reads positions directly; FK helper kept as a fallback validator                             |

---

## 9. Model structural review (2026-05-06)

A direct inspection of `GolfSwing3D_Kinetic.slx` and the three joint sub-models
turned up the following — none are bugs, but several are worth knowing about:

| Item                                                                    | Status             | Note                                                                                                                                                                                                              |
| ----------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SimscapeLogType='all'` persistent                                      | ✅ saved           | Line 4256 of MDL; live `get_param` confirmed                                                                                                                                                                      |
| `SignalLogging='on'` persistent                                         | ✅ saved           | `logsout` is rarely populated; CSB does the heavy lifting                                                                                                                                                         |
| `SaveOutput='off'` persistent                                           | ⚠ check           | Motion-matching turns it on per call; OK                                                                                                                                                                          |
| `StopTime='0.1'` persistent                                             | ⚠ confusing       | Cost-function default is 0.30s; the persisted value is the model's editor default, not a fitting target                                                                                                           |
| `MaxStep='0.001'` persistent                                            | 🔴 perf            | This forces ≥300 solver steps per 0.3 s sim; the largest single perf lever (see §4)                                                                                                                               |
| 27× `*StartPosition*` + 27× `*StartVelocity*` workspace vars            | ✅ as expected     | Drives the Stage-1 starting-pose solver                                                                                                                                                                           |
| 16× Transform Sensor blocks                                             | ✅ as expected     | These are why `LSLogs.GlobalPosition`, `LFLogs.GlobalPosition`, etc. land in the CSB without us adding any virtual signals                                                                                        |
| 15× top-level BusCreator blocks                                         | ✅ as expected     | Build the `CombinedSignalBus` from the 16 Transform Sensors + every joint subsystem's `SignalBus`                                                                                                                 |
| 98 referenced subsystems                                                | ✅ clean           | Each joint is one of three reusable sub-blocks; good decomposition                                                                                                                                                |
| `LocalDampeningEnable` parameter                                        | ✅ wired correctly | Stateflow PARAMETER_DATA + model-workspace Parameter; gates between local-vs-global damping. **Never overwrite from per-trial inputs.**                                                                           |
| Two `Mechanical_Rotational_Reference` blocks per Universal/Gimbal joint | ⚠ redundant?      | Each axis has its own reference — appears intentional but worth verifying that they aren't both contributing damping inadvertently when `LocalDampeningEnable=1`                                                  |
| Some "duplicate signal" warnings in older diagnostic scripts            | 🟡 cosmetic        | The scripts in `apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/` reference identifying duplicate logged signals; not a model defect, just a side-effect of the bus topology |

The main thing to watch out for: **don't try to "clean up" the
CombinedSignalBus by removing groups you don't think you need.** The bus is
built once per sim regardless of how much downstream code reads from it, and
the dataset generator's ~1856 columns relies on the full surface being
present.

---

## 10. Spec / contract docs

- [CLUB_IK_SPEC.md](motion_matching/shared/CLUB_IK_SPEC.md) — target schema (grip-primary)
- [COST_FUNCTION_SPEC.md](motion_matching/shared/COST_FUNCTION_SPEC.md) — cost terms and weights
- [CODING_STANDARDS.md](motion_matching/shared/CODING_STANDARDS.md) — DRY/DbC/LOD enforcement
- [DATASET_SCHEMA.md](motion_matching/shared/DATASET_SCHEMA.md) — dataset-generator output schema
- [VISUALIZATION_SPEC.md](motion_matching/shared/VISUALIZATION_SPEC.md) — required figures
- [src/model/mdl_reference/README.md](src/model/mdl_reference/README.md) — agent-readable text snapshots of the .slx files

Each `option*` directory has its own `README.md`, `APPROACH.md`, `RUNBOOK.md`,
and `TESTING.md`.

---

## 12. Glossary

| Term              | Meaning in this codebase                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| `theta`           | flat polynomial coefficient vector, `n_joints × 7`, ordered `[A B C D E F G]` per joint                 |
| `target`          | canonical struct containing the measured swing (CLUB_IK_SPEC)                                           |
| `result`          | canonical struct returned by an optimizer (CODING_STANDARDS)                                            |
| `sim_out`         | canonical struct returned by `simulate_with_coefficients`                                               |
| `grip`            | mid-hands position on the shaft (rigid body→club interface; primary motion-matching anchor)             |
| `mp`              | mid-grip in model code (= grip; legacy short name)                                                      |
| `butt`            | end of grip (backward-compat alias of `grip` in measured-data structs; distinct point in model structs) |
| `clubhead` / `ch` | striking face of the club                                                                               |
| `hub`             | top of spine / base of neck                                                                             |
| `CSB`             | `CombinedSignalBus` (top-level structured output bus from the model)                                    |
| `simlog`          | full Simscape per-block log (enabled by `SimscapeLogType='all'`)                                        |

---

_Last updated 2026-05-06 alongside the grip-primary motion-matching refactor (PR #4071)._

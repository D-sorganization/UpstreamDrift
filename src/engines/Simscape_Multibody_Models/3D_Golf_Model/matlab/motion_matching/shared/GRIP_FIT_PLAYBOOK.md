# Grip-Fit Playbook

How to drive the model so the **mid-hands grip** tracks a measured club path,
while minimising total mechanical work — the canonical motion-matching workflow.

This is the practical companion to [CLUB_IK_SPEC.md](CLUB_IK_SPEC.md) and
[COST_FUNCTION_SPEC.md](COST_FUNCTION_SPEC.md). Read those first if you want
the formal definitions; this doc is the recipe.

---

## What we are matching, and why

|                             | What                                                                | Why                                                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Primary**                 | `target.grip` (mid-hands position on the shaft, world-frame metres) | Rigid body→club interface — independent of player club length and shaft flex                                                                                 |
| **Primary**                 | `target.grip_quat` (mid-hands orientation, [w x y z])               | Sets the hand's rotational pose; clubhead position is derivable from grip pose + modeled shaft geometry                                                      |
| **Secondary** (default off) | `target.clubhead`, `target.club_quat`                               | Subject to club-length differences and shaft flex — penalising them in the cost forces the optimizer to chase noise. Available if you want soft supervision. |

The cost function defaults
([default_cost_options.m](default_cost_options.m)) are already grip-primary:

```
w_position_grip      = 1.0     ← grip position term (m^-2), primary
w_orientation_grip   = 0.5     ← grip orientation term (rad^-2), primary
w_position_clubhead  = 0.0     ← off; raise if you want soft clubhead supervision
w_orientation_club   = 0.0     ← off
w_anchor_impact      = 10.0    ← grip-position penalty at the impact frame
lambda               = 1e-4    ← total-work regularizer strength
```

---

## The two-stage workflow

```
┌────────────────────────────────────────────────────────────────────┐
│ Stage 1 (warm start, seconds)                                       │
│   - Pick a starting-pose input MAT (3DModelInputs_Impact.mat as base)│
│   - Solve for the small set of *_StartPosition* / *_StartVelocity*  │
│     overrides that put the body where the swing data says it should │
│     be at the swing arc centre / address frame                      │
│   - Output: an `input_overrides` struct                              │
├────────────────────────────────────────────────────────────────────┤
│ Stage 2 (the fit, ~10 minutes per swing)                            │
│   - With the starting pose fixed, solve for theta (polynomial       │
│     torque coefficients) that minimise:                             │
│       w_pg·||grip_sim - grip_meas||²                                 │
│     + w_og·d_geo(grip_quat_sim, grip_quat_meas)²                     │
│     + w_a ·||grip_sim(impact) - grip_meas(impact)||²                 │
│     + λ·W_total(θ)                                                   │
│   - Use fit_swing_fmincon (Option 1) as the workhorse                │
│   - Use prepare_fast_sim_input to enable FastRestart (~2× speedup)  │
└────────────────────────────────────────────────────────────────────┘
```

### Why two stages?

If you let `fit_swing_fmincon` solve theta with a wildly wrong starting pose,
the body has to do a lot of pre-impact "windup" work to even reach the
measured grip path. The solver wastes iterations and the total-work
regularizer fights with the position term. Decoupling fixes that:

1. Stage 1 puts the body at a reasonable address pose.
2. Stage 2 is then about timing and amplitude only — the regime where the
   polynomial-torque formulation is well-conditioned.

---

## Stage 1 — initial pose

### What you control (the inputs)

The input MATs (e.g. `3DModelInputs_Impact.mat`) contain ~595 variables.
The ones that move the _initial pose_ are the `*StartPosition*` and
`*StartVelocity*` family for the floating root (Hip translation+rotation) plus
each rotational joint:

| Family          | Joint        | Variables                                                        |
| --------------- | ------------ | ---------------------------------------------------------------- |
| Hip translation | World        | `TranslationStartPositionX/Y/Z`, `TranslationStartVelocityX/Y/Z` |
| Hip rotation    | Hip          | `HipStartPositionX/Y/Z`, `HipStartVelocityX/Y/Z`                 |
| Spine           | Spine        | `SpineStartPositionX/Y`, `SpineStartVelocityX/Y`                 |
| Torso           | Torso        | `TorsoStartPosition`, `TorsoStartVelocity`                       |
| Scapulae        | LScap, RScap | `LScapStartPositionX/Y`, `RScapStartPositionX/Y` (+ velocities)  |
| Shoulders       | LS, RS       | `LSStartPositionX/Y/Z`, `RSStartPositionX/Y/Z` (+ velocities)    |
| Elbows          | LE, RE       | `LEStartPosition`, `REStartPosition` (+ velocities)              |
| Forearms        | LF, RF       | `LFStartPosition`, `RFStartPosition` (+ velocities)              |
| Wrists          | LW, RW       | `LWStartPositionX/Y`, `RWStartPositionX/Y` (+ velocities)        |

(Search your loaded MAT for `*StartPosition*` to enumerate exactly what's
present; the names follow the joint subsystem names listed in the
[architecture guide §2.1](../../MATLAB_GOLF_MODEL_GUIDE.md).)

### How to set it analytically

Fastest path: don't run an optimizer. Use forward kinematics directly.

1. **Pick the address frame** of the measured swing — usually
   `target.events.A_sample` from the row-1 header (sample 240 in the Wiffle
   ProV1 sheet at 240 Hz).
2. **Set the hub at swing centre.** From the measured clubhead trace,
   compute the centroid of the swing arc; that's roughly where the hub
   (top of spine) wants to be. Then translate the world frame so the hub
   lands at, say, `(0, 0, 1.5)` m (chest height of a standing golfer).
3. **Place the hands.** From the measured grip position at the address
   frame, you know where the hands need to be relative to the hub.
   Fix `LSStartPosition*` and `RSStartPosition*` to canonical address-pose
   shoulder rotations, then let the elbows + wrists hang so the grip is
   at the measured location (use [compute_skeleton_fk.m](compute_skeleton_fk.m)
   to verify).

### How to set it numerically

If the analytic approach gives a residual >5 cm at the grip:

```matlab
% Pseudocode — wrap the existing tooling
function overrides = solve_starting_pose(skel0, target, base_input_mat)
    % skel0 is the model's t=0 skeleton (load_impact_starting_position)
    % target is the measured swing
    % base_input_mat is the path to the Impact MAT we layer onto

    % Decision variables: a small set of starting-pose perturbations
    %   (e.g. the 6 hip-translation + hip-rotation start positions, plus
    %   the four shoulder + elbow scalar starts).
    vars = {'TranslationStartPositionX','TranslationStartPositionY', ...
            'TranslationStartPositionZ','HipStartPositionZ', ...
            'LSStartPositionY','RSStartPositionY', ...
            'LEStartPosition','REStartPosition'};
    x0 = zeros(numel(vars), 1);

    cost_fn = @(x) local_pose_cost(x, vars, target, base_input_mat);
    x_opt = fminsearch(cost_fn, x0);

    overrides = local_vec_to_struct(x_opt, vars);
end
```

Inside `local_pose_cost` you call
[`prepare_fast_sim_input`](prepare_fast_sim_input.m) with `stop_time=0.005`
(just enough to extract t=0), pull the model's grip from
`CombinedSignalBus.MidpointCalcsLogs.MPGlobalPosition`, and return
`||model.grip(0) - target.grip(impact_idx_or_address)||²`. The whole
inner loop is fast because the sim is 5 ms.

This is on the roadmap (see §"What's next") but not yet wired in this PR.

---

## Stage 2 — solve for torques

This is what [`fit_swing_fmincon`](../option1_direct_optimization/fit_swing_fmincon.m) already does.
With the new grip-primary cost it just works:

```matlab
addpath(genpath('motion_matching/shared'))
addpath(genpath('motion_matching/option1_direct_optimization'))

target = load_club_target_excel( ...
    "src/apps/golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx", ...
    "TW_ProV1");

opts = default_option1_options();
opts.cost                       = default_cost_options();
opts.cost.w_position_grip       = 1.0;       % primary
opts.cost.w_position_clubhead   = 0.0;       % default — keep off
opts.cost.w_orientation_grip    = 0.5;       % primary
opts.cost.w_orientation_club    = 0.0;       % default — keep off
opts.cost.lambda                = 1e-4;      % total-work regularizer
opts.cost.regularizer           = "total_work";

% Apply Stage-1 starting-pose overrides if you computed them:
% opts.sim.input_overrides = overrides;

result = fit_swing_fmincon(target, opts);
fprintf('grip RMSE: %.2f mm\n', 1000 * result.final_rmse_m);
plot_trajectory_overlay(result, target);
plot_error_timecourse(result, target);
```

### Why total-work regularization

The polynomial-torque parameterisation is rich (`23 × 7 = 161` decision
variables). Without a regularizer, infinitely many `theta` vectors track the
same grip path equally well, but most of them produce wildly oscillating
joint torques. Penalising total mechanical work
$$W_\text{total}(\theta) = \int_0^T \sum_j |\tau_j \omega_j| dt$$
selects the smoothest, lowest-effort torque profile that still hits the
target — the closest analogue to "what a human would actually do".

Other options if total-work doesn't behave: `peak_power`, `torque_l2`,
`coeff_l2`. See [COST_FUNCTION_SPEC.md §Regularizer](COST_FUNCTION_SPEC.md).

### Speed up the fit with FastRestart

```matlab
% Inside fit_swing_fmincon (already wired) or for your own loop:
in = prepare_fast_sim_input(theta, struct( ...
    'model_name', 'GolfSwing3D_Kinetic', ...
    'stop_time',  0.30, ...
    'simscape_log', 'all', ...
    'input_overrides', overrides));   % ← Stage-1 overrides go here
sim_out = sim(in);
```

Empirical: ~7 s/sim warm vs. ~15 s cold. For a 200-iteration fmincon run
that's 23 minutes vs. 50 minutes — not life-changing but noticeable.

---

## Picking the right option (1 / 2 / 3 / 4)

| Situation                                                                    | Pick                                                                              |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| First time fitting a swing, want a baseline you can trust                    | **Option 1** (fmincon)                                                            |
| Need to fit ≥10 swings interactively, can spare 8+ hours of training upfront | **Option 2** (NN surrogate) — train on 5k+ trials, then sub-second forward passes |
| Need real-time inverse "given a club path, give me theta"                    | **Option 3** (inverse cVAE) — same training cost, then one forward pass per swing |
| Want Python-side optimization (JAX, scipy.optimize)                          | **Option 4** (bridge) — solver in Python, sim still in MATLAB                     |

For now Option 1 is the only fully working path. Options 2/3 have scaffolding
but no trained models in the tree; Option 4 is spec-only.

---

## Validation

After a fit, sanity-check:

1. **Final grip RMSE < 5 mm** (`result.final_rmse_m`) — primary success metric.
2. **Final clubhead RMSE** — should equal model-vs-measured shaft-length difference (≈27 mm for TW_ProV1). If it's much larger, your starting pose is off; if much smaller, you've over-fit on a weighted clubhead term.
3. **Total work < 500 J** — typical full swings deliver ~280 J. Outliers indicate the regularizer needs tuning.
4. **Visual** — `plot_trajectory_overlay(result, target)` should show the simulated and measured grip paths overlapping; the model and measured clubheads should be parallel but offset by the shaft-length difference.

---

## What's next (not in this PR)

- [ ] Implement `solve_starting_pose.m` (the Stage-1 numerical solver sketched above).
- [ ] Wire `opts.sim.input_overrides` through `simulate_with_coefficients` so Stage-1 results plumb through to Stage 2 cleanly.
- [ ] Train an Option-2 surrogate on the dataset_generator output and add a `fit_swing_surrogate.m` analogous to `fit_swing_fmincon.m`.
- [ ] Multi-start Option 1 with the n best basins kept; hybrid finishing pass to polish.
- [ ] Try looser solver tolerances (`RelTol=1e-2`, `AbsTol=1e-4`) for Stage-2 inner sims; refine final answer with tight tolerances.

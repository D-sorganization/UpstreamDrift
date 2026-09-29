# Cost Function Specification

The cost function is the heart of the inverse problem. All four options consume the **same** cost function spec; that is what lets us compare results fairly.

## Symbols

| Symbol        | Meaning                                                              |
| ------------- | -------------------------------------------------------------------- |
| `θ`           | coefficient vector, `n_joints × 7` flattened, decision variables     |
| `q_sim(t; θ)` | simulated kinematic trajectory at time t given coefficients θ        |
| `q_meas(t)`   | measured kinematic trajectory (the target)                           |
| `T`           | simulation duration (~0.3 s); we evaluate at the simulation timestep |
| `N`           | number of timesteps over T                                           |

For Phase 1 (club-only), `q` is a 12-vector per timestep:

- `r_grip ∈ ℝ³` (mid-hands / grip position, metres) — **PRIMARY** matching anchor
- `R_grip ∈ SO(3)` (mid-hands / grip orientation) — **PRIMARY** matching anchor
- `r_clubhead ∈ ℝ³` (clubhead position, metres) — **SECONDARY** (optional)
- `R_club ∈ SO(3)` (club orientation) — **SECONDARY** (optional)

The grip is the rigid body→club interface; the clubhead is a non-rigid extension because of (a) shaft flex during the swing and (b) the player's actual club length almost never matches the modeled club length to the millimetre. We therefore weight the **grip terms primary** and treat clubhead+club-orientation as low-weight (default 0) secondary signals that callers can opt into when they want soft clubhead supervision.

## Primary cost (club-only)

$$
J(\theta) = w_{pg}\,\frac{1}{N}\sum_{n} \|r^\text{grip}_\text{sim}(t_n) - r^\text{grip}_\text{meas}(t_n)\|^2
       \;+\; w_{pc}\,\frac{1}{N}\sum_{n} \|r^\text{ch}_\text{sim}(t_n) - r^\text{ch}_\text{meas}(t_n)\|^2
       \;+\; w_{og}\,\frac{1}{N}\sum_{n} d_\text{geo}(R^\text{grip}_\text{sim}, R^\text{grip}_\text{meas})^2
       \;+\; w_{oc}\,\frac{1}{N}\sum_{n} d_\text{geo}(R^\text{club}_\text{sim}, R^\text{club}_\text{meas})^2
       \;+\; \lambda\,W_\text{total}(\theta)
$$

Where:

- `d_geo(R1, R2) = ‖log(R1ᵀ R2)‖_F / √2` — the geodesic angle between two rotations (radians). Implementation: convert to quaternions and use `2·acos(|q1·q2|)`.
- `W_total(θ)` — total mechanical work, defined below.
- `w_pg`, `w_pc`, `w_og`, `w_oc`, `λ` — weights, defaults in [DEFAULTS](#defaults). By default `w_pc = w_oc = 0` so the cost ignores the clubhead entirely.

### Endpoint-anchor variant

When the optimizer is starting cold, the position term can dominate the orientation term and produce drift. Add an **anchor** term that is large at impact (the most kinematically constrained instant), pinned to the **grip** (the body→club rigid contact, which the body has to deliver in the right place at the right time):

$$
J_\text{anchor}(\theta) = J(\theta) + w_a\,\big\|r_\text{grip,sim}(t_\text{impact}) - r_\text{grip,meas}(t_\text{impact})\big\|^2
$$

`w_a` defaults to `10·w_pg`. `t_impact` is taken from the documented event-marker `I_sample` in the source xlsx when present, falling back to the speed-argmax heuristic on the resampled trace.

### Backward compatibility

Older callers that set only `opts.w_position` and `opts.w_orientation` get the legacy butt+clubhead behaviour (both summed into the position term, club orientation only). New callers should set the explicit `w_position_grip` / `w_position_clubhead` / `w_orientation_grip` / `w_orientation_club` weights.

## Regularizer: minimum total mechanical work

This is what makes the under-determined club-only fit produce a unique answer.

$$
W_\text{total}(\theta) = \int_0^T \sum_{j=1}^{n_\text{joints}} \big|\tau_j(t; \theta) \cdot \omega_j(t; \theta)\big|\, dt
$$

Where `τ_j(t; θ)` is the polynomial torque at joint `j` (from the coefficients), and `ω_j(t; θ)` is the joint angular velocity emitted by Simscape. `|·|` is absolute value (we count both positive and negative work — eccentric and concentric — equally).

In MATLAB:

```matlab
function W = compute_total_work(sim_out)
    arguments
        sim_out (1,1) struct {mustHaveFields(sim_out, ["time","tau","omega"])}
    end
    integrand = sum(abs(sim_out.tau .* sim_out.omega), 2);  % Nx1
    W = trapz(sim_out.time, integrand);
    assert(W >= 0, "Postcondition: total work must be non-negative");
end
```

### Variants

| Regularizer            | Formula                     | When useful                                                          |
| ---------------------- | --------------------------- | -------------------------------------------------------------------- |
| `total_work` (default) | `∫ Σ\|τω\| dt`              | Bias toward physiologically efficient swings                         |
| `peak_power`           | `max_t Σ\|τω\|`             | Bias against torque spikes                                           |
| `torque_l2`            | `∫ Σ τ² dt`                 | Bias against large torques regardless of motion                      |
| `coeff_l2`             | `‖θ‖²`                      | Trivial bias toward the model's nominal                              |
| `effort_l2` (new)      | `mean Σ (τ − τ_ref)² · w_j` | Penalise deviation from a reference torque profile (PR #3966 parity) |
| `smoothness_l2` (new)  | `mean Σ (Δτ)² · w_j`        | Penalise jerky control inputs (PR #3966 parity)                      |

The default is **total work**. Other variants are exposed via `options.regularizer = "total_work" | "peak_power" | "torque_l2" | "coeff_l2" | "effort_l2" | "smoothness_l2"`.

The `effort_l2` and `smoothness_l2` variants additionally consult two new options fields:

- `opts.tau_reference` (default `[]` / `None`): reference torque profile shaped like `sim_out.tau`; falls back to a zero reference when empty. With a zero reference, `effort_l2` reduces to `mean(τ²)`.
- `opts.regularizer_weights` (default `[]` / `None`): per-joint weight vector of length `n_joints`; defaults to ones. Lets callers down-weight wrist torques relative to shoulder torques, etc.

These match the discrete cost used by `MachineLearning/optimize_torque_sequence_for_club.py` (PR #3966), where the cost contains `α · MSE(u − u₀) + β · MSE(Δu)`. The scaffold absorbs `α` and `β` into `lambda` — pick one regularizer per run.

## Defaults

```matlab
function opts = default_cost_options()
    opts = struct();
    % --- New grip-primary weights (recommended) -----------------------------
    opts.w_position_grip      = 1.0;        % grip position weight (m^-2)
    opts.w_position_clubhead  = 0.0;        % clubhead position weight (default 0; raise for soft supervision)
    opts.w_orientation_grip   = 0.5;        % grip orientation weight (rad^-2)
    opts.w_orientation_club   = 0.0;        % club orientation weight (default 0)
    opts.w_anchor_impact      = 10.0;       % impact-anchor multiplier on grip position term
    % --- Common / shared -----------------------------------------------------
    opts.regularizer        = "total_work";
    opts.lambda             = 1e-4;
    opts.q_orientation_repr = "quaternion"; % "quaternion" | "rotmat"
    opts.time_alignment     = "impact";     % "impact" | "address" | "none"
    opts.resample_to_hz     = 1000;
    % --- Backward-compat aliases (older callers) -----------------------------
    opts.w_position    = 1.0;        % legacy: w_position_grip + w_position_clubhead
    opts.w_orientation = 0.1;        % legacy: w_orientation_club
end
```

These defaults are **not** sacred — they are starting points. Every optimization run records the full `opts` in its result struct so a fit can be reproduced or re-tuned.

## Numerical considerations

- **Units consistency.** Position in metres, time in seconds, orientation as unit quaternion. The dataset loader is responsible for converting from inches/Excel-time/rotation-matrices.
- **Time alignment.** The measured swing and the simulation must share the same timegrid. Default behaviour: align the measured swing's max-clubhead-speed instant to the simulation's expected impact time, then resample both to the simulation's `sample_rate`. See [CLUB_IK_SPEC.md](CLUB_IK_SPEC.md).
- **Quaternion sign ambiguity.** Quaternions q and −q represent the same rotation. When computing `d_geo` use `2·acos(|q1·q2|)` (note the absolute value).
- **Numerical noise floor.** With 1 kHz sampling over 0.3 s and metre-scale measurements, expect optimizer floor near `final_rmse_m ≈ 1e-3` (1 mm) before mocap noise dominates. Don't chase below that without filtering.

## What the function signature looks like

The reference MATLAB implementation is in `motion_matching/shared/compute_cost.m` (to be implemented under issue #015):

```matlab
function [J, terms] = compute_cost(theta, target, sim_fn, opts)
%COMPUTE_COST  Scalar swing-matching cost as defined in COST_FUNCTION_SPEC.md.
%
%   [J, terms] = compute_cost(THETA, TARGET, SIM_FN, OPTS) returns the scalar
%   cost J for coefficient vector THETA against measured trajectory TARGET,
%   evaluating the forward dynamics through callback SIM_FN with options OPTS.
%
%   TERMS is a struct with the breakdown:
%     .position, .orientation, .impact_anchor, .regularizer, .total
%   so callers can log and visualize each contribution separately.
%
%   Preconditions:
%     - THETA is a real, finite vector with length n_joints*7.
%     - TARGET is a struct conforming to CLUB_IK_SPEC.md output schema.
%     - SIM_FN is a function handle: sim_out = SIM_FN(theta).
%     - OPTS is the result of default_cost_options() with optional overrides.
%
%   Postconditions:
%     - J is a finite, non-negative scalar.
%     - terms.total == J (within eps).
```

The Python mirror is at `src/shared/python/motion_matching/cost.py`. Implementations must produce **identical** numeric results on the same inputs (cross-checked in Issue #016).

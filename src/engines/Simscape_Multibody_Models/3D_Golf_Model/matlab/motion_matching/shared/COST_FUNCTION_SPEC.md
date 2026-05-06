# Cost Function Specification

The cost function is the heart of the inverse problem. All four options consume the **same** cost function spec; that is what lets us compare results fairly.

## Symbols

| Symbol | Meaning |
|---|---|
| `θ` | coefficient vector, `n_joints × 7` flattened, decision variables |
| `q_sim(t; θ)` | simulated kinematic trajectory at time t given coefficients θ |
| `q_meas(t)` | measured kinematic trajectory (the target) |
| `T` | simulation duration (~0.3 s); we evaluate at the simulation timestep |
| `N` | number of timesteps over T |

For Phase 1 (club-only), `q` is a 12-vector per timestep:
- `r_butt ∈ ℝ³` (butt position, metres)
- `r_clubhead ∈ ℝ³` (clubhead position, metres)
- `R_club ∈ SO(3)` (club orientation, expressed as a flattened rotation matrix or a unit quaternion — see `q_orientation_repr` option)

## Primary cost (club-only)

$$
J(\theta) = w_p\,\underbrace{\frac{1}{N}\sum_{n} \big(\|r^\text{butt}_\text{sim}(t_n) - r^\text{butt}_\text{meas}(t_n)\|^2 + \|r^\text{ch}_\text{sim}(t_n) - r^\text{ch}_\text{meas}(t_n)\|^2\big)}_{\text{position term}} + w_o\,\underbrace{\frac{1}{N}\sum_{n} d_\text{geo}(R_\text{sim}(t_n), R_\text{meas}(t_n))^2}_{\text{orientation term}} + \lambda\,\underbrace{W_\text{total}(\theta)}_{\text{regularizer}}
$$

Where:

- `d_geo(R1, R2) = ‖log(R1ᵀ R2)‖_F / √2` — the geodesic angle between two rotations (radians). Implementation: convert to quaternions and use `2·acos(|q1·q2|)`.
- `W_total(θ)` — total mechanical work, defined below.
- `w_p`, `w_o`, `λ` — weights, defaults in [DEFAULTS](#defaults).

### Endpoint-anchor variant

When the optimizer is starting cold, the position term can dominate the orientation term and produce drift. Add an **anchor** term that is large at impact (the most kinematically constrained instant):

$$
J_\text{anchor}(\theta) = J(\theta) + w_a\,\big\|r_\text{ch,sim}(t_\text{impact}) - r_\text{ch,meas}(t_\text{impact})\big\|^2
$$

`w_a` defaults to `10·w_p`. `t_impact` is detected from the measured swing as the time of maximum clubhead speed.

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

| Regularizer | Formula | When useful |
|---|---|---|
| Total work (default) | `∫ Σ|τω| dt` | Bias toward physiologically efficient swings |
| Peak power | `max_t Σ|τω|` | Bias against torque spikes |
| Squared torque norm | `∫ Σ τ²  dt` | Bias against large torques regardless of motion |
| Coefficient L2 | `‖θ‖²` | Trivial bias toward the model's nominal |

The default is **total work**. Other variants are exposed via `options.regularizer = "total_work" | "peak_power" | "torque_l2" | "coeff_l2"`.

## Defaults

```matlab
function opts = default_cost_options()
    opts = struct();
    opts.w_position        = 1.0;       % position weight
    opts.w_orientation     = 0.1;       % orientation weight (radians^2 vs metres^2)
    opts.w_anchor_impact   = 10.0;      % impact-anchor multiplier on w_position
    opts.regularizer       = "total_work";
    opts.lambda            = 1e-4;      % regularizer strength
    opts.q_orientation_repr = "quaternion";  % "quaternion" | "rotmat"
    opts.time_alignment    = "impact";  % "impact" | "address" | "none"
    opts.resample_to_hz    = 1000;      % match simulation sample_rate
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

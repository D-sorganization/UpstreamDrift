# Issue: Implement compute_cost.m, compute_total_work.m, and +validators Package

## Summary

Implement the reference MATLAB cost function defined in `COST_FUNCTION_SPEC.md`,
its work-regularizer companion, and the `+validators` package of `mustHaveFields`,
`mustBeFiniteVector`, etc. used by `arguments` blocks across the motion-matching
codebase.

## Motivation

See `motion_matching/shared/COST_FUNCTION_SPEC.md`. Every option's optimizer
minimises this cost; without it nothing else is testable. The function must
return both a scalar `J` and a breakdown `terms` struct so the visualization
dashboard (#020–#022) can plot per-component contributions.

## Dependencies

None — foundational shared infrastructure. Note that the function takes a
`sim_fn` callback so it does not depend on #018 directly; tests mock the
callback.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\compute_cost.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\compute_total_work.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\default_cost_options.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\+validators\mustHaveFields.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\+validators\mustBeFiniteVector.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\+validators\mustBeUnitQuaternion.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\+validators\mustBeWithinCoefficientBounds.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_compute_cost.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_compute_total_work.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_validators.m`

## Public API

Verbatim from `COST_FUNCTION_SPEC.md`:

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

function W = compute_total_work(sim_out)
    arguments
        sim_out (1,1) struct {mustHaveFields(sim_out, ["time","tau","omega"])}
    end
    integrand = sum(abs(sim_out.tau .* sim_out.omega), 2);  % Nx1
    W = trapz(sim_out.time, integrand);
    assert(W >= 0, "Postcondition: total work must be non-negative");
end

function opts = default_cost_options()
    opts = struct();
    opts.w_position        = 1.0;
    opts.w_orientation     = 0.1;
    opts.w_anchor_impact   = 10.0;
    opts.regularizer       = "total_work";
    opts.lambda            = 1e-4;
    opts.q_orientation_repr = "quaternion";
    opts.time_alignment    = "impact";
    opts.resample_to_hz    = 1000;
end
```

## Required tests (TDD)

For `compute_cost`:

- `test_zero_residual_yields_only_regularizer_term`
- `test_position_term_is_mean_squared_butt_plus_clubhead_distance`
- `test_orientation_term_uses_geodesic_quaternion_distance_with_abs`
- `test_quaternion_sign_flip_does_not_change_orientation_term`
- `test_impact_anchor_term_only_active_when_w_anchor_impact_nonzero`
- `test_regularizer_total_work_matches_compute_total_work_output`
- `test_regularizer_peak_power_returns_max_t_sum_abs_tau_omega`
- `test_regularizer_torque_l2_returns_integral_of_tau_squared`
- `test_regularizer_coeff_l2_returns_squared_norm_of_theta`
- `test_terms_total_equals_J_within_eps`
- `test_J_finite_nonneg_for_random_finite_theta`
- `test_rejects_nan_theta_with_arguments_block_error`
- `test_rejects_target_missing_butt_field_with_validator_error`

For `compute_total_work`:

- `test_total_work_zero_for_zero_torque`
- `test_total_work_matches_handcalc_for_constant_torque_constant_omega`
- `test_total_work_uses_absolute_value_so_eccentric_counts_positive`

For validators:

- `test_must_have_fields_passes_for_complete_struct`
- `test_must_have_fields_errors_with_clear_message_listing_missing_fields`
- `test_must_be_finite_vector_rejects_inf_and_nan`
- `test_must_be_unit_quaternion_rejects_norms_outside_1e_minus_6`
- `test_must_be_within_coefficient_bounds_uses_generateRandomCoefficients_limits`

## DbC contract

`compute_cost` preconditions (in `arguments` block):

- `theta (:,1) double {mustBeFiniteVector}`
- `target (1,1) struct {validators.mustHaveFields(target, ["time","butt","clubhead","club_quat","impact_idx"])}`
- `sim_fn (1,1) function_handle`
- `opts (1,1) struct = default_cost_options()`

`compute_cost` postconditions:

- `J` is a finite non-negative scalar.
- `terms.total == J` to within `eps`.
- Every field of `terms` is non-negative.

`compute_total_work` postconditions:

- `W >= 0`.

## Acceptance Criteria

- [ ] `compute_cost.m` matches the math in `COST_FUNCTION_SPEC.md` and supports
      all four regularizer modes (`total_work`, `peak_power`, `torque_l2`, `coeff_l2`).
- [ ] `compute_total_work.m` and `default_cost_options.m` implemented per spec.
- [ ] Five validators implemented under `+validators/`.
- [ ] All listed tests pass via `runtests('motion_matching/shared/tests')`.
- [ ] `arguments` blocks present on every public function.
- [ ] `assert(...)` postconditions present.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `shared`, `matlab`, `tdd`, `dbc`, `infra`

## Effort estimate

M (1-3 days). Cost function math and validators are mechanical; the test suite
is the time sink.

# Issue: Implement simulate_with_coefficients.m — The Single Simscape Forward Wrapper

## Summary

Implement the one-and-only Simscape forward-call wrapper used by every motion-
matching option. Given a coefficient vector `theta`, run `GolfSwing3D_Kinetic.slx`
and return a populated `sim_out` struct (time, q, qd, qdd, tau, omega, club
kinematics). This is the most critical infrastructure issue in the backlog —
**every** other Simscape touchpoint must call this function.

## Motivation

See `motion_matching/shared/README.md` and the architecture diagram in
`shared/README.md`. The DRY rule (`CODING_STANDARDS.md`) prohibits any other
file from opening the .slx, configuring the model workspace, or calling `sim()`
directly. If two options each implement their own forward call, they will
silently diverge on solver settings and the leaderboard becomes meaningless.

## Dependencies

None — but **#014, #024, #025, #026, #030, #037 all depend on this**, so it
should be done first.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\simulate_with_coefficients.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\default_sim_options.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\theta_to_polynomial_struct.m` (reshape flat vector → polynomial struct expected by the model workspace)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\extract_sim_out.m` (post-process Simscape `Out` into canonical struct)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_simulate_with_coefficients.m`

## Public API

```matlab
function sim_out = simulate_with_coefficients(theta, opts)
%SIMULATE_WITH_COEFFICIENTS  Single Simscape forward call used by all options.
%
%   sim_out = SIMULATE_WITH_COEFFICIENTS(THETA, OPTS) runs
%   GolfSwing3D_Kinetic.slx with the polynomial torque coefficients THETA and
%   returns a canonical struct with fields:
%
%     .time        (N,1) double   simulation timegrid (s)
%     .q           (N, n_joints)  joint angles (rad)
%     .qd          (N, n_joints)  joint angular velocities (rad/s)
%     .qdd         (N, n_joints)  joint angular accelerations (rad/s^2)
%     .tau         (N, n_joints)  joint torques (N*m)
%     .omega       (N, n_joints)  alias for qd, kept for cost-function clarity
%     .r_butt      (N,3)          butt position (m)
%     .r_clubhead  (N,3)          clubhead position (m)
%     .q_club      (N,4)          club orientation quaternion [w x y z]
%     .v_clubhead  (N,3)          clubhead linear velocity (m/s)
%     .omega_club  (N,3)          club angular velocity (rad/s)
%     .joint_names (1, n_joints) string  ordering of joints
%     .solver_status (1,1) string  "success" | "warning" | "failed"
%
%   THETA is a real, finite vector of length n_joints*7 with ordering matching
%   getPolynomialParameterInfo.m: [A B C D E F G] per joint, joints in canonical
%   order.
%
%   OPTS is the result of default_sim_options() with optional overrides.
%
%   Preconditions:
%     - THETA is finite, length n_joints*7, within bounds from
%       generateRandomCoefficients.m.
%     - GolfSwing3D_Kinetic.slx is on the MATLAB path.
%
%   Postconditions:
%     - sim_out has all documented fields.
%     - sim_out.time is monotonic, starts at 0.
%     - sim_out.solver_status is one of {"success","warning","failed"}.

function opts = default_sim_options()
    opts = struct();
    opts.simulation_time   = 0.3;       % seconds
    opts.sample_rate       = 1000;      % Hz
    opts.solver            = "ode23t";  % matches model default
    opts.fast_restart      = true;      % keep model loaded between calls
    opts.parallel_safe     = false;     % set true for parsim/parfor use
    opts.verbosity         = "Silent";  % "Silent" | "Normal" | "Verbose" | "Debug"
end
```

## Required tests (TDD)

- `test_simulate_returns_canonical_struct_with_all_documented_fields`
- `test_simulate_time_starts_at_zero_and_is_monotonic`
- `test_simulate_field_lengths_consistent_N_rows_for_all_arrays`
- `test_simulate_clubhead_butt_distance_is_plausible_shaft_length_at_every_timestep`
- `test_simulate_quaternion_rows_are_unit_norm`
- `test_simulate_with_zero_theta_runs_without_error_and_returns_success_status`
- `test_simulate_rejects_non_finite_theta_with_arguments_block_error`
- `test_simulate_rejects_theta_outside_coefficient_bounds`
- `test_simulate_two_calls_with_same_theta_return_bit_identical_results`
- `test_simulate_fast_restart_reduces_wall_time_for_repeated_calls`
- `test_simulate_parallel_safe_mode_works_inside_parfor_loop`
- `test_theta_to_polynomial_struct_round_trips_through_getPolynomialParameterInfo`

## DbC contract

Preconditions (`arguments` block, using validators from #015):

- `theta (:,1) double {validators.mustBeFiniteVector, validators.mustBeWithinCoefficientBounds}`
- `opts (1,1) struct = default_sim_options()`

Postconditions (`assert(...)` after the sim run):

- `numel(unique([size(sim_out.q,1), size(sim_out.qd,1), numel(sim_out.time)]))==1`
- `sim_out.time(1) == 0`; `sim_out.time` is monotonic non-decreasing.
- `sim_out.solver_status` is a string in {"success","warning","failed"}.
- All `(N,3)` and `(N,4)` arrays free of NaN/Inf when status == "success".
- `vecnorm(sim_out.q_club, 2, 2)` is within `1e-6` of `1`.

## Acceptance Criteria

- [ ] Function lives at the file target above and is the only place in the
      motion-matching tree that calls `sim()` on the .slx.
- [ ] All listed tests pass.
- [ ] `arguments` block enforces preconditions; `assert(...)` checks postconditions.
- [ ] Fast restart implemented and verified to materially reduce repeated-call latency.
- [ ] `parallel_safe=true` mode tested inside a `parfor` of size >= 4.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.
- [ ] `motion_matching/shared/README.md` "Code that will live here" table marks the row done.

## Labels

`motion-matching`, `shared`, `matlab`, `tdd`, `dbc`, `infra`

## Effort estimate

L (3-7 days). The wrapper itself is short, but plumbing fast-restart, the
parallel-safe path, and verifying bit-identical reproducibility across calls is
where the time goes.

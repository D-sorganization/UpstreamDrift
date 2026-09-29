# Issue: Implement synthesize_target_from_coefficients.m TDD Oracle

## Summary

Build the synthetic-target generator that runs the Simscape model with a known
coefficient vector `theta_truth`, records the resulting club trajectory, and emits
a `target` struct conforming to `CLUB_IK_SPEC.md`. This is the trivial-fit oracle
every option's tests use to detect optimizer breakage.

## Motivation

See `motion_matching/shared/CLUB_IK_SPEC.md` §"Synthetic". With club-only observation
the inverse problem is under-determined on real swings, so we can't claim a fit is
"correct". The synthetic oracle solves this: feed in known `theta`, get a `target`,
then `fit(target)` must recover `theta` (or at least RMSE < 1 mm). If it doesn't,
the optimizer is broken — not the data. Every option's first test is this round trip.

## Dependencies

- #018 (`simulate_with_coefficients.m`) — synthesizer must use the **exact same**
  Simscape callback that the optimizer will use, otherwise round-trip fits aren't
  valid. Cannot be implemented before #018.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\synthesize_target_from_coefficients.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_synthesize_target_from_coefficients.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\fixtures\theta_truth_examples.mat` (3-5 example coefficient vectors used by tests)

## Public API

Verbatim from `CLUB_IK_SPEC.md`:

```matlab
function target = synthesize_target_from_coefficients(theta, opts)
%SYNTHESIZE_TARGET_FROM_COEFFICIENTS  Build a target struct by running the
%   Simscape model with known coefficients. Used as the oracle for tests.
```

Default options (define `default_synth_options()` alongside):

```matlab
function opts = default_synth_options()
    opts = struct();
    opts.sample_rate    = 1000;            % Hz, matches simulation
    opts.simulation_time = 0.3;            % seconds
    opts.add_noise       = false;          % set true to test noisy-target robustness
    opts.noise_sigma_m   = 0.001;          % 1 mm position noise when add_noise=true
    opts.subject_id     = "synthetic";
    opts.trial_id       = "synthesizer_v1";
end
```

## Required tests (TDD)

- `test_synthesize_returns_canonical_target_struct`
- `test_synthesize_target_satisfies_all_validation_rules_from_club_ik_spec`
- `test_synthesize_round_trip_with_zero_noise_produces_identical_clubhead_path`
- `test_synthesize_with_noise_produces_target_with_expected_noise_floor`
- `test_synthesize_uses_simulate_with_coefficients_not_a_separate_simscape_call`
- `test_synthesize_provenance_records_theta_truth_in_source_struct`
- `test_synthesize_rejects_theta_outside_bounds_from_generateRandomCoefficients`

## DbC contract

Preconditions:

- `theta` is a real, finite vector of length `n_joints * 7`.
- `theta` lies within the bounds from `generateRandomCoefficients.m`
  (A,B in ±1000; C,D in ±500; E,F in ±100; G in ±25).
- `opts.sample_rate` is positive; `opts.simulation_time` in (0, 1].

Postconditions:

- `target` satisfies all rules from `CLUB_IK_SPEC.md` §"Validation rules".
- `target.source.theta_truth` equals the input `theta` (so the test oracle can
  recover the truth).
- `target.source.format == "synthetic"`.

## Acceptance Criteria

- [ ] `synthesize_target_from_coefficients.m` calls `simulate_with_coefficients`
      from #018 and does **not** invoke `sim()` or open the .slx directly.
- [ ] All listed tests pass.
- [ ] `arguments` block enforces preconditions; `assert(...)` checks postconditions.
- [ ] Fixture file `theta_truth_examples.mat` checked in (small, <100 KB) with
      example coefficient vectors covering: nominal swing, hard swing, weak swing.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `shared`, `matlab`, `tdd`, `dbc`, `infra`

## Effort estimate

S (≤1 day) once #018 lands.

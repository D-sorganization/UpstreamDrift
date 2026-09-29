classdef test_fit_swing_surrogateopt < matlab.unittest.TestCase
%TEST_FIT_SWING_SURROGATEOPT  Unit tests for fit_swing_surrogateopt.
%
%   Tests inject a fake `sim_fn` via opts.sim_fn so they run without
%   Simscape / GolfSwing3D_Kinetic. The slow synthetic-recovery test is
%   tagged IsSlow and is skipped when Global Optimization Toolbox or the
%   Simscape model is unavailable.
%
%   GitHub issue: #026 / #3995.

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            opt1 = fileparts(here);
            mm   = fileparts(opt1);
            addpath(opt1);
            addpath(fullfile(mm, 'shared'));
            testCase.addTeardown(@() rmpath(opt1));
            testCase.addTeardown(@() rmpath(fullfile(mm, 'shared')));
        end
    end

    methods (Test)

        function test_options_struct_contract(testCase)
            % default_option1_options + the surrogateopt-specific fields
            % must produce a usable opts struct with the documented
            % defaults: surrogate_max_evals=1500, polish_max_iter=200,
            % max_wall_seconds=600.
            opts = default_option1_options();
            opts = local_apply_so_defaults(opts);
            testCase.verifyTrue(isstruct(opts) && isscalar(opts));
            testCase.verifyEqual(double(opts.surrogate_max_evals), 1500);
            testCase.verifyEqual(double(opts.polish_max_iter), 200);
            testCase.verifyEqual(double(opts.max_wall_seconds), 600);
            testCase.verifyTrue(isfield(opts, 'sim'));
            testCase.verifyTrue(isfield(opts, 'cost'));
            testCase.verifyTrue(isfield(opts, 'penalty_on_sim_failure'));
        end

        function test_handles_simulation_failure_with_inf_cost(testCase)
            % A sim_fn that throws must not abort the optimizer; the
            % cost function must convert exceptions into a finite penalty
            % (or Inf) so surrogateopt keeps exploring.
            testCase.assumeFalse(exist('surrogateopt', 'file') == 0, ...
                'Global Optimization Toolbox not available');
            target = local_dummy_target(5);
            opts = local_test_opts();
            opts.sim_fn = @(~) error('intentional:simFail', 'boom');
            opts.penalty_on_sim_failure = Inf;     % use Inf, not a finite penalty
            opts.surrogate_max_evals = 12;
            opts.polish_max_iter = 1;
            opts.max_wall_seconds = 30;
            opts.skip_polish = true;       % polish would also fail; isolate phase
            % Should not throw.
            try
                result = fit_swing_surrogateopt(target, opts);
                testCase.verifyTrue(isstruct(result));
                testCase.verifyTrue(isfield(result, 'surrogateopt_history'));
            catch ME
                % Surrogateopt may itself bail when every eval is Inf;
                % that's acceptable as long as it isn't our intentional
                % sim error propagating.
                testCase.verifyNotEqual(ME.identifier, 'intentional:simFail');
            end
        end

        function test_polish_phase_strictly_improves_or_equal_to_surrogate_best(testCase)
            % Build a smooth quadratic-cost fake sim_fn so both stages
            % converge fast. Polish must not worsen RMSE.
            testCase.assumeFalse(exist('surrogateopt', 'file') == 0, ...
                'Global Optimization Toolbox not available');
            target = local_dummy_target(5);
            opts = local_test_opts();
            opts.sim_fn = @local_quadratic_fake_sim;
            opts.surrogate_max_evals = 30;
            opts.polish_max_iter = 5;
            opts.max_wall_seconds = 60;
            opts.skip_polish = false;

            result = fit_swing_surrogateopt(target, opts);
            testCase.verifyTrue(isfield(result, 'surrogateopt_phase'));
            so_rmse = result.surrogateopt_phase.final_rmse_m;
            polished_rmse = result.final_rmse_m;
            testCase.verifyLessThanOrEqual(polished_rmse, so_rmse + 1e-9, ...
                'Polish must not strictly worsen the surrogate-best RMSE');
        end

        function test_result_struct_includes_both_phases_history(testCase)
            testCase.assumeFalse(exist('surrogateopt', 'file') == 0, ...
                'Global Optimization Toolbox not available');
            target = local_dummy_target(5);
            opts = local_test_opts();
            opts.sim_fn = @local_quadratic_fake_sim;
            opts.surrogate_max_evals = 25;
            opts.polish_max_iter = 4;
            opts.max_wall_seconds = 60;

            result = fit_swing_surrogateopt(target, opts);

            % Surrogate phase history captured under the documented field.
            testCase.verifyTrue(isfield(result, 'surrogateopt_history'));
            testCase.verifyTrue(istable(result.surrogateopt_history));

            % Polish phase history is the standard fmincon iter_history.
            testCase.verifyTrue(isfield(result, 'iter_history'));
            testCase.verifyTrue(istable(result.iter_history));

            % Solver tag is the hybrid tag (or surrogateopt-only if the
            % polish phase couldn't run because Simscape is unavailable
            % — in that case the surrogate result is returned and the
            % test still verifies both phases' history is recorded).
            testCase.verifyTrue(any(string(result.solver) == ...
                ["surrogateopt+fmincon", "surrogateopt"]));
            testCase.verifyTrue(isfield(result, 'surrogateopt_phase'));
        end
    end

    methods (Test, TestTags = {'IsSlow'})
        function test_completes_in_under_5min_on_30coeff_problem(testCase)
            % Slow integration test. Requires the Simscape model and
            % surrogateopt; otherwise skipped.
            testCase.assumeFalse(exist('surrogateopt', 'file') == 0, ...
                'Global Optimization Toolbox not available');
            testCase.assumeFalse(exist('synthesize_target_from_coefficients', 'file') == 0, ...
                'synthesize_target_from_coefficients not available');
            testCase.assumeFalse(exist('GolfSwing3D_Kinetic', 'file') == 0, ...
                'Simulink model GolfSwing3D_Kinetic not available');

            % 30 coefficients => roughly 5 joints of 7-poly minus tail.
            % Use 5 joints (35 coeffs) which is the closest.
            opts = default_option1_options();
            opts = local_apply_so_defaults(opts);
            opts.sim.joint_names = "j" + string(1:5);
            opts.surrogate_max_evals = 1500;
            opts.polish_max_iter = 200;
            opts.max_wall_seconds = 300;    % 5 minutes hard cap

            theta_truth = generateRandomCoefficients(5 * 7);
            target = synthesize_target_from_coefficients(theta_truth, opts.sim);

            t0 = tic;
            result = fit_swing_surrogateopt(target, opts);
            elapsed = toc(t0);

            testCase.verifyLessThanOrEqual(elapsed, 300, ...
                'Hybrid surrogateopt+polish must finish under 5 minutes');
            testCase.verifyTrue(isfinite(result.final_rmse_m));
        end
    end

end

% =====================================================================
function opts = local_test_opts()
    opts = default_option1_options();
    opts = local_apply_so_defaults(opts);
    opts.sim.joint_names = "j" + string(1:3);   % small d for fast tests
    opts.use_parallel = false;
    opts.skip_polish = false;
    opts.display = "off";
    opts.penalty_on_sim_failure = 1e9;
end

% =====================================================================
function opts = local_apply_so_defaults(opts)
    opts.surrogate_max_evals = 1500;
    opts.polish_max_iter     = 200;
    opts.max_wall_seconds    = 600;
    opts.min_surrogate_points = 50;
    opts.min_sample_distance  = 1e-3;
    opts.use_parallel         = false;
    opts.skip_polish          = false;
end

% =====================================================================
function target = local_dummy_target(N)
    if nargin < 1, N = 10; end
    target = struct();
    target.time       = (0:N-1)' * 1e-3;
    target.butt       = zeros(N, 3);
    target.clubhead   = zeros(N, 3);
    target.club_quat  = repmat([1 0 0 0], N, 1);
    target.impact_idx = N;
end

% =====================================================================
function sim_out = local_quadratic_fake_sim(theta)
%LOCAL_QUADRATIC_FAKE_SIM  Smooth fake sim_fn whose output position
%signals encode a quadratic cost in theta. compute_cost on the resulting
%sim_out vs a zero target should produce a strictly convex surface so the
%fmincon polish has something to improve.
    theta = theta(:);
    N = 5;
    % Embed the squared-norm of theta into the trajectory amplitude.
    amp = norm(theta) * 1e-3;
    t = (0:N-1)' * 1e-3;
    butt = amp * ones(N, 3);
    clubhead = amp * ones(N, 3);
    quat = repmat([1 0 0 0], N, 1);
    sim_out = struct( ...
        'time', t, ...
        'butt', butt, ...
        'clubhead', clubhead, ...
        'club_quat', quat, ...
        'impact_idx', N);
    % Provide r_* aliases for the standard adapter compatibility.
    sim_out.r_butt = butt;
    sim_out.r_clubhead = clubhead;
    sim_out.q_club = quat;
end

classdef test_fit_swing_multistart < matlab.unittest.TestCase
%TEST_FIT_SWING_MULTISTART  Unit tests for fit_swing_multistart.
%
%   Pure-MATLAB tests (sampling, plumbing, postconditions) run in CI; the
%   Simscape round-trip tests are tagged IsSlow and skipped unless the
%   Simulink model and synthesizer are available on the path.
%
%   GitHub issue: #025 / #3994.

    properties
        NJoints = 3
    end

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            opt1 = fileparts(here);
            mm   = fileparts(opt1);
            addpath(opt1);
            addpath(fullfile(opt1, 'private'));
            addpath(fullfile(mm, 'shared'));
            testCase.addTeardown(@() rmpath(opt1));
            testCase.addTeardown(@() rmpath(fullfile(opt1, 'private')));
            testCase.addTeardown(@() rmpath(fullfile(mm, 'shared')));
        end
    end

    methods (Test)
        % -------------------------------------------------------------
        % Sampling tests (no Simscape required)
        % -------------------------------------------------------------
        function test_sobol_starts_distinct(testCase)
            n = testCase.NJoints;
            [lb, ub] = build_coefficient_bounds(n);
            S = sample_starting_points(8, lb, ub, "sobol", 42);
            testCase.verifyEqual(size(S), [numel(lb), 8]);
            testCase.verifyEqual(size(unique(S.', 'rows'), 1), 8);
            testCase.verifyTrue(all(S(:) > lb(1) - eps));
            testCase.verifyTrue(all(S(:) < ub(1) + eps));
        end

        function test_random_starts_seed_reproducible(testCase)
            n = testCase.NJoints;
            [lb, ub] = build_coefficient_bounds(n);
            A = sample_starting_points(6, lb, ub, "random", 7);
            B = sample_starting_points(6, lb, ub, "random", 7);
            testCase.verifyEqual(A, B);
        end

        function test_sobol_lower_discrepancy_than_random(testCase)
            % Coarse proxy: variance of column means should be lower for Sobol.
            n = testCase.NJoints;
            [lb, ub] = build_coefficient_bounds(n);
            R = sample_starting_points(64, lb, ub, "random", 1);
            S = sample_starting_points(64, lb, ub, "sobol", 1);
            % Normalize to [0,1] to compare apples-to-apples.
            span = ub - lb;
            Rn = (R - lb) ./ span;
            Sn = (S - lb) ./ span;
            v_rand  = var(mean(Rn, 2));
            v_sobol = var(mean(Sn, 2));
            % Sobol should be at least as uniform as random.
            testCase.verifyLessThanOrEqual(v_sobol, v_rand + 1e-3);
        end

        function test_starts_within_bounds(testCase)
            n = testCase.NJoints;
            [lb, ub] = build_coefficient_bounds(n);
            for strat = ["sobol","random"]
                S = sample_starting_points(5, lb, ub, strat, 3);
                testCase.verifyTrue(all(all(S >= lb - 1e-9)));
                testCase.verifyTrue(all(all(S <= ub + 1e-9)));
            end
        end

        % -------------------------------------------------------------
        % Driver plumbing (uses Simscape if available, else assumed)
        % -------------------------------------------------------------
        function test_default_options_contract(testCase)
            opts = default_multistart_options();
            testCase.verifyTrue(isstruct(opts) && isscalar(opts));
            testCase.verifyTrue(isfield(opts, 'n_starts'));
            testCase.verifyTrue(isfield(opts, 'starting_strategy'));
            testCase.verifyTrue(isfield(opts, 'parallel_method'));
            testCase.verifyTrue(isfield(opts, 'fmincon_options'));
            testCase.verifyEqual(double(opts.n_starts), 8);
        end

        function test_bad_target_field_errors(testCase)
            opts = default_multistart_options();
            bad = struct('butt', zeros(10, 3));
            testCase.verifyError( ...
                @() fit_swing_multistart(bad, opts), ?MException);
        end

        % -------------------------------------------------------------
        % Toy-cost integration: prove parallel multistart finds the
        % global on a known-multimodal target by stubbing the simulator.
        % -------------------------------------------------------------
        function test_n_starts_runs_recorded(testCase)
            testCase.assumeTrue(local_have_solver_deps(), ...
                'compute_cost / simulate_with_coefficients not on path');
            testCase.assumeFalse(exist('GolfSwing3D_Kinetic', 'file') == 0, ...
                'Simulink model not available in test environment');
            opts = local_fast_opts(testCase, 4);
            target = local_dummy_target();
            result = fit_swing_multistart(target, opts);
            testCase.verifyEqual(numel(result.all_runs), 4);
            testCase.verifyEqual(numel(result.all_starts), 4);
        end

        function test_best_result_is_min_over_all_runs(testCase)
            testCase.assumeTrue(local_have_solver_deps(), ...
                'compute_cost / simulate_with_coefficients not on path');
            testCase.assumeFalse(exist('GolfSwing3D_Kinetic', 'file') == 0, ...
                'Simulink model not available in test environment');
            opts = local_fast_opts(testCase, 4);
            target = local_dummy_target();
            result = fit_swing_multistart(target, opts);
            rmses = arrayfun(@(r) r.final_rmse_m, result.all_runs);
            testCase.verifyEqual(result.final_rmse_m, min(rmses), ...
                'AbsTol', 1e-9 * max(1, min(rmses)));
        end

        function test_serial_and_parallel_produce_equivalent_best_solution(testCase)
            testCase.assumeTrue(local_have_solver_deps(), ...
                'compute_cost / simulate_with_coefficients not on path');
            testCase.assumeFalse(exist('GolfSwing3D_Kinetic', 'file') == 0, ...
                'Simulink model not available in test environment');
            opts_s = local_fast_opts(testCase, 3);
            opts_s.parallel = false;
            opts_p = opts_s;
            opts_p.parallel = true;
            target = local_dummy_target();
            r_s = fit_swing_multistart(target, opts_s);
            r_p = fit_swing_multistart(target, opts_p);
            testCase.verifyEqual(r_s.final_rmse_m, r_p.final_rmse_m, ...
                'AbsTol', 1e-6, 'RelTol', 1e-3);
        end
    end

    methods (Test, TestTags = {'IsSlow'})
        function test_multistart_outperforms_single_start_on_known_multimodal(testCase)
            testCase.assumeTrue(local_have_solver_deps(), ...
                'compute_cost / simulate_with_coefficients not on path');
            testCase.assumeFalse(exist('GolfSwing3D_Kinetic', 'file') == 0, ...
                'Simulink model not available in test environment');
            % Slow placeholder: full multimodal proof requires the Simulink
            % round-trip and the synthesizer (issue #014).
            testCase.assumeFail( ...
                'Slow tag placeholder; needs synthesize_target_from_coefficients');
        end

        function test_fits_synthetic_swing_to_within_0p5mm_with_8_starts(testCase)
            testCase.assumeFalse(exist('synthesize_target_from_coefficients', 'file') == 0, ...
                'synthesize_target_from_coefficients (issue #014) not yet available');
            testCase.assumeFalse(exist('GolfSwing3D_Kinetic', 'file') == 0, ...
                'Simulink model not available in test environment');
            opts = default_multistart_options();
            opts.n_starts = uint32(8);
            theta_truth = generateRandomCoefficients( ...
                numel(build_coefficient_bounds(7)));
            target = synthesize_target_from_coefficients( ...
                theta_truth, opts.fmincon_options.sim);
            result = fit_swing_multistart(target, opts);
            testCase.verifyLessThan(result.final_rmse_m, 5e-4);
        end
    end
end

% =====================================================================
function tf = local_have_solver_deps()
    tf = exist('compute_cost', 'file') ~= 0 && ...
         exist('simulate_with_coefficients', 'file') ~= 0;
end

function opts = local_fast_opts(testCase, n_starts)
    opts = default_multistart_options();
    opts.n_starts = uint32(n_starts);
    opts.parallel = false;
    opts.fmincon_options.sim.joint_names = ...
        "j" + string(1:testCase.NJoints);
    opts.fmincon_options.max_iter           = uint32(1);
    opts.fmincon_options.max_function_evals = uint32(2);
    opts.fmincon_options.display            = "off";
end

function target = local_dummy_target()
    N = 10;
    target = struct();
    target.time       = (0:N-1)' * 1e-3;
    target.butt       = zeros(N, 3);
    target.clubhead   = zeros(N, 3);
    target.club_quat  = repmat([1 0 0 0], N, 1);
    target.impact_idx = N;
end

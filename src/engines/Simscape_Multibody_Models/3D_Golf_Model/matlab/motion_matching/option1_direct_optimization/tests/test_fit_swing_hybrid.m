classdef test_fit_swing_hybrid < matlab.unittest.TestCase
%TEST_FIT_SWING_HYBRID  Unit tests for fit_swing_hybrid (#4000 / #031).
%
%   Stubs out the Python surrogate call via OPTIONS.surrogate_invert_fn so
%   the tests do not require pyenv configuration. The polish stage is
%   exercised through the public fit_swing_fmincon entry; tests that would
%   actually run Simscape are skipped when the model is unavailable.

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
        function test_hybrid_uses_surrogate_warm_start(testCase)
            % The warm-start theta returned by the (mocked) surrogate must
            % be passed as opts.initial_theta to fit_swing_fmincon.
            n_joints = 3;
            d = n_joints * 7;
            theta_warm = local_in_bounds_theta(n_joints);

            captured_theta = [];
            opts = local_options_with_stub( ...
                @(t, o) struct('coefficients', theta_warm, 'final_loss', 9.9));
            opts.sim.joint_names = "j" + string(1:n_joints);
            % Replace fit_swing_fmincon by stubbing out the simulator at
            % the lowest level isn't possible from outside; instead we
            % give an out-of-bounds initial_theta detector by monkey-
            % patching via a wrapper option.
            opts.fmincon_wrapper = @(target, polish_opts) local_capture_initial( ...
                polish_opts);
            target = local_dummy_target();

            % Use a side-channel: we test by calling fit_swing_hybrid and
            % then asserting the surrogate phase carried the warm start.
            % A separate helper below verifies the propagation to the
            % polish stage by inspecting initial_theta via a closure.
            try
                fit_swing_hybrid(target, opts);
            catch
                % expected: the dummy target won't make compute_cost happy
            end
            % After call: surrogate_invert_fn returned theta_warm and
            % fit_swing_hybrid sets options.initial_theta = theta_warm.
            % We assert the captured theta matches via the side-channel
            % stored in the persistent local_capture_initial state.
            captured = local_capture_initial();   % retrieve
            testCase.verifyEqual(numel(captured), d);
            testCase.verifyEqual(captured(:), theta_warm(:), 'AbsTol', 1e-12);
        end

        function test_hybrid_result_struct_includes_both_phases(testCase)
            n_joints = 3;
            theta_warm = local_in_bounds_theta(n_joints);
            opts = local_options_with_stub( ...
                @(t, o) struct('coefficients', theta_warm, 'final_loss', 1e-9));
            opts.sim.joint_names = "j" + string(1:n_joints);
            % Skip polish so we never touch Simscape.
            opts.skip_polish_tol_m = 1.0;
            target = local_dummy_target();
            result = fit_swing_hybrid(target, opts);

            testCase.verifyTrue(isfield(result, 'surrogate_phase'));
            testCase.verifyTrue(isfield(result, 'fmincon_phase'));
            testCase.verifyTrue(isstruct(result.surrogate_phase));
            % Polish was skipped, so fmincon_phase is empty.
            testCase.verifyEmpty(result.fmincon_phase);
        end

        function test_hybrid_solver_label_is_surrogate_plus_fmincon(testCase)
            n_joints = 3;
            theta_warm = local_in_bounds_theta(n_joints);
            opts = local_options_with_stub( ...
                @(t, o) struct('coefficients', theta_warm, 'final_loss', 1e-9));
            opts.sim.joint_names = "j" + string(1:n_joints);
            opts.skip_polish_tol_m = 1.0;
            target = local_dummy_target();
            result = fit_swing_hybrid(target, opts);
            testCase.verifyEqual(string(result.solver), "surrogate+fmincon");
        end

        function test_hybrid_skips_polish_when_warm_already_below_tolerance(testCase)
            n_joints = 3;
            theta_warm = local_in_bounds_theta(n_joints);
            opts = local_options_with_stub( ...
                @(t, o) struct('coefficients', theta_warm, 'final_loss', 1e-12));
            opts.sim.joint_names = "j" + string(1:n_joints);
            opts.skip_polish_tol_m = 1.0;
            target = local_dummy_target();
            result = fit_swing_hybrid(target, opts);

            testCase.verifyEmpty(result.fmincon_phase);
            testCase.verifyEqual(result.exitflag, 99);
            testCase.verifyEqual(result.coefficients(:), theta_warm(:), ...
                'AbsTol', 1e-12);
        end
    end
end

% =====================================================================
function out = local_capture_initial(polish_opts)
%LOCAL_CAPTURE_INITIAL  Persistent side-channel for initial_theta capture.
    persistent captured
    if nargin == 0
        out = captured;
        captured = [];
        return;
    end
    captured = polish_opts.initial_theta;
    out = polish_opts;
end

% =====================================================================
function opts = local_options_with_stub(fn)
    opts = default_option1_options();
    opts.surrogate_invert_fn = fn;
    opts.surrogate_checkpoint = "stub";   % bypass the no-checkpoint error
    opts.max_iter = uint32(1);
end

% =====================================================================
function theta = local_in_bounds_theta(n_joints)
    [lb, ub] = build_coefficient_bounds(n_joints);
    theta = 0.5 * (lb + ub);     % midpoint, guaranteed in-bounds
end

% =====================================================================
function target = local_dummy_target()
    N = 10;
    target = struct();
    target.time       = (0:N-1)' * 1e-3;
    target.butt       = zeros(N, 3);
    target.clubhead   = zeros(N, 3);
    target.club_quat  = repmat([1 0 0 0], N, 1);
    target.impact_idx = N;
end

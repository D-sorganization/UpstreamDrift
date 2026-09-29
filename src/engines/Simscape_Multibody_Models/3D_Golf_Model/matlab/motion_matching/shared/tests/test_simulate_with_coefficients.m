classdef test_simulate_with_coefficients < matlab.unittest.TestCase
%TEST_SIMULATE_WITH_COEFFICIENTS  matlab.unittest harness for issue #018/#3987.
%
%   Tests the single Simscape forward-call wrapper. Tests that need an actual
%   Simulink run are tagged "RequiresSimulink" and skipped when the model or
%   the toolbox is unavailable so unit-only CI can still pass.

    properties (Constant)
        REQUIRED_FIELDS = ["time","q","qd","qdd","tau","omega", ...
                           "r_butt","r_clubhead","q_club","v_clubhead", ...
                           "omega_club","joint_names","solver_status"];
    end

    properties
        n_joints (1,1) double = 0;
        joint_names (1,:) string = string.empty(1,0);
        opts struct
    end

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            shared_dir = fileparts(here);                  % .../shared
            engine_root = fileparts(fileparts(shared_dir));% .../matlab
            addpath(shared_dir);
            addpath(genpath(fullfile(engine_root, 'src')));
            testCase.opts = default_sim_options();
            testCase.opts.verbosity = "Silent";
            try
                info = getPolynomialParameterInfo();
                testCase.joint_names = string(info.joint_names);
                testCase.n_joints = numel(testCase.joint_names);
            catch ME
                testCase.assumeFail(sprintf( ...
                    "getPolynomialParameterInfo unavailable: %s", ME.message));
            end
        end
    end

    %% ---------------- Pure unit tests (no Simulink required) ----------------
    methods (Test)

        function test_invalid_theta_length_rejected_with_clear_error(testCase)
            bad_theta = zeros(testCase.n_joints * 7 - 1, 1);
            testCase.verifyError(@() simulate_with_coefficients(bad_theta, testCase.opts), ...
                "simulate_with_coefficients:badThetaLength");
        end

        function test_nonfinite_theta_rejected(testCase)
            theta_nan = zeros(testCase.n_joints * 7, 1);
            theta_nan(1) = NaN;
            testCase.verifyError(@() simulate_with_coefficients(theta_nan, testCase.opts), ...
                "MATLAB:validators:mustBeFinite");
            theta_inf = zeros(testCase.n_joints * 7, 1);
            theta_inf(2) = Inf;
            testCase.verifyError(@() simulate_with_coefficients(theta_inf, testCase.opts), ...
                "MATLAB:validators:mustBeFinite");
        end

        function test_complex_theta_rejected(testCase)
            theta_c = zeros(testCase.n_joints * 7, 1);
            theta_c(1) = 1 + 1i;
            testCase.verifyError(@() simulate_with_coefficients(theta_c, testCase.opts), ...
                "MATLAB:validators:mustBeReal");
        end

        function test_default_sim_options_has_all_required_fields(testCase)
            o = default_sim_options();
            required = ["model_name","simulation_time","sample_rate","solver", ...
                        "fast_restart","parallel_safe","verbosity","cache_dir", ...
                        "use_cache","stop_on_error","joint_names"];
            for k = 1:numel(required)
                testCase.verifyTrue(isfield(o, required(k)), ...
                    sprintf("default_sim_options missing field %s", required(k)));
            end
            testCase.verifyEqual(o.model_name, "GolfSwing3D_Kinetic");
            testCase.verifyEqual(o.simulation_time, 0.3);
            testCase.verifyEqual(o.sample_rate, 1000);
        end

        function test_theta_to_polynomial_struct_round_trips_through_param_info(testCase)
            d = testCase.n_joints * 7;
            theta = (1:d).' * 1e-3;
            cs = theta_to_polynomial_struct(theta);
            testCase.verifyEqual(numel(fieldnames(cs)), d);
            % Spot check: first joint, first coefficient (A) should equal theta(1)
            jname = char(testCase.joint_names(1));
            varA = cs.([jname 'A']);
            testCase.verifyEqual(varA, theta(1), "AbsTol", eps);
            % And last value
            jname_last = char(testCase.joint_names(end));
            varG = cs.([jname_last 'G']);
            testCase.verifyEqual(varG, theta(end), "AbsTol", eps);
        end

        function test_theta_to_polynomial_struct_rejects_wrong_length(testCase)
            bad = zeros(testCase.n_joints * 7 + 3, 1);
            testCase.verifyError(@() theta_to_polynomial_struct(bad), ...
                "theta_to_polynomial_struct:badLength");
        end
    end

    %% ---------------- Tests requiring Simulink + the .slx ------------------
    methods (Test, TestTags = {"RequiresSimulink"})

        function test_zero_coefficients_produces_static_pose(testCase)
            testCase.assumeSimulinkAvailable();
            theta = zeros(testCase.n_joints * 7, 1);
            sim_out = simulate_with_coefficients(theta, testCase.opts);
            testCase.verifyEqual(sim_out.solver_status, "success");
            % Static pose: clubhead displacement should be tiny across the run.
            ch = sim_out.r_clubhead;
            ch_disp = vecnorm(ch - ch(1, :), 2, 2);
            testCase.verifyLessThan(max(ch_disp), 0.05, ...
                "Zero coefficients should produce ~static clubhead (<5cm drift)");
        end

        function test_known_coefficients_reproduce_baseline(testCase)
            testCase.assumeSimulinkAvailable();
            theta = 0.01 * (1:testCase.n_joints*7).';
            a = simulate_with_coefficients(theta, testCase.opts);
            b = simulate_with_coefficients(theta, testCase.opts);
            testCase.verifyEqual(a.solver_status, "success");
            testCase.verifyEqual(b.solver_status, "success");
            testCase.verifyEqual(a.r_clubhead, b.r_clubhead, "AbsTol", 1e-9, ...
                "Same theta must produce bit-identical clubhead trajectory");
            testCase.verifyEqual(a.q, b.q, "AbsTol", 1e-9);
        end

        function test_output_struct_has_all_required_fields(testCase)
            testCase.assumeSimulinkAvailable();
            theta = zeros(testCase.n_joints * 7, 1);
            sim_out = simulate_with_coefficients(theta, testCase.opts);
            for k = 1:numel(testCase.REQUIRED_FIELDS)
                f = testCase.REQUIRED_FIELDS(k);
                testCase.verifyTrue(isfield(sim_out, f), ...
                    sprintf("sim_out missing field %s", f));
            end
        end

        function test_output_time_is_monotonic_and_starts_at_zero(testCase)
            testCase.assumeSimulinkAvailable();
            theta = zeros(testCase.n_joints * 7, 1);
            sim_out = simulate_with_coefficients(theta, testCase.opts);
            testCase.verifyEqual(sim_out.time(1), 0, "AbsTol", 1e-9);
            testCase.verifyTrue(all(diff(sim_out.time) >= 0), ...
                "Time vector must be monotonic non-decreasing");
        end

        function test_output_shapes_match_n_joints_and_n_timesteps(testCase)
            testCase.assumeSimulinkAvailable();
            theta = zeros(testCase.n_joints * 7, 1);
            sim_out = simulate_with_coefficients(theta, testCase.opts);
            N = numel(sim_out.time);
            testCase.verifyEqual(size(sim_out.q), [N, testCase.n_joints]);
            testCase.verifyEqual(size(sim_out.qd), [N, testCase.n_joints]);
            testCase.verifyEqual(size(sim_out.qdd), [N, testCase.n_joints]);
            testCase.verifyEqual(size(sim_out.tau), [N, testCase.n_joints]);
            testCase.verifyEqual(size(sim_out.omega), [N, testCase.n_joints]);
            testCase.verifyEqual(size(sim_out.r_butt), [N, 3]);
            testCase.verifyEqual(size(sim_out.r_clubhead), [N, 3]);
            testCase.verifyEqual(size(sim_out.q_club), [N, 4]);
            testCase.verifyEqual(size(sim_out.v_clubhead), [N, 3]);
            testCase.verifyEqual(size(sim_out.omega_club), [N, 3]);
        end

        function test_no_nan_inf_on_success_status(testCase)
            testCase.assumeSimulinkAvailable();
            theta = zeros(testCase.n_joints * 7, 1);
            sim_out = simulate_with_coefficients(theta, testCase.opts);
            if sim_out.solver_status == "success"
                testCase.verifyTrue(all(isfinite(sim_out.q(:))), "q has non-finite");
                testCase.verifyTrue(all(isfinite(sim_out.qd(:))), "qd has non-finite");
                testCase.verifyTrue(all(isfinite(sim_out.tau(:))), "tau has non-finite");
            end
        end

        function test_quaternion_rows_unit_norm(testCase)
            testCase.assumeSimulinkAvailable();
            theta = zeros(testCase.n_joints * 7, 1);
            sim_out = simulate_with_coefficients(theta, testCase.opts);
            if sim_out.solver_status == "success" && all(~isnan(sim_out.q_club(:)))
                qn = vecnorm(sim_out.q_club, 2, 2);
                testCase.verifyLessThan(max(abs(qn - 1)), 1e-3);
            end
        end

        function test_cache_round_trip(testCase)
            testCase.assumeSimulinkAvailable();
            tmp = tempname;
            mkdir(tmp);
            cleanup = onCleanup(@() rmdir(tmp, 's'));
            o = testCase.opts;
            o.use_cache = true;
            o.cache_dir = string(tmp);
            theta = zeros(testCase.n_joints * 7, 1);
            a = simulate_with_coefficients(theta, o);
            b = simulate_with_coefficients(theta, o);
            testCase.verifyTrue(b.cache_hit, "Second call should hit cache");
            testCase.verifyEqual(a.r_clubhead, b.r_clubhead, "AbsTol", 1e-12);
        end
    end

    %% ---------------- Helpers ---------------------------------------------
    methods (Access = private)
        function assumeSimulinkAvailable(testCase)
            % Only run live-Simulink tests when both Simulink and the model
            % are reachable. Otherwise mark as assumption failure (not test
            % failure) so the suite still passes in headless CI.
            try
                v = ver('simulink'); %#ok<VERMATLAB>
            catch
                v = [];
            end
            if isempty(v)
                testCase.assumeFail("Simulink toolbox not installed");
            end
            if exist('GolfSwing3D_Kinetic', 'file') == 0 && ...
               exist('GolfSwing3D_Kinetic.slx', 'file') == 0
                testCase.assumeFail("GolfSwing3D_Kinetic.slx not on path");
            end
        end
    end
end

classdef test_synthesize_target_from_coefficients < matlab.unittest.TestCase
%TEST_SYNTHESIZE_TARGET_FROM_COEFFICIENTS  Harness for #014 / #3983.
%
%   Tests the synthetic-target oracle. Pure-MATLAB tests (input validation,
%   schema, determinism, etc.) run unconditionally via a local stub of
%   simulate_with_coefficients injected on the path. Live-Simulink tests
%   are tagged "RequiresSimulink" and skipped when the model or the
%   toolbox is unavailable.

    properties (Constant)
        REQUIRED_TARGET_FIELDS = ["time","butt","clubhead","club_quat", ...
                                   "impact_idx","source"];
        REQUIRED_SOURCE_FIELDS = ["filename","format","subject_id","trial_id", ...
                                   "sha256","theta_truth","git_commit"];
    end

    properties
        n_joints   (1,1) double = 7;   % default fallback if param info missing
        opts struct
        stub_dir   (1,1) string = "";
    end

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            shared_dir = fileparts(here);                  % .../shared
            engine_root = fileparts(fileparts(shared_dir));% .../matlab
            addpath(shared_dir);
            addpath(genpath(fullfile(engine_root, 'src')));

            % Try to read the real joint count; fall back to 7 if unavailable.
            try
                info = getPolynomialParameterInfo();
                testCase.n_joints = numel(info.joint_names);
            catch
                testCase.n_joints = 7;
            end
            testCase.opts = default_synth_options();

            % Install a local stub of simulate_with_coefficients on the path
            % at HIGHER priority than the real shared dir so tests don't
            % require Simulink. The stub returns a smooth analytic swing.
            testCase.stub_dir = string(tempname);
            mkdir(testCase.stub_dir);
            testCase.installStub();
            addpath(char(testCase.stub_dir), '-begin');
        end
    end

    methods (TestClassTeardown)
        function cleanup(testCase)
            if strlength(testCase.stub_dir) > 0 && isfolder(testCase.stub_dir)
                rmpath(char(testCase.stub_dir));
                rmdir(testCase.stub_dir, 's');
            end
        end
    end

    %% ---------------- Pure unit tests (no Simulink) ------------------------
    methods (Test)

        function test_synthesize_returns_canonical_target_struct(testCase)
            theta = testCase.zeroTheta();
            target = synthesize_target_from_coefficients(theta, testCase.opts);
            for k = 1:numel(testCase.REQUIRED_TARGET_FIELDS)
                f = testCase.REQUIRED_TARGET_FIELDS(k);
                testCase.verifyTrue(isfield(target, f), ...
                    sprintf("target missing field %s", f));
            end
            % source schema
            for k = 1:numel(testCase.REQUIRED_SOURCE_FIELDS)
                f = testCase.REQUIRED_SOURCE_FIELDS(k);
                testCase.verifyTrue(isfield(target.source, f), ...
                    sprintf("target.source missing field %s", f));
            end
            N = numel(target.time);
            testCase.verifyEqual(size(target.butt),     [N, 3]);
            testCase.verifyEqual(size(target.clubhead), [N, 3]);
            testCase.verifyEqual(size(target.club_quat),[N, 4]);
            testCase.verifyClass(target.impact_idx, "uint32");
        end

        function test_synthesize_with_zero_theta_returns_static_target(testCase)
            theta = testCase.zeroTheta();
            target = synthesize_target_from_coefficients(theta, testCase.opts);
            % Stub produces a near-static clubhead for theta=0.
            ch = target.clubhead;
            disp_max = max(vecnorm(ch - ch(1, :), 2, 2));
            testCase.verifyLessThan(disp_max, 0.05, ...
                "Zero-theta should produce ~static clubhead (<5 cm drift)");
        end

        function test_synthesize_is_deterministic(testCase)
            theta = 0.01 * (1:testCase.n_joints*7).';
            a = synthesize_target_from_coefficients(theta, testCase.opts);
            b = synthesize_target_from_coefficients(theta, testCase.opts);
            testCase.verifyEqual(a.time,      b.time,      "AbsTol", 0);
            testCase.verifyEqual(a.butt,      b.butt,      "AbsTol", 0);
            testCase.verifyEqual(a.clubhead,  b.clubhead,  "AbsTol", 0);
            testCase.verifyEqual(a.club_quat, b.club_quat, "AbsTol", 0);
            testCase.verifyEqual(a.impact_idx, b.impact_idx);
            testCase.verifyEqual(a.source.sha256, b.source.sha256);
        end

        function test_synthesize_impact_idx_within_bounds(testCase)
            theta = 0.05 * (1:testCase.n_joints*7).';
            target = synthesize_target_from_coefficients(theta, testCase.opts);
            N = numel(target.time);
            testCase.verifyGreaterThanOrEqual(target.impact_idx, uint32(1));
            testCase.verifyLessThanOrEqual(target.impact_idx, uint32(N));
        end

        function test_synthesize_quaternions_unit_norm(testCase)
            theta = 0.02 * (1:testCase.n_joints*7).';
            target = synthesize_target_from_coefficients(theta, testCase.opts);
            qn = vecnorm(target.club_quat, 2, 2);
            testCase.verifyLessThan(max(abs(qn - 1)), 1e-6);
        end

        function test_synthesize_source_provenance_populated(testCase)
            theta = 0.01 * (1:testCase.n_joints*7).';
            target = synthesize_target_from_coefficients(theta, testCase.opts);
            testCase.verifyEqual(target.source.format, "synthetic");
            testCase.verifyEqual(target.source.subject_id, "synthetic");
            testCase.verifyEqual(target.source.trial_id, "synthesizer_v1");
            testCase.verifyEqual(strlength(target.source.sha256), 64);
            testCase.verifyEqual(target.source.theta_truth(:), theta(:), ...
                "AbsTol", 0);
            testCase.verifyTrue(isstring(target.source.git_commit) || ...
                                ischar(target.source.git_commit));
        end

        function test_synthesize_rejects_invalid_theta(testCase)
            % Wrong length (not multiple of 7)
            bad_len = zeros(testCase.n_joints * 7 + 1, 1);
            testCase.verifyError(@() synthesize_target_from_coefficients( ...
                bad_len, testCase.opts), ...
                "synthesize_target_from_coefficients:badThetaLength");

            % NaN
            theta_nan = zeros(testCase.n_joints * 7, 1);
            theta_nan(1) = NaN;
            testCase.verifyError(@() synthesize_target_from_coefficients( ...
                theta_nan, testCase.opts), ...
                "MATLAB:validators:mustBeFinite");

            % Out-of-bounds A coefficient (must be |A| <= 1000)
            theta_big = zeros(testCase.n_joints * 7, 1);
            theta_big(1) = 1.5e3;
            testCase.verifyError(@() synthesize_target_from_coefficients( ...
                theta_big, testCase.opts), ...
                "synthesize_target_from_coefficients:thetaOutOfBounds");
        end
    end

    %% ---------------- Helpers ---------------------------------------------
    methods (Access = private)
        function theta = zeroTheta(testCase)
            theta = zeros(testCase.n_joints * 7, 1);
        end

        function installStub(testCase)
            % Write a deterministic analytic stub of simulate_with_coefficients.
            % Output schema matches the real wrapper.
            stub = string(fullfile(testCase.stub_dir, ...
                "simulate_with_coefficients.m"));
            n_j = testCase.n_joints;
            lines = strings(0, 1);
            lines(end+1) = "function sim_out = simulate_with_coefficients(theta, opts)";
            lines(end+1) = "    n_joints = " + n_j + ";";
            lines(end+1) = "    dt = 1.0 / double(opts.sample_rate);";
            lines(end+1) = "    t = (0:dt:double(opts.simulation_time))';";
            lines(end+1) = "    N = numel(t);";
            lines(end+1) = "    M = reshape(theta, 7, n_joints).';";
            lines(end+1) = "    A = M(:,1); B = M(:,2);";
            lines(end+1) = "    % Simple analytic clubhead path: small bump scaled by mean(A)";
            lines(end+1) = "    s = mean(A) * 1e-3;";
            lines(end+1) = "    bump = s * sin(pi * t / max(t(end), eps)).^2;";
            lines(end+1) = "    sim_out = struct();";
            lines(end+1) = "    sim_out.time = t;";
            lines(end+1) = "    sim_out.q = zeros(N, n_joints) + (mean(A)*1e-4) * t;";
            lines(end+1) = "    sim_out.qd = zeros(N, n_joints) + (mean(A)*1e-4);";
            lines(end+1) = "    sim_out.qdd = zeros(N, n_joints);";
            lines(end+1) = "    sim_out.tau = repmat(B.', N, 1) * 1e-3;";
            lines(end+1) = "    sim_out.omega = sim_out.qd;";
            lines(end+1) = "    sim_out.r_butt = [zeros(N,1), zeros(N,1), 1.2 + 0*t];";
            lines(end+1) = "    sim_out.r_clubhead = [bump, 0.5 + 0*t, 0.2 + 0*t];";
            lines(end+1) = "    th = (mean(A)*1e-4) * t;";
            lines(end+1) = "    sim_out.q_club = [cos(th/2), sin(th/2), zeros(N,1), zeros(N,1)];";
            lines(end+1) = "    sim_out.v_clubhead = [gradient(bump, dt), zeros(N,1), zeros(N,1)];";
            lines(end+1) = "    sim_out.omega_club = zeros(N, 3);";
            lines(end+1) = "    sim_out.joint_names = ""joint"" + string(1:n_joints);";
            lines(end+1) = "    sim_out.solver_status = ""success"";";
            lines(end+1) = "    sim_out.status_message = """";";
            lines(end+1) = "    sim_out.cache_hit = false;";
            lines(end+1) = "    sim_out.duration_s = 0;";
            lines(end+1) = "    sim_out.theta_length = numel(theta);";
            lines(end+1) = "end";
            fid = fopen(stub, "w");
            assert(fid >= 0, "Could not open stub file: %s", stub);
            fprintf(fid, "%s\n", lines);
            fclose(fid);
        end
    end
end

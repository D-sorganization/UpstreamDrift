classdef test_solve_starting_pose < matlab.unittest.TestCase
%TEST_SOLVE_STARTING_POSE  matlab.unittest harness for issue #4072.
%
%   Covers the Stage-1 starting-pose solver.  Pure-MATLAB tests run
%   everywhere; the live-Simulink tests are tagged "RequiresSimulink" and
%   skip cleanly when the toolbox or model are unavailable so unit-only
%   CI keeps passing.

    properties
        opts          struct
        base_input    char
        target_stub   struct
        default_vars  cell
    end

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            shared_dir  = fileparts(here);
            engine_root = fileparts(fileparts(shared_dir));
            addpath(shared_dir);
            addpath(genpath(fullfile(engine_root, 'src')));

            % Locate the Impact MAT used as Stage-1 base.  Skip the whole
            % suite if it isn't present (pure-checkout dev env).
            candidate = fullfile(engine_root, 'src', 'model', 'inputs', ...
                                  '3DModelInputs_Impact.mat');
            if ~isfile(candidate)
                testCase.assumeFail(sprintf( ...
                    "Impact MAT not found at %s — required for Stage-1 tests", ...
                    candidate));
            end
            testCase.base_input = char(candidate);

            % Stub target with the canonical fields the solver consumes.
            testCase.target_stub = local_make_stub_target();

            testCase.default_vars = { ...
                'TranslationStartPositionX', ...
                'TranslationStartPositionY', ...
                'TranslationStartPositionZ', ...
                'HipStartPositionZ',         ...
                'LSStartPositionY',          ...
                'RSStartPositionY',          ...
                'LEStartPosition',           ...
                'REStartPosition'};

            testCase.opts = struct( ...
                'verbose',   false, ...
                'rng_seed',  42, ...
                'max_iters', 8, ...        % small for unit tests
                'stop_time', 0.005);
        end
    end

    %% ---------------- Pure unit tests (no Simulink required) ----------------
    methods (Test)

        function test_target_validation_rejects_missing_grip(testCase)
            bad = struct('events', struct('A_sample', 1));
            testCase.verifyError( ...
                @() solve_starting_pose(bad, string(testCase.base_input), testCase.opts), ...
                "solve_starting_pose:badTarget");
        end

        function test_target_validation_rejects_missing_A_sample(testCase)
            bad = testCase.target_stub;
            bad.events = struct();   % drop A_sample
            testCase.verifyError( ...
                @() solve_starting_pose(bad, string(testCase.base_input), testCase.opts), ...
                "solve_starting_pose:noAddressFrame");
        end

        function test_base_input_must_be_a_file(testCase)
            o = testCase.opts;
            % mustBeFile precondition surfaces as MATLAB:validators:mustBeFile
            testCase.verifyError( ...
                @() solve_starting_pose(testCase.target_stub, "not_a_real_file.mat", o), ...
                "MATLAB:validators:mustBeFile");
        end

        function test_bounds_must_match_vars_count(testCase)
            o = testCase.opts;
            o.vars = testCase.default_vars;
            o.lb   = zeros(3, 1);    % wrong length
            testCase.verifyError( ...
                @() solve_starting_pose(testCase.target_stub, ...
                                        string(testCase.base_input), o), ...
                "solve_starting_pose:badBounds");
        end

        function test_x0_must_match_vars_count(testCase)
            o = testCase.opts;
            o.vars = testCase.default_vars;
            o.x0   = zeros(3, 1);   % wrong length (default vars list is 8)
            testCase.verifyError( ...
                @() solve_starting_pose(testCase.target_stub, ...
                                        string(testCase.base_input), o), ...
                "solve_starting_pose:badX0");
        end
    end

    %% ---------------- Tests requiring Simulink + the .slx ------------------
    methods (Test, TestTags = {'RequiresSimulink'})

        function test_tdd_oracle_recovers_known_overrides(testCase)
            % Synthesize a target by running the model with KNOWN
            % perturbations; then solve and verify recovery within tol.
            testCase.assumeSimulinkAvailable();

            % Ground-truth perturbations: small but non-trivial.
            true_x = [0.02; -0.01; 0.015; 0.05; 0.03; -0.04; 0.02; -0.03];

            target = local_synthesize_target_at_pose( ...
                testCase.base_input, testCase.default_vars, true_x);

            o = testCase.opts;
            o.vars      = testCase.default_vars;
            o.max_iters = 200;
            o.rng_seed  = 42;
            o.verbose   = false;

            overrides = solve_starting_pose(target, string(testCase.base_input), o);

            % Recovery: each component within 5 cm / 5 deg, and the
            % position residual at the recovered pose < 5 mm.
            recovered = zeros(numel(testCase.default_vars), 1);
            for k = 1:numel(testCase.default_vars)
                recovered(k) = overrides.(testCase.default_vars{k});
            end

            % Re-simulate at the recovered overrides and compare grip pos.
            grip_recovered = local_simulate_grip_at_pose( ...
                testCase.base_input, testCase.default_vars, recovered);
            grip_target = target.grip(target.events.A_sample, :);
            residual_mm = 1000 * norm(grip_recovered - grip_target);

            testCase.verifyLessThan(residual_mm, 5.0, ...
                sprintf("Stage-1 grip residual %.2f mm exceeds 5 mm tol", residual_mm));
        end

        function test_bounded_deviation_no_override_outside_envelope(testCase)
            testCase.assumeSimulinkAvailable();

            target = local_synthesize_target_at_pose( ...
                testCase.base_input, testCase.default_vars, ...
                zeros(numel(testCase.default_vars), 1));

            o = testCase.opts;
            o.vars      = testCase.default_vars;
            o.max_iters = 50;
            o.verbose   = false;

            overrides = solve_starting_pose(target, string(testCase.base_input), o);

            % Translations: ±2 m. Angles: ±pi.
            for k = 1:numel(testCase.default_vars)
                name = testCase.default_vars{k};
                v    = overrides.(name);
                if startsWith(name, 'TranslationStartPosition')
                    testCase.verifyGreaterThanOrEqual(v, -2.0);
                    testCase.verifyLessThanOrEqual(v,    2.0);
                else
                    testCase.verifyGreaterThanOrEqual(v, -pi);
                    testCase.verifyLessThanOrEqual(v,    pi);
                end
            end
        end

        function test_determinism_identical_seed_identical_result(testCase)
            testCase.assumeSimulinkAvailable();

            target = local_synthesize_target_at_pose( ...
                testCase.base_input, testCase.default_vars, ...
                zeros(numel(testCase.default_vars), 1));

            o = testCase.opts;
            o.vars      = testCase.default_vars;
            o.max_iters = 30;
            o.rng_seed  = 1234;
            o.verbose   = false;

            a = solve_starting_pose(target, string(testCase.base_input), o);
            b = solve_starting_pose(target, string(testCase.base_input), o);

            for k = 1:numel(testCase.default_vars)
                name = testCase.default_vars{k};
                testCase.verifyEqual(a.(name), b.(name), ...
                    sprintf("Determinism: %s differs between runs (%g vs %g)", ...
                            name, a.(name), b.(name)));
            end
        end
    end

    %% ---------------- Helpers ---------------------------------------------
    methods (Access = private)
        function assumeSimulinkAvailable(testCase)
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


%% =====================================================================
function target = local_make_stub_target()
%LOCAL_MAKE_STUB_TARGET  Minimal target struct for *unit* (no-Simulink) tests.
%   The unit tests only exercise the precondition gates, so the contents
%   beyond the documented fields don't matter.
    N = 5;
    target = struct( ...
        'time',      (0:N-1).' / 240, ...
        'grip',      repmat([0.10 0.20 0.95], N, 1), ...
        'grip_quat', repmat([1 0 0 0], N, 1), ...
        'butt',      repmat([0.10 0.20 0.95], N, 1), ...
        'clubhead',  repmat([0.10 0.20 0.05], N, 1), ...
        'club_quat', repmat([1 0 0 0], N, 1), ...
        'impact_idx', 3, ...
        'events',    struct('A_sample', 1, 'T_sample', 2, ...
                            'I_sample', 3, 'F_sample', 4, 'CHS_mph', 100));
end


%% =====================================================================
function target = local_synthesize_target_at_pose(base_input_mat, vars, x_true)
%LOCAL_SYNTHESIZE_TARGET_AT_POSE  Run the model with known perturbations,
%   build a target struct whose grip(A_sample) == the model's grip(0).
    grip = local_simulate_grip_at_pose(base_input_mat, vars, x_true);
    R    = local_simulate_grip_R_at_pose(base_input_mat, vars, x_true);
    q    = local_R_to_quat(R);

    N = 3;
    target = struct( ...
        'time',       (0:N-1).' / 240, ...
        'grip',       repmat(grip, N, 1), ...
        'grip_quat',  repmat(q,    N, 1), ...
        'butt',       repmat(grip, N, 1), ...
        'clubhead',   repmat(grip, N, 1), ...
        'club_quat',  repmat([1 0 0 0], N, 1), ...
        'impact_idx', 1, ...
        'events',     struct('A_sample', 1, 'T_sample', 2, ...
                             'I_sample', 3, 'F_sample', 3, 'CHS_mph', 100));
end


%% =====================================================================
function grip = local_simulate_grip_at_pose(base_input_mat, vars, x)
%LOCAL_SIMULATE_GRIP_AT_POSE  Helper — same inner sim solve_starting_pose runs.
    base = load(base_input_mat);
    overrides = struct();
    f = fieldnames(base);
    for k = 1:numel(f); overrides.(f{k}) = base.(f{k}); end
    for k = 1:numel(vars)
        if isfield(overrides, vars{k})
            overrides.(vars{k}) = overrides.(vars{k}) + x(k);
        else
            overrides.(vars{k}) = x(k);
        end
    end
    in = prepare_fast_sim_input([], struct( ...
        'model_name',      'GolfSwing3D_Kinetic', ...
        'stop_time',       0.005, ...
        'simscape_log',    'all', ...
        'input_overrides', overrides));
    simOut = sim(in);
    d = double(simOut.CombinedSignalBus.MidpointCalcsLogs.MPGlobalPosition.Data);
    grip = reshape(d(1, 1:3), 1, 3);
end


%% =====================================================================
function R = local_simulate_grip_R_at_pose(base_input_mat, vars, x)
    base = load(base_input_mat);
    overrides = struct();
    f = fieldnames(base);
    for k = 1:numel(f); overrides.(f{k}) = base.(f{k}); end
    for k = 1:numel(vars)
        if isfield(overrides, vars{k})
            overrides.(vars{k}) = overrides.(vars{k}) + x(k);
        else
            overrides.(vars{k}) = x(k);
        end
    end
    in = prepare_fast_sim_input([], struct( ...
        'model_name',      'GolfSwing3D_Kinetic', ...
        'stop_time',       0.005, ...
        'simscape_log',    'all', ...
        'input_overrides', overrides));
    simOut = sim(in);
    R = nan(3,3);
    try
        d = double(simOut.CombinedSignalBus.MomentandCoupleLogs.RotationTransformMP.Data);
        if ndims(d) == 3 && size(d,1) == 3 && size(d,2) == 3
            R = squeeze(d(:, :, 1));
        end
    catch
    end
    if any(isnan(R(:)))
        R = eye(3);
    end
end


%% =====================================================================
function q = local_R_to_quat(R)
    tr = R(1,1) + R(2,2) + R(3,3);
    if tr > 0
        S = 2 * sqrt(tr + 1);
        w = 0.25 * S;
        x = (R(3,2) - R(2,3)) / S;
        y = (R(1,3) - R(3,1)) / S;
        z = (R(2,1) - R(1,2)) / S;
    elseif (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
        S = 2 * sqrt(1 + R(1,1) - R(2,2) - R(3,3));
        w = (R(3,2) - R(2,3)) / S;  x = 0.25*S;
        y = (R(1,2) + R(2,1)) / S;  z = (R(1,3) + R(3,1)) / S;
    elseif R(2,2) > R(3,3)
        S = 2 * sqrt(1 + R(2,2) - R(1,1) - R(3,3));
        w = (R(1,3) - R(3,1)) / S;  x = (R(1,2) + R(2,1)) / S;
        y = 0.25*S;                 z = (R(2,3) + R(3,2)) / S;
    else
        S = 2 * sqrt(1 + R(3,3) - R(1,1) - R(2,2));
        w = (R(2,1) - R(1,2)) / S;  x = (R(1,3) + R(3,1)) / S;
        y = (R(2,3) + R(3,2)) / S;  z = 0.25*S;
    end
    q = [w x y z];
    q = q / max(norm(q), eps);
    if q(1) < 0; q = -q; end
end

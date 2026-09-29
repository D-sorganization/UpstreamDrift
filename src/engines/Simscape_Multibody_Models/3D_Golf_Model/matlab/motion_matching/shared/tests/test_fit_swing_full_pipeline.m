classdef test_fit_swing_full_pipeline < matlab.unittest.TestCase
%TEST_FIT_SWING_FULL_PIPELINE  Tests for the single-call M1 driver.
%
%   Mirrors the gating pattern from test_load_club_target_excel.m: tests
%   that need MATLAB's Simulink toolbox or the Wiffle xlsx call
%   testCase.assumeFail when those resources are absent so unit-only CI
%   keeps passing.
%
%   GitHub issue: #4083.

    properties
        save_root  char
    end

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            shared_dir = fileparts(here);
            engine_root = fileparts(fileparts(shared_dir));
            addpath(shared_dir);
            addpath(genpath(fullfile(engine_root, 'motion_matching', ...
                                     'option1_direct_optimization')));
            testCase.addTeardown(@() rmpath(shared_dir));

            % Tests scribble into a unique tempdir so we don't pollute
            % the repo's output/ tree.
            testCase.save_root = tempname;
            mkdir(testCase.save_root);
            testCase.addTeardown(@() local_safe_rmdir(testCase.save_root));
        end
    end

    methods (Test)

        %% ---------- Pure unit tests (no Simulink required) ------------

        function test_unknown_optimizer_rejected(testCase)
            target = local_make_stub_target();
            opts = struct('option', 'totally_made_up_solver', ...
                          'render_figures', false, ...
                          'skip_stage1', true, ...
                          'verbose', false, ...
                          'save_dir', fullfile(testCase.save_root, "unknown"));
            testCase.verifyError( ...
                @() fit_swing_full_pipeline(target, opts), ...
                "fit_swing_full_pipeline:unknownOption");
        end

        function test_target_struct_missing_required_fields(testCase)
            % Strip 'grip_quat' from a stub; resolver should raise.
            bad_target = local_make_stub_target();
            bad_target = rmfield(bad_target, 'grip_quat');
            opts = struct('render_figures', false, ...
                          'skip_stage1', true, ...
                          'verbose', false, ...
                          'save_dir', fullfile(testCase.save_root, "missing_field"));
            testCase.verifyError( ...
                @() fit_swing_full_pipeline(bad_target, opts), ...
                "fit_swing_full_pipeline:badTarget");
        end

        function test_xlsx_path_must_exist(testCase)
            opts = struct('render_figures', false, ...
                          'skip_stage1', true, ...
                          'verbose', false, ...
                          'save_dir', fullfile(testCase.save_root, "no_xlsx"));
            testCase.verifyError( ...
                @() fit_swing_full_pipeline("nonexistent_file.xlsx", opts), ...
                "fit_swing_full_pipeline:targetNotFound");
        end

        function test_stage1_opts_must_be_struct(testCase)
            target = local_make_stub_target();
            opts = struct('stage1_opts', 'not a struct', ...
                          'render_figures', false, ...
                          'skip_stage1', true, ...
                          'verbose', false, ...
                          'save_dir', fullfile(testCase.save_root, "bad_stage1"));
            testCase.verifyError( ...
                @() fit_swing_full_pipeline(target, opts), ...
                "fit_swing_full_pipeline:badStage1Opts");
        end

        function test_dispatch_uses_named_optimizer(testCase)
            % Inject a fake fit_swing_<option> via opts trick: we can't
            % easily monkeypatch in MATLAB, so this test verifies that
            % the dispatcher accepts each documented option name without
            % rejecting it during precondition validation.  The actual
            % dispatched function will then attempt to run; we stop
            % before that by setting skip_stage1=true and intercepting
            % via verifyError on the *first* expected failure (Simulink
            % missing) — but unit-only CI doesn't have that, so we
            % verify the precondition gate accepts the names.
            for option = ["fmincon", "multistart", "surrogate", ...
                          "surrogateopt", "hybrid"]
                target = local_make_stub_target();
                opts = struct('option', option, ...
                              'render_figures', false, ...
                              'skip_stage1', true, ...
                              'verbose', false, ...
                              'save_dir', fullfile(testCase.save_root, ...
                                                   "dispatch_" + option));
                % We expect the dispatch *itself* to succeed (i.e., not
                % error out with unknownOption). Whatever happens
                % downstream — Simulink missing, etc. — is not an
                % unknownOption error.
                err = local_capture_error(@() ...
                    fit_swing_full_pipeline(target, opts));
                if ~isempty(err)
                    testCase.verifyNotEqual(string(err.identifier), ...
                        "fit_swing_full_pipeline:unknownOption", ...
                        sprintf("Option '%s' unexpectedly hit unknownOption", option));
                end
            end
        end

        %% ---------- Live tests (require Simulink + model) -------------

        function test_returns_augmented_result_struct(testCase)
            % End-to-end: target -> Stage-1 -> Stage-2 -> figures ->
            % report.  Verifies the augmented result struct contract.
            testCase.assumeSimulinkAvailable();
            xlsx = locate_xlsx_or_skip(testCase);
            input_mat = locate_input_mat_or_skip(testCase);

            save_dir = fullfile(testCase.save_root, "e2e");
            opts = struct( ...
                'sheet',          "TW_ProV1", ...
                'option',         "fmincon", ...
                'save_dir',       save_dir, ...
                'render_figures', true, ...
                'save_animation', false, ...
                'input_mat',      input_mat, ...
                'verbose',        false, ...
                'stage1_opts',    struct('max_iters', 4, 'verbose', false), ...
                'stage2_opts',    struct('max_iter',  uint32(2), ...
                                         'max_function_evals', uint32(8), ...
                                         'display', "off"));

            result = fit_swing_full_pipeline(xlsx, opts);

            for f = ["stage1_overrides", "figure_paths", "report_path", ...
                     "save_dir", "target", "solver", "final_rmse_m"]
                testCase.verifyTrue(isfield(result, f), ...
                    sprintf("result missing field: %s", f));
            end
            testCase.verifyTrue(iscell(result.figure_paths));
            testCase.verifyTrue(isstruct(result.stage1_overrides));
            testCase.verifyTrue(isfile(char(result.report_path)));
            testCase.verifyEqual(string(result.solver), "fmincon");
            testCase.verifyTrue(isfolder(char(result.save_dir)));
        end
    end

    %% --------------------- Helpers ------------------------------------
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
function err = local_capture_error(fn)
%LOCAL_CAPTURE_ERROR  Run fn(); return the MException it threw, or [] if
%   it didn't throw.
    err = [];
    try
        fn();
    catch ME
        err = ME;
    end
end


%% =====================================================================
function target = local_make_stub_target()
%LOCAL_MAKE_STUB_TARGET  Minimal canonical target struct for unit tests.
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
                            'I_sample', 3, 'F_sample', 4, 'CHS_mph', 100), ...
        'source',    struct('filename', "stub.xlsx", 'format', "stub", ...
                            'subject_id', "stub", 'trial_id', "stub", ...
                            'sha256', repmat('0', 1, 64)));
end


%% =====================================================================
function xlsx = locate_xlsx_or_skip(testCase)
    here = fileparts(mfilename("fullpath"));
    candidate = fullfile(here, ...
        "..", "..", "..", "src", "apps", "golf_gui", ...
        "Motion Capture Plotter", "Wiffle_ProV1_club_3D_data.xlsx");
    if exist(candidate, "file") ~= 2
        testCase.assumeFail( ...
            "skipped — Wiffle_ProV1_club_3D_data.xlsx not present at " + ...
            string(candidate));
    end
    xlsx = string(candidate);
end


%% =====================================================================
function p = locate_input_mat_or_skip(testCase)
    here = fileparts(mfilename("fullpath"));
    shared_dir = fileparts(here);
    engine_root = fileparts(fileparts(shared_dir));
    candidate = fullfile(engine_root, "src", "model", "inputs", ...
                         "3DModelInputs_Impact.mat");
    if ~isfile(candidate)
        testCase.assumeFail( ...
            "skipped — 3DModelInputs_Impact.mat not present at " + ...
            string(candidate));
    end
    p = char(candidate);
end


%% =====================================================================
function local_safe_rmdir(p)
    try
        if isfolder(p)
            rmdir(p, 's');
        end
    catch
    end
end

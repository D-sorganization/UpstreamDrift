classdef test_smoke_fit_wiffle_proV1 < matlab.unittest.TestCase
%TEST_SMOKE_FIT_WIFFLE_PROV1  End-to-end CI smoke fit against measured data.
%
%   Asserts the production-grade success criterion from PROJECT_SPEC.md §2:
%   grip-position RMSE across the 0.30 s impact window < 5 mm. This is the
%   regression gate that catches silent breakage in compute_cost,
%   fit_swing_fmincon, the loaders, or the model itself.
%
%   Tagged ``IsSlow`` because a 50-iter fmincon over the impact window runs
%   for several minutes wall-clock. The MaxIterations cap depends on the
%   Stage-1 starting pose (issue #4072) being a good warm start.
%
%   GitHub issue: #4073.

    properties (Constant)
        % tests/ -> option1_direct_optimization/ -> motion_matching/ -> matlab/ -> 3D_Golf_Model/
        XLSX_RELPATH = fullfile("..", "..", "..", "src", "apps", "golf_gui", ...
            "Motion Capture Plotter", "Wiffle_ProV1_club_3D_data.xlsx");
        IMPACT_MAT_RELPATH = fullfile("..", "..", "..", "src", "model", ...
            "inputs", "3DModelInputs_Impact.mat");
        MAX_GRIP_RMSE_MM = 5.0;
    end

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename("fullpath"));
            opt1 = fileparts(here);
            mm   = fileparts(opt1);
            shared = fullfile(mm, "shared");
            addpath(opt1);
            addpath(shared);
            testCase.addTeardown(@() rmpath(opt1));
            testCase.addTeardown(@() rmpath(shared));
        end
    end

    methods (Test, TestTags = {'IsSlow'})
        function test_wiffle_ProV1_grip_rmse_below_5mm(testCase)
            xlsx = locate_relpath(testCase, testCase.XLSX_RELPATH, ...
                "Wiffle_ProV1_club_3D_data.xlsx");
            impact_mat = locate_relpath(testCase, testCase.IMPACT_MAT_RELPATH, ...
                "3DModelInputs_Impact.mat");
            testCase.assumeTrue(exist("GolfSwing3D_Kinetic", "file") ~= 0, ...
                "Simulink model GolfSwing3D_Kinetic not available in test environment");
            testCase.assumeTrue(exist("fit_swing_full_pipeline", "file") ~= 0, ...
                "fit_swing_full_pipeline not on path (issue #4083)");

            opts = struct();
            opts.sheet = "TW_ProV1";
            opts.option = "fmincon";
            opts.input_mat = impact_mat;
            opts.render_figures = false;
            opts.save_animation = false;
            opts.stage2_opts = struct();
            opts.stage2_opts.fmincon = struct();
            opts.stage2_opts.fmincon.MaxIterations = 50;

            result = fit_swing_full_pipeline(xlsx, opts);

            grip_rmse_mm = 1000 * result.grip_rmse_m;

            % Print the per-component cost breakdown so any regression is
            % debuggable from the CI log alone (acceptance criterion 3).
            fprintf("smoke fit grip RMSE: %.3f mm (limit %.1f mm)\n", ...
                grip_rmse_mm, testCase.MAX_GRIP_RMSE_MM);
            fprintf("optimizer: %s, MaxIterations: %d, duration: %.1f s\n", ...
                result.solver, opts.stage2_opts.fmincon.MaxIterations, ...
                result.duration_s);
            if isfield(result, "final_cost_terms")
                terms = result.final_cost_terms;
                term_names = fieldnames(terms);
                for i = 1:numel(term_names)
                    fprintf("  terms.%s = %.6g\n", term_names{i}, terms.(term_names{i}));
                end
            end

            testCase.verifyLessThan(grip_rmse_mm, testCase.MAX_GRIP_RMSE_MM);
        end
    end
end


function p = locate_relpath(testCase, relpath, missing_label)
%LOCATE_RELPATH  Resolve a test-relative path or skip with a clear message.
    here = fileparts(mfilename("fullpath"));
    candidate = fullfile(here, relpath);
    if exist(candidate, "file") ~= 2
        testCase.assumeFail(sprintf( ...
            "skipped — %s not present at %s", missing_label, candidate));
    end
    p = string(candidate);
end

classdef test_run_leaderboard < matlab.unittest.TestCase
%TEST_RUN_LEADERBOARD  Tests for scripts/run_leaderboard.m (#4080).
%
%   Exercises the contract documented in run_leaderboard.m:
%     - rejects unknown options with a clear error,
%     - skips missing optimizers gracefully (no crash),
%     - writes LEADERBOARD.md with the canonical schema header.
%
%   Tests that need Simulink / the Wiffle xlsx / a real solver are
%   gated with assumeFail so the suite stays runnable in headless CI.

    properties
        results_dir (1,1) string
    end

    methods (TestClassSetup)
        function add_paths(testCase)
            here   = fileparts(mfilename('fullpath'));
            shared = fileparts(here);
            scripts_dir = fullfile(shared, 'scripts');
            addpath(shared);
            addpath(scripts_dir);
            testCase.addTeardown(@() rmpath(scripts_dir));
            testCase.addTeardown(@() rmpath(shared));
        end
    end

    methods (TestMethodSetup)
        function make_tempdir(testCase)
            testCase.results_dir = string(tempname);
            mkdir(testCase.results_dir);
            testCase.addTeardown(@() local_rmdir(testCase.results_dir));
        end
    end

    methods (Test)
        function test_writes_leaderboard_md(testCase)
            % Dry-run mode bypasses the real fit but still emits the MD.
            summary = run_leaderboard( ...
                'Trials',     "TW_ProV1", ...
                'Options',    ["fmincon", "surrogate", "inverse", "hybrid"], ...
                'ResultsDir', testCase.results_dir, ...
                'DryRun',     true, ...
                'Verbose',    false);

            md_path = fullfile(testCase.results_dir, "LEADERBOARD.md");
            testCase.verifyTrue(isfile(md_path), ...
                "Driver must write LEADERBOARD.md");
            testCase.verifyEqual(summary.leaderboard_md, string(md_path));

            txt = string(fileread(char(md_path)));
            testCase.verifySubstring(char(txt), "swing_id");
            testCase.verifySubstring(char(txt), "TW_ProV1");
            testCase.verifySubstring(char(txt), "rmse_mm");
        end

        function test_rejects_unknown_option(testCase)
            testCase.verifyError(@() run_leaderboard( ...
                'Trials',     "TW_ProV1", ...
                'Options',    "magic-bullet", ...
                'ResultsDir', testCase.results_dir, ...
                'DryRun',     true, ...
                'Verbose',    false), ...
                "run_leaderboard:unknownOption");
        end

        function test_rejects_empty_trials(testCase)
            testCase.verifyError(@() run_leaderboard( ...
                'Trials',     string.empty(), ...
                'Options',    "fmincon", ...
                'ResultsDir', testCase.results_dir, ...
                'DryRun',     true, ...
                'Verbose',    false), ...
                "run_leaderboard:noTrials");
        end

        function test_skips_missing_optimizer_gracefully(testCase)
            % Force a trial × option pair that hits the "not on path" path
            % by passing a bogus xlsx so even fmincon can't run, while
            % asking for an option whose solver is *not* on the MATLAB
            % path. The driver should mark the cell skipped, not crash.
            tmp_xlsx = fullfile(testCase.results_dir, "does_not_exist.xlsx");

            summary = run_leaderboard( ...
                'Trials',      "TW_ProV1", ...
                'Options',     "inverse", ...
                'ResultsDir',  testCase.results_dir, ...
                'XlsxPath',    tmp_xlsx, ...
                'DryRun',      false, ...
                'SkipMissing', true, ...
                'Verbose',     false);

            md_path = fullfile(testCase.results_dir, "LEADERBOARD.md");
            testCase.verifyTrue(isfile(md_path));
            testCase.verifyEqual(numel(summary.rows), 1);
            % The row must be a "pending" placeholder, not a real fit.
            testCase.verifyNotEqual(summary.rows(1).status, "ok");
        end

        function test_grid_cell_renders_pending_for_missing_pairs(testCase)
            summary = run_leaderboard( ...
                'Trials',     "TW_ProV1", ...
                'Options',    "fmincon", ...
                'ResultsDir', testCase.results_dir, ...
                'DryRun',     true, ...
                'Verbose',    false);
            txt = string(fileread(char(summary.leaderboard_md)));
            % In dry-run we expect the cross-option grid to mark the
            % single cell as "not run".
            testCase.verifySubstring(char(txt), "not run");
        end

        function test_real_fit_smoke(testCase)
            % Real Stage-2 fit requires Simulink + xlsx; skip when missing.
            if ~exist("fit_swing_full_pipeline", "file")
                testCase.assumeFail("fit_swing_full_pipeline not on path");
            end
            if ~license("test", "Simulink")
                testCase.assumeFail("Simulink license unavailable");
            end
            here = fileparts(mfilename("fullpath"));
            engine_root = fileparts(fileparts(fileparts(fileparts(fileparts(here)))));
            xlsx = fullfile(engine_root, "matlab", "src", "apps", ...
                "golf_gui", "Motion Capture Plotter", ...
                "Wiffle_ProV1_club_3D_data.xlsx");
            if ~isfile(xlsx)
                testCase.assumeFail("Wiffle xlsx not found at expected path");
            end
            summary = run_leaderboard( ...
                'Trials',     "TW_ProV1", ...
                'Options',    "fmincon", ...
                'XlsxPath',   xlsx, ...
                'ResultsDir', testCase.results_dir, ...
                'DryRun',     false, ...
                'Verbose',    false);
            testCase.verifyTrue(isfile(summary.leaderboard_md));
        end
    end
end


function local_rmdir(p)
    if isfolder(p)
        rmdir(p, 's');
    end
end

classdef test_leaderboard < matlab.unittest.TestCase
%TEST_LEADERBOARD  Tests for cross-option leaderboard comparison (#3992).
%
%   Synthesises minimal-but-valid result structs into a tempdir and
%   exercises the contract documented in
%   VISUALIZATION_SPEC.md §"Comparison across options".

    properties
        results_dir (1,1) string
    end

    methods (TestClassSetup)
        function add_paths(testCase)
            here   = fileparts(mfilename('fullpath'));
            shared = fileparts(here);
            addpath(shared);
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
        % --- canonical issue tests ----------------------------------------

        function test_empty_results_dir_returns_empty_table_with_correct_schema(testCase)
            tbl = leaderboard(testCase.results_dir);
            testCase.verifyEqual(height(tbl), 0);
            testCase.verifyEqual(tbl.Properties.VariableNames, ...
                {'swing_id','option','solver','rmse_mm','work_J','wall_s', ...
                 'commit','timestamp'});
        end

        function test_loads_multiple_results_and_orders_by_rmse(testCase)
            local_write_result(testCase.results_dir, "TW_ProV1", 1, "fmincon",      0.0050);
            local_write_result(testCase.results_dir, "TW_ProV1", 2, "nn-surrogate", 0.0023);
            local_write_result(testCase.results_dir, "TW_ProV1", 3, "inverse-cvae", 0.0070);

            tbl = leaderboard(testCase.results_dir);
            testCase.verifyEqual(height(tbl), 3);
            testCase.verifyTrue(issorted(tbl.rmse_mm));
            testCase.verifyEqual(tbl.rmse_mm(1), 2.3, 'AbsTol', 1e-9);
        end

        function test_filter_swing_id_narrows_rows(testCase)
            local_write_result(testCase.results_dir, "TW_ProV1", 1, "fmincon", 0.0050);
            local_write_result(testCase.results_dir, "MI_TP5",   2, "nn",      0.0030);
            local_write_result(testCase.results_dir, "TW_ProV1", 4, "bridge",  0.0040);

            opts = default_leaderboard_options();
            opts.filter_swing_id = "TW_ProV1";
            tbl = leaderboard(testCase.results_dir, opts);
            testCase.verifyEqual(height(tbl), 2);
            testCase.verifyTrue(all(tbl.swing_id == "TW_ProV1"));
        end

        function test_markdown_format_produces_pipe_table(testCase)
            local_write_result(testCase.results_dir, "TW_ProV1", 1, "fmincon", 0.0050);
            opts = default_leaderboard_options();
            opts.format = "markdown";
            [~, md] = leaderboard(testCase.results_dir, opts);
            testCase.verifyClass(md, 'string');
            testCase.verifySubstring(char(md), "| swing_id |");
            testCase.verifySubstring(char(md), "| --- |");
            testCase.verifySubstring(char(md), "TW_ProV1");
        end

        function test_csv_format_round_trips_via_readtable(testCase)
            local_write_result(testCase.results_dir, "TW_ProV1", 1, "fmincon", 0.0050);
            local_write_result(testCase.results_dir, "TW_ProV1", 2, "nn",      0.0030);

            opts = default_leaderboard_options();
            opts.write_csv = true;
            opts.csv_path  = fullfile(testCase.results_dir, "leaderboard.csv");
            tbl_out = leaderboard(testCase.results_dir, opts);
            testCase.assertTrue(isfile(opts.csv_path));

            tbl_in = readtable(opts.csv_path, 'TextType', 'string');
            testCase.verifyEqual(height(tbl_in), height(tbl_out));
            testCase.verifyEqual( ...
                sort(string(tbl_in.Properties.VariableNames)), ...
                sort(string(tbl_out.Properties.VariableNames)));
        end

        function test_handles_missing_optional_fields_gracefully(testCase)
            % swing_id and option are optional — the rest can be derived
            % from filename; the row must still appear.
            local_write_result_no_optional(testCase.results_dir, ...
                "fmincon", 0.0050);
            tbl = leaderboard(testCase.results_dir);
            testCase.verifyEqual(height(tbl), 1);
            testCase.verifyEqual(tbl.solver(1), "fmincon");
        end

        % --- additional spec tests from issue #3992 -----------------------

        function test_columns_are_swing_id_option_solver_rmse_mm_work_J_wall_s_commit(testCase)
            local_write_result(testCase.results_dir, "TW_ProV1", 1, "fmincon", 0.0050);
            tbl = leaderboard(testCase.results_dir);
            cols = string(tbl.Properties.VariableNames);
            for required = ["swing_id","option","solver","rmse_mm","work_J","wall_s","commit"]
                testCase.verifyTrue(any(cols == required), ...
                    sprintf("Missing column: %s", required));
            end
        end

        function test_skips_result_structs_missing_provenance_fields_with_warning(testCase)
            local_write_result(testCase.results_dir, "TW_ProV1", 1, "fmincon", 0.0050);
            local_write_malformed(testCase.results_dir, "bad_result.mat");

            warned = false;
            try
                lastwarn('');
                tbl = leaderboard(testCase.results_dir);
                [msg, id] = lastwarn;
                warned = contains(string(id), "leaderboard:") || strlength(msg) > 0;
            catch err
                testCase.verifyFail(['Should not error: ', err.message]);
            end
            testCase.verifyEqual(height(tbl), 1, ...
                'Malformed result must be skipped, not crash the scan.');
            testCase.verifyTrue(warned, 'Expected a warning for malformed result.');
        end

        function test_converts_final_rmse_m_to_rmse_mm_in_table(testCase)
            local_write_result(testCase.results_dir, "TW_ProV1", 1, "fmincon", 0.0023);
            tbl = leaderboard(testCase.results_dir);
            testCase.verifyEqual(tbl.rmse_mm(1), 2.3, 'AbsTol', 1e-9);
        end
    end
end

% =========================================================================
% Local helpers (TDD synthetic-fixture builders)
% =========================================================================

function local_write_result(root, swing_id, option, solver, rmse_m)
    swing_dir = fullfile(root, swing_id);
    if ~isfolder(swing_dir); mkdir(swing_dir); end
    result = local_make_result(swing_id, option, solver, rmse_m); %#ok<NASGU>
    name = sprintf("option%d.mat", option);
    save(fullfile(swing_dir, name), 'result');
end

function local_write_result_no_optional(root, solver, rmse_m)
    % Place the file directly in root; no swing_id/option in struct.
    result = local_make_result("", 0, solver, rmse_m);
    result = rmfield(result, 'swing_id');
    result = rmfield(result, 'option'); %#ok<NASGU>
    save(fullfile(root, "loose_result.mat"), 'result');
end

function local_write_malformed(root, fname)
    result = struct('solver', 'fmincon');  %#ok<NASGU>  % missing all provenance
    save(fullfile(root, fname), 'result');
end

function r = local_make_result(swing_id, option, solver, rmse_m)
    r = struct();
    r.swing_id           = string(swing_id);
    r.option             = double(option);
    r.coefficients       = zeros(7, 7);
    r.final_rmse_m       = double(rmse_m);
    r.final_total_work_J = 285.0;
    r.solver             = string(solver);
    r.solver_options     = struct('iterations', 100);
    r.target_hash        = "abcdef0123456789aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    r.git_commit         = "deadbeefcafebabe0000";
    r.matlab_version     = string(version);
    r.duration_s         = 12.5;
    r.timestamp_utc      = "2026-05-05T00:00:00Z";
end

function local_rmdir(p)
    if isfolder(p)
        rmdir(p, 's');
    end
end

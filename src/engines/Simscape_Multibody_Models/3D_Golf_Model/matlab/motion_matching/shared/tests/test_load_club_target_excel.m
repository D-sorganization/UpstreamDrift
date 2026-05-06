classdef test_load_club_target_excel < matlab.unittest.TestCase
%TEST_LOAD_CLUB_TARGET_EXCEL  Excel loader tests (gated on real file presence).

    properties (Constant)
        XLSX_RELPATH = fullfile("..", "..", "src", "apps", "golf_gui", ...
            "Motion Capture Plotter", "Wiffle_ProV1_club_3D_data.xlsx");
    end

    methods (TestClassSetup)
        function add_shared_to_path(testCase)
            here = fileparts(mfilename("fullpath"));
            shared_dir = fullfile(here, "..");
            addpath(shared_dir);
            testCase.addTeardown(@() rmpath(shared_dir));
        end
    end

    methods (Test)
        function test_loads_TW_ProV1_sheet_to_canonical_target(testCase)
            xlsx = locate_xlsx(testCase);
            target = load_club_target_excel(xlsx, "TW_ProV1");
            testCase.verifyTrue(isstruct(target));
            for f = ["time","butt","clubhead","club_quat","impact_idx","source"]
                testCase.verifyTrue(isfield(target, f), ...
                    sprintf("Missing field: %s", f));
            end
            testCase.verifyEqual(size(target.butt, 2), 3);
            testCase.verifyEqual(size(target.clubhead, 2), 3);
            testCase.verifyEqual(size(target.club_quat, 2), 4);
        end

        function test_inches_converted_to_metres(testCase)
            xlsx = locate_xlsx(testCase);
            target = load_club_target_excel(xlsx, "TW_ProV1");
            % Sanity: max ||r|| of butt should be < 5 m (postcondition) and
            % > 0.1 m (so we know we did *some* conversion, not feet, etc).
            r_butt = vecnorm(target.butt, 2, 2);
            testCase.verifyGreaterThan(max(r_butt), 0.1);
            testCase.verifyLessThan(max(r_butt), 5.0);
            % Shaft-length plausibility: butt-to-clubhead distance roughly
            % within a typical driver shaft (0.7 - 1.4 m).
            shaft = vecnorm(target.clubhead - target.butt, 2, 2);
            testCase.verifyGreaterThan(median(shaft), 0.5);
            testCase.verifyLessThan(median(shaft), 2.0);
        end

        function test_rotation_matrices_become_unit_quaternions_with_q0_nonneg(testCase)
            xlsx = locate_xlsx(testCase);
            target = load_club_target_excel(xlsx, "TW_ProV1");
            qn = sqrt(sum(target.club_quat.^2, 2));
            testCase.verifyLessThan(max(abs(qn - 1)), 1e-6);
            testCase.verifyTrue(all(target.club_quat(:, 1) >= 0));
        end

        function test_impact_index_within_bounds(testCase)
            xlsx = locate_xlsx(testCase);
            target = load_club_target_excel(xlsx, "TW_ProV1");
            N = numel(target.time);
            testCase.verifyGreaterThanOrEqual(target.impact_idx, 1);
            testCase.verifyLessThanOrEqual(target.impact_idx, N);
        end

        function test_source_provenance_populated(testCase)
            xlsx = locate_xlsx(testCase);
            target = load_club_target_excel(xlsx, "TW_ProV1");
            testCase.verifyEqual(target.source.format, "xlsx");
            testCase.verifyTrue(strlength(target.source.filename) > 0);
            testCase.verifyEqual(strlength(target.source.sha256), 64);
        end

        function test_missing_file_raises_clear_error(testCase)
            testCase.verifyError( ...
                @() load_club_target_excel("does_not_exist.xlsx", "TW_ProV1"), ...
                ?MException);
        end
    end
end


function xlsx = locate_xlsx(testCase)
    here = fileparts(mfilename("fullpath"));
    candidate = fullfile(here, ...
        "..", "..", "src", "apps", "golf_gui", ...
        "Motion Capture Plotter", "Wiffle_ProV1_club_3D_data.xlsx");
    if exist(candidate, "file") ~= 2
        testCase.assumeFail( ...
            "skipped — Wiffle_ProV1_club_3D_data.xlsx not present at " + ...
            string(candidate));
    end
    xlsx = string(candidate);
end

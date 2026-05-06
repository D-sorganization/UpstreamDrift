classdef test_load_club_target_excel < matlab.unittest.TestCase
%TEST_LOAD_CLUB_TARGET_EXCEL  Excel loader tests (gated on real file presence).

    properties (Constant)
        % tests/ → shared/ → motion_matching/ → matlab/ → src/apps/...
        XLSX_RELPATH = fullfile("..", "..", "..", "src", "apps", "golf_gui", ...
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
            % `grip` and `grip_quat` are the new canonical names for the
            % mid-hands position + orientation; `butt` and `club_quat`
            % are kept as backward-compat aliases / clubhead orientation.
            for f = ["time","grip","grip_quat","butt","clubhead","club_quat","impact_idx","events","source"]
                testCase.verifyTrue(isfield(target, f), ...
                    sprintf("Missing field: %s", f));
            end
            testCase.verifyEqual(size(target.grip,      2), 3);
            testCase.verifyEqual(size(target.grip_quat, 2), 4);
            testCase.verifyEqual(size(target.clubhead,  2), 3);
            testCase.verifyEqual(size(target.club_quat, 2), 4);
            testCase.verifyEqual(target.butt, target.grip);   % alias is faithful
        end

        function test_position_units_are_plausible(testCase)
            % Sanity check that the loader's unit conversion lands in metres.
            % Definitions tab claims inches but values are actually cm —
            % see the loader header.  Values that are off by a 2.54x
            % factor would push these well outside the plausible range.
            xlsx = locate_xlsx(testCase);
            target = load_club_target_excel(xlsx, "TW_ProV1");
            r_grip = vecnorm(target.grip, 2, 2);
            testCase.verifyGreaterThan(max(r_grip), 0.1);
            testCase.verifyLessThan(max(r_grip), 5.0);
            % Shaft-length plausibility: mid-hands → clubhead within
            % a typical iron / fairway-wood / driver length.
            shaft = vecnorm(target.clubhead - target.grip, 2, 2);
            testCase.verifyGreaterThan(median(shaft), 0.7);
            testCase.verifyLessThan(median(shaft), 1.4);
        end

        function test_event_markers_parsed_from_header(testCase)
            xlsx = locate_xlsx(testCase);
            target = load_club_target_excel(xlsx, "TW_ProV1");
            testCase.verifyTrue(isfield(target, 'events'));
            ev = target.events;
            for f = ["A_sample","T_sample","I_sample","F_sample","CHS_mph"]
                testCase.verifyTrue(isfield(ev, f));
                testCase.verifyTrue(isfinite(ev.(f)));
            end
            % TW_ProV1 sheet documents A=240, T=418, I=525, F=725, CHS=114.5.
            testCase.verifyEqual(ev.I_sample, 525);
            testCase.verifyEqual(ev.A_sample, 240);
            testCase.verifyEqual(ev.CHS_mph,  114.5, "AbsTol", 0.1);
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
        "..", "..", "..", "src", "apps", "golf_gui", ...
        "Motion Capture Plotter", "Wiffle_ProV1_club_3D_data.xlsx");
    if exist(candidate, "file") ~= 2
        testCase.assumeFail( ...
            "skipped — Wiffle_ProV1_club_3D_data.xlsx not present at " + ...
            string(candidate));
    end
    xlsx = string(candidate);
end

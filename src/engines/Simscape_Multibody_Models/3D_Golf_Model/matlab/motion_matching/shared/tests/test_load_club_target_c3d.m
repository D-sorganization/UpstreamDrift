classdef test_load_club_target_c3d < matlab.unittest.TestCase
%TEST_LOAD_CLUB_TARGET_C3D  C3D loader tests (gated on real file presence).

    methods (TestClassSetup)
        function add_shared_to_path(testCase)
            here = fileparts(mfilename("fullpath"));
            shared_dir = fullfile(here, "..");
            addpath(shared_dir);
            testCase.addTeardown(@() rmpath(shared_dir));
        end
    end

    methods (Test)
        function test_loads_existing_cluster_c3d_file_without_error(testCase)
            c3d = locate_c3d(testCase);
            opts = default_align_options();
            opts.verbosity = "Verbose";
            target = load_club_target_c3d(c3d, opts);
            testCase.verifyTrue(isstruct(target));
            for f = ["time","butt","clubhead","club_quat","impact_idx","source"]
                testCase.verifyTrue(isfield(target, f), ...
                    sprintf("Missing field: %s", f));
            end
        end

        function test_marker_mapping_documented_in_log(testCase)
            c3d = locate_c3d(testCase);
            opts = default_align_options();
            opts.verbosity = "Verbose";
            % Capture console output to verify the loader logs the marker
            % names it discovered.  This is the documentation hook.
            log = evalc("load_club_target_c3d(c3d, opts);");
            testCase.verifyTrue(contains(log, "C3D markers present"), ...
                "Loader must log discovered marker names");
            testCase.verifyTrue(contains(log, "Marker mapping"), ...
                "Loader must log butt/clubhead mapping");
        end

        function test_quaternions_unit_norm(testCase)
            c3d = locate_c3d(testCase);
            target = load_club_target_c3d(c3d);
            qn = sqrt(sum(target.club_quat.^2, 2));
            testCase.verifyLessThan(max(abs(qn - 1)), 1e-6);
        end
    end
end


function c3d = locate_c3d(testCase)
    here = fileparts(mfilename("fullpath"));
    base = fullfile(here, "..", "..", "Data", "Mocap C3D Files");
    candidates = [
        fullfile(base, "C3DExport Tour average.c3d"), ...
        fullfile(base, "C3DExport tour average iron.c3d")];
    for i = 1:numel(candidates)
        if exist(candidates(i), "file") == 2
            c3d = string(candidates(i));
            return;
        end
    end
    testCase.assumeFail("skipped — no cluster-marker C3D file present at " + base);
end

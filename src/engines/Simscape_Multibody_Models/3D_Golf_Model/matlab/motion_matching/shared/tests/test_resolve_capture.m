classdef test_resolve_capture < matlab.unittest.TestCase
%TEST_RESOLVE_CAPTURE  Unit tests for resolve_capture.m (#11162).

    methods (TestClassSetup)
        function add_shared_to_path(testCase)
            here = fileparts(mfilename("fullpath"));
            shared_dir = fullfile(here, "..");
            addpath(shared_dir);
            testCase.addTeardown(@() rmpath(shared_dir));
        end
    end

    methods (Test)
        function test_resolve_capture_A(testCase)
            p = resolve_capture("capture-A");
            testCase.verifyTrue(exist(p, 'file') == 2);
            [~, fname, fext] = fileparts(p);
            testCase.verifyEqual([fname, fext], 'C3D_TA_Driver.c3d');
        end

        function test_resolve_capture_B(testCase)
            p = resolve_capture("capture-B");
            testCase.verifyTrue(exist(p, 'file') == 2);
            [~, fname, fext] = fileparts(p);
            testCase.verifyEqual([fname, fext], 'C3D_TA_Iron.c3d');
        end

        function test_resolve_capture_unknown_throws(testCase)
            testCase.verifyError(@() resolve_capture("unknown-id-12345"), ...
                'capture:unknown');
        end

        function test_resolve_capture_private_unavailable_without_env(testCase)
            % Ensure CAPTURE_DATA_DIR is empty for this test
            orig_env = getenv("CAPTURE_DATA_DIR");
            setenv("CAPTURE_DATA_DIR", "");
            testCase.addTeardown(@() setenv("CAPTURE_DATA_DIR", orig_env));

            testCase.verifyError(@() resolve_capture("capture-O"), ...
                'capture:unavailable');
            testCase.verifyError(@() resolve_capture("club-workbook-main"), ...
                'capture:unavailable');
            testCase.verifyError(@() resolve_capture("club-workbook-wiffle"), ...
                'capture:unavailable');
        end

        function test_resolve_capture_private_missing_file_throws_unavailable(testCase)
            temp_dir = tempname();
            mkdir(temp_dir);
            testCase.addTeardown(@() rmdir(temp_dir, 's'));

            testCase.verifyError(@() resolve_capture("capture-O", data_dir=temp_dir), ...
                'capture:unavailable');
        end

        function test_resolve_capture_integrity_mismatch_throws(testCase)
            temp_dir = tempname();
            mkdir(temp_dir);
            testCase.addTeardown(@() rmdir(temp_dir, 's'));

            % Create corrupted file at expected relative path
            target_file = fullfile(temp_dir, 'datasets', 'capture-O', 'capture-O_driver.c3d');
            [parent_dir, ~, ~] = fileparts(target_file);
            mkdir(parent_dir);
            fid = fopen(target_file, 'w');
            fprintf(fid, 'corrupted data');
            fclose(fid);

            testCase.verifyError(@() resolve_capture("capture-O", data_dir=temp_dir), ...
                'capture:integrity');
        end
    end
end

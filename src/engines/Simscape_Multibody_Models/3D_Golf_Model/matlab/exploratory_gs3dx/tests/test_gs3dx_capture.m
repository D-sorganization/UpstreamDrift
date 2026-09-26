classdef test_gs3dx_capture < matlab.unittest.TestCase
%TEST_GS3DX_CAPTURE  The tour-average capture and the stance taken from it (#10985).
%
%   Pins what the data audit found in data/C3D_TA_Driver.c3d (identity,
%   rate, no force plates) and that GS3DX_LEG_TABLE's .stance is what
%   GS3DX_CAPTURE_STANCE measures.  Skipped when MATLAB's Python has no
%   ezc3d.

    properties
        cap struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            gs3dx_setup();
            try
                py.importlib.import_module('ezc3d');
                has_ezc3d = true;
            catch
                has_ezc3d = false;
            end
            testCase.assumeTrue(has_ezc3d, 'Python ezc3d is not available to MATLAB (pyenv)');
            testCase.cap = gs3dx_capture_stance();
        end
    end

    methods (Test)
        function capture_is_the_audited_file(testCase)
            c = testCase.cap;
            testCase.verifyEqual(c.sha256, '545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba');
            testCase.verifyEqual(c.n_frames, 654);
            testCase.verifyEqual(c.rate_hz, 360);
        end

        function capture_has_no_ground_reaction_data(testCase)
            testCase.verifyEqual(testCase.cap.force_plates_used, 0);
            testCase.verifyEqual(testCase.cap.n_analog, 0);
        end

        function leg_and_foot_markers_are_complete(testCase)
            m = testCase.cap.missing;
            for name = ["LKneeOut", "RKneeOut", "LAnkleOut", "RAnkleOut", "LToeIn", "LToeOut", "RToeIn", "RToeOut"]
                testCase.verifyEqual(m.(name), 0, name);
            end
        end

        function leg_table_stance_matches_capture(testCase)
            c = testCase.cap;
            st = gs3dx_leg_table().stance;
            for s = ["L", "R"]
                testCase.verifyEqual(st.("ankle_" + s), c.("ankle_" + s)(1:2), 'AbsTol', 1e-3, "ankle_" + s);
                testCase.verifyEqual(st.("foot_yaw_" + s), c.("foot_yaw_" + s), 'AbsTol', 0.1, "foot_yaw_" + s);
            end
            drop = -mean([c.ankle_L(3), c.ankle_R(3)]);
            testCase.verifyEqual(st.drop, drop, 'AbsTol', 1e-3);
        end
    end
end

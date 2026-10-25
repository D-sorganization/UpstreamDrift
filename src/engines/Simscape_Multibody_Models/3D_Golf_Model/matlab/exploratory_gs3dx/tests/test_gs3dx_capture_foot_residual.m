classdef test_gs3dx_capture_foot_residual < matlab.unittest.TestCase
%TEST_GS3DX_CAPTURE_FOOT_RESIDUAL  Capture foot-split residual metrics (#11173).
%
%   Synthetic stance geometry only; no capture file, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function balanced_stance_near_zero_residual(testCase)
            m = 80;
            g = 9.80665;
            n = 5;
            k.t = (0:n - 1) / 100;
            k.grf = repmat([0; 0; m * g], 1, n);
            k.com = repmat([0; 0; 1], 1, n);
            k.mass = m;
            k.impact_frame = 3;
            jc.t = k.t;
            jc.ankle_L = repmat([-0.15; 0; 0], 1, n);
            jc.ankle_R = repmat([0.15; 0; 0], 1, n);
            r = gs3dx_capture_foot_residual(k, jc);
            testCase.verifyLessThan(r.force_rms_bw, 1e-9);
            testCase.verifyLessThan(r.moment_rms_nm, 1e-6);
            testCase.verifyLessThan(r.moment_vertical_rms_nm, 1e-6);
            testCase.verifyLessThan(r.moment_horizontal_rms_nm, 1e-6);
        end

        function com_outside_left_foot_moment(testCase)
            m = 80;
            g = 9.80665;
            n = 5;
            k.t = (0:n - 1) / 100;
            k.grf = repmat([0; 0; m * g], 1, n);
            k.com = repmat([-0.25; 0; 1], 1, n);
            k.mass = m;
            k.impact_frame = 3;
            jc.t = k.t;
            jc.ankle_L = repmat([-0.15; 0; 0], 1, n);
            jc.ankle_R = repmat([0.15; 0; 0], 1, n);
            r = gs3dx_capture_foot_residual(k, jc);
            testCase.verifyEqual(r.moment_rms_nm, m * g * 0.1, 'AbsTol', 1e-4);
            testCase.verifyEqual(r.moment_vertical_rms_nm, m * g * 0.1, 'AbsTol', 1e-4);
            testCase.verifyLessThan(r.moment_horizontal_rms_nm, 1e-4);
            testCase.verifyLessThan(r.force_rms_bw, 1e-9);
        end

        function centred_stance_horizontal_force_lever_moment(testCase)
            m = 80;
            g = 9.80665;
            n = 5;
            k.t = (0:n - 1) / 100;
            k.grf = repmat([100; 0; m * g], 1, n);
            k.com = repmat([0; 0; 1], 1, n);
            k.mass = m;
            k.impact_frame = 3;
            jc.t = k.t;
            jc.ankle_L = repmat([-0.15; 0; 0], 1, n);
            jc.ankle_R = repmat([0.15; 0; 0], 1, n);
            r = gs3dx_capture_foot_residual(k, jc);
            testCase.verifyLessThan(r.moment_vertical_rms_nm, 1e-4);
            testCase.verifyEqual(r.moment_horizontal_rms_nm, 100 * 1, 'AbsTol', 1e-4);
        end

        function com_offset_cancels_constant_com_shift(testCase)
            m = 80;
            g = 9.80665;
            n = 5;
            k.t = (0:n - 1) / 100;
            k.grf = repmat([0; 0; m * g], 1, n);
            k.com = repmat([0; 0; 1], 1, n);
            k.mass = m;
            k.impact_frame = 3;
            jc.t = k.t;
            jc.ankle_L = repmat([-0.15; 0; 0], 1, n);
            jc.ankle_R = repmat([0.15; 0; 0], 1, n);
            r0 = gs3dx_capture_foot_residual(k, jc);
            delta = [1.2; -0.4; 0.8];
            k.com = k.com + delta;
            r1 = gs3dx_capture_foot_residual(k, jc, 'com_offset', delta);
            testCase.verifyEqual(r1.force_rms_bw, r0.force_rms_bw, 'AbsTol', 1e-6);
            testCase.verifyEqual(r1.moment_rms_nm, r0.moment_rms_nm, 'AbsTol', 1e-6);
        end
    end
end

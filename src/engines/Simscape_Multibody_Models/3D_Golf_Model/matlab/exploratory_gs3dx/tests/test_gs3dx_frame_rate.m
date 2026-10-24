classdef test_gs3dx_frame_rate < matlab.unittest.TestCase
%TEST_GS3DX_FRAME_RATE  Body-frame spin rate from rotations (#11173).
%
%   Synthetic rotations only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function constant_identity_gives_zero_spin(testCase)
            dt = 0.01;
            n = 5;
            R = repmat(eye(3), 1, 1, n);
            w = gs3dx_frame_rate(R, dt);
            testCase.verifyEqual(w, zeros(3, n), 'AbsTol', 1e-12);
        end

        function rotation_about_z_matches_rate(testCase)
            dt = 0.01;
            n = 21;
            R = zeros(3, 3, n);
            for k = 0:(n - 1)
                theta = 0.5 * k * dt;
                R(:, :, k + 1) = [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1];
            end
            w = gs3dx_frame_rate(R, dt);
            testCase.verifyEqual(w(:, 11), [0; 0; 0.5], 'AbsTol', 1e-6);
        end
    end
end

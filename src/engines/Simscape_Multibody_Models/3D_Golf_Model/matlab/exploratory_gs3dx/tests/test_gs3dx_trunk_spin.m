classdef test_gs3dx_trunk_spin < matlab.unittest.TestCase
%TEST_GS3DX_TRUNK_SPIN  Trunk spin angular momentum (#11173).
%
%   Synthetic geometry only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function long_axis_spin_matches_inertia(testCase)
            dt = 0.01;
            n = 21;
            mass = 30;
            gyration = [0.3, 0.3, 0.2];
            I_long = mass * (gyration(3) * 0.5)^2;

            hip_L = zeros(3, n);
            hip_R = zeros(3, n);
            shoulder_L = zeros(3, n);
            shoulder_R = zeros(3, n);
            base = [0, -0.1, 0; 0, 0.1, 0; 0, -0.1, 0.5; 0, 0.1, 0.5].';
            for k = 1:n
                t = (k - 1) * dt;
                ct = cos(t);
                st = sin(t);
                Rz = [ct, -st, 0; st, ct, 0; 0, 0, 1];
                hip_L(:, k) = Rz * base(:, 1);
                hip_R(:, k) = Rz * base(:, 2);
                shoulder_L(:, k) = Rz * base(:, 3);
                shoulder_R(:, k) = Rz * base(:, 4);
            end

            h = gs3dx_trunk_spin(hip_L, hip_R, shoulder_L, shoulder_R, mass, gyration, dt);
            k = 11;
            testCase.verifyGreaterThanOrEqual(norm(h(:, k)), I_long - 1e-4);
            testCase.verifyEqual(h(3, k), I_long, 'AbsTol', 1e-4);
        end
    end
end

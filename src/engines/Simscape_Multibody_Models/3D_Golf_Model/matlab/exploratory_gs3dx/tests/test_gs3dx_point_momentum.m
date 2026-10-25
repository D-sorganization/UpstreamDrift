classdef test_gs3dx_point_momentum < matlab.unittest.TestCase
%TEST_GS3DX_POINT_MOMENTUM  Point-mass angular momentum (#11173).
%
%   Synthetic geometry only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function single_particle_has_zero_angular_momentum(testCase)
            dt = 0.01;
            n = 8;
            t = (0:n - 1) * dt;
            mass = 2.5;
            pos = zeros(3, 1, n);
            pos(1, 1, :) = sin(t);
            pos(3, 1, :) = cos(2 * t);
            [h, hdot] = gs3dx_point_momentum(mass, pos, dt);
            testCase.verifyEqual(h, zeros(3, n), 'AbsTol', 1e-12);
            testCase.verifyEqual(hdot, zeros(3, n), 'AbsTol', 1e-12);
        end

        function two_particles_match_hand_formula(testCase)
            dt = 0.01;
            n = 10;
            mass = [1; 1];
            pos = zeros(3, 2, n);
            for k = 1:n
                tk = (k - 1) * dt;
                pos(:, 1, k) = [0; 0; 0];
                pos(:, 2, k) = [1; tk; 0];
            end
            [h, ~] = gs3dx_point_momentum(mass, pos, dt);
            k = 5;
            com = (pos(:, 1, k) + pos(:, 2, k)) / 2;
            r12 = pos(:, 2, k) - pos(:, 1, k);
            v12 = [0; 1; 0];
            rB = pos(:, 2, k) - com;
            vRel = v12 - v12 / 2;
            signZ = sign(cross(rB, vRel));
            if signZ(3) == 0
                signZ(3) = 1;
            end
            lever = abs(r12(1));
            wantZ = 0.5 * lever * norm(v12) * signZ(3);
            testCase.verifyEqual(h(3, k), wantZ, 'AbsTol', 1e-9);
            testCase.verifyEqual(h(:, k), 0.5 * cross(r12, v12), 'AbsTol', 1e-9);
        end
    end
end

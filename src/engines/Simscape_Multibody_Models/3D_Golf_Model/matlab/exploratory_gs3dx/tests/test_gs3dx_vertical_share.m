classdef test_gs3dx_vertical_share < matlab.unittest.TestCase
%TEST_GS3DX_VERTICAL_SHARE  Vertical GRF share across contacts (#11173).
%
%   Synthetic geometry only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function cop_at_square_centre(testCase)
            total = 800;
            cop = [0; 0];
            points = local_square_points();
            fz = gs3dx_vertical_share(cop, points, total);
            testCase.verifyEqual(sum(fz), total, 'AbsTol', 1e-8);
            testCase.verifyGreaterThanOrEqual(min(fz), -1e-12);
            pts2 = points(1:2, :, 1);
            c = pts2 * fz(:, 1) / sum(fz(:, 1));
            testCase.verifyEqual(c, [0; 0], 'AbsTol', 1e-8);
        end

        function cop_outside_square_left_edge(testCase)
            total = 800;
            cop = [-0.2; 0];
            points = local_square_points();
            fz = gs3dx_vertical_share(cop, points, total);
            pts2 = points(1:2, :, 1);
            c = pts2 * fz(:, 1) / sum(fz(:, 1));
            testCase.verifyEqual(c(1), -0.1, 'AbsTol', 1e-8);
            testCase.verifyEqual(c(2), 0, 'AbsTol', 1e-8);
            testCase.verifyEqual(fz(2), 0, 'AbsTol', 1e-12);
            testCase.verifyEqual(fz(3), 0, 'AbsTol', 1e-12);
            testCase.verifyGreaterThan(fz(1), 0);
            testCase.verifyGreaterThan(fz(4), 0);
        end

        function zero_total(testCase)
            cop = [0; 0];
            points = local_square_points();
            fz = gs3dx_vertical_share(cop, points, 0);
            testCase.verifyEqual(fz, zeros(4, 1), 'AbsTol', 1e-12);
        end
    end
end

function points = local_square_points()
    points = zeros(3, 4, 1);
    points(1, :, 1) = [-0.1, 0.1, 0.1, -0.1];
    points(2, :, 1) = [-0.05, -0.05, 0.05, 0.05];
    points(3, :, 1) = 0;
end

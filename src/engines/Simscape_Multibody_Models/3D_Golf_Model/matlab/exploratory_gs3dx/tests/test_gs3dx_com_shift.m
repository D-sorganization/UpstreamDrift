classdef test_gs3dx_com_shift < matlab.unittest.TestCase
%TEST_GS3DX_COM_SHIFT  Horizontal COM shift to hull COP (#11173).
%
%   Synthetic geometry only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function vertical_grf_at_com(testCase)
            f = [0; 0; 800];
            com = [0; 0; 1];
            points = local_square_points();
            shift = gs3dx_com_shift(f, com, points, 0.8);
            testCase.verifyEqual(shift, zeros(3, 1), 'AbsTol', 1e-9);
        end

        function shear_cop_outside_hull(testCase)
            f = [160; 0; 800];
            com = [0; 0; 1];
            points = local_square_points();
            shift = gs3dx_com_shift(f, com, points, 0.8);
            testCase.verifyEqual(shift, [0.1; 0; 0], 'AbsTol', 1e-9);
            s2 = gs3dx_support_residual(f, com + shift, points, 0.8);
            testCase.verifyEqual(s2.moment, 0, 'AbsTol', 1e-9);
        end

        function negative_vertical_grf(testCase)
            f = [0; 0; -100];
            com = [0; 0; 1];
            points = local_square_points();
            shift = gs3dx_com_shift(f, com, points, 0.8);
            testCase.verifyEqual(shift, zeros(3, 1), 'AbsTol', 1e-9);
        end
    end
end

function points = local_square_points()
    points = zeros(3, 4, 1);
    points(1, :, 1) = [-0.1, 0.1, 0.1, -0.1];
    points(2, :, 1) = [-0.05, -0.05, 0.05, 0.05];
    points(3, :, 1) = 0;
end

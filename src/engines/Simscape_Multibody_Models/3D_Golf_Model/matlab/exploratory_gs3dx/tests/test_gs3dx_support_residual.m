classdef test_gs3dx_support_residual < matlab.unittest.TestCase
%TEST_GS3DX_SUPPORT_RESIDUAL  Shear COP tip residual (#11173).
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
            s = gs3dx_support_residual(f, com, points, 0.8);
            testCase.verifyEqual(s.moment, 0, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.force, 0, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.cop, [0; 0], 'AbsTol', 1e-9);
        end

        function shear_cop_on_hull_edge(testCase)
            f = [80; 0; 800];
            com = [0; 0; 1];
            points = local_square_points();
            s = gs3dx_support_residual(f, com, points, 0.8);
            testCase.verifyEqual(s.cop, [-0.1; 0], 'AbsTol', 1e-9);
            testCase.verifyEqual(s.moment, 0, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.force, 0, 'AbsTol', 1e-9);
        end

        function shear_cop_outside_hull(testCase)
            f = [160; 0; 800];
            com = [0; 0; 1];
            points = local_square_points();
            s = gs3dx_support_residual(f, com, points, 0.8);
            testCase.verifyEqual(s.cop, [-0.2; 0], 'AbsTol', 1e-9);
            testCase.verifyEqual(s.moment, 80, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.force, 0, 'AbsTol', 1e-9);
        end

        function negative_vertical_grf(testCase)
            f = [0; 0; -100];
            com = [0; 0; 1];
            points = local_square_points();
            s = gs3dx_support_residual(f, com, points, 0.8);
            testCase.verifyEqual(s.force, 100, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.moment, 0, 'AbsTol', 1e-9);
            testCase.verifyTrue(all(isnan(s.cop)));
        end

        function required_moment_shifts_cop_inside_hull(testCase)
            f = [0; 0; 800];
            com = [0; 0; 1];
            points = local_square_points();
            m = [0; 80; 0];
            s = gs3dx_support_residual(f, com, points, 0.8, m);
            testCase.verifyEqual(s.cop, [-0.1; 0], 'AbsTol', 1e-9);
            testCase.verifyEqual(s.moment, 0, 'AbsTol', 1e-9);
        end

        function required_moment_outside_hull(testCase)
            f = [0; 0; 800];
            com = [0; 0; 1];
            points = local_square_points();
            m = [0; 160; 0];
            s = gs3dx_support_residual(f, com, points, 0.8, m);
            testCase.verifyEqual(s.cop, [-0.2; 0], 'AbsTol', 1e-9);
            testCase.verifyEqual(s.moment, 80, 'AbsTol', 1e-9);
        end

        function friction_limited_shear(testCase)
            f = [800; 0; 100];
            com = [0; 0; 1];
            points = local_square_points();
            s = gs3dx_support_residual(f, com, points, 0.8);
            testCase.verifyEqual(s.cop(1), -0.8, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.moment, 70, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.force, 720, 'AbsTol', 1e-9);
        end
    end
end

function points = local_square_points()
    points = zeros(3, 4, 1);
    points(1, :, 1) = [-0.1, 0.1, 0.1, -0.1];
    points(2, :, 1) = [-0.05, -0.05, 0.05, 0.05];
    points(3, :, 1) = 0;
end

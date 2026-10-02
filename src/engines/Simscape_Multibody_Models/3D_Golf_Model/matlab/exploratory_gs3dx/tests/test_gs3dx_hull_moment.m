classdef test_gs3dx_hull_moment < matlab.unittest.TestCase
%TEST_GS3DX_HULL_MOMENT  COM-to-support-hull moment (#11173).
%
%   Synthetic geometry only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function com_at_square_centre(testCase)
            fz = 800;
            p_com = [0; 0; 0];
            points = local_square_points();
            m = gs3dx_hull_moment(p_com, points, fz);
            testCase.verifyEqual(m, 0, 'AbsTol', 1e-9);
        end

        function com_outside_square_edge(testCase)
            fz = 800;
            p_com = [-0.2; 0; 0];
            points = local_square_points();
            m = gs3dx_hull_moment(p_com, points, fz);
            testCase.verifyEqual(m, fz * 0.1, 'AbsTol', 1e-9);
        end

        function com_on_square_vertex(testCase)
            fz = 800;
            p_com = [0.1; 0.05; 0];
            points = local_square_points();
            m = gs3dx_hull_moment(p_com, points, fz);
            testCase.verifyEqual(m, 0, 'AbsTol', 1e-9);
        end
    end
end

function points = local_square_points()
    points = zeros(3, 4, 1);
    points(1, :, 1) = [-0.1, 0.1, 0.1, -0.1];
    points(2, :, 1) = [-0.05, -0.05, 0.05, 0.05];
    points(3, :, 1) = 0;
end

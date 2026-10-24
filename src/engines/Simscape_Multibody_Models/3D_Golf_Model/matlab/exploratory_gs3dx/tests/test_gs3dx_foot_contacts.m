classdef test_gs3dx_foot_contacts < matlab.unittest.TestCase
%TEST_GS3DX_FOOT_CONTACTS  Rectangular foot contact corners (#11173).
%
%   Synthetic geometry only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function axis_aligned_rectangle_corners(testCase)
            ankle = [0; 0; 0];
            toe = [1; 0; 0];
            points = gs3dx_foot_contacts(ankle, toe, 0.4, 0.1);
            testCase.verifyEqual(points(1, :, 1), [-0.1, -0.1, 0.3, 0.3], 'AbsTol', 1e-12);
            testCase.verifyEqual(points(2, :, 1), [-0.05, 0.05, -0.05, 0.05], 'AbsTol', 1e-12);
        end

        function hull_moment_at_centroid_and_outside(testCase)
            ankle = [0; 0; 0];
            toe = [1; 0; 0];
            points = gs3dx_foot_contacts(ankle, toe, 0.4, 0.1);
            fz = 800;
            p_in = [0.1; 0; 0];
            m_in = gs3dx_hull_moment(p_in, points, fz);
            testCase.verifyEqual(m_in, 0, 'AbsTol', 1e-9);
            p_out = [0.1; 0.05 + 0.1; 0];
            m_out = gs3dx_hull_moment(p_out, points, fz);
            testCase.verifyEqual(m_out, fz * 0.1, 'AbsTol', 1e-9);
        end

        function zero_foot_width_rejected(testCase)
            ankle = [0; 0; 0];
            toe = [1; 0; 0];
            testCase.verifyError(@() gs3dx_foot_contacts(ankle, toe, 0.4, 0), ...
                'gs3dx:foot');
        end
    end
end

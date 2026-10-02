classdef test_gs3dx_trail_foot_reference < matlab.unittest.TestCase
%TEST_GS3DX_TRAIL_FOOT_REFERENCE  Trail-foot reference lowering (#10979).
%
%   Synthetic feet/preload only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function zero_preload_leaves_feet_unchanged(testCase)
            feet = ones(6, 5);
            t = 0:0.01:2;
            fz = 200 * ones(size(t));
            hPre = gs3dx_trail_preload(t, fz);
            testCase.verifyEqual(hPre, zeros(size(t)), 'AbsTol', 1e-12);
            h = zeros(1, 5);
            out = gs3dx_trail_foot_reference(feet, h);
            testCase.verifyEqual(out, feet, 'AbsTol', 1e-12);
        end

        function subtracts_h_from_trail_foot_z_only(testCase)
            feet = zeros(6, 4);
            feet(6, :) = 0.05;
            h = [0 0 0.01 0.02];
            out = gs3dx_trail_foot_reference(feet, h);
            testCase.verifyEqual(out(1:5, :), zeros(5, 4), 'AbsTol', 1e-12);
            testCase.verifyEqual(out(6, :), [0.05 0.05 0.04 0.03], 'AbsTol', 1e-12);
        end

        function negative_h_throws(testCase)
            feet = zeros(6, 1);
            testCase.verifyError(@() gs3dx_trail_foot_reference(feet, -0.01), 'gs3dx:preload');
        end
    end
end

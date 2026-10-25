classdef test_gs3dx_trail_load < matlab.unittest.TestCase
%TEST_GS3DX_TRAIL_LOAD  Trail-foot preload on foot references (#10979).
%
%   Synthetic feet/force/time only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function zero_preload_before_t_on_leaves_feet_unchanged(testCase)
            feet = ones(6, 5);
            t = 0:0.25:1;
            fz = zeros(size(t));
            out = gs3dx_trail_load(feet, t, fz);
            testCase.verifyEqual(out, feet, 'AbsTol', 1e-12);
        end

        function integrates_preload_into_trail_foot_z(testCase)
            n = 236;
            t = 0:0.01:2.35;
            testCase.verifyEqual(numel(t), n);
            fz = zeros(size(t));
            feet = zeros(6, n);
            feet(6, :) = 0.2;
            feet(3, :) = 0.2;
            t_on = 1.35;
            preload = 78.48;
            ki = 1e-4;
            out = gs3dx_trail_load(feet, t, fz, t_on, preload, ki);
            testCase.verifyEqual(out(1:5, :), feet(1:5, :), 'AbsTol', 1e-12);
            k = find(abs(t - 2.35) <= 1e-12, 1);
            testCase.verifyEqual(out(6, k), 0.2 - ki * preload * 1, 'AbsTol', 1e-6);
        end

        function column_count_mismatch_throws(testCase)
            feet = ones(6, 3);
            t = [0 0.5];
            fz = zeros(size(t));
            testCase.verifyError(@() gs3dx_trail_load(feet, t, fz), 'gs3dx:preload');
        end
    end
end

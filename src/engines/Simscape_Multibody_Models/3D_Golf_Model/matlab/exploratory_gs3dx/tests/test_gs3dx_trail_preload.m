classdef test_gs3dx_trail_preload < matlab.unittest.TestCase
%TEST_GS3DX_TRAIL_PRELOAD  Finish-phase trail-foot preload law (#10979).
%
%   Synthetic force/time only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function high_contact_force_keeps_zero_preload(testCase)
            t = 0:0.01:2;
            fz = 200 * ones(size(t));
            h = gs3dx_trail_preload(t, fz);
            testCase.verifyEqual(h, zeros(size(t)), 'AbsTol', 1e-12);
        end

        function zero_force_integrates_after_t_on(testCase)
            t = 0:0.01:2.35;
            fz = zeros(size(t));
            t_on = 1.35;
            preload = 78.48;
            ki = 1e-4;
            h = gs3dx_trail_preload(t, fz, t_on, preload, ki);
            before = t < t_on;
            testCase.verifyEqual(h(before), zeros(1, sum(before)), 'AbsTol', 1e-12);
            k = find(abs(t - (t_on + 1)) <= 1e-12, 1);
            testCase.verifyEqual(h(k), ki * preload * 1, 'AbsTol', 1e-6);
        end

        function zero_ki_keeps_zero_preload(testCase)
            t = 0:0.01:2;
            fz = zeros(size(t));
            h = gs3dx_trail_preload(t, fz, 1.35, 78.48, 0);
            testCase.verifyEqual(h, zeros(size(t)), 'AbsTol', 1e-12);
        end
    end
end

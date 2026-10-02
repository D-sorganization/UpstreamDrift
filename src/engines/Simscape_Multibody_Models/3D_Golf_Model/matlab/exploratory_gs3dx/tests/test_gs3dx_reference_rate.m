classdef test_gs3dx_reference_rate < matlab.unittest.TestCase
%TEST_GS3DX_REFERENCE_RATE  Reference angle rates from gradient (#11173).
%
%   Synthetic angles only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function constant_angles_give_zero_rates(testCase)
            t = 0:0.1:1;
            q = ones(3, numel(t));
            qd = gs3dx_reference_rate(t, q);
            testCase.verifyEqual(qd, zeros(size(q)), 'AbsTol', 1e-12);
        end

        function linear_ramp_matches_unit_slope(testCase)
            t = 0:0.1:1;
            q = repmat(t, 2, 1);
            qd = gs3dx_reference_rate(t, q);
            testCase.verifyEqual(qd, ones(size(q)), 'AbsTol', 1e-12);
        end

        function non_increasing_time_throws(testCase)
            t = [0 0.1 0.05 0.2];
            q = zeros(2, numel(t));
            testCase.verifyError(@() gs3dx_reference_rate(t, q), 'gs3dx:rate');
        end
    end
end

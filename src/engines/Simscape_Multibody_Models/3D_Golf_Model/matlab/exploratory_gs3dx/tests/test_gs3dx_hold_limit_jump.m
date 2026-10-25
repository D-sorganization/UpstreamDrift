classdef test_gs3dx_hold_limit_jump < matlab.unittest.TestCase
%TEST_GS3DX_HOLD_LIMIT_JUMP  Remove ROM limit jumps on reference rows (#11173).
%
%   Synthetic angles only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function hi_then_lo_holds_at_hi(testCase)
            lo = -5;
            hi = 150;
            q = [140 150 -5 10];
            testCase.verifyEqual(gs3dx_hold_limit_jump(q, lo, hi), [140 150 150 10]);
        end

        function lo_then_hi_holds_at_lo(testCase)
            lo = -5;
            hi = 150;
            q = [-5 150 140];
            testCase.verifyEqual(gs3dx_hold_limit_jump(q, lo, hi), [-5 -5 140]);
        end

        function interior_samples_unchanged(testCase)
            lo = -5;
            hi = 150;
            q = [10 20 30];
            testCase.verifyEqual(gs3dx_hold_limit_jump(q, lo, hi), q);
        end

        function invalid_limits_throw(testCase)
            testCase.verifyError(@() gs3dx_hold_limit_jump([0 1], 10, 10), 'gs3dx:rate');
            testCase.verifyError(@() gs3dx_hold_limit_jump([0 1], 10, 5), 'gs3dx:rate');
        end
    end
end

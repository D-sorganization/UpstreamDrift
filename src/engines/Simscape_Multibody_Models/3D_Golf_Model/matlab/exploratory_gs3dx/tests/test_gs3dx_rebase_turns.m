classdef test_gs3dx_rebase_turns < matlab.unittest.TestCase
%TEST_GS3DX_REBASE_TURNS  Whole-turn rebasing for arc reference rows (#11173).
%
%   Synthetic angles only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function whole_turn_offset_rebases(testCase)
            q = [-1094 -1093];
            out = gs3dx_rebase_turns(q);
            testCase.verifyEqual(out, [-14 -13], 'AbsTol', 1e-9);
            testCase.verifyEqual(max(out) - min(out), max(q) - min(q), 'AbsTol', 1e-9);
        end

        function interior_row_unchanged(testCase)
            q = [10 20 30];
            testCase.verifyEqual(gs3dx_rebase_turns(q), q);
        end

        function positive_offset_rebases(testCase)
            q = [200 210];
            out = gs3dx_rebase_turns(q);
            testCase.verifyEqual(out, [-160 -150], 'AbsTol', 1e-9);
        end
    end
end

classdef test_default_impact_input_mat < matlab.unittest.TestCase
%TEST_DEFAULT_IMPACT_INPUT_MAT  The shared impact-pose input resolves to the
%   committed file, so callers that omit .input_mat (the leaderboard) do not
%   fall back to a bare filename that MATLAB cannot find.

    methods (TestClassSetup)
        function add_paths(testCase)
            shared = fileparts(fileparts(mfilename('fullpath')));
            addpath(shared);
            testCase.addTeardown(@() rmpath(shared));
        end
    end

    methods (Test)
        function resolves_to_the_committed_file(testCase)
            p = default_impact_input_mat();
            testCase.verifyClass(p, 'char');
            testCase.verifyTrue(isfile(p), sprintf('missing: %s', p));
            testCase.verifyTrue(endsWith(p, fullfile('matlab', 'src', 'model', 'inputs', ...
                '3DModelInputs_Impact.mat')));
        end
    end
end

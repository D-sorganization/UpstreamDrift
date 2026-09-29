% TEST_EXAMPLE  Example test file using MATLAB's function-based testing framework
%   This file demonstrates the use of functiontests for unit testing.
%
%   Usage:
%       tests = test_example
%       results = runtests('test_example')
%
%   See also: functiontests, runtests

function tests = test_example()
    % TEST_EXAMPLE  Create test suite
    %
    %   Returns:
    %       tests - Test suite object
    
    % Validate inputs (none required)
    arguments
        % No input arguments
    end
    
    tests = functiontests(localfunctions);
end

function test_truth(testCase)
    % TEST_TRUTH  Test basic arithmetic
    %   Verifies that 1 + 1 equals 2
    
    verifyEqual(testCase, 1+1, 2);
end

% File: matlab/tests/test_example.m
function tests = test_example
    % TEST_EXAMPLE Example test suite for MATLAB Unit Testing Framework
    %
    % This is a simple example test file demonstrating the basic structure
    % of MATLAB unit tests using the functiontests framework.
    %
    % Outputs:
    %   tests - Test suite created from local functions
    %
    % Usage:
    %   results = runtests('test_example')
    %
    % See also: functiontests, matlab.unittest.TestCase

    tests = functiontests(localfunctions);
end

function test_truth(testCase)
    % TEST_TRUTH Test basic mathematical truth
    %
    % This test verifies that basic arithmetic operations work correctly.
    %
    % Inputs:
    %   testCase - Test case object from MATLAB Unit Testing Framework

    arguments
        testCase (1,1) matlab.unittest.TestCase
    end

    verifyEqual(testCase, 1+1, 2);
end

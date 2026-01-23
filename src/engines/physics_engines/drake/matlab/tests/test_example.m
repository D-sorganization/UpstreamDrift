// File: matlab/tests/test_example.m
function tests = test_example()
    % Test suite for example functionality.
    %
    % Returns:
    %   tests: Test suite structure
    
    tests = functiontests(localfunctions);
end

function test_truth(testCase)
    % Test basic arithmetic truth.
    %
    % Args:
    %   testCase: Test case object from functiontests
    
    arguments
        testCase (1,1) matlab.unittest.TestCase
    end
    
    verifyEqual(testCase, 1+1, 2);
end

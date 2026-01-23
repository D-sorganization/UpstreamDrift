// File: matlab/tests/test_example.m
function tests = test_example
    tests = functiontests(localfunctions);
end

function test_truth(testCase)
    verifyEqual(testCase, 1+1, 2);
end

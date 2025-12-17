% File: matlab/tests/test_quality_checks.m
function tests = test_quality_checks
    % TEST_QUALITY_CHECKS Test suite for MATLAB quality check functions
    %
    % This test suite validates that the MATLAB quality checking infrastructure
    % is properly configured and functional.
    %
    % Outputs:
    %   tests - Test suite created from local functions
    %
    % Usage:
    %   results = runtests('test_quality_checks')
    %
    % See also: functiontests, matlab.unittest.TestCase

    tests = functiontests(localfunctions);
end

function test_quality_config_function_exists(testCase)
    % TEST_QUALITY_CONFIG_FUNCTION_EXISTS Verify quality config function exists
    %
    % This test verifies that the matlab_quality_config function exists and
    % can be called without errors.
    %
    % Inputs:
    %   testCase - Test case object from MATLAB Unit Testing Framework

    arguments
        testCase (1,1) matlab.unittest.TestCase
    end

    % Test that the quality config function exists and can be called
    verifyTrue(testCase, exist('matlab_quality_config', 'file') == 2);

    % Test that it can be called without errors
    try
        matlab_quality_config();
        verifyTrue(testCase, true);
    catch ME
        verifyFail(testCase, sprintf('matlab_quality_config failed: %s', ME.message));
    end
end

function test_test_runner_function_exists(testCase)
    % TEST_TEST_RUNNER_FUNCTION_EXISTS Verify test runner function exists
    %
    % This test verifies that the run_matlab_tests function exists and
    % can be called without errors.
    %
    % Inputs:
    %   testCase - Test case object from MATLAB Unit Testing Framework

    arguments
        testCase (1,1) matlab.unittest.TestCase
    end

    % Test that the test runner function exists and can be called
    verifyTrue(testCase, exist('run_matlab_tests', 'file') == 2);

    % Test that it can be called without errors
    try
        run_matlab_tests();
        verifyTrue(testCase, true);
    catch ME
        verifyFail(testCase, sprintf('run_matlab_tests failed: %s', ME.message));
    end
end

function test_basic_math(testCase)
    % TEST_BASIC_MATH Basic mathematical operations test
    %
    % This test verifies that basic arithmetic operations work correctly.
    %
    % Inputs:
    %   testCase - Test case object from MATLAB Unit Testing Framework

    arguments
        testCase (1,1) matlab.unittest.TestCase
    end

    % Basic sanity test
    verifyEqual(testCase, 1+1, 2);
    verifyEqual(testCase, 2*3, 6);
    verifyEqual(testCase, 10/2, 5);
end

function test_string_operations(testCase)
    % TEST_STRING_OPERATIONS String manipulation operations test
    %
    % This test verifies that string operations work correctly.
    %
    % Inputs:
    %   testCase - Test case object from MATLAB Unit Testing Framework

    arguments
        testCase (1,1) matlab.unittest.TestCase
    end

    % Test string operations
    test_str = 'Hello World';
    verifyEqual(testCase, length(test_str), 11);
    verifyEqual(testCase, upper(test_str), 'HELLO WORLD');
end

function test_array_operations(testCase)
    % TEST_ARRAY_OPERATIONS Array operations test
    %
    % This test verifies that array operations work correctly.
    %
    % Inputs:
    %   testCase - Test case object from MATLAB Unit Testing Framework

    arguments
        testCase (1,1) matlab.unittest.TestCase
    end

    % Test array operations
    test_array = [1, 2, 3, 4, 5];
    verifyEqual(testCase, length(test_array), 5);
    verifyEqual(testCase, sum(test_array), 15);
    verifyEqual(testCase, mean(test_array), 3);
end

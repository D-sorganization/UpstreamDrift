function test_screw_theory()
% TEST_SCREW_THEORY  Unit tests for screw theory functions
%   Tests exponential map, logarithmic map, screw axes, and adjoint transforms
%
%   Usage:
%       test_screw_theory()
%
%   See also: exponential_map, logarithmic_map, screw_axis, constants

% Validate inputs (none required)
arguments
    % No input arguments
end

% Load constants (after arguments block)
addpath(genpath('../'));
constants;

fprintf('Running screw theory tests...\n');

num_tests = 0;
num_passed = 0;

% Note: In production, manage paths externally via startup.m or MATLAB path
% For testing, we add paths here
addpath('../screw_theory');
addpath('../spatial_v2');

%% Test 1: SCREW_AXIS - Pure rotation
num_tests = num_tests + 1;
try
    axis = [0; 0; 1];  % z-axis
    point = [0; 0; 0];  % Through origin
    S = screw_axis(axis, point, 0);

    % Should be [0; 0; 1; 0; 0; 0]
    expected = [0; 0; 1; 0; 0; 0];
    assert(norm(S - expected) < 1e-10, 'SCREW_AXIS: Wrong result for z-rotation');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 1: SCREW_AXIS pure rotation\n');
catch ME
    fprintf('  [FAIL] Test 1: SCREW_AXIS pure rotation - %s\n', ME.message);
end

%% Test 2: SCREW_AXIS - Pure translation
num_tests = num_tests + 1;
try
    axis = [1; 0; 0];
    point = [0; 0; 0];
    S = screw_axis(axis, point, inf);  % Infinite pitch = translation

    % Should be [0; 0; 0; 1; 0; 0]
    expected = [0; 0; 0; 1; 0; 0];
    assert(norm(S - expected) < 1e-10, 'SCREW_AXIS: Wrong result for translation');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 2: SCREW_AXIS pure translation\n');
catch ME
    fprintf('  [FAIL] Test 2: SCREW_AXIS pure translation - %s\n', ME.message);
end

%% Test 3: EXPONENTIAL_MAP - Pure rotation
num_tests = num_tests + 1;
try
    % 90° rotation about z-axis
    S = [0; 0; 1; 0; 0; 0];
    theta = pi/2;
    T = exponential_map(S, theta);

    % Expected rotation matrix
    R_expected = [0, -1, 0;
                  1,  0, 0;
                  0,  0, 1];

    assert(norm(T(1:3, 1:3) - R_expected, 'fro') < 1e-10, ...
        'EXPONENTIAL_MAP: Wrong rotation matrix');
    assert(norm(T(1:3, 4)) < 1e-10, 'EXPONENTIAL_MAP: Should have no translation');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 3: EXPONENTIAL_MAP pure rotation\n');
catch ME
    fprintf('  [FAIL] Test 3: EXPONENTIAL_MAP pure rotation - %s\n', ME.message);
end

%% Test 4: EXPONENTIAL_MAP - Pure translation
num_tests = num_tests + 1;
try
    % Translation along x by TEST_ANGLE_3_RAD meters
    S = [0; 0; 0; 1; 0; 0];
    theta = TEST_ANGLE_3_RAD;
    T = exponential_map(S, theta);

    assert(norm(T(1:3, 1:3) - eye(3), 'fro') < 1e-10, ...
        'EXPONENTIAL_MAP: Should have identity rotation');

    p_expected = [TEST_ANGLE_3_RAD; 0; 0];
    assert(norm(T(1:3, 4) - p_expected) < 1e-10, ...
        'EXPONENTIAL_MAP: Wrong translation');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 4: EXPONENTIAL_MAP pure translation\n');
catch ME
    fprintf('  [FAIL] Test 4: EXPONENTIAL_MAP pure translation - %s\n', ME.message);
end

%% Test 5: LOGARITHMIC_MAP and EXPONENTIAL_MAP inverse relationship
num_tests = num_tests + 1;
try
    % Start with a known screw
    S_orig = screw_axis([0; 0; 1], [1; 0; 0], 0);
    theta_orig = pi/3;

    % Apply exponential map
    T = exponential_map(S_orig, theta_orig);

    % Apply logarithmic map
    [S_recovered, theta_recovered] = logarithmic_map(T);

    % S can differ by sign, so check S*theta
    screw_orig = S_orig * theta_orig;
    screw_recovered = S_recovered * theta_recovered;

    assert(norm(screw_orig - screw_recovered) < 1e-8, ...
        'LOGARITHMIC_MAP: Failed to recover screw parameters');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 5: LOGARITHMIC_MAP inverse\n');
catch ME
    fprintf('  [FAIL] Test 5: LOGARITHMIC_MAP inverse - %s\n', ME.message);
end

%% Test 6: ADJOINT_TRANSFORM structure
num_tests = num_tests + 1;
try
    T = [0, -1,  0,  1;
         1,  0,  0,  2;
         0,  0,  1,  3;
         0,  0,  0,  1];

    Ad = adjoint_transform(T);

    % Check size
    assert(all(size(Ad) == [SPATIAL_DIM, SPATIAL_DIM]), 'ADJOINT_TRANSFORM: Wrong size');

    % Bottom-left should be zero
    assert(norm(Ad(4:6, 1:3), 'fro') < 1e-10, ...
        'ADJOINT_TRANSFORM: Bottom-left should be zero');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 6: ADJOINT_TRANSFORM structure\n');
catch ME
    fprintf('  [FAIL] Test 6: ADJOINT_TRANSFORM structure - %s\n', ME.message);
end

%% Test 7: ADJOINT_TRANSFORM composition property
num_tests = num_tests + 1;
try
    % Create two transforms
    T1 = exponential_map([0;0;1;0;0;0], pi/4);
    T2 = exponential_map([0;0;0;1;0;0], 0.5);

    Ad1 = adjoint_transform(T1);
    Ad2 = adjoint_transform(T2);
    Ad_comp = adjoint_transform(T1 * T2);

    % Property: Ad(T1*T2) = Ad(T1) * Ad(T2)
    assert(norm(Ad_comp - Ad1 * Ad2, 'fro') < 1e-10, ...
        'ADJOINT_TRANSFORM: Composition property failed');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 7: ADJOINT_TRANSFORM composition\n');
catch ME
    fprintf('  [FAIL] Test 7: ADJOINT_TRANSFORM composition - %s\n', ME.message);
end

%% Test 8: TWIST_TO_SPATIAL basic conversion
num_tests = num_tests + 1;
try
    omega = [1; 0; 0];
    v = [0; 1; 0];
    V = twist_to_spatial(omega, v);

    assert(norm(V(1:3) - omega) < 1e-10, 'TWIST_TO_SPATIAL: Wrong angular part');
    assert(norm(V(4:6) - v) < 1e-10, 'TWIST_TO_SPATIAL: Wrong linear part');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 8: TWIST_TO_SPATIAL conversion\n');
catch ME
    fprintf('  [FAIL] Test 8: TWIST_TO_SPATIAL conversion - %s\n', ME.message);
end

%% Test 9: WRENCH_TO_SPATIAL basic conversion
num_tests = num_tests + 1;
try
    moment = [0; 0; 1];
    force = [10; 0; 0];
    F = wrench_to_spatial(moment, force);

    assert(norm(F(1:3) - moment) < 1e-10, 'WRENCH_TO_SPATIAL: Wrong moment part');
    assert(norm(F(4:6) - force) < 1e-10, 'WRENCH_TO_SPATIAL: Wrong force part');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 9: WRENCH_TO_SPATIAL conversion\n');
catch ME
    fprintf('  [FAIL] Test 9: WRENCH_TO_SPATIAL conversion - %s\n', ME.message);
end

%% Test 10: SCREW_TO_TRANSFORM convenience function
num_tests = num_tests + 1;
try
    % Should match exponential_map(screw_axis(...), theta)
    axis = [0; 0; 1];
    point = [1; 0; 0];
    pitch = 0;
    theta = pi/2;

    T1 = screw_to_transform(axis, point, pitch, theta);

    S = screw_axis(axis, point, pitch);
    T2 = exponential_map(S, theta);

    assert(norm(T1 - T2, 'fro') < 1e-10, ...
        'SCREW_TO_TRANSFORM: Should match screw_axis + exponential_map');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 10: SCREW_TO_TRANSFORM convenience\n');
catch ME
    fprintf('  [FAIL] Test 10: SCREW_TO_TRANSFORM convenience - %s\n', ME.message);
end

%% Summary
fprintf('\n=== Test Summary ===\n');
fprintf('Passed: %d/%d tests\n', num_passed, num_tests);
if num_passed == num_tests
    fprintf('All tests PASSED!\n');
else
    fprintf('Some tests FAILED.\n');
end
end

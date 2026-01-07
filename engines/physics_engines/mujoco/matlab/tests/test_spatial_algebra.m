function test_spatial_algebra()
% TEST_SPATIAL_ALGEBRA  Unit tests for spatial algebra functions
%   Comprehensive tests for spatial_v2 library including:
%   - Spatial cross product operators (CRM, CRF)
%   - Spatial transformations (XROT, XLT, XTRANS)
%   - Spatial inertia (MCI, TRANSFORM_SPATIAL_INERTIA)
%   - Joint calculations (JCALC)

fprintf('Running spatial algebra tests...\n');

% Test counter
num_tests = 0;
num_passed = 0;

% Add paths
addpath('../spatial_v2');
addpath('..');
constants; % Load constants into workspace

%% Test 1: CRM - Spatial cross product operator for motion
num_tests = num_tests + 1;
try
    v = [1; 0; 0; 0; 1; 0];  % Angular [1,0,0], linear [0,1,0]
    X = crm(v);

    % Check dimensions
    assert(all(size(X) == [SPATIAL_DIM, SPATIAL_DIM]), 'CRM: Wrong size');

    % Check structure: top-left and bottom-right should be skew(w)
    assert(abs(X(1,1)) < eps, 'CRM: Diagonal should be zero');

    % Verify antisymmetry of blocks
    w_skew = X(1:3, 1:3);
    assert(norm(w_skew + w_skew', 'fro') < 1e-10, 'CRM: Should be skew-symmetric');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 1: CRM operator\n');
catch ME
    fprintf('  [FAIL] Test 1: CRM operator - %s\n', ME.message);
end

%% Test 2: CRF - Dual spatial cross product operator
num_tests = num_tests + 1;
try
    v = [1; 0; 0; 0; 1; 0];
    X_crf = crf(v);
    X_crm = crm(v);

    % Verify relationship: crf(v) = -crm(v)'
    assert(norm(X_crf + X_crm', 'fro') < 1e-10, ...
        'CRF: Should equal -CRM transpose');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 2: CRF operator\n');
catch ME
    fprintf('  [FAIL] Test 2: CRF operator - %s\n', ME.message);
end

%% Test 3: XROT - Rotation transformation
num_tests = num_tests + 1;
try
    % 90 degree rotation about z-axis
    theta = pi/2;
    E = [cos(theta), -sin(theta), 0;
         sin(theta),  cos(theta), 0;
         0,           0,          1];
    X = xrot(E);

    % Check block diagonal structure
    assert(norm(X(1:SPATIAL_LIN_DIM, SPATIAL_LIN_DIM+1:SPATIAL_DIM), 'fro') < 1e-10, 'XROT: Off-diagonal should be zero');
    assert(norm(X(SPATIAL_LIN_DIM+1:SPATIAL_DIM, 1:SPATIAL_LIN_DIM), 'fro') < 1e-10, 'XROT: Off-diagonal should be zero');
    assert(norm(X(1:SPATIAL_LIN_DIM, 1:SPATIAL_LIN_DIM) - E, 'fro') < 1e-10, 'XROT: Top block wrong');
    assert(norm(X(SPATIAL_LIN_DIM+1:SPATIAL_DIM, SPATIAL_LIN_DIM+1:SPATIAL_DIM) - E, 'fro') < 1e-10, 'XROT: Bottom block wrong');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 3: XROT transformation\n');
catch ME
    fprintf('  [FAIL] Test 3: XROT transformation - %s\n', ME.message);
end

%% Test 4: XLT - Translation transformation
num_tests = num_tests + 1;
try
    r = [1; 2; 3];
    X = xlt(r);

    % Check structure: top-right should be zero, others identity or -r_skew
    assert(norm(X(1:SPATIAL_LIN_DIM, SPATIAL_LIN_DIM+1:SPATIAL_DIM), 'fro') < 1e-10, 'XLT: Top-right should be zero');
    assert(norm(X(1:SPATIAL_LIN_DIM, 1:SPATIAL_LIN_DIM) - eye(SPATIAL_LIN_DIM), 'fro') < 1e-10, 'XLT: Top-left should be I');
    assert(norm(X(SPATIAL_LIN_DIM+1:SPATIAL_DIM, SPATIAL_LIN_DIM+1:SPATIAL_DIM) - eye(SPATIAL_LIN_DIM), 'fro') < 1e-10, 'XLT: Bottom-right should be I');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 4: XLT transformation\n');
catch ME
    fprintf('  [FAIL] Test 4: XLT transformation - %s\n', ME.message);
end

%% Test 5: XTRANS and INV_XTRANS
num_tests = num_tests + 1;
try
    E = [0, -1, 0; 1, 0, 0; 0, 0, 1];  % 90° rotation about z
    r = [1; 2; 0];

    X = xtrans(E, r);
    X_inv = inv_xtrans(E, r);

    % Verify inverse relationship
    prod = X * X_inv;
    assert(norm(prod - eye(SPATIAL_DIM), 'fro') < 1e-10, ...
        'XTRANS: X * X_inv should equal identity');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 5: XTRANS and inverse\n');
catch ME
    fprintf('  [FAIL] Test 5: XTRANS and inverse - %s\n', ME.message);
end

%% Test 6: MCI - Spatial inertia construction
num_tests = num_tests + 1;
try
    % Simple sphere at origin
    mass = 1.0;
    radius = 0.1;
    com = [0; 0; 0];
    I_sphere = (2/5) * mass * radius^2 * eye(3);

    I_spatial = mcI(mass, com, I_sphere);

    % Check symmetry
    assert(norm(I_spatial - I_spatial', 'fro') < 1e-10, ...
        'MCI: Spatial inertia should be symmetric');

    % Check positive definiteness (all eigenvalues positive)
    eigvals = eig(I_spatial);
    assert(all(eigvals > 0), 'MCI: Should be positive definite');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 6: MCI spatial inertia\n');
catch ME
    fprintf('  [FAIL] Test 6: MCI spatial inertia - %s\n', ME.message);
end

%% Test 7: JCALC - Revolute joint
num_tests = num_tests + 1;
try
    [Xj, S] = jcalc('Rz', pi/4);

    % Check motion subspace for Rz joint
    S_expected = [0; 0; 1; 0; 0; 0];
    assert(norm(S - S_expected) < 1e-10, 'JCALC: Wrong motion subspace for Rz');

    % Verify Xj is a valid spatial transform
    assert(all(size(Xj) == [SPATIAL_DIM, SPATIAL_DIM]), 'JCALC: Wrong transform size');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 7: JCALC revolute joint\n');
catch ME
    fprintf('  [FAIL] Test 7: JCALC revolute joint - %s\n', ME.message);
end

%% Test 8: JCALC - Prismatic joint
num_tests = num_tests + 1;
try
    [Xj, S] = jcalc('Px', 0.5);

    % Check motion subspace for Px joint
    S_expected = [0; 0; 0; 1; 0; 0];
    assert(norm(S - S_expected) < 1e-10, 'JCALC: Wrong motion subspace for Px');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 8: JCALC prismatic joint\n');
catch ME
    fprintf('  [FAIL] Test 8: JCALC prismatic joint - %s\n', ME.message);
end

%% Test 9: SPATIAL_CROSS - Motion cross product
num_tests = num_tests + 1;
try
    v1 = [1; 0; 0; 0; 0; 0];  % Pure angular velocity about x
    v2 = [0; 1; 0; 0; 0; 0];  % Pure angular velocity about y

    result = spatial_cross(v1, v2, 'motion');

    % v1 x v2 should give angular velocity about z
    expected = [0; 0; 1; 0; 0; 0];
    assert(norm(result - expected) < 1e-10, ...
        'SPATIAL_CROSS: Wrong result for motion cross product');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 9: SPATIAL_CROSS motion\n');
catch ME
    fprintf('  [FAIL] Test 9: SPATIAL_CROSS motion - %s\n', ME.message);
end

%% Test 10: TRANSFORM_SPATIAL_INERTIA
num_tests = num_tests + 1;
try
    % Create a simple inertia
    mass = 1.0;
    I_com = 0.01 * eye(3);
    com = [0; 0; 0];
    I_B = mcI(mass, com, I_com);

    % Identity transform should not change inertia
    X = eye(SPATIAL_DIM);
    I_A = transform_spatial_inertia(I_B, X);

    assert(norm(I_A - I_B, 'fro') < 1e-10, ...
        'TRANSFORM_SPATIAL_INERTIA: Identity transform should preserve inertia');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 10: TRANSFORM_SPATIAL_INERTIA\n');
catch ME
    fprintf('  [FAIL] Test 10: TRANSFORM_SPATIAL_INERTIA - %s\n', ME.message);
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

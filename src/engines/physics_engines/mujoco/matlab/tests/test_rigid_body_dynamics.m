function test_rigid_body_dynamics()
% TEST_RIGID_BODY_DYNAMICS  Unit tests for Featherstone's RBD algorithms
%   Tests RNEA, CRBA, and ABA on simple kinematic chains
%
%   Usage:
%       test_rigid_body_dynamics()
%
%   See also: aba, rnea, crba, constants

% Validate inputs (none required)
arguments
    % No input arguments
end

% Load constants (after arguments block)
addpath(genpath('../'));
constants;

fprintf('Running rigid body dynamics tests...\n');

num_tests = 0;
num_passed = 0;

% Note: In production, manage paths externally via startup.m or MATLAB path
% For testing, we add paths here
addpath('../rigid_body_dynamics');
addpath('../spatial_v2');

%% Create a simple 2-link planar robot for testing
function model = create_2link_model()
    % Constants are already loaded in parent function scope
    
    model.NB = 2;  % Two bodies
    model.parent = [0, 1];  % Body 1 attached to base, body 2 to body 1
    model.jtype = {'Rz', 'Rz'};  % Two revolute joints about z-axis
    model.gravity = [0; 0; 0; 0; 0; -GRAVITY_STANDARD_M_S2];

    % Link parameters
    L1 = LINK_LENGTH_1_M;  % Length of link 1 (m)
    L2 = LINK_LENGTH_2_M;  % Length of link 2 (m)
    m1 = MASS_1_KG;  % Mass of link 1 (kg)
    m2 = MASS_2_KG;  % Mass of link 2 (kg)

    % Inertia of uniform density rod: I = (1/12)*m*L^2
    I1 = (1/12) * m1 * L1^2;
    I2 = (1/12) * m2 * L2^2;

    % Joint transforms (Xtree): from predecessor to joint frame
    % Both joints at base of their respective links
    model.Xtree = zeros(SPATIAL_DIM, SPATIAL_DIM, 2);
    model.Xtree(:, :, 1) = eye(SPATIAL_DIM);  % Joint 1 at origin
    model.Xtree(:, :, 2) = xlt([L1; 0; 0]);  % Joint 2 at end of link 1

    % Spatial inertias about joint frames
    model.I = zeros(SPATIAL_DIM, SPATIAL_DIM, 2);

    % Link 1: COM at L1/2 from joint
    com1 = [L1/2; 0; 0];
    I_rot1 = diag([0, 0, I1]);  % Inertia about COM (rod along x)
    model.I(:, :, 1) = mcI(m1, com1, I_rot1);

    % Link 2: COM at L2/2 from joint
    com2 = [L2/2; 0; 0];
    I_rot2 = diag([0, 0, I2]);
    model.I(:, :, 2) = mcI(m2, com2, I_rot2);
end

%% Test 1: CRBA - Mass matrix symmetry
num_tests = num_tests + 1;
try
    model = create_2link_model();
    q = [0; 0];  % Zero configuration

    H = crba(model, q);

    % Check symmetry
    assert(norm(H - H', 'fro') < 1e-10, 'CRBA: Mass matrix should be symmetric');

    % Check positive definiteness
    eigvals = eig(H);
    assert(all(eigvals > 0), 'CRBA: Mass matrix should be positive definite');

    % Check size
    assert(all(size(H) == [2, 2]), 'CRBA: Wrong size');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 1: CRBA mass matrix properties\n');
catch ME
    fprintf('  [FAIL] Test 1: CRBA mass matrix properties - %s\n', ME.message);
end

%% Test 2: RNEA - Zero motion should give gravity terms
num_tests = num_tests + 1;
try
    model = create_2link_model();
    q = [pi/4; -pi/6];  % Some configuration
    qd = [0; 0];   % Zero velocity
    qdd = [0; 0];  % Zero acceleration

    tau = rnea(model, q, qd, qdd);

    % With no motion, tau should only contain gravity terms
    % Both should be non-zero due to gravity
    assert(abs(tau(1)) > 1e-6, 'RNEA: Should have gravity torque on joint 1');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 2: RNEA gravity compensation\n');
catch ME
    fprintf('  [FAIL] Test 2: RNEA gravity compensation - %s\n', ME.message);
end

%% Test 3: RNEA and CRBA consistency (H*qdd = tau - C - g)
num_tests = num_tests + 1;
try
    model = create_2link_model();
    q = [0.5; -0.3];
    qd = [0.1; 0.2];
    qdd = [0.5; -0.2];

    % Compute mass matrix
    H = crba(model, q);

    % Compute full inverse dynamics
    tau_full = rnea(model, q, qd, qdd);

    % Compute gravity and Coriolis terms (qdd = 0)
    tau_bias = rnea(model, q, qd, zeros(2, 1));

    % Check: H*qdd = tau_full - tau_bias
    H_qdd = H * qdd;
    tau_inertial = tau_full - tau_bias;

    assert(norm(H_qdd - tau_inertial) < 1e-8, ...
        'RNEA/CRBA: Inconsistent dynamics computation');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 3: RNEA/CRBA consistency\n');
catch ME
    fprintf('  [FAIL] Test 3: RNEA/CRBA consistency - %s\n', ME.message);
end

%% Test 4: ABA - Forward dynamics basic test
num_tests = num_tests + 1;
try
    model = create_2link_model();
    q = [0; 0];
    qd = [0; 0];
    tau = [0; 0];  % No applied torques

    qdd = aba(model, q, qd, tau);

    % Check output size
    assert(length(qdd) == 2, 'ABA: Wrong output size');

    % With no applied torque, should have negative acceleration due to gravity
    % (robot will fall down)
    assert(qdd(1) < 0, 'ABA: Should accelerate downward under gravity');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 4: ABA forward dynamics\n');
catch ME
    fprintf('  [FAIL] Test 4: ABA forward dynamics - %s\n', ME.message);
end

%% Test 5: ABA and RNEA inverse relationship
num_tests = num_tests + 1;
try
    model = create_2link_model();
    q = [0.3; -0.5];
    qd = [0.1; -0.2];
    tau = [1.5; 0.5];

    % Forward dynamics: compute qdd from tau
    qdd_fd = aba(model, q, qd, tau);

    % Inverse dynamics: compute tau from qdd
    tau_id = rnea(model, q, qd, qdd_fd);

    % Should recover original torques
    assert(norm(tau - tau_id) < 1e-8, ...
        'ABA/RNEA: Forward and inverse dynamics inconsistent');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 5: ABA/RNEA inverse relationship\n');
catch ME
    fprintf('  [FAIL] Test 5: ABA/RNEA inverse relationship - %s\n', ME.message);
end

%% Test 6: ABA vs mass matrix inversion
num_tests = num_tests + 1;
try
    model = create_2link_model();
    q = [0.2; -0.3];
    qd = [0.05; 0.1];
    tau = [1.0; 0.3];

    % Method 1: ABA (O(n) algorithm)
    qdd_aba = aba(model, q, qd, tau);

    % Method 2: Explicit inversion qdd = H^-1 * (tau - C - g)
    H = crba(model, q);
    tau_bias = rnea(model, q, qd, zeros(2, 1));
    qdd_inv = H \ (tau - tau_bias);

    % Should give same result
    assert(norm(qdd_aba - qdd_inv) < 1e-8, ...
        'ABA: Should match explicit mass matrix inversion');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 6: ABA vs explicit inversion\n');
catch ME
    fprintf('  [FAIL] Test 6: ABA vs explicit inversion - %s\n', ME.message);
end

%% Test 7: CRBA configuration dependence
num_tests = num_tests + 1;
try
    model = create_2link_model();

    % Mass matrix should change with configuration
    H1 = crba(model, [0; 0]);
    H2 = crba(model, [pi/2; 0]);

    % Should be different
    assert(norm(H1 - H2, 'fro') > 1e-6, ...
        'CRBA: Mass matrix should depend on configuration');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 7: CRBA configuration dependence\n');
catch ME
    fprintf('  [FAIL] Test 7: CRBA configuration dependence - %s\n', ME.message);
end

%% Test 8: RNEA with external forces
num_tests = num_tests + 1;
try
    model = create_2link_model();
    q = [0; 0];
    qd = [0; 0];
    qdd = [0; 0];

    % Apply external force on link 2
    f_ext = zeros(SPATIAL_DIM, 2);
    f_ext(:, 2) = [0; 0; 0; 10; 0; 0];  % 10N force in x direction

    tau_no_ext = rnea(model, q, qd, qdd);
    tau_with_ext = rnea(model, q, qd, qdd, f_ext);

    % External force should change required torques
    assert(norm(tau_no_ext - tau_with_ext) > 1e-6, ...
        'RNEA: External forces should affect torques');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 8: RNEA with external forces\n');
catch ME
    fprintf('  [FAIL] Test 8: RNEA with external forces - %s\n', ME.message);
end

%% Test 9: Single-body system (edge case)
num_tests = num_tests + 1;
try
    model_single.NB = 1;
    model_single.parent = 0;
    model_single.jtype = {'Rz'};
    model_single.gravity = [0; 0; 0; 0; 0; -GRAVITY_STANDARD_M_S2];
    model_single.Xtree = eye(SPATIAL_DIM, SPATIAL_DIM, 1);

    % Simple pendulum
    mass = 1.0;
    length = 0.5;
    com = [length; 0; 0];
    I_com = zeros(3, 3);
    I_com(3, 3) = mass * length^2;
    model_single.I = mcI(mass, com, I_com);

    q = 0;
    qd = 0;
    qdd = 0;

    tau = rnea(model_single, q, qd, qdd);
    H = crba(model_single, q);
    qdd_test = aba(model_single, q, qd, tau);

    % Should execute without error and give scalar results
    assert(isscalar(tau), 'Single body: tau should be scalar');
    assert(isscalar(H), 'Single body: H should be scalar');
    assert(isscalar(qdd_test), 'Single body: qdd should be scalar');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 9: Single-body system\n');
catch ME
    fprintf('  [FAIL] Test 9: Single-body system - %s\n', ME.message);
end

%% Test 10: Energy conservation check
num_tests = num_tests + 1;
try
    model = create_2link_model();
    q = [pi/4; -pi/3];
    qd = [0.5; -0.3];

    % Compute kinetic energy: T = 0.5 * qd' * H * qd
    H = crba(model, q);
    T = 0.5 * qd' * H * qd;

    % Kinetic energy should be positive
    assert(T > 0, 'Energy: Kinetic energy should be positive');

    % Mass matrix gives correct kinetic energy
    assert(T < 100, 'Energy: Kinetic energy seems unreasonably large');

    num_passed = num_passed + 1;
    fprintf('  [PASS] Test 10: Energy computation\n');
catch ME
    fprintf('  [FAIL] Test 10: Energy computation - %s\n', ME.message);
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

% EXAMPLE_2LINK_ROBOT  Demonstrate Featherstone algorithms on 2-link robot
%   This script demonstrates the RNEA, CRBA, and ABA algorithms on a simple
%   2-link planar robot and compares the results.
%
%   Usage:
%       example_2link_robot
%
%   See also: aba, rnea, crba, constants

clear; close all; clc;

% Add paths (note: in production, manage paths externally via startup.m)
addpath('..');  % Add parent directory to access constants.m
addpath('../spatial_v2');
addpath('../rigid_body_dynamics');

% Load constants
constants;

fprintf('=== 2-Link Robot Dynamics Example ===\n\n');

%% Define robot model
fprintf('Creating 2-link planar robot model...\n');

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

% Joint transforms
model.Xtree = zeros(SPATIAL_DIM, SPATIAL_DIM, 2);
model.Xtree(:, :, 1) = eye(SPATIAL_DIM);
model.Xtree(:, :, 2) = xlt([L1; 0; 0]);

% Spatial inertias
com1 = [L1/2; 0; 0];
I_rot1 = diag([0, 0, I1]);
model.I(:, :, 1) = mcI(m1, com1, I_rot1);

com2 = [L2/2; 0; 0];
I_rot2 = diag([0, 0, I2]);
model.I(:, :, 2) = mcI(m2, com2, I_rot2);

fprintf('  Link 1: L=%.2fm, m=%.2fkg\n', L1, m1);
fprintf('  Link 2: L=%.2fm, m=%.2fkg\n\n', L2, m2);

%% Test CRBA - Mass Matrix
fprintf('Testing CRBA (Composite Rigid Body Algorithm)...\n');

q = [pi/6; -pi/4];  % Configuration
fprintf('  Configuration: q = [%.3f, %.3f] rad\n', q(1), q(2));

H = crba(model, q);
fprintf('  Mass matrix H:\n');
disp(H);

% Verify properties
fprintf('  Properties:\n');
fprintf('    Symmetric: %s\n', mat2str(norm(H - H', 'fro') < 1e-10));
fprintf('    Positive definite: %s\n', mat2str(all(eig(H) > 0)));

%% Test RNEA - Inverse Dynamics
fprintf('\nTesting RNEA (Recursive Newton-Euler Algorithm)...\n');

q = [pi/4; -pi/3];
qd = [0.5; -0.3];
qdd = [0.2; 0.1];

fprintf('  q   = [%.3f, %.3f] rad\n', q(1), q(2));
fprintf('  qd  = [%.3f, %.3f] rad/s\n', qd(1), qd(2));
fprintf('  qdd = [%.3f, %.3f] rad/s^2\n', qdd(1), qdd(2));

tau = rnea(model, q, qd, qdd);
fprintf('  Required torques: tau = [%.4f, %.4f] N*m\n', tau(1), tau(2));

% Decompose into components
tau_gravity = rnea(model, q, zeros(2, 1), zeros(2, 1));
tau_coriolis = rnea(model, q, qd, zeros(2, 1)) - tau_gravity;
H = crba(model, q);
tau_inertial = H * qdd;

fprintf('  Torque decomposition:\n');
fprintf('    Inertial:  [%.4f, %.4f] N*m\n', tau_inertial(1), tau_inertial(2));
fprintf('    Coriolis:  [%.4f, %.4f] N*m\n', tau_coriolis(1), tau_coriolis(2));
fprintf('    Gravity:   [%.4f, %.4f] N*m\n', tau_gravity(1), tau_gravity(2));

%% Test ABA - Forward Dynamics
fprintf('\nTesting ABA (Articulated Body Algorithm)...\n');

tau_applied = [1.0; 0.5];
fprintf('  Applied torques: tau = [%.3f, %.3f] N*m\n', ...
    tau_applied(1), tau_applied(2));

qdd_computed = aba(model, q, qd, tau_applied);
fprintf('  Computed accelerations: qdd = [%.4f, %.4f] rad/s^2\n', ...
    qdd_computed(1), qdd_computed(2));

%% Verify ABA and RNEA are inverses
fprintf('\nVerifying ABA and RNEA are inverses...\n');

% Forward dynamics -> inverse dynamics
tau_recovered = rnea(model, q, qd, qdd_computed);
fprintf('  Original tau:   [%.6f, %.6f] N*m\n', tau_applied(1), tau_applied(2));
fprintf('  Recovered tau:  [%.6f, %.6f] N*m\n', tau_recovered(1), tau_recovered(2));
fprintf('  Error: %.2e N*m\n', norm(tau_applied - tau_recovered));

%% Compare ABA with explicit mass matrix inversion
fprintf('\nComparing ABA with explicit H^-1 method...\n');

H = crba(model, q);
tau_bias = rnea(model, q, qd, zeros(2, 1));
qdd_explicit = H \ (tau_applied - tau_bias);

fprintf('  ABA result:      [%.6f, %.6f] rad/s^2\n', ...
    qdd_computed(1), qdd_computed(2));
fprintf('  H^-1 result:     [%.6f, %.6f] rad/s^2\n', ...
    qdd_explicit(1), qdd_explicit(2));
fprintf('  Error: %.2e rad/s^2\n', norm(qdd_computed - qdd_explicit));

fprintf('\n=== All tests completed successfully! ===\n');

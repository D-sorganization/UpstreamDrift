% EXAMPLE_SCREW_THEORY  Demonstrate screw theory operations
%   This script demonstrates screw axes, exponential map, logarithmic map,
%   and adjoint transformations.

clear; close all; clc;

% Add paths
addpath('..');  % Add parent directory to access constants.m
addpath('../screw_theory');
addpath('../spatial_v2');

fprintf('=== Screw Theory Examples ===\n\n');

%% Example 1: Pure rotation
fprintf('Example 1: Pure rotation about z-axis\n');

axis = [0; 0; 1];  % z-axis
point = [1; 0; 0];  % Point on x-axis
pitch = 0;  % Pure rotation
theta = pi/2;  % 90 degrees

S = screw_axis(axis, point, pitch);
fprintf('  Screw axis S = [%.3f %.3f %.3f %.3f %.3f %.3f]''\n', S);

T = exponential_map(S, theta);
fprintf('  Transformation matrix T:\n');
disp(T);

fprintf('  Rotation extracts:\n');
fprintf('    R(1,1:3) = [%.3f %.3f %.3f]\n', T(1, 1:3));
fprintf('    R(2,1:3) = [%.3f %.3f %.3f]\n', T(2, 1:3));
fprintf('    R(3,1:3) = [%.3f %.3f %.3f]\n', T(3, 1:3));
fprintf('  Translation p = [%.3f %.3f %.3f]''\n\n', T(1:3, 4));

%% Example 2: Pure translation
fprintf('Example 2: Pure translation along x-axis\n');

% Load constants
constants;

axis = [1; 0; 0];
point = [0; 0; 0];
pitch = inf;  % Infinite pitch = translation
theta = TEST_ANGLE_3_RAD;  % Test angle (rad or m)

S = screw_axis(axis, point, pitch);
fprintf('  Screw axis S = [%.3f %.3f %.3f %.3f %.3f %.3f]''\n', S);

T = exponential_map(S, theta);
fprintf('  Transformation matrix T:\n');
disp(T);

fprintf('  Translation p = [%.3f %.3f %.3f]''\n\n', T(1:3, 4));

%% Example 3: Screw motion (rotation + translation)
fprintf('Example 3: Screw motion (rotation with pitch)\n');

axis = [0; 0; 1];  % z-axis
point = [0; 0; 0];
pitch = 0.1;  % 0.1 m per radian
theta = 2*pi;  % One full rotation

S = screw_axis(axis, point, pitch);
fprintf('  Screw axis S = [%.3f %.3f %.3f %.3f %.3f %.3f]''\n', S);
fprintf('  Pitch h = %.3f m/rad\n', pitch);
fprintf('  Angle theta = %.3f rad (360 degrees)\n', theta);

T = exponential_map(S, theta);
fprintf('  After one full rotation:\n');
fprintf('    Translation along z = %.3f m\n', T(3, 4));
fprintf('    Expected (h*theta) = %.3f m\n\n', pitch * theta);

%% Example 4: Logarithmic map (inverse of exponential)
fprintf('Example 4: Logarithmic map\n');

% Create a transformation
S_orig = screw_axis([0; 0; 1], [1; 0; 0], 0);
theta_orig = pi/3;
T = exponential_map(S_orig, theta_orig);

fprintf('  Original screw parameters:\n');
fprintf('    theta = %.6f rad\n', theta_orig);

% Extract screw parameters
[S_recovered, theta_recovered] = logarithmic_map(T);

fprintf('  Recovered screw parameters:\n');
fprintf('    theta = %.6f rad\n', theta_recovered);

% Compare S*theta (allows for sign difference in S)
screw_orig = S_orig * theta_orig;
screw_recovered = S_recovered * theta_recovered;

fprintf('  Error in S*theta: %.2e\n\n', norm(screw_orig - screw_recovered));

%% Example 5: Adjoint transformation
fprintf('Example 5: Adjoint transformation\n');

% Create transformation
T = screw_to_transform([0; 0; 1], [1; 0; 0], 0, pi/4);
Ad = adjoint_transform(T);

fprintf('  Transformation T represents:\n');
fprintf('    45° rotation about z-axis\n');
fprintf('    Through point [1, 0, 0]\n\n');

% Transform a twist
V_b = [0; 0; 1; 0; 0; 0];  % Angular velocity about z in frame b
V_a = Ad * V_b;

fprintf('  Twist in frame b: [%.3f %.3f %.3f %.3f %.3f %.3f]''\n', V_b);
fprintf('  Twist in frame a: [%.3f %.3f %.3f %.3f %.3f %.3f]''\n', V_a);

%% Example 6: Composition property of adjoint
fprintf('\nExample 6: Composition property Ad(T1*T2) = Ad(T1)*Ad(T2)\n');

T1 = exponential_map([0; 0; 1; 0; 0; 0], pi/4);
T2 = exponential_map([0; 0; 0; 1; 0; 0], 0.5);

Ad1 = adjoint_transform(T1);
Ad2 = adjoint_transform(T2);
Ad_comp = adjoint_transform(T1 * T2);
Ad_product = Ad1 * Ad2;

fprintf('  Error in composition: %.2e\n', norm(Ad_comp - Ad_product, 'fro'));

fprintf('\n=== All examples completed! ===\n');

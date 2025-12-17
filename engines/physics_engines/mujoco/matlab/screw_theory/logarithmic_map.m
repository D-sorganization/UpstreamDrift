function [S, theta] = logarithmic_map(T)
% LOGARITHMIC_MAP  Extract screw axis and displacement from transformation
%   [S, theta] = LOGARITHMIC_MAP(T) computes the screw axis S and
%   displacement theta from a homogeneous transformation matrix T.
%
%   This implements the logarithmic map from SE(3) to se(3):
%   [S]*theta = log(T)
%
%   This is the inverse of the exponential map.
%
% Inputs:
%   T     - 4x4 homogeneous transformation matrix
%
% Outputs:
%   S     - 6x1 normalized screw axis [omega; v]
%   theta - Scalar displacement (radians or meters)
%
% Algorithm:
%   1. Extract rotation matrix R and position p from T
%   2. If R = I, this is pure translation
%   3. Otherwise, compute rotation axis and angle
%   4. Compute linear velocity component from position
%
% Examples:
%   % Extract screw from a rotation matrix
%   T = [0 -1  0  1;
%        1  0  0  0;
%        0  0  1  0;
%        0  0  0  1];
%   [S, theta] = logarithmic_map(T);
%
% References:
%   Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
%   Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.
%
% See also: EXPONENTIAL_MAP, SCREW_AXIS

% Validate input
arguments
    T (4,4) {mustBeNumeric, mustBeFinite}
end

% Load constants (after arguments block)
addpath('..');
constants;

validateattributes(T, {'numeric'}, {'size', [4, 4], 'finite'}, ...
    'logarithmic_map', 'T');

% Verify it's a valid transformation matrix
if norm(T(4, :) - [0, 0, 0, 1]) > 1e-10
    warning('logarithmic_map:InvalidTransform', ...
        'Bottom row of T should be [0 0 0 1]');
end

% Extract rotation and position
R = T(1:3, 1:3);
p = T(1:3, 4);

% Check if rotation is identity (pure translation)
if norm(R - eye(SPATIAL_LIN_DIM), 'fro') < 1e-10
    % Pure translation
    S = [zeros(SPATIAL_LIN_DIM, 1); p / norm(p)];
    theta = norm(p);
    return;
end

% Compute rotation angle using trace
% trace(R) = 1 + 2*cos(theta)
cos_theta = (trace(R) - 1) / 2;
cos_theta = max(-1, min(1, cos_theta));  % Clamp to [-1, 1]
theta = acos(cos_theta);

% Handle special cases
if abs(theta) < eps
    % No rotation, pure translation
    S = [zeros(3, 1); p / norm(p)];
    theta = norm(p);
    return;
end

if abs(theta - PI) < eps
    % 180 degree rotation - need special handling
    % Find the eigenvector corresponding to eigenvalue 1
    [V, D] = eig(R);
    [~, idx] = min(abs(diag(D) - 1));
    omega = real(V(:, idx));
    omega = omega / norm(omega);
else
    % General case: extract axis from skew-symmetric part
    % omega_hat = (R - R') / (2*sin(theta))
    omega_skew = (R - R') / (2 * sin(theta));
    omega = [omega_skew(3, 2); omega_skew(1, 3); omega_skew(2, 1)];
end

% Compute linear velocity component
% p = (I*theta + (1-cos(theta))*omega_hat + (theta-sin(theta))*omega_hat^2) * v
% Solve for v using the inverse transformation
omega_hat = skew(omega);
G_inv = (1 / theta) * eye(SPATIAL_LIN_DIM) - 0.5 * omega_hat + ...
        (1 / theta - 0.5 / tan(theta / 2)) * (omega_hat * omega_hat);
v = G_inv * p;

% Construct screw axis
S = [omega; v];
end

function S = skew(v)
% SKEW  Create 3x3 skew-symmetric matrix from 3x1 vector

S = [    0,  -v(3),   v(2);
      v(3),      0,  -v(1);
     -v(2),   v(1),      0];
end

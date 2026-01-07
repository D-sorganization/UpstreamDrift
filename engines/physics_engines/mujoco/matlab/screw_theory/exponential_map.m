function T = exponential_map(S, theta)
% EXPONENTIAL_MAP  Screw motion via matrix exponential
%   T = EXPONENTIAL_MAP(S, theta) computes the rigid body transformation
%   resulting from moving along screw axis S by amount theta.
%
%   This implements the exponential map from se(3) to SE(3):
%   T = exp([S]*theta)
%
%   where [S] is the 4x4 matrix representation of the screw axis,
%   and T is a 4x4 homogeneous transformation matrix.
%
%   For pure rotation (||omega|| = 1, pitch = 0):
%   T = [exp(omega_hat*theta),  (I - exp(omega_hat*theta))(omega x v) + omega*omega'*v*theta]
%       [        0                                    1                                      ]
%
%   For pure translation (omega = 0):
%   T = [I,  v*theta]
%       [0,    1    ]
%
% Inputs:
%   S     - 6x1 screw axis [omega; v]
%   theta - Scalar displacement along screw (radians for rotation, meters for translation)
%
% Outputs:
%   T     - 4x4 homogeneous transformation matrix in SE(3)
%
% Examples:
%   % Rotation of 90 degrees about z-axis
%   S = screw_axis([0; 0; 1], [0; 0; 0]);
%   T = exponential_map(S, pi/2);
%
%   % Translation of 0.5m along x-axis
%   S = screw_axis([1; 0; 0], [0; 0; 0], inf);
%   T = exponential_map(S, 0.5);
%
% References:
%   Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
%   Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.
%   Chapter 3: Rigid-Body Motions
%
% See also: SCREW_AXIS, LOGARITHMIC_MAP, ADJOINT_TRANSFORM

% Validate inputs
arguments
    S (6,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    theta (1,1) {mustBeNumeric, mustBeFinite, mustBeScalar}
end

% Load constants (after arguments block)
addpath('..');
constants;

validateattributes(S, {'numeric'}, {'size', [SPATIAL_DIM, 1], 'finite'}, ...
    'exponential_map', 'S');
validateattributes(theta, {'numeric'}, {'scalar', 'finite'}, ...
    'exponential_map', 'theta');

% Extract angular and linear components
omega = S(1:3);
v = S(4:6);

omega_norm = norm(omega);

if omega_norm < eps
    % Pure translation (prismatic motion)
    R = eye(3);
    p = v * theta;
else
    % Rotation or screw motion
    % Normalize for computation
    omega_hat = skew(omega);

    % Rodrigues' formula for rotation matrix
    % R = I + sin(theta)*omega_hat + (1-cos(theta))*omega_hat^2
    R = eye(3) + sin(theta) * omega_hat + ...
        (1 - cos(theta)) * (omega_hat * omega_hat);

    % Position component (Proposition 3.14 in Lynch & Park)
    % p = (I*theta + (1-cos(theta))*omega_hat + (theta-sin(theta))*omega_hat^2) * v
    p = (eye(3) * theta + (1 - cos(theta)) * omega_hat + ...
         (theta - sin(theta)) * (omega_hat * omega_hat)) * v;
end

% Construct homogeneous transformation matrix
T = [R, p;
     zeros(1, 3), 1];
end

function S = skew(v)
% SKEW  Create 3x3 skew-symmetric matrix from 3x1 vector

S = [    0,  -v(3),   v(2);
      v(3),      0,  -v(1);
     -v(2),   v(1),      0];
end

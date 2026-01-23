function Ad = adjoint_transform(T)
% ADJOINT_TRANSFORM  Compute adjoint transformation matrix
%   Ad = ADJOINT_TRANSFORM(T) computes the 6x6 adjoint transformation
%   matrix from a 4x4 homogeneous transformation matrix T.
%
%   The adjoint transformation maps twists (spatial velocities) from one
%   frame to another:
%   V_a = Ad_{T_ab} * V_b
%
%   The adjoint matrix has the form:
%   Ad = [R,      p_hat*R]
%        [0,         R   ]
%
%   where R is the 3x3 rotation matrix and p_hat is the skew-symmetric
%   matrix of the position vector p.
%
%   This is equivalent to Featherstone's spatial transformation matrix.
%
% Inputs:
%   T  - 4x4 homogeneous transformation matrix
%
% Outputs:
%   Ad - 6x6 adjoint transformation matrix
%
% Properties:
%   - Ad(T1 * T2) = Ad(T1) * Ad(T2)
%   - Ad(T^-1) = Ad(T)^-1
%   - For transforming wrenches: use Ad'
%
% Example:
%   % Create transformation: 90° rotation about z, translate [1;0;0]
%   T = [0 -1  0  1;
%        1  0  0  0;
%        0  0  1  0;
%        0  0  0  1];
%   Ad = adjoint_transform(T);
%
%   % Transform a twist
%   V_b = [0; 0; 1; 0; 0; 0];  % Angular velocity about z
%   V_a = Ad * V_b;
%
% References:
%   Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
%   Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.
%   Chapter 3: Rigid-Body Motions, Section 3.3.3
%
% See also: XTRANS, EXPONENTIAL_MAP, TWIST_TO_SPATIAL

% Validate input
arguments
    T (4,4) {mustBeNumeric, mustBeFinite}
end

% Load constants (after arguments block)
addpath('..');
constants;

validateattributes(T, {'numeric'}, {'size', [4, 4], 'finite'}, ...
    'adjoint_transform', 'T');

% Extract rotation and position
R = T(1:3, 1:3);
p = T(1:3, 4);

% Create skew-symmetric matrix of position
p_hat = skew(p);

% Build adjoint matrix (SPATIAL_DIM x SPATIAL_DIM)
Ad = [R,        p_hat * R;
      zeros(SPATIAL_LIN_DIM), R        ];
end

function S = skew(v)
% SKEW  Create 3x3 skew-symmetric matrix from 3x1 vector

S = [    0,  -v(3),   v(2);
      v(3),      0,  -v(1);
     -v(2),   v(1),      0];
end

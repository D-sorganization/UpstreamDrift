function X = xtrans(E, r)
% XTRANS  General spatial coordinate transformation (Plücker transform)
%   X = XTRANS(E, r) returns the 6x6 spatial transformation matrix for a
%   general rigid body transformation consisting of rotation E and
%   translation r.
%
%   This transforms spatial vectors from frame B to frame A, where:
%   - Frame B is rotated from frame A by rotation matrix E
%   - Frame B origin is translated by vector r from frame A origin
%
%   The transformation matrix (Plücker transform) is:
%   X = [ E        0     ]
%       [ -E*rx    E     ]
%
%   where rx is the skew-symmetric matrix of r.
%
%   For motion vectors: v_A = X * v_B
%   For force vectors:  f_B = X' * f_A  (note the transpose)
%
% Inputs:
%   E - 3x3 rotation matrix from frame A to frame B
%   r - 3x1 translation vector from A to B origin, expressed in frame A
%
% Outputs:
%   X - 6x6 spatial transformation matrix (Plücker transform)
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 2: Spatial Vector Algebra, Section 2.6
%
% See also: XLT, XROT, PLUCKER_TRANSFORM, INV_XTRANS

% Validate inputs
arguments
    E (3,3) {mustBeNumeric, mustBeFinite}
    r (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
end

validateattributes(E, {'numeric'}, {'size', [3, 3]}, 'xtrans', 'E');
validateattributes(r, {'numeric'}, {'size', [3, 1]}, 'xtrans', 'r');

% Load constants (after arguments block)
addpath('..');
constants;

% Create skew-symmetric matrix
r_skew = skew(r);

% Build the SPATIAL_DIM x SPATIAL_DIM Plücker transformation matrix
X = [E,              zeros(SPATIAL_LIN_DIM, SPATIAL_LIN_DIM);
     -E * r_skew,    E          ];
end

function S = skew(v)
% SKEW  Create 3x3 skew-symmetric matrix from 3x1 vector

S = [    0,  -v(3),   v(2);
      v(3),      0,  -v(1);
     -v(2),   v(1),      0];
end

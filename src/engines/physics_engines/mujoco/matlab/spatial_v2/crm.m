function X = crm(v)
% CRM  Spatial cross product operator for motion vectors
%   X = CRM(v) returns the 6x6 matrix X such that X*m = v x m for any
%   spatial motion vector m, where x is the spatial cross product.
%
%   Following Featherstone's spatial vector algebra notation:
%   v = [w; v] where w is angular velocity, v is linear velocity
%
%   The cross product operator is:
%   crm(v) = [ skew(w)    0      ]
%            [ skew(v)  skew(w)  ]
%
% Inputs:
%   v - 6x1 spatial motion vector [angular; linear]
%
% Outputs:
%   X - 6x6 spatial cross product matrix
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 2: Spatial Vector Algebra
%
% See also: CRF, SKEW, SPATIAL_CROSS

% Validate input
arguments
    v (6,1) {mustBeNumeric, mustBeFinite, mustBeVector}
end

validateattributes(v, {'numeric'}, {'size', [6, 1]}, 'crm', 'v');

% Load constants (after arguments block)
addpath('..');
constants;

% Extract angular and linear components
w = v(1:3);
vlin = v(4:6);

% Construct skew-symmetric matrices
w_skew = skew(w);
v_skew = skew(vlin);

% Build the SPATIAL_DIM x SPATIAL_DIM spatial cross product operator
X = [w_skew,           zeros(SPATIAL_LIN_DIM, SPATIAL_LIN_DIM);
     v_skew,           w_skew       ];
end

function S = skew(v)
% SKEW  Create 3x3 skew-symmetric matrix from 3x1 vector
%   S = SKEW(v) returns the skew-symmetric matrix such that
%   S*u = cross(v, u) for any 3x1 vector u
%
% Inputs:
%   v - 3x1 vector
%
% Outputs:
%   S - 3x3 skew-symmetric matrix

S = [    0,  -v(3),   v(2);
      v(3),      0,  -v(1);
     -v(2),   v(1),      0];
end

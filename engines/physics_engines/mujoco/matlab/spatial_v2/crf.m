function X = crf(v)
% CRF  Spatial cross product operator for force vectors
%   X = CRF(v) returns the 6x6 matrix X such that X*f = v x* f for any
%   spatial force vector f, where x* is the dual spatial cross product.
%
%   Following Featherstone's spatial vector algebra notation:
%   v = [w; v] where w is angular velocity, v is linear velocity
%   f = [n; f] where n is torque, f is force
%
%   The dual cross product operator is:
%   crf(v) = -crm(v)' = [ skew(w)   skew(v) ]
%                       [   0       skew(w) ]
%
% Inputs:
%   v - 6x1 spatial motion vector [angular; linear]
%
% Outputs:
%   X - 6x6 dual spatial cross product matrix
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 2: Spatial Vector Algebra
%
% See also: CRM, SKEW, SPATIAL_CROSS

% Validate input
arguments
    v (6,1) {mustBeNumeric, mustBeFinite, mustBeVector}
end

validateattributes(v, {'numeric'}, {'size', [6, 1]}, 'crf', 'v');

% Load constants (after arguments block)
addpath('..');
constants;

% Extract angular and linear components
w = v(1:3);
vlin = v(4:6);

% Construct skew-symmetric matrices
w_skew = skew(w);
v_skew = skew(vlin);

% Build the SPATIAL_DIM x SPATIAL_DIM dual spatial cross product operator
X = [w_skew,           v_skew;
     zeros(SPATIAL_LIN_DIM, SPATIAL_LIN_DIM),      w_skew];
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

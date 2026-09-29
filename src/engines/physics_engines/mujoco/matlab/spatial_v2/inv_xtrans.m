function X_inv = inv_xtrans(E, r)
% INV_XTRANS  Inverse of spatial coordinate transformation
%   X_inv = INV_XTRANS(E, r) returns the inverse of the spatial
%   transformation matrix computed by XTRANS(E, r).
%
%   This is more efficient than computing inv(xtrans(E, r)) directly,
%   as we can exploit the special structure of the transformation.
%
%   Given X transforms from B to A, X_inv transforms from A to B.
%
%   X_inv = [ E'       0      ]
%           [ rx*E'    E'     ]
%
%   where rx is the skew-symmetric matrix of r.
%
% Inputs:
%   E - 3x3 rotation matrix from frame A to frame B
%   r - 3x1 translation vector from A to B, expressed in frame A
%
% Outputs:
%   X_inv - 6x6 inverse spatial transformation matrix
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 2: Spatial Vector Algebra
%
% See also: XTRANS, XLT, XROT

% Validate inputs
arguments
    E (3,3) {mustBeNumeric, mustBeFinite}
    r (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
end

validateattributes(E, {'numeric'}, {'size', [3, 3]}, 'inv_xtrans', 'E');
validateattributes(r, {'numeric'}, {'size', [3, 1]}, 'inv_xtrans', 'r');

% Load constants (after arguments block)
addpath('..');
constants;

% Compute transpose (inverse) of rotation
E_T = E';

% Create skew-symmetric matrix
r_skew = skew(r);

% Build the SPATIAL_DIM x SPATIAL_DIM inverse transformation matrix
X_inv = [E_T,              zeros(SPATIAL_LIN_DIM, SPATIAL_LIN_DIM);
         r_skew * E_T,     E_T        ];
end

function S = skew(v)
% SKEW  Create 3x3 skew-symmetric matrix from 3x1 vector

S = [    0,  -v(3),   v(2);
      v(3),      0,  -v(1);
     -v(2),   v(1),      0];
end

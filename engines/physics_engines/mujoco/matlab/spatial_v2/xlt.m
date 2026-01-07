function X = xlt(r)
% XLT  Spatial coordinate transformation for pure translation
%   X = XLT(r) returns the 6x6 spatial transformation matrix for a pure
%   translation by vector r.
%
%   This transforms spatial vectors from frame B to frame A, where frame B
%   is translated from frame A by vector r (expressed in frame A).
%
%   The transformation matrix is:
%   X = [ I      0   ]
%       [ -rx    I   ]
%
%   where rx is the skew-symmetric matrix of r.
%
% Inputs:
%   r - 3x1 translation vector from A to B, expressed in frame A
%
% Outputs:
%   X - 6x6 spatial transformation matrix
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 2: Spatial Vector Algebra
%
% See also: XROT, XTRANS, PLUCKER_TRANSFORM

% Validate input
arguments
    r (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
end

validateattributes(r, {'numeric'}, {'size', [3, 1]}, 'xlt', 'r');

% Load constants (after arguments block)
addpath('..');
constants;

% Create skew-symmetric matrix
r_skew = skew(r);

% Build the SPATIAL_DIM x SPATIAL_DIM spatial transformation matrix
X = [eye(SPATIAL_LIN_DIM),     zeros(SPATIAL_LIN_DIM, SPATIAL_LIN_DIM);
     -r_skew,    eye(SPATIAL_LIN_DIM)      ];
end

function S = skew(v)
% SKEW  Create 3x3 skew-symmetric matrix from 3x1 vector

S = [    0,  -v(3),   v(2);
      v(3),      0,  -v(1);
     -v(2),   v(1),      0];
end

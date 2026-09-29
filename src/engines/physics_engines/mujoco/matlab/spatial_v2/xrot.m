function X = xrot(E)
% XROT  Spatial coordinate transformation for pure rotation
%   X = XROT(E) returns the 6x6 spatial transformation matrix for a pure
%   rotation described by the 3x3 rotation matrix E.
%
%   This transforms spatial vectors from frame B to frame A, where frame B
%   is rotated from frame A by rotation matrix E (from A to B).
%
%   The transformation matrix is:
%   X = [ E    0 ]
%       [ 0    E ]
%
% Inputs:
%   E - 3x3 rotation matrix from frame A to frame B
%
% Outputs:
%   X - 6x6 spatial transformation matrix
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 2: Spatial Vector Algebra
%
% See also: XLT, XTRANS, PLUCKER_TRANSFORM

% Validate input
arguments
    E (3,3) {mustBeNumeric, mustBeFinite}
end

validateattributes(E, {'numeric'}, {'size', [3, 3]}, 'xrot', 'E');

% Load constants (after arguments block)
addpath('..');
constants;

% Verify it's a valid rotation matrix (optional, can be expensive)
if abs(det(E) - 1) > 1e-10 || norm(E * E' - eye(SPATIAL_LIN_DIM), 'fro') > 1e-10
    warning('xrot:InvalidRotation', ...
        'Input matrix E may not be a valid rotation matrix');
end

% Build the SPATIAL_DIM x SPATIAL_DIM spatial transformation matrix
X = [E,           zeros(SPATIAL_LIN_DIM, SPATIAL_LIN_DIM);
     zeros(SPATIAL_LIN_DIM, SPATIAL_LIN_DIM), E          ];
end

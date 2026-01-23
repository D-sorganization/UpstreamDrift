function I_A = transform_spatial_inertia(I_B, X)
% TRANSFORM_SPATIAL_INERTIA  Transform spatial inertia between frames
%   I_A = TRANSFORM_SPATIAL_INERTIA(I_B, X) transforms the spatial inertia
%   matrix from frame B to frame A using spatial transformation X.
%
%   The transformation formula is:
%   I_A = X' * I_B * X^(-T)
%
%   However, for spatial transforms, X^(-T) = X, so:
%   I_A = X' * I_B * X
%
%   This preserves the symmetric positive-definite properties of the
%   spatial inertia matrix.
%
% Inputs:
%   I_B - 6x6 spatial inertia matrix in frame B
%   X   - 6x6 spatial transformation from B to A
%
% Outputs:
%   I_A - 6x6 spatial inertia matrix in frame A
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 2: Spatial Vector Algebra, Section 2.9
%
% See also: MCI, XTRANS, INV_XTRANS

% Validate inputs
arguments
    I_B (6,6) {mustBeNumeric, mustBeFinite}
    X (6,6) {mustBeNumeric, mustBeFinite}
end

% Load constants (after arguments block)
addpath('..');
constants;

validateattributes(I_B, {'numeric'}, {'size', [SPATIAL_DIM, SPATIAL_DIM], 'finite'}, ...
    'transform_spatial_inertia', 'I_B');
validateattributes(X, {'numeric'}, {'size', [SPATIAL_DIM, SPATIAL_DIM], 'finite'}, ...
    'transform_spatial_inertia', 'X');

% Verify I_B is symmetric
if norm(I_B - I_B', 'fro') > 1e-10
    warning('transform_spatial_inertia:AsymmetricInertia', ...
        'Spatial inertia I_B should be symmetric. Symmetrizing.');
    I_B = (I_B + I_B') / 2;
end

% Transform the inertia matrix
% For spatial transforms: I_A = X' * I_B * X
I_A = X' * I_B * X;

% Ensure result is symmetric (numerical precision)
I_A = (I_A + I_A') / 2;
end

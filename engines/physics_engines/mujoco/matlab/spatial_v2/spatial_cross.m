function result = spatial_cross(v, u, type)
% SPATIAL_CROSS  Compute spatial cross product
%   result = SPATIAL_CROSS(v, u) computes the spatial cross product
%   v x u (motion-to-motion cross product) by default.
%
%   result = SPATIAL_CROSS(v, u, type) specifies the type of cross product:
%     'motion'  - Motion cross product: v x u (default)
%     'force'   - Force cross product:  v x* f (dual cross product)
%
%   This is a convenience function that uses CRM and CRF operators.
%
% Inputs:
%   v    - 6x1 spatial motion vector [angular; linear]
%   u    - 6x1 spatial vector (motion or force depending on type)
%   type - String: 'motion' or 'force' (optional, default: 'motion')
%
% Outputs:
%   result - 6x1 spatial vector resulting from cross product
%
% Examples:
%   % Motion cross product (acceleration)
%   v = [1; 0; 0; 0; 1; 0];  % Angular and linear velocity
%   a = [0; 1; 0; 0; 0; 1];  % Angular and linear acceleration
%   bias = spatial_cross(v, a, 'motion');
%
%   % Force cross product (wrench transformation)
%   v = [1; 0; 0; 0; 1; 0];  % Velocity
%   f = [0; 0; 10; 0; 0; 0]; % Force/torque
%   f_transformed = spatial_cross(v, f, 'force');
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 2: Spatial Vector Algebra
%
% See also: CRM, CRF

% Validate inputs
arguments
    v (6,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    u (6,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    type (1,1) {mustBeMember(type, {'motion', 'force'})} = 'motion'
end

% Load constants (after arguments block)
addpath('..');
constants;

validateattributes(v, {'numeric'}, {'size', [SPATIAL_DIM, 1]}, 'spatial_cross', 'v');
validateattributes(u, {'numeric'}, {'size', [SPATIAL_DIM, 1]}, 'spatial_cross', 'u');

% Compute cross product based on type
switch lower(type)
    case 'motion'
        % Motion cross product: v x u
        result = crm(v) * u;

    case 'force'
        % Force cross product: v x* f
        result = crf(v) * u;

    otherwise
        error('spatial_cross:InvalidType', ...
            'Type must be ''motion'' or ''force'', got: %s', type);
end
end

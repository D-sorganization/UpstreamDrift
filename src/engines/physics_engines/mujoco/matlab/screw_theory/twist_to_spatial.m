function V = twist_to_spatial(omega, v, point)
% TWIST_TO_SPATIAL  Convert twist (angular + linear velocity) to spatial vector
%   V = TWIST_TO_SPATIAL(omega, v) creates a 6x1 spatial motion vector
%   from angular velocity omega and linear velocity v at the origin.
%
%   V = TWIST_TO_SPATIAL(omega, v, point) creates a spatial motion vector
%   at a reference point.
%
%   A twist represents the instantaneous motion of a rigid body:
%   - omega: 3x1 angular velocity vector
%   - v: 3x1 linear velocity of a point on the body
%
%   The spatial vector format is: V = [omega; v]
%
% Inputs:
%   omega - 3x1 angular velocity vector (rad/s)
%   v     - 3x1 linear velocity vector (m/s)
%   point - 3x1 reference point (optional, default: origin)
%
% Outputs:
%   V     - 6x1 spatial motion vector
%
% Example:
%   % Body rotating about z-axis at 1 rad/s
%   omega = [0; 0; 1];
%   v = [0; 0; 0];
%   V = twist_to_spatial(omega, v);
%
% References:
%   Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
%   Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.
%
% See also: WRENCH_TO_SPATIAL, SCREW_TO_TWIST, SPATIAL_TO_TWIST

% Validate inputs
arguments
    omega (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    v (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    point (3,1) {mustBeNumeric, mustBeFinite, mustBeVector} = zeros(3, 1)
end

% Load constants (after arguments block)
addpath('..');
constants;

validateattributes(omega, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'twist_to_spatial', 'omega');
validateattributes(v, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'twist_to_spatial', 'v');
validateattributes(point, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'twist_to_spatial', 'point');

% Adjust linear velocity if reference point is not the origin
if norm(point) > eps
    % v_new = v_old - omega x point
    v = v - cross(omega, point);
end

% Construct spatial vector [angular; linear]
V = [omega; v];
end

function S = screw_axis(axis, point, pitch)
% SCREW_AXIS  Compute screw axis representation
%   S = SCREW_AXIS(axis, point) returns the 6x1 screw axis for pure
%   rotation about an axis passing through a point.
%
%   S = SCREW_AXIS(axis, point, pitch) returns the screw axis with
%   specified pitch (ratio of translation to rotation).
%
%   A screw axis represents the instantaneous motion of a rigid body
%   rotating about and/or translating along an axis in space.
%
%   For pure rotation (pitch = 0):
%   S = [omega; -omega x q]
%   where omega is the unit axis direction and q is a point on the axis.
%
%   For pure translation (pitch = inf):
%   S = [0; v]
%   where v is the unit direction of translation.
%
%   For general screw motion:
%   S = [omega; v + h*omega]
%   where h is the pitch (distance translated per radian).
%
% Inputs:
%   axis  - 3x1 unit direction vector of the screw axis
%   point - 3x1 point on the screw axis (m)
%   pitch - Scalar pitch value (m/rad) (optional, default: 0 for pure rotation)
%           Use inf for pure translation
%
% Outputs:
%   S     - 6x1 normalized screw axis
%
% Examples:
%   % Rotation about z-axis passing through origin
%   S = screw_axis([0; 0; 1], [0; 0; 0]);
%
%   % Rotation about z-axis passing through point [1; 0; 0]
%   S = screw_axis([0; 0; 1], [1; 0; 0]);
%
%   % Translation along x-axis
%   S = screw_axis([1; 0; 0], [0; 0; 0], inf);
%
%   % Screw motion: rotation about z with pitch 0.1 m/rad
%   S = screw_axis([0; 0; 1], [0; 0; 0], 0.1);
%
% References:
%   Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
%   Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.
%
% See also: TWIST_TO_SPATIAL, EXPONENTIAL_MAP, SCREW_TO_TRANSFORM

% Validate inputs
arguments
    axis (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    point (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    pitch (1,1) {mustBeNumeric, mustBeScalar} = 0
end

% Load constants (after arguments block)
addpath('..');
constants;

validateattributes(axis, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'screw_axis', 'axis');
validateattributes(point, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'screw_axis', 'point');
validateattributes(pitch, {'numeric'}, {'scalar'}, 'screw_axis', 'pitch');

% Normalize axis direction
omega = axis / norm(axis);

if isinf(pitch)
    % Pure translation (prismatic joint)
    S = [zeros(SPATIAL_LIN_DIM, 1); omega];
else
    % Rotation or screw motion
    % Linear velocity component: v = -omega x q + h*omega
    % where q is point on axis, h is pitch
    v = -cross(omega, point) + pitch * omega;
    S = [omega; v];
end
end

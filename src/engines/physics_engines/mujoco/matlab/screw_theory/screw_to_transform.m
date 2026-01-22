function T = screw_to_transform(axis, point, pitch, theta)
% SCREW_TO_TRANSFORM  Compute transformation from screw motion parameters
%   T = SCREW_TO_TRANSFORM(axis, point, pitch, theta) computes the 4x4
%   homogeneous transformation matrix resulting from a screw motion.
%
%   This is a convenience function combining SCREW_AXIS and EXPONENTIAL_MAP.
%
% Inputs:
%   axis  - 3x1 unit direction vector of screw axis
%   point - 3x1 point on the screw axis
%   pitch - Scalar pitch (m/rad), use inf for pure translation
%   theta - Displacement along screw (rad or m)
%
% Outputs:
%   T     - 4x4 homogeneous transformation matrix
%
% Examples:
%   % 90° rotation about z-axis through origin
%   T = screw_to_transform([0;0;1], [0;0;0], 0, pi/2);
%
%   % Translation of 1m along x-axis
%   T = screw_to_transform([1;0;0], [0;0;0], inf, 1.0);
%
%   % Screw motion: rotate pi/4 about z with 0.1m/rad pitch
%   T = screw_to_transform([0;0;1], [0;0;0], 0.1, pi/4);
%
% See also: SCREW_AXIS, EXPONENTIAL_MAP

% Validate inputs
arguments
    axis (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    point (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    pitch (1,1) {mustBeNumeric, mustBeScalar}
    theta (1,1) {mustBeNumeric, mustBeFinite, mustBeScalar}
end

validateattributes(axis, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'screw_to_transform', 'axis');
validateattributes(point, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'screw_to_transform', 'point');
validateattributes(pitch, {'numeric'}, {'scalar'}, ...
    'screw_to_transform', 'pitch');
validateattributes(theta, {'numeric'}, {'scalar', 'finite'}, ...
    'screw_to_transform', 'theta');

% Compute screw axis
S = screw_axis(axis, point, pitch);

% Apply exponential map
T = exponential_map(S, theta);
end

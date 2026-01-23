function F = wrench_to_spatial(moment, force, point)
% WRENCH_TO_SPATIAL  Convert wrench (moment + force) to spatial force vector
%   F = WRENCH_TO_SPATIAL(moment, force) creates a 6x1 spatial force vector
%   from moment and force applied at the origin.
%
%   F = WRENCH_TO_SPATIAL(moment, force, point) creates a spatial force
%   vector at a reference point.
%
%   A wrench represents forces and moments acting on a rigid body:
%   - moment: 3x1 moment (torque) vector
%   - force: 3x1 force vector
%
%   The spatial force format is: F = [moment; force]
%
% Inputs:
%   moment - 3x1 moment vector (N*m)
%   force  - 3x1 force vector (N)
%   point  - 3x1 reference point (optional, default: origin)
%
% Outputs:
%   F      - 6x1 spatial force vector
%
% Example:
%   % Pure force along x-axis applied at origin
%   moment = [0; 0; 0];
%   force = [10; 0; 0];
%   F = wrench_to_spatial(moment, force);
%
% References:
%   Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
%   Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.
%
% See also: TWIST_TO_SPATIAL, SCREW_TO_WRENCH

% Validate inputs
arguments
    moment (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    force (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    point (3,1) {mustBeNumeric, mustBeFinite, mustBeVector} = zeros(3, 1)
end

% Load constants (after arguments block)
addpath('..');
constants;

validateattributes(moment, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'wrench_to_spatial', 'moment');
validateattributes(force, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'wrench_to_spatial', 'force');
validateattributes(point, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'wrench_to_spatial', 'point');

% Adjust moment if reference point is not the origin
if norm(point) > eps
    % moment_new = moment_old + point x force
    moment = moment + cross(point, force);
end

% Construct spatial force vector [moment; force]
F = [moment; force];
end

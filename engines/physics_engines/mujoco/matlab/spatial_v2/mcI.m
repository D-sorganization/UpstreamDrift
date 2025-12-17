function I_spatial = mcI(mass, com, I_com)
% MCI  Construct spatial inertia matrix from mass, COM, and rotational inertia
%   I_spatial = MCI(mass, com, I_com) constructs the 6x6 spatial inertia
%   matrix from the body's mass, center of mass location, and rotational
%   inertia tensor about the center of mass.
%
%   The spatial inertia matrix has the form:
%   I_spatial = [ I_com + m*c_skew*c_skew'    m*c_skew ]
%               [        m*c_skew'              m*I_3  ]
%
%   where c is the COM vector and c_skew is its skew-symmetric matrix.
%
%   This follows Featherstone's convention where the spatial inertia
%   represents the inertia of a rigid body about a reference point.
%
% Inputs:
%   mass  - Scalar mass of the body (kg)
%   com   - 3x1 vector from reference point to center of mass (m)
%   I_com - 3x3 rotational inertia tensor about COM (kg*m^2)
%
% Outputs:
%   I_spatial - 6x6 spatial inertia matrix
%
% Example:
%   % Uniform density sphere of radius 0.1m and mass 1kg
%   mass = 1.0;
%   radius = 0.1;
%   com = [0; 0; 0];  % COM at reference point
%   I_sphere = (2/5) * mass * radius^2 * eye(3);
%   I_spatial = mcI(mass, com, I_sphere);
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 2: Spatial Vector Algebra, Section 2.8
%
% See also: XTRANS, CRM, CRF

% Validate inputs
arguments
    mass (1,1) {mustBeNumeric, mustBePositive, mustBeFinite, mustBeScalar}
    com (3,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    I_com (3,3) {mustBeNumeric, mustBeFinite}
end

validateattributes(mass, {'numeric'}, {'scalar', 'positive', 'finite'}, ...
    'mcI', 'mass');
validateattributes(com, {'numeric'}, {'size', [3, 1], 'finite'}, ...
    'mcI', 'com');
validateattributes(I_com, {'numeric'}, {'size', [3, 3], 'finite'}, ...
    'mcI', 'I_com');

% Load constants (after arguments block)
addpath('..');
constants;

% Verify inertia tensor is symmetric
if norm(I_com - I_com', 'fro') > 1e-10
    warning('mcI:AsymmetricInertia', ...
        'Inertia tensor I_com should be symmetric. Symmetrizing.');
    I_com = (I_com + I_com') / 2;
end

% Create skew-symmetric matrix for COM vector
c_skew = skew(com);

% Parallel axis theorem: transform inertia to reference point
% I = I_com + m * c_skew * c_skew'
I_ref = I_com + mass * (c_skew * c_skew');

% Build the SPATIAL_DIM x SPATIAL_DIM spatial inertia matrix
I_spatial = [I_ref,                mass * c_skew;
             mass * c_skew',       mass * eye(SPATIAL_LIN_DIM)];
end

function S = skew(v)
% SKEW  Create 3x3 skew-symmetric matrix from 3x1 vector

S = [    0,  -v(3),   v(2);
      v(3),      0,  -v(1);
     -v(2),   v(1),      0];
end

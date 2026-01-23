function [Xj, S] = jcalc(jtype, q)
% JCALC  Calculate joint transform and motion subspace
%   [Xj, S] = JCALC(jtype, q) calculates the joint transformation matrix
%   and motion subspace vector for a given joint type and position.
%
%   This function supports various joint types and returns:
%   - Xj: 6x6 spatial transformation from successor to predecessor
%   - S:  6x1 motion subspace vector (joint axis)
%
% Inputs:
%   jtype - String specifying joint type:
%           'Rx' - Revolute joint about x-axis
%           'Ry' - Revolute joint about y-axis
%           'Rz' - Revolute joint about z-axis
%           'Px' - Prismatic joint along x-axis
%           'Py' - Prismatic joint along y-axis
%           'Pz' - Prismatic joint along z-axis
%   q     - Scalar joint position (radians for revolute, meters for prismatic)
%
% Outputs:
%   Xj - 6x6 spatial transformation matrix
%   S  - 6x1 motion subspace vector
%
% Examples:
%   % Revolute joint about z-axis at 45 degrees
%   [Xj, S] = jcalc('Rz', pi/4);
%
%   % Prismatic joint along x-axis extended 0.5m
%   [Xj, S] = jcalc('Px', 0.5);
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 4: Kinematics
%
% See also: XTRANS, XROT, XLT

% Validate inputs
arguments
    jtype (1,1) {mustBeText}
    q (1,1) {mustBeNumeric, mustBeFinite, mustBeScalar}
end

validateattributes(jtype, {'char', 'string'}, {}, 'jcalc', 'jtype');
validateattributes(q, {'numeric'}, {'scalar', 'finite'}, 'jcalc', 'q');

% Load constants (after arguments block)
addpath('..');
constants;

switch jtype
    case 'Rx'  % Revolute about x-axis
        c = cos(q);
        s = sin(q);
        E = [1,  0,  0;
             0,  c, -s;
             0,  s,  c];
        Xj = xrot(E);
        S = [1; 0; 0; 0; 0; 0];  % Angular velocity about x

    case 'Ry'  % Revolute about y-axis
        c = cos(q);
        s = sin(q);
        E = [ c,  0,  s;
              0,  1,  0;
             -s,  0,  c];
        Xj = xrot(E);
        S = [0; 1; 0; 0; 0; 0];  % Angular velocity about y

    case 'Rz'  % Revolute about z-axis
        c = cos(q);
        s = sin(q);
        E = [c, -s,  0;
             s,  c,  0;
             0,  0,  1];
        Xj = xrot(E);
        S = [0; 0; 1; 0; 0; 0];  % Angular velocity about z

    case 'Px'  % Prismatic along x-axis
        r = [q; 0; 0];
        Xj = xlt(r);
        S = [0; 0; 0; 1; 0; 0];  % Linear velocity along x

    case 'Py'  % Prismatic along y-axis
        r = [0; q; 0];
        Xj = xlt(r);
        S = [0; 0; 0; 0; 1; 0];  % Linear velocity along y

    case 'Pz'  % Prismatic along z-axis
        r = [0; 0; q];
        Xj = xlt(r);
        S = [0; 0; 0; 0; 0; 1];  % Linear velocity along z

    otherwise
        error('jcalc:UnsupportedJointType', ...
            'Unsupported joint type: %s\nSupported types: Rx, Ry, Rz, Px, Py, Pz', ...
            jtype);
end
end

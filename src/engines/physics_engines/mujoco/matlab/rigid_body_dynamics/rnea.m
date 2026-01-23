function tau = rnea(model, q, qd, qdd, f_ext)
% RNEA  Recursive Newton-Euler Algorithm for inverse dynamics
%   tau = RNEA(model, q, qd, qdd) computes the inverse dynamics of a
%   kinematic tree using the Recursive Newton-Euler Algorithm.
%
%   tau = RNEA(model, q, qd, qdd, f_ext) includes external forces.
%
%   Given joint positions q, velocities qd, and accelerations qdd, this
%   algorithm computes the joint forces/torques tau required to produce
%   that motion.
%
%   This is Featherstone's O(n) algorithm for computing inverse dynamics.
%
% Inputs:
%   model - Structure containing robot model with fields:
%           .NB        - Number of bodies
%           .parent    - Parent body indices (1xNB)
%           .jtype     - Joint types cell array (1xNB)
%           .Xtree     - Joint transforms (6x6xNB)
%           .I         - Spatial inertias (6x6xNB)
%           .gravity   - 6x1 spatial gravity vector (optional)
%   q      - NB x 1 vector of joint positions
%   qd     - NB x 1 vector of joint velocities
%   qdd    - NB x 1 vector of joint accelerations
%   f_ext  - 6 x NB matrix of external forces (optional)
%
% Outputs:
%   tau    - NB x 1 vector of joint forces/torques
%
% Algorithm:
%   Forward pass: compute velocities and accelerations
%   Backward pass: compute forces and project to joint torques
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 5: Independent Joint Equations of Motion, Algorithm 5.1
%
% See also: CRBA, ABA, FD_ABA

% Validate inputs
arguments
    model (1,1) struct
    q (:,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    qd (:,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    qdd (:,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    f_ext (6,:) {mustBeNumeric, mustBeFinite} = []
end

validateattributes(q, {'numeric'}, {'vector', 'finite'}, 'rnea', 'q');
validateattributes(qd, {'numeric'}, {'vector', 'finite'}, 'rnea', 'qd');
validateattributes(qdd, {'numeric'}, {'vector', 'finite'}, 'rnea', 'qdd');

% Load constants (after arguments block)
addpath('..');
constants;

NB = model.NB;
if isempty(f_ext)
    f_ext = zeros(SPATIAL_DIM, NB);
end

% Ensure column vectors
q = q(:);
qd = qd(:);
qdd = qdd(:);

% Check dimensions
assert(length(q) == NB, 'q must have length NB');
assert(length(qd) == NB, 'qd must have length NB');
assert(length(qdd) == NB, 'qdd must have length NB');

% Get gravity vector (default: -9.80665 m/s^2 in z-direction, NIST standard)
if isfield(model, 'gravity')
    a_grav = model.gravity;
else
    a_grav = [0; 0; 0; 0; 0; -GRAVITY_STANDARD_M_S2];
end

% Initialize arrays
v = zeros(SPATIAL_DIM, NB);      % Spatial velocities
a = zeros(SPATIAL_DIM, NB);      % Spatial accelerations
f = zeros(SPATIAL_DIM, NB);      % Spatial forces
tau = zeros(NB, 1);    % Joint torques

% --- Forward pass: kinematics ---
for i = 1:NB
    % Calculate joint transform and motion subspace
    [Xj, S] = jcalc(model.jtype{i}, q(i));

    % Joint velocity in joint frame
    vJ = S * qd(i);

    % Composite transform from body i to base
    if model.parent(i) == 0
        % Body i is connected to base
        v(:, i) = vJ;
        a(:, i) = Xj * (-a_grav) + S * qdd(i);
    else
        % Body i has a parent
        p = model.parent(i);
        Xp = Xj * model.Xtree(:, :, i);  % Transform from parent to i

        % Velocity: transform parent velocity and add joint velocity
        v(:, i) = Xp * v(:, p) + vJ;

        % Acceleration: transform parent accel + bias accel + joint accel
        a(:, i) = Xp * a(:, p) + S * qdd(i) + crm(v(:, i)) * vJ;
    end
end

% --- Backward pass: dynamics ---
for i = NB:-1:1
    % Newton-Euler equation: f = I*a + v x* I*v - f_ext
    f(:, i) = model.I(:, :, i) * a(:, i) + ...
              crf(v(:, i)) * (model.I(:, :, i) * v(:, i)) - ...
              f_ext(:, i);

    % Project force to joint torque
    [~, S] = jcalc(model.jtype{i}, q(i));
    tau(i) = S' * f(:, i);

    % Propagate force to parent
    if model.parent(i) ~= 0
        p = model.parent(i);
        Xp = jcalc(model.jtype{i}, q(i)) * model.Xtree(:, :, i);
        f(:, p) = f(:, p) + Xp' * f(:, i);
    end
end
end

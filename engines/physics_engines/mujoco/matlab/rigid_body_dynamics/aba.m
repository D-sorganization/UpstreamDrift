function qdd = aba(model, q, qd, tau, f_ext)
% ABA  Articulated Body Algorithm for forward dynamics
%   qdd = ABA(model, q, qd, tau) computes the forward dynamics of a
%   kinematic tree using the Articulated Body Algorithm.
%
%   qdd = ABA(model, q, qd, tau, f_ext) includes external forces.
%
%   Given joint positions q, velocities qd, and applied torques tau, this
%   algorithm computes the resulting joint accelerations qdd.
%
%   This is Featherstone's O(n) algorithm for computing forward dynamics,
%   which is much more efficient than inverting the mass matrix.
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
%   tau    - NB x 1 vector of joint forces/torques
%   f_ext  - 6 x NB matrix of external forces (optional)
%
% Outputs:
%   qdd    - NB x 1 vector of joint accelerations
%
% Algorithm:
%   Pass 1: Kinematics - compute velocities and bias accelerations
%   Pass 2: Articulated bodies - compute articulated-body inertias
%   Pass 3: Accelerations - compute joint and spatial accelerations
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 7: Articulated-Body Algorithm, Algorithm 7.1
%
% See also: RNEA, CRBA, FD_ABA

% Validate inputs
arguments
    model (1,1) struct
    q (:,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    qd (:,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    tau (:,1) {mustBeNumeric, mustBeFinite, mustBeVector}
    f_ext (6,:) {mustBeNumeric, mustBeFinite} = []
end

validateattributes(q, {'numeric'}, {'vector', 'finite'}, 'aba', 'q');
validateattributes(qd, {'numeric'}, {'vector', 'finite'}, 'aba', 'qd');
validateattributes(tau, {'numeric'}, {'vector', 'finite'}, 'aba', 'tau');

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
tau = tau(:);

% Check dimensions
assert(length(q) == NB, 'q must have length NB');
assert(length(qd) == NB, 'qd must have length NB');
assert(length(tau) == NB, 'tau must have length NB');

% Get gravity vector
if isfield(model, 'gravity')
    a_grav = model.gravity;
else
    % Default gravity: -9.80665 m/s^2 in z-direction (NIST standard)
    a_grav = [0; 0; 0; 0; 0; -GRAVITY_STANDARD_M_S2];
end

% Initialize arrays
Xup = zeros(SPATIAL_DIM, SPATIAL_DIM, NB);     % Transforms from body to parent
S = zeros(SPATIAL_DIM, NB);          % Motion subspaces
v = zeros(SPATIAL_DIM, NB);          % Spatial velocities
c = zeros(SPATIAL_DIM, NB);          % Velocity-product accelerations (bias)
IA = zeros(SPATIAL_DIM, SPATIAL_DIM, NB);      % Articulated-body inertias
pA = zeros(SPATIAL_DIM, NB);         % Articulated-body bias forces
U = zeros(SPATIAL_DIM, NB);          % IA * S
d = zeros(NB, 1);          % S' * U (joint-space inertia)
u = zeros(NB, 1);          % tau - S' * pA (bias force)
a = zeros(SPATIAL_DIM, NB);          % Spatial accelerations
qdd = zeros(NB, 1);        % Joint accelerations

% --- Pass 1: Forward kinematics ---
for i = 1:NB
    [Xj, S(:, i)] = jcalc(model.jtype{i}, q(i));
    Xup(:, :, i) = Xj * model.Xtree(:, :, i);

    vJ = S(:, i) * qd(i);  % Joint velocity

    if model.parent(i) == 0
        v(:, i) = vJ;
        c(:, i) = zeros(6, 1);  % No bias for base-connected bodies
    else
        p = model.parent(i);
        v(:, i) = Xup(:, :, i) * v(:, p) + vJ;
        c(:, i) = crm(v(:, i)) * vJ;  % Velocity-product acceleration
    end

    % Initialize articulated-body inertia with rigid-body inertia
    IA(:, :, i) = model.I(:, :, i);

    % Bias force: Coriolis + external forces
    pA(:, i) = crf(v(:, i)) * (model.I(:, :, i) * v(:, i)) - f_ext(:, i);
end

% --- Pass 2: Backward recursion (articulated-body inertias) ---
for i = NB:-1:1
    U(:, i) = IA(:, :, i) * S(:, i);
    d(i) = S(:, i)' * U(:, i);  % Joint-space inertia
    u(i) = tau(i) - S(:, i)' * pA(:, i);  % Bias torque

    % Articulated-body inertia update for parent
    if model.parent(i) ~= 0
        p = model.parent(i);

        % Inverse of joint-space inertia
        % For 1-DOF joints, this is just 1/d(i)
        if abs(d(i)) < 1e-10
            warning('aba:SingularInertia', ...
                'Near-singular joint-space inertia at joint %d', i);
            d(i) = sign(d(i)) * 1e-10;
        end
        dinv = 1 / d(i);

        % Update articulated inertia
        Ia_update = U(:, i) * dinv * U(:, i)';
        IA(:, :, p) = IA(:, :, p) + ...
                      Xup(:, :, i)' * (IA(:, :, i) - Ia_update) * Xup(:, :, i);

        % Update bias force
        pa_update = pA(:, i) + IA(:, :, i) * c(:, i) + U(:, i) * dinv * u(i);
        pA(:, p) = pA(:, p) + Xup(:, :, i)' * pa_update;
    end
end

% --- Pass 3: Forward recursion (accelerations) ---
for i = 1:NB
    if model.parent(i) == 0
        a(:, i) = Xup(:, :, i) * (-a_grav) + c(:, i);
    else
        p = model.parent(i);
        a(:, i) = Xup(:, :, i) * a(:, p) + c(:, i);
    end

    qdd(i) = (u(i) - U(:, i)' * a(:, i)) / d(i);
    a(:, i) = a(:, i) + S(:, i) * qdd(i);
end
end

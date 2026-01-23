function H = crba(model, q)
% CRBA  Composite Rigid Body Algorithm for computing mass matrix
%   H = CRBA(model, q) computes the joint-space mass matrix (inertia matrix)
%   H(q) of a kinematic tree using the Composite Rigid Body Algorithm.
%
%   The mass matrix H satisfies the equation of motion:
%   H(q) * qdd + C(q, qd) * qd + g(q) = tau
%
%   This is Featherstone's O(n^2) algorithm for computing the symmetric
%   positive-definite mass matrix.
%
% Inputs:
%   model - Structure containing robot model with fields:
%           .NB        - Number of bodies
%           .parent    - Parent body indices (1xNB)
%           .jtype     - Joint types cell array (1xNB)
%           .Xtree     - Joint transforms (6x6xNB)
%           .I         - Spatial inertias (6x6xNB)
%   q      - NB x 1 vector of joint positions
%
% Outputs:
%   H      - NB x NB symmetric positive-definite mass matrix
%
% Algorithm:
%   1. Compute composite inertias (backward pass)
%   2. Compute mass matrix elements using motion subspaces
%
% References:
%   Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
%   Chapter 6: Operational Space Dynamics, Algorithm 6.1
%
% See also: RNEA, ABA, FD_ABA

% Validate inputs
arguments
    model (1,1) struct
    q (:,1) {mustBeNumeric, mustBeFinite, mustBeVector}
end

validateattributes(q, {'numeric'}, {'vector', 'finite'}, 'crba', 'q');

% Load constants (after arguments block)
addpath('..');
constants;

NB = model.NB;
q = q(:);  % Ensure column vector

assert(length(q) == NB, 'q must have length NB');

% Initialize arrays
Xup = zeros(SPATIAL_DIM, SPATIAL_DIM, NB);     % Transforms from body to parent
S = zeros(SPATIAL_DIM, NB);          % Motion subspaces
Ic = zeros(SPATIAL_DIM, SPATIAL_DIM, NB);      % Composite inertias
H = zeros(NB, NB);         % Mass matrix

% --- Forward pass: compute transforms and motion subspaces ---
for i = 1:NB
    [Xj, S(:, i)] = jcalc(model.jtype{i}, q(i));
    Xup(:, :, i) = Xj * model.Xtree(:, :, i);
end

% --- Backward pass: compute composite inertias ---
% Initialize composite inertias with body inertias
for i = 1:NB
    Ic(:, :, i) = model.I(:, :, i);
end

% Accumulate inertias from children to parents
for i = NB:-1:1
    if model.parent(i) ~= 0
        p = model.parent(i);
        % Transform composite inertia to parent frame and add
        Ic(:, :, p) = Ic(:, :, p) + Xup(:, :, i)' * Ic(:, :, i) * Xup(:, :, i);
    end
end

% --- Compute mass matrix ---
% H(i,j) represents the coupling between joints i and j
for i = 1:NB
    % F is the force transmitted through joint i due to unit acceleration
    % at joint i, affecting the composite body rooted at i
    F = Ic(:, :, i) * S(:, i);
    H(i, i) = S(:, i)' * F;  % Diagonal element

    % Propagate force up the tree to compute off-diagonal elements
    j = i;
    while model.parent(j) > 0
        p = model.parent(j);
        F = Xup(:, :, j)' * F;  % Transform force to parent frame
        H(i, p) = S(:, p)' * F;  % Off-diagonal element
        H(p, i) = H(i, p);       % Symmetric
        j = p;
    end
end

% Ensure exact symmetry (numerical precision)
H = (H + H') / 2;
end

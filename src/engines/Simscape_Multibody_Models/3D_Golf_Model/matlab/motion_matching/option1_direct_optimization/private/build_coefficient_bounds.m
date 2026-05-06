function [lb, ub] = build_coefficient_bounds(n_joints)
%BUILD_COEFFICIENT_BOUNDS  Per-coefficient lower/upper bounds.
%
%   [LB, UB] = BUILD_COEFFICIENT_BOUNDS(N_JOINTS) returns column vectors of
%   length n_joints*7 giving the box constraints used by Option 1 fits.
%
%   The bounds mirror the ranges in
%   src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/functions/
%   dataset_generator/generateRandomCoefficients.m. Coefficient ordering is
%   [A B C D E F G] per joint:
%
%       A, B (t^6, t^5):  +/- 1000
%       C, D (t^4, t^3):  +/- 500
%       E, F (t^2, t^1):  +/- 100
%       G    (constant):  +/- 25
%
%   Preconditions:
%     - N_JOINTS is a positive integer scalar.
%
%   Postconditions:
%     - size(lb) == size(ub) == [n_joints*7, 1]
%     - all(lb < ub)
%
%   GitHub issue: #024 / #3993.
    arguments
        n_joints (1,1) double {mustBePositive, mustBeInteger}
    end

    per_joint_max = [1000; 1000; 500; 500; 100; 100; 25];   % A B C D E F G
    block_max = repmat(per_joint_max, n_joints, 1);
    lb = -block_max;
    ub =  block_max;

    assert(numel(lb) == n_joints * 7 && numel(ub) == n_joints * 7, ...
        "build_coefficient_bounds:badShape", ...
        "Postcondition: lb/ub must have length n_joints*7");
    assert(all(lb < ub), ...
        "build_coefficient_bounds:degenerate", ...
        "Postcondition: every lb must be strictly less than ub");
end

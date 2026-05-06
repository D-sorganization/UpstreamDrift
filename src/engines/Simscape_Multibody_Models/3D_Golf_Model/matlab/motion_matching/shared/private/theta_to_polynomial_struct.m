function [coeff_struct, joint_names, coeff_letters] = theta_to_polynomial_struct(theta, joint_names_override)
%THETA_TO_POLYNOMIAL_STRUCT  Reshape flat theta vector to model-workspace struct.
%
%   [COEFF_STRUCT, JOINT_NAMES, COEFF_LETTERS] = ...
%       THETA_TO_POLYNOMIAL_STRUCT(THETA) reshapes the flat theta vector into
%   a struct whose field names are the per-coefficient model-workspace
%   variables expected by GolfSwing3D_Kinetic.slx. Joints are discovered via
%   getPolynomialParameterInfo; theta is laid out [A B C D E F G] per joint
%   in the joint ordering returned by that function.
%
%   THETA_TO_POLYNOMIAL_STRUCT(THETA, JOINT_NAMES_OVERRIDE) uses the supplied
%   joint ordering instead of the discovered one. Each joint must still have
%   the canonical 7-letter coefficient set ABCDEFG and the same length.
%
%   Preconditions:
%     - THETA is finite, real, with length n_joints*7.
%
%   Postconditions:
%     - numel(fieldnames(COEFF_STRUCT)) == n_joints*7.
%     - JOINT_NAMES is a string row vector of length n_joints.
%     - COEFF_LETTERS is the cellstr ordering used per joint.

    arguments
        theta (:,1) double {mustBeReal, mustBeFinite}
        joint_names_override string = string.empty(1, 0)
    end

    param_info = getPolynomialParameterInfo();

    if isempty(joint_names_override)
        joint_names = string(param_info.joint_names);
    else
        joint_names = string(joint_names_override(:)).';
    end

    n_joints = numel(joint_names);
    expected_len = n_joints * 7;

    assert(numel(theta) == expected_len, ...
        "theta_to_polynomial_struct:badLength", ...
        "Precondition: theta length %d does not match n_joints*7 = %d", ...
        numel(theta), expected_len);

    % Determine coefficient letter order per joint. Default to ABCDEFG.
    coeff_letters_canonical = {'A','B','C','D','E','F','G'};

    coeff_struct = struct();
    idx = 1;
    for j = 1:n_joints
        joint_name = char(joint_names(j));

        % If joint exists in param_info, use its ordering for safety.
        if isempty(joint_names_override)
            local_idx = j;
            local_letters = param_info.joint_coeffs{local_idx};
            if ischar(local_letters)
                local_letters = num2cell(local_letters);
            end
        else
            local_letters = coeff_letters_canonical;
        end

        assert(numel(local_letters) == 7, ...
            "theta_to_polynomial_struct:badCoeffCount", ...
            "Joint %s does not have 7 coefficients", joint_name);

        for k = 1:7
            var_name = sprintf('%s%s', joint_name, char(local_letters{k}));
            coeff_struct.(var_name) = theta(idx);
            idx = idx + 1;
        end
    end

    coeff_letters = coeff_letters_canonical;

    % Postcondition
    assert(numel(fieldnames(coeff_struct)) == expected_len, ...
        "theta_to_polynomial_struct:postcondCount", ...
        "Postcondition: produced %d fields, expected %d", ...
        numel(fieldnames(coeff_struct)), expected_len);
end

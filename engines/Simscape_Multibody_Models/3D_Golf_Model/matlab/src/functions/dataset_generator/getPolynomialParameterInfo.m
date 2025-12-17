function param_info = getPolynomialParameterInfo()
    % External function for getting polynomial parameter information - can be used in parallel processing
    % This function reads the actual parameter structure from the model file

    % Use persistent variable to only show loading message once
    persistent has_shown_loading_message;
    if isempty(has_shown_loading_message)
        has_shown_loading_message = false;
    end

    % Try to load the actual parameter structure from the model file
    try
        % Try multiple possible locations for the parameter file
        possible_paths = {
            'Model/PolynomialInputValues.mat',
            'PolynomialInputValues.mat',
            fullfile(pwd, '..', '..', 'Model', 'PolynomialInputValues.mat'),
            fullfile(pwd, '..', 'Model', 'PolynomialInputValues.mat'),
            fullfile(pwd, 'Model', 'PolynomialInputValues.mat'),
            fullfile(pwd, 'PolynomialInputValues.mat')
        };

        model_path = '';
        for i = 1:length(possible_paths)
            if exist(possible_paths{i}, 'file')
                model_path = possible_paths{i};
                break;
            end
        end

        if ~isempty(model_path)
            if ~has_shown_loading_message
                fprintf('Loading parameter structure from: %s\n', model_path);
                has_shown_loading_message = true;
            end
            loaded_data = load(model_path);
            var_names = fieldnames(loaded_data);

            % Parse variable names to find joints with 7 coefficients (ABCDEFG)
            joint_map = containers.Map();

            for i = 1:length(var_names)
                name = var_names{i};
                if length(name) > 1
                    coeff = name(end);
                    base_name = name(1:end-1);

                    if isKey(joint_map, base_name)
                        joint_map(base_name) = [joint_map(base_name), coeff];
                    else
                        joint_map(base_name) = coeff;
                    end
                end
            end

            % Filter to only 7-coefficient joints
            all_joint_names = keys(joint_map);
            filtered_joint_names = {};
            filtered_coeffs = {};

            for i = 1:length(all_joint_names)
                joint_name = all_joint_names{i};
                coeffs = sort(joint_map(joint_name));

                if length(coeffs) == 7 && strcmp(coeffs, 'ABCDEFG')
                    filtered_joint_names{end+1} = joint_name;
                    filtered_coeffs{end+1} = coeffs;
                end
            end

            param_info.joint_names = sort(filtered_joint_names);
            param_info.joint_coeffs = cell(size(param_info.joint_names));

            for i = 1:length(param_info.joint_names)
                joint_name = param_info.joint_names{i};
                idx = find(strcmp(filtered_joint_names, joint_name));
                param_info.joint_coeffs{i} = filtered_coeffs{idx};
            end

            param_info.total_params = length(param_info.joint_names) * 7;
            if ~has_shown_loading_message
                fprintf('Loaded %d joints with 7 coefficients each = %d total coefficients\n', ...
                    length(param_info.joint_names), param_info.total_params);
            end

        else
            error('CRITICAL ERROR: PolynomialInputValues.mat not found in any of the expected locations. This file is required for proper data extraction. No fallback data will be used.');
        end

    catch ME
        error('CRITICAL ERROR loading parameter structure: %s. No fallback data will be used.', ME.message);
    end
end

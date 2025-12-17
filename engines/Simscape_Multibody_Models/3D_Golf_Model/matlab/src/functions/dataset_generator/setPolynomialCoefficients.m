function simIn = setPolynomialCoefficients(simIn, coefficients, config)
    % Get parameter info for coefficient mapping
    param_info = getPolynomialParameterInfo();

    % Basic validation
    if isempty(param_info.joint_names)
        error('No joint names found in polynomial parameter info');
    end

    % Handle parallel worker coefficient format issues
    if iscell(coefficients)
        try
            % Check if cells contain strings or numbers
            if all(cellfun(@ischar, coefficients))
                % Convert string cells to numeric
                coefficients = cellfun(@str2double, coefficients);
            elseif all(cellfun(@isnumeric, coefficients))
                % Convert numeric cells to array
                coefficients = cell2mat(coefficients);
            else
                % Mixed content or other issues
                numeric_coeffs = zeros(size(coefficients));
                for i = 1:numel(coefficients)
                    if ischar(coefficients{i})
                        numeric_coeffs(i) = str2double(coefficients{i});
                    elseif isnumeric(coefficients{i})
                        numeric_coeffs(i) = coefficients{i};
                    else
                        numeric_coeffs(i) = NaN;
                    end
                end
                coefficients = numeric_coeffs;
            end
        catch ME
            fprintf('Error: Could not convert cell coefficients to numeric: %s\n', ME.message);
            % Try one more approach - flatten and convert
            try
                coefficients = str2double(coefficients(:));
            catch
                fprintf('Error: All conversion attempts failed\n');
                return;
            end
        end
    end

    % Ensure coefficients are numeric
    if ~isnumeric(coefficients)
        fprintf('Error: Coefficients must be numeric, got %s\n', class(coefficients));
        return;
    end

    % Ensure coefficients are a row vector if needed
    if size(coefficients, 1) > 1 && size(coefficients, 2) == 1
        coefficients = coefficients';
    end

    % Set coefficients as model variables
    global_coeff_idx = 1;
    variables_set = 0;

    for joint_idx = 1:length(param_info.joint_names)
        joint_name = param_info.joint_names{joint_idx};
        coeffs = param_info.joint_coeffs{joint_idx};

        for local_coeff_idx = 1:length(coeffs)
            coeff_letter = coeffs(local_coeff_idx);
            var_name = sprintf('%s%s', joint_name, coeff_letter);

            if global_coeff_idx <= length(coefficients)
                try
                    simIn = simIn.setVariable(var_name, coefficients(global_coeff_idx));
                    variables_set = variables_set + 1;
                catch ME
                    fprintf('  Warning: Failed to set %s: %s\n', var_name, ME.message);
                end
            else
                fprintf('  Warning: Not enough coefficients for %s (need %d, have %d)\n', var_name, global_coeff_idx, length(coefficients));
            end
            global_coeff_idx = global_coeff_idx + 1;
        end
    end
end

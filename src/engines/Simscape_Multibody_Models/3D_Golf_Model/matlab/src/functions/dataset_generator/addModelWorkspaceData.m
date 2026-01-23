function data_table = addModelWorkspaceData(data_table, simOut, num_rows)
% Extract model workspace variables and add as constant columns
% These include segment lengths, masses, inertias, and other model parameters
%
% This is the working version for comprehensive data extraction
% Uses direct table modification instead of buggy helper functions

try
    % Get model workspace from simulation output
    model_name = simOut.SimulationMetadata.ModelInfo.ModelName;

    % Check if model is loaded
    if ~bdIsLoaded(model_name)
        fprintf('Warning: Model %s not loaded, skipping workspace data\n', model_name);
        return;
    end

    model_workspace = get_param(model_name, 'ModelWorkspace');
    try
        variables = model_workspace.getVariableNames;
    catch
        % For older MATLAB versions, try alternative method
        try
            variables = model_workspace.whos;
            variables = {variables.name};
        catch
            fprintf('Warning: Could not retrieve model workspace variable names\n');
            return;
        end
    end

    if ~isempty(variables)
        fprintf('Adding %d model workspace variables to CSV...\n', length(variables));
    else
        fprintf('No model workspace variables found\n');
        return;
    end

    for i = 1:length(variables)
        var_name = variables{i};

        try
            var_value = model_workspace.getVariable(var_name);

            % Handle different variable types - DIRECT approach (working version)
            if isnumeric(var_value) && isscalar(var_value)
                % Scalar numeric values (lengths, masses, etc.)
                column_name = sprintf('model_%s', var_name);
                data_table.(column_name) = repmat(var_value, num_rows, 1);

            elseif isnumeric(var_value) && isvector(var_value)
                % Vector values (3D coordinates, etc.)
                for j = 1:length(var_value)
                    column_name = sprintf('model_%s_%d', var_name, j);
                    data_table.(column_name) = repmat(var_value(j), num_rows, 1);
                end

            elseif isnumeric(var_value) && ismatrix(var_value)
                % Matrix values (inertia matrices, etc.)
                [rows, cols] = size(var_value);
                for r = 1:rows
                    for c = 1:cols
                        column_name = sprintf('model_%s_%d_%d', var_name, r, c);
                        data_table.(column_name) = repmat(var_value(r,c), num_rows, 1);
                    end
                end

            elseif isa(var_value, 'Simulink.Parameter')
                % Handle Simulink Parameters
                param_val = var_value.Value;
                if isnumeric(param_val) && isscalar(param_val)
                    column_name = sprintf('model_%s', var_name);
                    data_table.(column_name) = repmat(param_val, num_rows, 1);
                end
            end

        catch ME
            % Skip variables that can't be extracted
            fprintf('Warning: Could not extract variable %s: %s\n', var_name, ME.message);
        end
    end

catch ME
    fprintf('Warning: Could not access model workspace: %s\n', ME.message);
end
end

% FIXED: Extract workspace outputs (tout, xout, etc.)
function workspace_data = extractWorkspaceOutputs(simOut)
    workspace_data = [];

    try
        fprintf('Debug: Extracting workspace outputs\n');

        % Get available properties
        if isa(simOut, 'Simulink.SimulationOutput')
            available = simOut.who;
        else
            available = fieldnames(simOut);
        end

        fprintf('Debug: Available outputs: %s\n', strjoin(available, ', '));

        % Look for tout (time output)
        time_data = [];
        if ismember('tout', available)
            if isa(simOut, 'Simulink.SimulationOutput')
                time_data = simOut.get('tout');
            else
                time_data = simOut.tout;
            end
            fprintf('Debug: Found tout with length: %d\n', length(time_data));
        end

        if isempty(time_data)
            fprintf('Debug: No time output found\n');
            return;
        end

        data_cells = {time_data(:)};
        var_names = {'time'};

        % Look for xout (state output)
        if ismember('xout', available)
            if isa(simOut, 'Simulink.SimulationOutput')
                xout = simOut.get('xout');
            else
                xout = simOut.xout;
            end

            if ~isempty(xout) && size(xout, 1) == length(time_data)
                for i = 1:size(xout, 2)
                    data_cells{end+1} = xout(:, i);
                    var_names{end+1} = sprintf('x%d', i);
                end
                fprintf('Debug: Added xout with %d states\n', size(xout, 2));
            end
        end

        % Look for other numeric outputs
        for i = 1:length(available)
            var_name = available{i};

            % Skip already processed variables
            if ismember(var_name, {'tout', 'xout', 'logsout', 'simlog', 'CombinedSignalBus'})
                continue;
            end

            try
                if isa(simOut, 'Simulink.SimulationOutput')
                    var_data = simOut.get(var_name);
                else
                    var_data = simOut.(var_name);
                end

                if isnumeric(var_data) && length(var_data) == length(time_data)
                    data_cells{end+1} = var_data(:);
                    var_names{end+1} = var_name;
                    fprintf('Debug: Added workspace output %s\n', var_name);
                end
            catch
                % Skip variables that can't be accessed
                continue;
            end
        end

        if length(data_cells) > 1
            workspace_data = table(data_cells{:}, 'VariableNames', var_names);
            fprintf('Debug: Created workspace outputs table with %d columns\n', width(workspace_data));
        end

    catch ME
        fprintf('Error extracting workspace outputs: %s\n', ME.message);
    end
end

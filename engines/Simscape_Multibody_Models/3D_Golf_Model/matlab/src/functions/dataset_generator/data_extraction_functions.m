% Data extraction functions for golf swing simulation output
% These functions extract data from Simulink simulation output

function logsout_data = extractLogsoutDataFixed(logsout)
    logsout_data = [];

    try
        fprintf('Debug: Extracting logsout data, type: %s\n', class(logsout));

        % Handle modern Simulink.SimulationData.Dataset format
        if isa(logsout, 'Simulink.SimulationData.Dataset')
            fprintf('Debug: Processing Dataset format with %d elements\n', logsout.numElements);

            if logsout.numElements == 0
                fprintf('Debug: Dataset is empty\n');
                return;
            end

            % Get time from first element
            first_element = logsout.getElement(1);  % Use getElement instead of {}

            % Handle Signal objects properly
            if isa(first_element, 'Simulink.SimulationData.Signal')
                time = first_element.Values.Time;
                fprintf('Debug: Using time from Signal object, length: %d\n', length(time));
            elseif isa(first_element, 'timeseries')
                time = first_element.Time;
                fprintf('Debug: Using time from timeseries, length: %d\n', length(time));
            else
                fprintf('Debug: Unknown first element type: %s\n', class(first_element));
                return;
            end

            % Pre-allocate with conservative estimate (performance optimization)
            % Assume 2-4 columns per signal on average
            estimated_total = 1 + (logsout.numElements * 3);
            data_cells = cell(estimated_total, 1);
            var_names = cell(estimated_total, 1);

            % Initialize with time
            data_cells{1} = time;
            var_names{1} = 'time';
            cell_idx = 1;
            expected_length = length(time);

            % Process each element in the dataset
            for i = 1:logsout.numElements
                element = logsout.getElement(i);  % Use getElement

                if isa(element, 'Simulink.SimulationData.Signal')
                    signalName = element.Name;
                    if isempty(signalName)
                        signalName = sprintf('Signal_%d', i);
                    end

                    % Extract data from Signal object
                    data = element.Values.Data;
                    signal_time = element.Values.Time;

                    % Ensure data matches time length and is valid
                    if isnumeric(data) && length(signal_time) == expected_length && ~isempty(data)
                        % Check if data has the right dimensions
                        if size(data, 1) == expected_length
                            if size(data, 2) > 1
                                % Multi-dimensional signal
                                for col = 1:size(data, 2)
                                    col_data = data(:, col);
                                    % Ensure the column data is the right length
                                    if length(col_data) == expected_length
                                        cell_idx = cell_idx + 1;
                                        data_cells{cell_idx} = col_data;
                                        var_names{cell_idx} = sprintf('%s_%d', signalName, col);
                                        fprintf('Debug: Added multi-dim signal %s_%d (length: %d)\n', signalName, col, length(col_data));
                                    else
                                        fprintf('Debug: Skipping column %d of signal %s (length mismatch: %d vs %d)\n', col, signalName, length(col_data), expected_length);
                                    end
                                end
                            else
                                % Single column signal
                                flat_data = data(:);
                                if length(flat_data) == expected_length
                                    cell_idx = cell_idx + 1;
                                    data_cells{cell_idx} = flat_data;
                                    var_names{cell_idx} = signalName;
                                    fprintf('Debug: Added signal %s (length: %d)\n', signalName, length(flat_data));
                                else
                                    fprintf('Debug: Skipping signal %s (flattened length mismatch: %d vs %d)\n', signalName, length(flat_data), expected_length);
                                end
                            end
                        else
                            fprintf('Debug: Skipping signal %s (row dimension mismatch: %d vs %d)\n', signalName, size(data, 1), expected_length);
                        end
                    else
                        fprintf('Debug: Skipping signal %s (time length mismatch: %d vs %d, or empty data)\n', signalName, length(signal_time), expected_length);
                    end

                elseif isa(element, 'timeseries')
                    signalName = element.Name;
                    data = element.Data;
                    if isnumeric(data) && length(data) == expected_length && ~isempty(data)
                        flat_data = data(:);
                        if length(flat_data) == expected_length
                            cell_idx = cell_idx + 1;
                            data_cells{cell_idx} = flat_data;
                            var_names{cell_idx} = signalName;
                            fprintf('Debug: Added timeseries %s (length: %d)\n', signalName, length(flat_data));
                        else
                            fprintf('Debug: Skipping timeseries %s (flattened length mismatch: %d vs %d)\n', signalName, length(flat_data), expected_length);
                        end
                    else
                        fprintf('Debug: Skipping timeseries %s (length mismatch: %d vs %d, or empty data)\n', signalName, length(data), expected_length);
                    end
                else
                    fprintf('Debug: Element %d is type: %s\n', i, class(element));
                end
            end

            % Trim to actual size (performance optimization)
            data_cells = data_cells(1:cell_idx);
            var_names = var_names(1:cell_idx);

            % Validate all data vectors have the same length before creating table
            if length(data_cells) > 1
                % Check that all vectors have the same length
                lengths = cellfun(@length, data_cells);
                if all(lengths == expected_length)
                    logsout_data = table(data_cells{:}, 'VariableNames', var_names);
                    fprintf('Debug: Created logsout table with %d columns, all vectors length %d\n', width(logsout_data), expected_length);
                else
                    fprintf('Debug: Cannot create table - vector length mismatch. Lengths: ');
                    fprintf('%d ', lengths);
                    fprintf('\n');
                    % Try to create table with only vectors of the correct length
                    valid_indices = find(lengths == expected_length);
                    if length(valid_indices) > 1
                        valid_cells = data_cells(valid_indices);
                        valid_names = var_names(valid_indices);
                        logsout_data = table(valid_cells{:}, 'VariableNames', valid_names);
                        fprintf('Debug: Created logsout table with %d valid columns (length %d)\n', width(logsout_data), expected_length);
                    else
                        fprintf('Debug: Not enough valid vectors to create table\n');
                    end
                end
            else
                fprintf('Debug: No valid data found in logsout Dataset\n');
            end

        else
            fprintf('Debug: Logsout format not supported: %s\n', class(logsout));
        end

    catch ME
        fprintf('Error extracting logsout data: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
end

function simscape_data = extractSimscapeDataFixed(simlog)
    simscape_data = [];

    try
        fprintf('Debug: Extracting Simscape data\n');

        if ~isempty(simlog) && isa(simlog, 'simscape.logging.Node')
            % Get series data - this is the correct way to access Simscape data
            try
                series_info = simlog.series();
                if ~isempty(series_info)
                    time = series_info.time;
                    fprintf('Debug: Found Simscape time series, length: %d\n', length(time));

                    data_cells = {time};
                    var_names = {'time'};

                    % Get all logged variables
                    logged_vars = simlog.listVariables('-all');
                    fprintf('Debug: Found %d Simscape variables\n', length(logged_vars));

                    for i = 1:length(logged_vars)
                        var_name = logged_vars{i};
                        try
                            var_data = simlog.find(var_name);
                            if ~isempty(var_data) && isprop(var_data, 'series')
                                var_series = var_data.series();
                                if ~isempty(var_series) && length(var_series.values) == length(time)
                                    data_cells{end+1} = var_series.values;
                                    var_names{end+1} = strrep(var_name, '.', '_');
                                    fprintf('Debug: Added Simscape variable %s\n', var_name);
                                end
                            end
                        catch
                            % Skip variables that can't be accessed
                            continue;
                        end
                    end

                    if length(data_cells) > 1
                        simscape_data = table(data_cells{:}, 'VariableNames', var_names);
                        fprintf('Debug: Created Simscape table with %d columns\n', width(simscape_data));
                    end
                end
            catch ME2
                fprintf('Debug: Could not access Simscape series data: %s\n', ME2.message);
            end
        else
            fprintf('Debug: Simlog is not a valid Simscape logging node\n');
        end

    catch ME
        fprintf('Error extracting Simscape data: %s\n', ME.message);
    end
end

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

% Extract from CombinedSignalBus with enhanced data extraction support
function data_table = extractFromCombinedSignalBus(combinedBus)
data_table = [];

try
    % CombinedSignalBus should be a struct with time and signals
    % Enhanced for comprehensive data extraction
    if ~isstruct(combinedBus)
        return;
    end

    % Look for time field
    bus_fields = fieldnames(combinedBus);

    time_field = '';
    time_data = [];

    % Find time data - check common time field patterns
    for i = 1:length(bus_fields)
        field_name = bus_fields{i};
        field_value = combinedBus.(field_name);

        % Check if this field contains time data
        if isstruct(field_value) && isfield(field_value, 'time')
            time_field = field_name;
            time_data = field_value.time(:);  % Extract time from struct
            fprintf('Debug: Found time in %s.time (length: %d)\n', field_name, length(time_data));
            break;
        elseif isstruct(field_value) && isfield(field_value, 'Time')
            time_field = field_name;
            time_data = field_value.Time(:);  % Extract Time from struct
            fprintf('Debug: Found time in %s.Time (length: %d)\n', field_name, length(time_data));
            break;
        elseif contains(lower(field_name), 'time') && isnumeric(field_value)
            time_field = field_name;
            time_data = field_value(:);  % Ensure column vector
            fprintf('Debug: Found time field: %s (length: %d)\n', field_name, length(time_data));
            break;
        end
    end

    % If still no time found, try the first field that looks like it has time data
    if isempty(time_data)
        for i = 1:length(bus_fields)
            field_name = bus_fields{i};
            field_value = combinedBus.(field_name);

            if isstruct(field_value)
                sub_fields = fieldnames(field_value);
                for j = 1:length(sub_fields)
                    if contains(lower(sub_fields{j}), 'time')
                        time_field = field_name;
                        time_data = field_value.(sub_fields{j})(:);
                        fprintf('Debug: Found time in %s.%s (length: %d)\n', field_name, sub_fields{j}, length(time_data));
                        break;
                    end
                end
                if ~isempty(time_data)
                    break;
                end
            end
        end
    end

    % First, try to find time data by examining the signal structures
    for i = 1:length(bus_fields)
        if ~isempty(time_data)
            break;
        end

        field_name = bus_fields{i};
        field_value = combinedBus.(field_name);

        if isstruct(field_value)
            % This field contains sub-signals
            sub_fields = fieldnames(field_value);

            % Try to get time from the first valid signal
            for j = 1:length(sub_fields)
                sub_field_name = sub_fields{j};
                signal_data = field_value.(sub_field_name);

                % Check if this is a timeseries or signal structure with time
                if isa(signal_data, 'timeseries')
                    time_data = signal_data.Time(:);
                    fprintf('Debug: Found time in %s.%s (timeseries), length: %d\n', field_name, sub_field_name, length(time_data));
                    break;
                elseif isstruct(signal_data) && isfield(signal_data, 'time')
                    time_data = signal_data.time(:);
                    fprintf('Debug: Found time in %s.%s.time, length: %d\n', field_name, sub_field_name, length(time_data));
                    break;
                elseif isstruct(signal_data) && isfield(signal_data, 'Time')
                    time_data = signal_data.Time(:);
                    fprintf('Debug: Found time in %s.%s.Time, length: %d\n', field_name, sub_field_name, length(time_data));
                    break;
                elseif isstruct(signal_data) && isfield(signal_data, 'Values')
                    % Could be a Simulink.SimulationData.Signal format
                    if isnumeric(signal_data.Values) && size(signal_data.Values, 1) > 1
                        % Assume first column is time if it exists
                        time_data = (0:size(signal_data.Values, 1)-1)' * 0.001; % Default 1ms sampling
                        fprintf('Debug: Generated time vector for %s.%s, length: %d\n', field_name, sub_field_name, length(time_data));
                        break;
                    end
                elseif isnumeric(signal_data) && length(signal_data) > 1
                    % Just numeric data, we'll need to generate time
                    time_data = (0:length(signal_data)-1)' * 0.001; % Default 1ms sampling
                    fprintf('Debug: Generated time vector from %s.%s numeric data, length: %d\n', field_name, sub_field_name, length(time_data));
                    break;
                end
            end
        end
    end

    if isempty(time_data)
        fprintf('Debug: No time data found in any signals\n');
        return;
    end

    % Now extract all signals using this time reference
    data_cells = {time_data};
    var_names = {'time'};
    expected_length = length(time_data);

    fprintf('Debug: Starting data extraction with time vector length: %d\n', expected_length);

    % Process each field in the bus
    for i = 1:length(bus_fields)
        field_name = bus_fields{i};
        field_value = combinedBus.(field_name);

        if isstruct(field_value)
            % This field contains sub-signals
            sub_fields = fieldnames(field_value);

            for j = 1:length(sub_fields)
                sub_field_name = sub_fields{j};
                signal_data = field_value.(sub_field_name);

                % Extract numeric data from various formats
                numeric_data = [];

                if isa(signal_data, 'timeseries')
                    numeric_data = signal_data.Data;
                elseif isstruct(signal_data) && isfield(signal_data, 'Data')
                    numeric_data = signal_data.Data;
                elseif isstruct(signal_data) && isfield(signal_data, 'Values')
                    numeric_data = signal_data.Values;
                elseif isnumeric(signal_data)
                    numeric_data = signal_data;
                end

                % Add the data - handle both time series and constant properties
                if ~isempty(numeric_data)
                    data_size = size(numeric_data);
                    num_elements = numel(numeric_data);

                    if size(numeric_data, 1) == expected_length
                        % TIME SERIES DATA - matches expected length
                        if size(numeric_data, 2) == 1
                            % Single column time series
                            data_cells{end+1} = numeric_data(:);
                            var_names{end+1} = sprintf('%s_%s', field_name, sub_field_name);
                            fprintf('Debug: Added time series %s_%s\n', field_name, sub_field_name);
                        elseif size(numeric_data, 2) > 1
                            % Multi-column time series
                            for col = 1:size(numeric_data, 2)
                                data_cells{end+1} = numeric_data(:, col);
                                var_names{end+1} = sprintf('%s_%s_%d', field_name, sub_field_name, col);
                                fprintf('Debug: Added time series %s_%s_%d\n', field_name, sub_field_name, col);
                            end
                        end

                    elseif num_elements == 3
                        % 3D VECTOR (e.g., COM position [x, y, z])
                        vector_data = numeric_data(:);  % Ensure column vector
                        for dim = 1:3
                            % Replicate constant value for all time steps
                            replicated_data = repmat(vector_data(dim), expected_length, 1);
                            data_cells{end+1} = replicated_data;
                            dim_labels = {'x', 'y', 'z'};
                            var_names{end+1} = sprintf('%s_%s_%s', field_name, sub_field_name, dim_labels{dim});
                            fprintf('Debug: Added 3D vector %s_%s_%s (replicated %g for %d timesteps)\n', ...
                                field_name, sub_field_name, dim_labels{dim}, vector_data(dim), expected_length);
                        end

                    elseif num_elements == 9
                        % 3x3 MATRIX (e.g., inertia matrix) - treat as potentially time-varying
                        if isequal(data_size, [3, 3])
                            % Convert constant 3x3 matrix to 3x3xN format for consistent handling
                            matrix_data = repmat(numeric_data, [1, 1, expected_length]);
                        else
                            % Reshape to 3x3 if it's a 9x1 vector, then expand to 3x3xN
                            matrix_data = reshape(numeric_data, 3, 3);
                            matrix_data = repmat(matrix_data, [1, 1, expected_length]);
                        end

                        % Now handle as 3x3xN time series (same as below)
                        n_steps = size(matrix_data,3);
                        % Flatten each 3x3 matrix at each timestep into 9 columns
                        flat_matrix = reshape(permute(matrix_data, [3 1 2]), n_steps, 9);
                        for idx = 1:9
                            [row, col] = ind2sub([3,3], idx);
                            data_cells{end+1} = flat_matrix(:,idx);
                            var_names{end+1} = sprintf('%s_%s_I%d%d', field_name, sub_field_name, row, col);
                            fprintf('Debug: Added 3x3 matrix %s_%s_I%d%d (N=%d)\n', field_name, sub_field_name, row, col, n_steps);
                        end

                    elseif num_elements == 1
                        % 1 ELEMENT DATA (scalar constants)
                        scalar_value = numeric_data(1);
                        replicated_data = repmat(scalar_value, expected_length, 1);
                        data_cells{end+1} = replicated_data;
                        var_names{end+1} = sprintf('%s_%s', field_name, sub_field_name);
                        fprintf('Debug: Added scalar data %s_%s (replicated %g for %d timesteps)\n', ...
                            field_name, sub_field_name, scalar_value, expected_length);

                    elseif num_elements == 6
                        % 6 ELEMENT DATA (e.g., 6DOF pose/twist)
                        vector_data = numeric_data(:);  % Ensure column vector
                        for dim = 1:6
                            replicated_data = repmat(vector_data(dim), expected_length, 1);
                            data_cells{end+1} = replicated_data;
                            var_names{end+1} = sprintf('%s_%s_dof%d', field_name, sub_field_name, dim);
                            fprintf('Debug: Added 6DOF data %s_%s_dof%d (replicated %g for %d timesteps)\n', ...
                                field_name, sub_field_name, dim, vector_data(dim), expected_length);
                        end

                        % Handle 3x1xN time series (3D vectors over time)
                    elseif ndims(numeric_data) == 3 && all(size(numeric_data,1:2) == [3 1])
                        n_steps = size(numeric_data,3);
                        if n_steps ~= expected_length
                            fprintf('Debug: Skipping %s.%s (3x1xN but N=%d, expected %d)\n', field_name, sub_field_name, n_steps, expected_length);
                        else
                            % Extract each component of the 3D vector over time
                            for dim = 1:3
                                data_cells{end+1} = squeeze(numeric_data(dim, 1, :));
                                var_names{end+1} = sprintf('%s_%s_dim%d', field_name, sub_field_name, dim);
                                fprintf('Debug: Added 3x1xN vector %s_%s_dim%d (N=%d)\n', field_name, sub_field_name, dim, n_steps);
                            end
                        end

                    elseif ndims(numeric_data) == 3 && all(size(numeric_data,1:2) == [3 3])
                        % Handle 3x3xN time series (e.g., inertia over time)
                        n_steps = size(numeric_data,3);
                        if n_steps ~= expected_length
                            fprintf('Debug: Skipping %s.%s (3x3xN but N=%d, expected %d)\n', field_name, sub_field_name, n_steps, expected_length);
                        else
                            % Flatten each 3x3 matrix at each timestep into 9 columns
                            flat_matrix = reshape(permute(numeric_data, [3 1 2]), n_steps, 9);
                            for idx = 1:9
                                [row, col] = ind2sub([3,3], idx);
                                data_cells{end+1} = flat_matrix(:,idx);
                                var_names{end+1} = sprintf('%s_%s_I%d%d', field_name, sub_field_name, row, col);
                                fprintf('Debug: Added 3x3xN matrix %s_%s_I%d%d (N=%d)\n', field_name, sub_field_name, row, col, n_steps);
                            end
                        end

                    else
                        % UNHANDLED SIZE - still skip but with better diagnostic
                        fprintf('Debug: Skipping %s.%s (size [%s] not supported - need time series, 3D vector, 3x3 matrix, 6DOF, or scalar)\n', ...
                            field_name, sub_field_name, num2str(data_size));
                    end
                end
            end
        elseif isnumeric(field_value) && length(field_value) == expected_length
            % Direct numeric field
            data_cells{end+1} = field_value(:);
            var_names{end+1} = field_name;
            fprintf('Debug: Added direct field %s\n', field_name);
        end
    end

    % Create table if we have data
    if length(data_cells) > 1
        data_table = table(data_cells{:}, 'VariableNames', var_names);
        fprintf('Debug: Created CombinedSignalBus table with %d columns, %d rows\n', width(data_table), height(data_table));
    else
        fprintf('Debug: No valid data found in CombinedSignalBus\n');
        fprintf('Debug: Total data_cells collected: %d\n', length(data_cells));
        fprintf('Debug: Variable names: %s\n', strjoin(var_names, ', '));
    end

catch ME
    fprintf('Error extracting CombinedSignalBus data: %s\n', ME.message);
end
end

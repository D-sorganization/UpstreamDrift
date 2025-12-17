% FIXED: Extract from logsout with proper Signal handling
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

        data_cells = {time};
        var_names = {'time'};
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
                    data_size = size(data);
                    num_elements = numel(data);

                    if size(data, 1) == expected_length
                        % Standard time series data (N x M)
                        if size(data, 2) > 1
                            % Multi-dimensional signal
                            for col = 1:size(data, 2)
                                col_data = data(:, col);
                                % Ensure the column data is the right length
                                if length(col_data) == expected_length
                                    data_cells{end+1} = col_data;
                                    var_names{end+1} = sprintf('%s_%d', signalName, col);
                                    fprintf('Debug: Added multi-dim signal %s_%d (length: %d)\n', signalName, col, length(col_data));
                                else
                                    fprintf('Debug: Skipping column %d of signal %s (length mismatch: %d vs %d)\n', col, signalName, length(col_data), expected_length);
                                end
                            end
                        else
                            % Single column signal
                            flat_data = data(:);
                            if length(flat_data) == expected_length
                                data_cells{end+1} = flat_data;
                                var_names{end+1} = signalName;
                                fprintf('Debug: Added signal %s (length: %d)\n', signalName, length(flat_data));
                            else
                                fprintf('Debug: Skipping signal %s (flattened length mismatch: %d vs %d)\n', signalName, length(flat_data), expected_length);
                            end
                        end



                    else
                        fprintf('Debug: Skipping signal %s (size [%s] not supported - need time series, [3 1 N], or [3 3 N])\n', ...
                            signalName, num2str(data_size));
                    end
                else
                    fprintf('Debug: Skipping signal %s (time length mismatch: %d vs %d, or empty data)\n', signalName, length(signal_time), expected_length);
                end

            elseif isa(element, 'timeseries')
                signalName = element.Name;
                data = element.Data;
                if isnumeric(data) && ~isempty(data)
                    data_size = size(data);
                    num_elements = numel(data);

                    if length(data) == expected_length
                        % Standard timeseries data
                        flat_data = data(:);
                        if length(flat_data) == expected_length
                            data_cells{end+1} = flat_data;
                            var_names{end+1} = signalName;
                            fprintf('Debug: Added timeseries %s (length: %d)\n', signalName, length(flat_data));
                        else
                            fprintf('Debug: Skipping timeseries %s (flattened length mismatch: %d vs %d)\n', signalName, length(flat_data), expected_length);
                        end



                    else
                        fprintf('Debug: Skipping timeseries %s (size [%s] not supported - need time series, [3 1 N], or [3 3 N])\n', ...
                            signalName, num2str(data_size));
                    end
                else
                    fprintf('Debug: Skipping timeseries %s (empty data)\n', signalName);
                end
            else
                fprintf('Debug: Skipping unknown element type: %s\n', class(element));
            end
        end

        % Create table if we have data
        if length(data_cells) > 1  % At least time + one signal
            logsout_data = table(data_cells{:}, 'VariableNames', var_names);
            fprintf('Debug: Created logsout table with %d columns and %d rows\n', width(logsout_data), height(logsout_data));
        else
            fprintf('Debug: No valid signals found in logsout\n');
        end

    else
        fprintf('Debug: Unsupported logsout format: %s\n', class(logsout));
    end

catch ME
    fprintf('Error extracting logsout data: %s\n', ME.message);
end
end

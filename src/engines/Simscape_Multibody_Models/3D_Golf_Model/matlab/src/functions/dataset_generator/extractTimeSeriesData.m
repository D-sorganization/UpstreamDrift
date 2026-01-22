% Helper function to extract time series data from various formats
function [time_data, values_data] = extractTimeSeriesData(data_obj, signal_path)
    time_data = [];
    values_data = [];

    try
        if isa(data_obj, 'timeseries')
            % MATLAB timeseries object
            time_data = data_obj.Time;
            values_data = data_obj.Data;
            fprintf('Debug: Extracted timeseries from %s\n', signal_path);
        elseif isstruct(data_obj)
            % Struct with time/data fields
            if isfield(data_obj, 'time') && isfield(data_obj, 'signals')
                time_data = data_obj.time;
                if isstruct(data_obj.signals) && isfield(data_obj.signals, 'values')
                    values_data = data_obj.signals.values;
                    fprintf('Debug: Extracted struct time/signals from %s\n', signal_path);
                end
            elseif isfield(data_obj, 'Time') && isfield(data_obj, 'Data')
                time_data = data_obj.Time;
                values_data = data_obj.Data;
                fprintf('Debug: Extracted struct Time/Data from %s\n', signal_path);
            else
                % Check for direct numeric data in struct (constant matrices/vectors)
                fields = fieldnames(data_obj);
                for i = 1:length(fields)
                    field_name = fields{i};
                    field_value = data_obj.(field_name);
                    if isnumeric(field_value)
                        % Found numeric data - treat as constant and create mock time
                        num_elements = numel(field_value);
                        if num_elements >= 1 && num_elements <= 9
                            % Create a default time vector for constant data
                            time_data = linspace(0, 3, 3006)';  % Default 3 second simulation
                            values_data = field_value;
                            fprintf('Debug: Extracted constant numeric data from %s.%s (%d elements)\n', signal_path, field_name, num_elements);
                            break;  % Use first numeric field found
                        end
                    end
                end
            end
        elseif isnumeric(data_obj)
            % Direct numeric data (constant values)
            num_elements = numel(data_obj);
            if num_elements >= 1 && num_elements <= 9
                % Create a default time vector for constant data
                time_data = linspace(0, 3, 3006)';  % Default 3 second simulation
                values_data = data_obj;
                fprintf('Debug: Extracted direct numeric data from %s (%d elements)\n', signal_path, num_elements);
            end
        end

        % Ensure column vectors
        if ~isempty(time_data)
            time_data = time_data(:);
        end
        if ~isempty(values_data)
            if isvector(values_data)
                values_data = values_data(:);
            end
        end

    catch ME
        fprintf('Debug: Error extracting time series from %s: %s\n', signal_path, ME.message);
    end
end

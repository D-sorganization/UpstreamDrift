function bus_data = extractCombinedSignalBusData(combined_bus)
    bus_data = [];

    try
        fprintf('Debug: Extracting CombinedSignalBus data\n');

        if ~isstruct(combined_bus)
            fprintf('Debug: CombinedSignalBus is not a struct\n');
            return;
        end

        fields = fieldnames(combined_bus);
        fprintf('Debug: Found %d fields in CombinedSignalBus\n', length(fields));

        % Get time from the first timeseries we find
        time_data = [];
        expected_length = 0;

        % Find time data from first available timeseries
        for i = 1:length(fields)
            field_name = fields{i};
            field_value = combined_bus.(field_name);

            if isstruct(field_value)
                sub_fields = fieldnames(field_value);
                for j = 1:length(sub_fields)
                    sub_field_name = sub_fields{j};
                    sub_field_value = field_value.(sub_field_name);

                    if isa(sub_field_value, 'timeseries')
                        time_data = sub_field_value.Time;
                        expected_length = length(time_data);
                        fprintf('Debug: Found time data from %s.%s (length: %d)\n', field_name, sub_field_name, expected_length);
                        break;
                    end
                end
                if ~isempty(time_data)
                    break;
                end
            elseif isa(field_value, 'timeseries')
                time_data = field_value.Time;
                expected_length = length(time_data);
                fprintf('Debug: Found time data from %s (length: %d)\n', field_name, expected_length);
                break;
            end
        end

        if isempty(time_data)
            fprintf('Debug: No time data found in CombinedSignalBus\n');
            return;
        end

        data_cells = {time_data(:)};
        var_names = {'time'};
        used_names = {'time'}; % Track used names to avoid duplicates

        % Extract all timeseries data
        for i = 1:length(fields)
            field_name = fields{i};
            field_value = combined_bus.(field_name);

            if isstruct(field_value)
                % Handle struct fields (most common case)
                sub_fields = fieldnames(field_value);
                fprintf('Debug: Processing struct field %s with %d sub-fields\n', field_name, length(sub_fields));

                for j = 1:length(sub_fields)
                    sub_field_name = sub_fields{j};
                    sub_field_value = field_value.(sub_field_name);

                    if isa(sub_field_value, 'timeseries')
                        % Extract data from timeseries
                        data = sub_field_value.Data;

                        if isnumeric(data) && ~isempty(data)
                            % Handle different data dimensions
                            if size(data, 1) == expected_length
                                if size(data, 2) > 1
                                    % Multi-dimensional data
                                    for col = 1:size(data, 2)
                                        col_data = data(:, col);
                                        if length(col_data) == expected_length
                                            % Create unique column name
                                            base_name = sprintf('%s_%s_%d', field_name, sub_field_name, col);
                                            unique_name = makeUniqueName(base_name, used_names);
                                            used_names{end+1} = unique_name;

                                            data_cells{end+1} = col_data;
                                            var_names{end+1} = unique_name;
                                            fprintf('Debug: Added %s (length: %d)\n', unique_name, length(col_data));
                                        end
                                    end
                                else
                                    % Single column data
                                    flat_data = data(:);
                                    if length(flat_data) == expected_length
                                        % Create unique column name
                                        base_name = sprintf('%s_%s', field_name, sub_field_name);
                                        unique_name = makeUniqueName(base_name, used_names);
                                        used_names{end+1} = unique_name;

                                        data_cells{end+1} = flat_data;
                                        var_names{end+1} = unique_name;
                                        fprintf('Debug: Added %s (length: %d)\n', unique_name, length(flat_data));
                                    end
                                end
                            else
                                fprintf('Debug: Skipping %s.%s (row dimension mismatch: %d vs %d)\n', field_name, sub_field_name, size(data, 1), expected_length);
                            end
                        end
                    end
                end
            elseif isa(field_value, 'timeseries')
                % Handle direct timeseries fields
                data = field_value.Data;

                if isnumeric(data) && ~isempty(data)
                    if size(data, 1) == expected_length
                        if size(data, 2) > 1
                            % Multi-dimensional data
                            for col = 1:size(data, 2)
                                col_data = data(:, col);
                                if length(col_data) == expected_length
                                    % Create unique column name
                                    base_name = sprintf('%s_%d', field_name, col);
                                    unique_name = makeUniqueName(base_name, used_names);
                                    used_names{end+1} = unique_name;

                                    data_cells{end+1} = col_data;
                                    var_names{end+1} = unique_name;
                                    fprintf('Debug: Added %s (length: %d)\n', unique_name, length(col_data));
                                end
                            end
                        else
                            % Single column data
                            flat_data = data(:);
                            if length(flat_data) == expected_length
                                % Create unique column name
                                unique_name = makeUniqueName(field_name, used_names);
                                used_names{end+1} = unique_name;

                                data_cells{end+1} = flat_data;
                                var_names{end+1} = unique_name;
                                fprintf('Debug: Added %s (length: %d)\n', unique_name, length(flat_data));
                            end
                        end
                    else
                        fprintf('Debug: Skipping %s (row dimension mismatch: %d vs %d)\n', field_name, size(data, 1), expected_length);
                    end
                end
            end
        end

        % Create table if we have data
        if length(data_cells) > 1
            % Validate all data vectors have the same length
            lengths = cellfun(@length, data_cells);
            if all(lengths == expected_length)
                bus_data = table(data_cells{:}, 'VariableNames', var_names);
                fprintf('Debug: Created CombinedSignalBus table with %d columns, all vectors length %d\n', width(bus_data), expected_length);
            else
                fprintf('Debug: Cannot create table - vector length mismatch. Lengths: ');
                fprintf('%d ', lengths);
                fprintf('\n');
                % Try to create table with only vectors of the correct length
                valid_indices = find(lengths == expected_length);
                if length(valid_indices) > 1
                    valid_cells = data_cells(valid_indices);
                    valid_names = var_names(valid_indices);
                    bus_data = table(valid_cells{:}, 'VariableNames', valid_names);
                    fprintf('Debug: Created CombinedSignalBus table with %d valid columns (length %d)\n', width(bus_data), expected_length);
                else
                    fprintf('Debug: Not enough valid vectors to create table\n');
                end
            end
        else
            fprintf('Debug: No valid data found in CombinedSignalBus\n');
        end

    catch ME
        fprintf('Error extracting CombinedSignalBus data: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
end

function unique_name = makeUniqueName(base_name, used_names)
    % Helper function to create unique variable names
    unique_name = base_name;
    counter = 1;

    while ismember(unique_name, used_names)
        unique_name = sprintf('%s_dup%d', base_name, counter);
        counter = counter + 1;
    end
end

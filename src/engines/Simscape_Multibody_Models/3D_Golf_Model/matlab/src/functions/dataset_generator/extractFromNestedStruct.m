function data_table = extractFromNestedStruct(nested_struct, struct_name, time_data)
    % External function for extracting data from nested structures - can be used in parallel processing
    % This function doesn't rely on config.verbosity

    data_table = table();

    try
        if isempty(nested_struct)
            return;
        end

        % Initialize data arrays
        all_data = {};
        var_names = {};

        % Get field names
        field_names = fieldnames(nested_struct);

        % Determine expected length from time data or first field
        expected_length = [];
        if ~isempty(time_data)
            expected_length = length(time_data);
        else
            % Try to find time data in the structure
            for i = 1:length(field_names)
                field_value = nested_struct.(field_names{i});
                if isstruct(field_value) && isfield(field_value, 'time')
                    expected_length = length(field_value.time);
                    break;
                elseif isnumeric(field_value) && isvector(field_value)
                    expected_length = length(field_value);
                    break;
                end
            end
        end

        if isempty(expected_length)
            fprintf('Could not determine expected data length\n');
            return;
        end

        % Process each field
        for i = 1:length(field_names)
            field_name = field_names{i};
            field_value = nested_struct.(field_name);

            % Create full field name
            if ~isempty(struct_name)
                full_field_name = sprintf('%s_%s', struct_name, field_name);
            else
                full_field_name = field_name;
            end

            % Extract data from this field
            field_data = extractDataFromField(field_value, expected_length);

            if ~isempty(field_data)
                % Add field name prefix to variable names
                prefixed_names = cellfun(@(name) sprintf('%s_%s', full_field_name, name), ...
                                       field_data.var_names, 'UniformOutput', false);

                all_data = [all_data, field_data.data_cells];
                var_names = [var_names, prefixed_names];
            end
        end

        % Create table
        if ~isempty(all_data)
            data_table = table(all_data{:}, 'VariableNames', var_names);
        end

    catch ME
        fprintf('Error extracting from nested struct: %s\n', ME.message);
    end
end

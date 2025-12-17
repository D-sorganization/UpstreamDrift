function simIn = loadInputFile(simIn, input_file)
    try
        % Load input data
        input_data = load(input_file);

        % Get field names and set as model variables
        field_names = fieldnames(input_data);
        for i = 1:length(field_names)
            field_name = field_names{i};
            field_value = input_data.(field_name);

            % Only set scalar values or small arrays
            if isscalar(field_value) || (isnumeric(field_value) && numel(field_value) <= 100)
                simIn = simIn.setVariable(field_name, field_value);
            end
        end
    catch ME
        warning('Could not load input file %s: %s', input_file, ME.message);
    end
end

function data_array = extractDataFromField(field_value, expected_length)
    % Extract numeric data from various field formats
    data_array = [];

    try
        if isa(field_value, 'timeseries')
            data_array = field_value.Data;
        elseif isa(field_value, 'Simulink.SimulationData.Signal')
            data_array = field_value.Values.Data;
        elseif isstruct(field_value)
            if isfield(field_value, 'Data')
                data_array = field_value.Data;
            elseif isfield(field_value, 'signals')
                % Handle nested signal structure
                return; % Let specialized function handle this
            elseif isfield(field_value, 'Values')
                data_array = field_value.Values;
            end
        elseif isnumeric(field_value)
            data_array = field_value;
        end

        % Validate data length
        if ~isempty(data_array) && size(data_array, 1) ~= expected_length
            data_array = [];
        end
    catch
        data_array = [];
    end
end

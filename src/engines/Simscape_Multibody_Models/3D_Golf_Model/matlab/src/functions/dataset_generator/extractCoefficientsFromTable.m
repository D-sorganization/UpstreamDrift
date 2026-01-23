function coefficient_values = extractCoefficientsFromTable(handles)
    try
        table_data = get(handles.coefficients_table, 'Data');

        if isempty(table_data)
            coefficient_values = [];
            return;
        end

        num_trials = size(table_data, 1);
        num_total_coeffs = size(table_data, 2) - 1;
        coefficient_values = zeros(num_trials, num_total_coeffs);

        for row = 1:num_trials
            for col = 2:(num_total_coeffs + 1)
                value_str = table_data{row, col};
                if ischar(value_str)
                    coefficient_values(row, col-1) = str2double(value_str);
                else
                    coefficient_values(row, col-1) = value_str;
                end
            end
        end

        if any(isnan(coefficient_values(:)))
            warning('Some coefficient values are invalid (NaN)');
        end

    catch ME
        warning('Error extracting coefficients: %s', ME.message);
        coefficient_values = [];
    end
end

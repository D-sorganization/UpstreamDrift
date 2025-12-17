function resampled_table = resampleDataToFrequency(data_table, target_freq, sim_time)
    resampled_table = data_table;

    try
        % Find time column
        time_col = find(contains(lower(data_table.Properties.VariableNames), 'time'), 1);
        if isempty(time_col)
            fprintf('Warning: No time column found, cannot resample\n');
            return;
        end

        original_time = data_table.(data_table.Properties.VariableNames{time_col});
        original_freq = 1 / mean(diff(original_time));

        fprintf('Original data: %d points at ~%.1f Hz\n', length(original_time), original_freq);
        fprintf('Target frequency: %.1f Hz\n', target_freq);

        % If target frequency is higher than original, no need to resample
        if target_freq >= original_freq
            fprintf('Target frequency >= original frequency, keeping original data\n');
            return;
        end

        % Calculate new time vector
        target_dt = 1 / target_freq;
        new_time = 0:target_dt:sim_time;

        % Ensure we don't exceed simulation time
        if new_time(end) > sim_time
            new_time = new_time(new_time <= sim_time);
        end

        fprintf('Resampling to %d points at %.1f Hz\n', length(new_time), target_freq);

        % Create new table with resampled data
        resampled_data = cell(1, width(data_table));
        resampled_data{time_col} = new_time';

        % Resample each column
        for col = 1:width(data_table)
            if col == time_col
                continue; % Already handled
            end

            original_data = data_table.(data_table.Properties.VariableNames{col});

            % Use interpolation for smooth resampling
            if isnumeric(original_data) && length(original_data) == length(original_time)
                try
                    % Use interp1 for interpolation
                    resampled_data{col} = interp1(original_time, original_data, new_time, 'linear', 'extrap')';
                catch
                    % Fallback to nearest neighbor if interpolation fails
                    fprintf('Warning: Using nearest neighbor interpolation for column %s\n', data_table.Properties.VariableNames{col});
                    resampled_data{col} = interp1(original_time, original_data, new_time, 'nearest', 'extrap')';
                end
            else
                % For non-numeric or mismatched data, use nearest neighbor
                resampled_data{col} = interp1(original_time, original_data, new_time, 'nearest', 'extrap')';
            end
        end

        % Create resampled table
        resampled_table = table(resampled_data{:}, 'VariableNames', data_table.Properties.VariableNames);

        fprintf('Successfully resampled data from %d to %d points (%.1f%% reduction)\n', ...
                length(original_time), length(new_time), ...
                100 * (1 - length(new_time) / length(original_time)));

    catch ME
        fprintf('Error resampling data: %s\n', ME.message);
        fprintf('Returning original data\n');
        resampled_table = data_table;
    end
end

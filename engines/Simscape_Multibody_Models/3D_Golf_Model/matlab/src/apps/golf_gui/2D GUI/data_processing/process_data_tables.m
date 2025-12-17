function [BASEQ, ZTCFQ, DELTAQ] = process_data_tables(config, BaseData, ZTCF)
% PROCESS_DATA_TABLES - Process base and ZTCF data into Q-tables for visualization
%
% Inputs:
%   config - Configuration structure from model_config()
%   BaseData - Base data table from generate_base_data()
%   ZTCF - ZTCF data table from generate_ztcf_data()
%
% Returns:
%   BASEQ - Processed base data table for visualization
%   ZTCFQ - Processed ZTCF data table for visualization
%   DELTAQ - Delta data table (BASEQ - ZTCFQ) for visualization
%
% This function:
%   1. Ensures BaseData and ZTCF have proper Time columns
%   2. Creates DELTAQ as the difference between BASEQ and ZTCFQ
%   3. Returns all three Q-tables ready for visualization

    fprintf('🔄 Processing data tables...\n');

    try
        % Ensure BaseData has a Time column
        if ~ismember('Time', BaseData.Properties.VariableNames)
            time_vector = (0:height(BaseData)-1) * config.sample_time;
            BaseData.Time = time_vector';
        end
        BASEQ = BaseData;

        % Ensure ZTCF has a Time column
        if ~ismember('Time', ZTCF.Properties.VariableNames)
            time_vector = (0:height(ZTCF)-1) * config.sample_time;
            ZTCF.Time = time_vector';
        end
        ZTCFQ = ZTCF;

        % Create DELTAQ as the difference between BASEQ and ZTCFQ
        fprintf('   Creating DELTAQ table...\n');
        DELTAQ = BASEQ;

        % Get numeric columns for difference calculation (excluding Time) - vectorized for performance
        var_names = BASEQ.Properties.VariableNames;
        is_numeric = cellfun(@(x) isnumeric(BASEQ.(x)), var_names);
        is_not_time = ~strcmp(var_names, 'Time');
        numeric_vars = var_names(is_numeric & is_not_time);

        % Calculate differences for numeric columns
        for i = 1:length(numeric_vars)
            var_name = numeric_vars{i};
            if ismember(var_name, ZTCFQ.Properties.VariableNames)
                try
                    DELTAQ.(var_name) = BASEQ.(var_name) - ZTCFQ.(var_name);
                catch ME
                    % Skip if calculation fails (e.g., different data types)
                    fprintf('   Warning: Could not calculate difference for %s: %s\n', var_name, ME.message);
                end
            end
        end

        fprintf('✅ Data tables processed successfully\n');
        fprintf('   BASEQ: %d frames\n', height(BASEQ));
        fprintf('   ZTCFQ: %d frames\n', height(ZTCFQ));
        fprintf('   DELTAQ: %d frames\n', height(DELTAQ));
        fprintf('   Time range: %.3f to %.3f seconds\n', BASEQ.Time(1), BASEQ.Time(end));

    catch ME
        fprintf('❌ Error processing data tables: %s\n', ME.message);
        rethrow(ME);
    end

end

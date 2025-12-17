function save_data_tables(config, BASEQ, ZTCFQ, DELTAQ)
% SAVE_DATA_TABLES - Save processed Q-tables to files
%
% Inputs:
%   config - Configuration structure from model_config()
%   BASEQ - Processed base data table
%   ZTCFQ - Processed ZTCF data table
%   DELTAQ - Processed DELTA data table
%
% This function:
%   1. Creates the Tables directory if it doesn't exist
%   2. Saves BASEQ, ZTCFQ, and DELTAQ to .mat files
%   3. Provides feedback on the save operation

    fprintf('💾 Saving data tables...\n');

    try
        % Create Tables directory if it doesn't exist
        if ~exist(config.tables_path, 'dir')
            fprintf('   Creating Tables directory: %s\n', config.tables_path);
            mkdir(config.tables_path);
        end

        % Save BASEQ
        baseq_file = fullfile(config.tables_path, 'BASEQ.mat');
        fprintf('   Saving BASEQ to: %s\n', baseq_file);
        save(baseq_file, 'BASEQ');

        % Save ZTCFQ
        ztcfq_file = fullfile(config.tables_path, 'ZTCFQ.mat');
        fprintf('   Saving ZTCFQ to: %s\n', ztcfq_file);
        save(ztcfq_file, 'ZTCFQ');

        % Save DELTAQ
        deltaq_file = fullfile(config.tables_path, 'DELTAQ.mat');
        fprintf('   Saving DELTAQ to: %s\n', deltaq_file);
        save(deltaq_file, 'DELTAQ');

        fprintf('✅ Data tables saved successfully\n');
        fprintf('   BASEQ: %s\n', baseq_file);
        fprintf('   ZTCFQ: %s\n', ztcfq_file);
        fprintf('   DELTAQ: %s\n', deltaq_file);

    catch ME
        fprintf('❌ Error saving data tables: %s\n', ME.message);
        rethrow(ME);
    end

end

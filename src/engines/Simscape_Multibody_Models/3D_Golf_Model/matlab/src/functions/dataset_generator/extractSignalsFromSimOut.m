function [data_table, signal_info] = extractSignalsFromSimOut(simOut, options)
    % Extract signals from simulation output based on specified options
    % This replaces the missing extractAllSignalsFromBus function

    data_table = [];
    signal_info = struct();

    try
        % Validate simOut input to prevent brace indexing errors
        if isempty(simOut)
            if options.verbose
                fprintf('Warning: Empty simulation output provided\n');
            end
            return;
        end

        % Check if simOut is a valid simulation output object
        if ~isobject(simOut) && ~isstruct(simOut)
            if options.verbose
                fprintf('Warning: Invalid simulation output type: %s\n', class(simOut));
            end
            return;
        end

        % Initialize data collection
        all_data = {};

        % Extract from CombinedSignalBus if enabled and available
        if options.extract_combined_bus && (isprop(simOut, 'CombinedSignalBus') || isfield(simOut, 'CombinedSignalBus'))
            if options.verbose
                fprintf('Extracting from CombinedSignalBus...\n');
            end

            try
                combinedBus = simOut.CombinedSignalBus;
                if ~isempty(combinedBus)
                    signal_bus_data = extractFromCombinedSignalBus(combinedBus);

                    if ~isempty(signal_bus_data)
                        all_data{end+1} = signal_bus_data;
                        if options.verbose
                            fprintf('CombinedSignalBus: %d columns extracted\n', width(signal_bus_data));
                        end
                    end
                end
            catch ME
                if contains(ME.message, 'brace indexing') || contains(ME.message, 'comma separated list')
                    if options.verbose
                        fprintf('Warning: Brace indexing error accessing CombinedSignalBus: %s\n', ME.message);
                    end
                else
                    if options.verbose
                        fprintf('Warning: Error extracting CombinedSignalBus: %s\n', ME.message);
                    end
                end
            end
        end

        % Extract from logsout if enabled and available
        if options.extract_logsout && (isprop(simOut, 'logsout') || isfield(simOut, 'logsout'))
            if options.verbose
                fprintf('Extracting from logsout...\n');
            end

            try
                logsout_data = extractLogsoutDataFixed(simOut.logsout);
                if ~isempty(logsout_data)
                    all_data{end+1} = logsout_data;
                    if options.verbose
                        fprintf('Logsout: %d columns extracted\n', width(logsout_data));
                    end
                end
            catch ME
                if contains(ME.message, 'brace indexing') || contains(ME.message, 'comma separated list')
                    if options.verbose
                        fprintf('Warning: Brace indexing error accessing logsout: %s\n', ME.message);
                    end
                else
                    if options.verbose
                        fprintf('Warning: Error extracting logsout: %s\n', ME.message);
                    end
                end
            end
        end

        % Extract from Simscape if enabled and available
        if options.extract_simscape
            if options.verbose
                fprintf('Checking for Simscape simlog...\n');
            end

            % Enhanced simlog access for parallel execution
            simlog_available = false;
            simlog_data = [];

            if isprop(simOut, 'simlog') || isfield(simOut, 'simlog')
                try
                    simlog_data = simOut.simlog;
                    if ~isempty(simlog_data)
                        simlog_available = true;
                        if options.verbose
                            fprintf('Found simlog (type: %s)\n', class(simlog_data));
                        end
                    end
                catch ME
                    if contains(ME.message, 'brace indexing') || contains(ME.message, 'comma separated list')
                        if options.verbose
                            fprintf('Warning: Brace indexing error accessing simlog: %s\n', ME.message);
                        end
                    else
                        if options.verbose
                            fprintf('Warning: Could not access simlog: %s\n', ME.message);
                        end
                    end
                end
            end

            % Try alternative access methods for parallel workers
            if ~simlog_available
                try
                    if isprop(simOut, 'SimulationMetadata') && isfield(simOut.SimulationMetadata, 'SimscapeLoggingInfo')
                        if options.verbose
                            fprintf('Attempting alternative simlog access...\n');
                        end
                    end
                catch
                    % Continue
                end
            end

            if simlog_available
                if options.verbose
                    fprintf('Extracting from Simscape simlog...\n');
                end

                simscape_data = extractSimscapeDataRecursive(simlog_data);
                if ~isempty(simscape_data)
                    all_data{end+1} = simscape_data;
                    if options.verbose
                        fprintf('Simscape: %d columns extracted\n', width(simscape_data));
                    end
                else
                    if options.verbose
                        fprintf('Warning: No Simscape data extracted despite simlog being available\n');
                    end
                end
            else
                if options.verbose
                    fprintf('Warning: No simlog found in simulation output\n');
                end
            end
        end

        % Combine all data sources
        if ~isempty(all_data)
            data_table = combineDataSources(all_data);
            signal_info.sources_found = length(all_data);
            signal_info.total_columns = width(data_table);
        else
            if options.verbose
                fprintf('Warning: No data extracted from any source\n');
            end
        end

    catch ME
        if options.verbose
            fprintf('Error in extractSignalsFromSimOut: %s\n', ME.message);
        end
        % Return empty results on error
        data_table = [];
        signal_info = struct();
    end
end

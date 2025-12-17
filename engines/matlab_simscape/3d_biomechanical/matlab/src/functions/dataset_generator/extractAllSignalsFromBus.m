function [signal_table, signal_info] = extractAllSignalsFromBus(simOut, options)
    % EXTRACTALLSIGNALSFROMBUS - Extract all signals from multiple data sources
    %
    % This function extracts all timeseries data from various Simulink output sources
    % including CombinedSignalBus, logsout, and Simscape Results Explorer data.
    %
    % Inputs:
    %   simOut - The Simulink simulation output object
    %   options - Structure with extraction options (optional)
    %     .extract_combined_bus - Boolean, extract from CombinedSignalBus (default: true)
    %     .extract_logsout - Boolean, extract from logsout (default: false)
    %     .extract_simscape - Boolean, extract from Simscape Results Explorer (default: false)
    %     .verbose - Boolean, enable verbose output (default: true)
    %
    % Outputs:
    %   signal_table - Table containing all extracted signals with time column
    %   signal_info  - Structure containing metadata about extracted signals
    %
    % Example:
    %   simOut = sim('YourModel');
    %   options.extract_combined_bus = true;
    %   options.extract_logsout = true;
    %   options.extract_simscape = false;
    %   [data, info] = extractAllSignalsFromBus(simOut, options);

    % Set default options if not provided
    if nargin < 2
        options = struct();
    end

    if ~isfield(options, 'extract_combined_bus')
        options.extract_combined_bus = true;
    end
    if ~isfield(options, 'extract_logsout')
        options.extract_logsout = false;
    end
    if ~isfield(options, 'extract_simscape')
        options.extract_simscape = false;
    end
    if ~isfield(options, 'verbose')
        options.verbose = true;
    end

    signal_table = [];
    signal_info = struct();

    try
        if options.verbose
            fprintf('=== Extracting Signals from Multiple Sources ===\n');
            fprintf('CombinedSignalBus: %s\n', mat2str(options.extract_combined_bus));
            fprintf('Logsout: %s\n', mat2str(options.extract_logsout));
            fprintf('Simscape Results Explorer: %s\n', mat2str(options.extract_simscape));
        end

        % Initialize containers for collecting data
        all_signals = struct();
        time_data = [];
        expected_length = 0;
        source_info = struct();
        source_info.combined_bus = struct('extracted', false, 'signals', 0, 'time_points', 0);
        source_info.logsout = struct('extracted', false, 'signals', 0, 'time_points', 0);
        source_info.simscape = struct('extracted', false, 'signals', 0, 'time_points', 0);

        % Extract from CombinedSignalBus
        if options.extract_combined_bus && isprop(simOut, 'CombinedSignalBus')
            if options.verbose
                fprintf('\n--- Extracting from CombinedSignalBus ---\n');
            end

            [all_signals, time_data, expected_length] = extractFromCombinedBus(...
                simOut.CombinedSignalBus, all_signals, time_data, expected_length, options.verbose);

            source_info.combined_bus.extracted = true;
            source_info.combined_bus.signals = length(fieldnames(all_signals));
            source_info.combined_bus.time_points = expected_length;

            if options.verbose
                fprintf('CombinedSignalBus: Extracted %d signals with %d time points\n', ...
                    source_info.combined_bus.signals, source_info.combined_bus.time_points);
            end
        elseif options.extract_combined_bus
            if options.verbose
                fprintf('Warning: CombinedSignalBus not found in simulation output\n');
            end
        end

        % Extract from logsout
        if options.extract_logsout && isprop(simOut, 'logsout')
            if options.verbose
                fprintf('\n--- Extracting from Logsout ---\n');
            end

            [all_signals, time_data, expected_length] = extractFromLogsout(...
                simOut.logsout, all_signals, time_data, expected_length, options.verbose);

            source_info.logsout.extracted = true;
            source_info.logsout.signals = length(fieldnames(all_signals)) - source_info.combined_bus.signals;
            source_info.logsout.time_points = expected_length;

            if options.verbose
                fprintf('Logsout: Extracted %d signals with %d time points\n', ...
                    source_info.logsout.signals, source_info.logsout.time_points);
            end
        elseif options.extract_logsout
            if options.verbose
                fprintf('Warning: Logsout not found in simulation output\n');
            end
        end

        % Extract from Simscape Results Explorer
        if options.extract_simscape
            if options.verbose
                fprintf('\n--- Extracting from Simscape Results Explorer ---\n');
            end

            [all_signals, time_data, expected_length] = extractFromSimscape(...
                all_signals, time_data, expected_length, options.verbose);

            source_info.simscape.extracted = true;
            source_info.simscape.signals = length(fieldnames(all_signals)) - ...
                source_info.combined_bus.signals - source_info.logsout.signals;
            source_info.simscape.time_points = expected_length;

            if options.verbose
                fprintf('Simscape Results Explorer: Extracted %d signals with %d time points\n', ...
                    source_info.simscape.signals, source_info.simscape.time_points);
            end
        end

        % Check if we have any data
        if isempty(time_data)
            if options.verbose
                fprintf('Error: No time data found in any source\n');
            end
            return;
        end

        total_signals = length(fieldnames(all_signals));
        if options.verbose
            fprintf('\n=== Summary ===\n');
            fprintf('Total signals extracted: %d\n', total_signals);
            fprintf('Time points: %d\n', expected_length);
        end

        % Create table from extracted signals
        if total_signals > 0
            [signal_table, signal_info] = createSignalTable(all_signals, time_data, expected_length, source_info, options.verbose);
        else
            if options.verbose
                fprintf('Warning: No valid signals found in any source\n');
            end
        end

    catch ME
        if options.verbose
            fprintf('Error extracting signals: %s\n', ME.message);
            fprintf('Stack trace:\n');
            for i = 1:length(ME.stack)
                fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
            end
        end
    end
end

function [all_signals, time_data, expected_length] = extractFromCombinedBus(combined_bus, all_signals, time_data, expected_length, verbose)
    % Extract signals from CombinedSignalBus

    if ~isstruct(combined_bus)
        if verbose
            fprintf('Error: CombinedSignalBus is not a struct\n');
        end
        return;
    end

    % Get all field names from the bus
    bus_fields = fieldnames(combined_bus);
    if verbose
        fprintf('Found %d top-level fields in CombinedSignalBus\n', length(bus_fields));
    end

    % Recursively extract all timeseries data
    [all_signals, time_data, expected_length] = extractSignalsRecursive(combined_bus, '', all_signals, time_data, expected_length, verbose);
end

function [all_signals, time_data, expected_length] = extractFromLogsout(logsout, all_signals, time_data, expected_length, verbose)
    % Extract signals from logsout

    if ~isa(logsout, 'Simulink.SimulationData.Dataset')
        if verbose
            fprintf('Warning: Logsout is not a Simulink.SimulationData.Dataset\n');
        end
        return;
    end

    if logsout.numElements == 0
        if verbose
            fprintf('Warning: Logsout Dataset is empty\n');
        end
        return;
    end

    if verbose
        fprintf('Processing logsout Dataset with %d elements\n', logsout.numElements);
    end

    % Get time from first element if not already set
    if isempty(time_data)
        first_element = logsout.getElement(1);
        if isa(first_element, 'Simulink.SimulationData.Signal')
            time_data = first_element.Values.Time;
        elseif isa(first_element, 'timeseries')
            time_data = first_element.Time;
        else
            if verbose
                fprintf('Warning: Cannot extract time from logsout\n');
            end
            return;
        end
        expected_length = length(time_data);
        if verbose
            fprintf('Using time data from logsout (length: %d)\n', expected_length);
        end
    end

    % Process each element in the dataset
    for i = 1:logsout.numElements
        element = logsout.getElement(i);

        if isa(element, 'Simulink.SimulationData.Signal')
            signalName = element.Name;
            if isempty(signalName)
                signalName = sprintf('LogsoutSignal_%d', i);
            end

            % Extract data from Signal object
            data = element.Values.Data;
            signal_time = element.Values.Time;

            % Ensure data matches time length and is valid
            if isnumeric(data) && length(signal_time) == expected_length && ~isempty(data)
                % Handle different data dimensions
                data_size = size(data);

                if length(data_size) == 2
                    % 2D data: [time_points, components] or [components, time_points]
                    if data_size(1) == expected_length
                        % Format: [time_points, components] - extract each component
                        unique_name = makeUniqueSignalName(signalName, fieldnames(all_signals));
                        all_signals.(unique_name) = data;

                        if data_size(2) == 1
                            if verbose
                                fprintf('Extracted 1D logsout signal: %s (length: %d)\n', unique_name, data_size(1));
                            end
                        elseif data_size(2) == 3
                            if verbose
                                fprintf('Extracted 3D vector logsout signal: %s (size: %dx%d)\n', unique_name, data_size(1), data_size(2));
                            end
                        else
                            if verbose
                                fprintf('Extracted multi-dimensional logsout signal: %s (size: %dx%d)\n', unique_name, data_size(1), data_size(2));
                            end
                        end
                    elseif data_size(2) == expected_length
                        % Format: [components, time_points] - transpose to get [time_points, components]
                        data_transposed = data';
                        unique_name = makeUniqueSignalName(signalName, fieldnames(all_signals));
                        all_signals.(unique_name) = data_transposed;

                        if verbose
                            fprintf('Extracted transposed logsout signal: %s (size: %dx%d)\n', unique_name, data_size(2), data_size(1));
                        end
                    else
                        if verbose
                            fprintf('Warning: Logsout signal %s skipped (dimension mismatch: %dx%d vs expected %d)\n', ...
                                signalName, data_size(1), data_size(2), expected_length);
                        end
                    end

                elseif length(data_size) == 3
                    % 3D data: [rows, cols, time_points] - this is for 3x3 matrices
                    if data_size(3) == expected_length
                        % Format: [rows, cols, time_points] - extract each matrix element
                        unique_name = makeUniqueSignalName(signalName, fieldnames(all_signals));
                        all_signals.(unique_name) = data;

                        if data_size(1) == 3 && data_size(2) == 3
                            if verbose
                                fprintf('Extracted 3x3 matrix logsout signal: %s (size: %dx%dx%d) - rotation matrix or inertia tensor\n', ...
                                    unique_name, data_size(1), data_size(2), data_size(3));
                            end
                        else
                            if verbose
                                fprintf('Extracted 3D matrix logsout signal: %s (size: %dx%dx%d)\n', ...
                                    unique_name, data_size(1), data_size(2), data_size(3));
                            end
                        end
                    else
                        if verbose
                            fprintf('Warning: Logsout signal %s skipped (3D dimension mismatch: %dx%dx%d vs expected %d)\n', ...
                                signalName, data_size(1), data_size(2), data_size(3), expected_length);
                        end
                    end

                else
                    if verbose
                        fprintf('Warning: Logsout signal %s skipped (unsupported dimensions: %s)\n', ...
                            signalName, mat2str(data_size));
                    end
                end
            else
                if verbose
                    fprintf('Warning: Logsout signal %s skipped (time length mismatch: %d vs expected %d, or empty data)\n', ...
                        signalName, length(signal_time), expected_length);
                end
            end
        else
            if verbose
                fprintf('Warning: Logsout element %d is not a Signal object (type: %s)\n', i, class(element));
            end
        end
    end
end

function [all_signals, time_data, expected_length] = extractFromSimscape(all_signals, time_data, expected_length, verbose)
    % Extract signals from Simscape Results Explorer

    try
        % Check if there are any Simscape runs
        simscapeRuns = Simulink.sdi.getAllRunIDs;
        if isempty(simscapeRuns)
            if verbose
                fprintf('No Simscape runs found in Results Explorer.\n');
            end
            return;
        end

        if verbose
            fprintf('Found %d Simscape runs in Results Explorer.\n', length(simscapeRuns));
        end

        % Get the most recent run
        latestRun = simscapeRuns(end);
        runObj = Simulink.sdi.getRun(latestRun);

        if verbose
            fprintf('Using latest run: %s (ID: %d)\n', runObj.Name, latestRun);
        end

        % Get all signals from the run
        signals = runObj.getAllSignals;
        if verbose
            fprintf('Total Simscape signals available: %d\n', length(signals));
        end

        % Get time from first signal if not already set
        if isempty(time_data) && length(signals) > 0
            first_signal = signals(1);
            time_data = first_signal.Time;
            expected_length = length(time_data);
            if verbose
                fprintf('Using time data from Simscape (length: %d)\n', expected_length);
            end
        end

        % Process each signal
        for i = 1:length(signals)
            signal = signals(i);
            signalName = signal.Name;

            % Skip if signal name is empty
            if isempty(signalName)
                signalName = sprintf('SimscapeSignal_%d', i);
            end

            % Extract data
            data = signal.Data;
            signal_time = signal.Time;

            % Ensure data matches time length and is valid
            if isnumeric(data) && length(signal_time) == expected_length && ~isempty(data)
                % Handle different data dimensions
                data_size = size(data);

                if length(data_size) == 2
                    % 2D data: [time_points, components] or [components, time_points]
                    if data_size(1) == expected_length
                        % Format: [time_points, components] - extract each component
                        unique_name = makeUniqueSignalName(signalName, fieldnames(all_signals));
                        all_signals.(unique_name) = data;

                        if data_size(2) == 1
                            if verbose
                                fprintf('Extracted 1D Simscape signal: %s (length: %d)\n', unique_name, data_size(1));
                            end
                        elseif data_size(2) == 3
                            if verbose
                                fprintf('Extracted 3D vector Simscape signal: %s (size: %dx%d)\n', unique_name, data_size(1), data_size(2));
                            end
                        else
                            if verbose
                                fprintf('Extracted multi-dimensional Simscape signal: %s (size: %dx%d)\n', unique_name, data_size(1), data_size(2));
                            end
                        end
                    elseif data_size(2) == expected_length
                        % Format: [components, time_points] - transpose to get [time_points, components]
                        data_transposed = data';
                        unique_name = makeUniqueSignalName(signalName, fieldnames(all_signals));
                        all_signals.(unique_name) = data_transposed;

                        if verbose
                            fprintf('Extracted transposed Simscape signal: %s (size: %dx%d)\n', unique_name, data_size(2), data_size(1));
                        end
                    else
                        if verbose
                            fprintf('Warning: Simscape signal %s skipped (dimension mismatch: %dx%d vs expected %d)\n', ...
                                signalName, data_size(1), data_size(2), expected_length);
                        end
                    end

                elseif length(data_size) == 3
                    % 3D data: [rows, cols, time_points] - this is for 3x3 matrices
                    if data_size(3) == expected_length
                        % Format: [rows, cols, time_points] - extract each matrix element
                        unique_name = makeUniqueSignalName(signalName, fieldnames(all_signals));
                        all_signals.(unique_name) = data;

                        if data_size(1) == 3 && data_size(2) == 3
                            if verbose
                                fprintf('Extracted 3x3 matrix Simscape signal: %s (size: %dx%dx%d) - rotation matrix or inertia tensor\n', ...
                                    unique_name, data_size(1), data_size(2), data_size(3));
                            end
                        else
                            if verbose
                                fprintf('Extracted 3D matrix Simscape signal: %s (size: %dx%dx%d)\n', ...
                                    unique_name, data_size(1), data_size(2), data_size(3));
                            end
                        end
                    else
                        if verbose
                            fprintf('Warning: Simscape signal %s skipped (3D dimension mismatch: %dx%dx%d vs expected %d)\n', ...
                                signalName, data_size(1), data_size(2), data_size(3), expected_length);
                        end
                    end

                else
                    if verbose
                        fprintf('Warning: Simscape signal %s skipped (unsupported dimensions: %s)\n', ...
                            signalName, mat2str(data_size));
                    end
                end
            else
                if verbose
                    fprintf('Warning: Simscape signal %s skipped (time length mismatch: %d vs expected %d, or empty data)\n', ...
                        signalName, length(signal_time), expected_length);
                end
            end
        end

    catch ME
        if verbose
            fprintf('Error accessing Simscape Results Explorer: %s\n', ME.message);
        end
    end
end

function [signal_table, signal_info] = createSignalTable(all_signals, time_data, expected_length, source_info, verbose)
    % Create the final signal table from all extracted signals

    % Pre-allocate with conservative estimate (performance optimization)
    % Assuming ~10 components per signal on average (some 3D matrices expand to 9 columns)
    signal_names = fieldnames(all_signals);
    max_possible_signals = (length(signal_names) * 10) + 1;
    table_data_pre = cell(max_possible_signals, 1);
    table_names_pre = cell(max_possible_signals, 1);

    % Initialize with time
    table_data_pre{1} = time_data;
    table_names_pre{1} = 'time';
    data_idx = 1;

    % Add all signal data
    valid_signals = 0;

    if verbose
        fprintf('Processing %d extracted signals for table creation...\n', length(signal_names));
    end

    for i = 1:length(signal_names)
        signal_name = signal_names{i};
        signal_data = all_signals.(signal_name);

        % Ensure data is a column vector and has correct length
        if isnumeric(signal_data) && ~isempty(signal_data)
            data_size = size(signal_data);

            if length(data_size) == 2
                % 2D data: [time_points, components]
                if data_size(1) == expected_length
                    % Check if this is 1D data
                    if data_size(2) == 1
                        % Single column data
                        signal_data = double(signal_data(:));

                        % Check for any issues with the data
                        if ~any(isnan(signal_data)) && ~any(isinf(signal_data))
                            data_idx = data_idx + 1;
                            table_data_pre{data_idx} = signal_data;
                            table_names_pre{data_idx} = signal_name;
                            valid_signals = valid_signals + 1;
                        else
                            if verbose
                                fprintf('Warning: Skipping %s (contains NaN or Inf values)\n', signal_name);
                            end
                        end
                    else
                        % Multi-column data, extract each column
                        for col = 1:data_size(2)
                            col_data = signal_data(:, col);
                            if length(col_data) == expected_length
                                % Create unique column name
                                base_name = sprintf('%s_%d', signal_name, col);
                                unique_name = makeUniqueSignalName(base_name, table_names_pre(1:data_idx));

                                % Ensure data is a double column vector
                                col_data = double(col_data(:));

                                % Check for any issues with the data
                                if ~any(isnan(col_data)) && ~any(isinf(col_data))
                                    data_idx = data_idx + 1;
                                    table_data_pre{data_idx} = col_data;
                                    table_names_pre{data_idx} = unique_name;
                                    valid_signals = valid_signals + 1;
                                    if verbose
                                        fprintf('Extracted multi-dimensional signal: %s (column %d)\n', signal_name, col);
                                    end
                                else
                                    if verbose
                                        fprintf('Warning: Skipping %s column %d (contains NaN or Inf values)\n', signal_name, col);
                                    end
                                end
                            end
                        end
                    end
                else
                    if verbose
                        fprintf('Warning: Skipping %s (dimension mismatch: %dx%d vs expected %d)\n', ...
                            signal_name, data_size(1), data_size(2), expected_length);
                    end
                end

            elseif length(data_size) == 3
                % 3D data: [rows, cols, time_points] - 3x3 matrices
                if data_size(3) == expected_length
                    % Extract each matrix element as a separate time series
                    for row = 1:data_size(1)
                        for col = 1:data_size(2)
                            % Extract the time series for this matrix element
                            element_data = squeeze(signal_data(row, col, :));

                            if length(element_data) == expected_length
                                % Create unique column name for this matrix element
                                base_name = sprintf('%s_%d_%d', signal_name, row, col);
                                unique_name = makeUniqueSignalName(base_name, table_names_pre(1:data_idx));

                                % Ensure data is a double column vector
                                element_data = double(element_data(:));

                                % Check for any issues with the data
                                if ~any(isnan(element_data)) && ~any(isinf(element_data))
                                    data_idx = data_idx + 1;
                                    table_data_pre{data_idx} = element_data;
                                    table_names_pre{data_idx} = unique_name;
                                    valid_signals = valid_signals + 1;
                                    if verbose
                                        fprintf('Extracted 3x3 matrix element: %s (row %d, col %d)\n', signal_name, row, col);
                                    end
                                else
                                    if verbose
                                        fprintf('Warning: Skipping %s element [%d,%d] (contains NaN or Inf values)\n', signal_name, row, col);
                                    end
                                end
                            end
                        end
                    end
                else
                    if verbose
                        fprintf('Warning: Skipping %s (3D dimension mismatch: %dx%dx%d vs expected %d)\n', ...
                            signal_name, data_size(1), data_size(2), data_size(3), expected_length);
                    end
                end

            else
                if verbose
                    fprintf('Warning: Skipping %s (unsupported dimensions: %s)\n', ...
                        signal_name, mat2str(data_size));
                end
            end
        end
    end

    if verbose
        fprintf('Valid signals for table: %d\n', valid_signals);
    end

    % Trim pre-allocated arrays to actual size (performance optimization)
    table_data = table_data_pre(1:data_idx);
    table_names = table_names_pre(1:data_idx);

    % Create the table only if we have valid data
    if length(table_data) > 1
        % Verify all vectors have the same length and are valid
        lengths = cellfun(@length, table_data);

        % Additional validation
        all_valid = true;
        for i = 1:length(table_data)
            if ~isnumeric(table_data{i}) || isempty(table_data{i})
                if verbose
                    fprintf('Error: Vector %d (%s) is not numeric or empty\n', i, table_names{i});
                end
                all_valid = false;
            elseif any(isnan(table_data{i})) || any(isinf(table_data{i}))
                if verbose
                    fprintf('Error: Vector %d (%s) contains NaN or Inf values\n', i, table_names{i});
                end
                all_valid = false;
            end
        end

        if all_valid && all(lengths == expected_length)
            if verbose
                fprintf('All vectors have correct length and are valid. Creating table...\n');
            end

            % Try to create table with explicit data types
            try
                signal_table = table(table_data{:}, 'VariableNames', table_names);

                % Create signal info structure
                signal_info.total_signals = valid_signals;
                signal_info.time_points = expected_length;
                signal_info.signal_names = table_names(2:end); % Exclude 'time'
                signal_info.source_info = source_info;
                signal_info.extraction_time = datetime('now');

                if verbose
                    fprintf('Successfully created table with %d columns (%d signals + time)\n', ...
                        width(signal_table), valid_signals);
                end
            catch table_error
                if verbose
                    fprintf('Error creating table: %s\n', table_error.message);
                end

                % Try alternative approach - create table step by step
                if verbose
                    fprintf('Trying alternative table creation method...\n');
                end
                try
                    % Create empty table first
                    signal_table = table();

                    % Add columns one by one
                    for i = 1:length(table_data)
                        signal_table.(table_names{i}) = table_data{i};
                    end

                    if verbose
                        fprintf('Successfully created table with alternative method (%d columns)\n', width(signal_table));
                    end
                catch alt_error
                    if verbose
                        fprintf('Alternative method also failed: %s\n', alt_error.message);
                    end
                end
            end
        else
            if verbose
                fprintf('Error: Vector validation failed. Expected: %d, Actual lengths: ', expected_length);
                fprintf('%d ', lengths);
                fprintf('\n');
            end
        end
    else
        if verbose
            fprintf('Warning: No valid signals found in any source\n');
        end
    end
end

function [all_signals, time_data, expected_length] = extractSignalsRecursive(obj, prefix, all_signals, time_data, expected_length, verbose)
    % Recursively extract all timeseries data from nested structures

    if isstruct(obj)
        fields = fieldnames(obj);

        for i = 1:length(fields)
            field_name = fields{i};
            field_value = obj.(field_name);

            % Create the full path name
            if isempty(prefix)
                full_name = field_name;
            else
                full_name = sprintf('%s_%s', prefix, field_name);
            end

            % Recursively process this field
            [all_signals, time_data, expected_length] = extractSignalsRecursive(...
                field_value, full_name, all_signals, time_data, expected_length, verbose);
        end

    elseif isa(obj, 'timeseries')
        % Extract data from timeseries
        data = obj.Data;
        time = obj.Time;

        if isnumeric(data) && ~isempty(data)
            % If this is the first timeseries, use its time data
            if isempty(time_data)
                time_data = time;
                expected_length = length(time);
                if verbose
                    fprintf('Using time data from %s (length: %d)\n', prefix, expected_length);
                end
            end

            % Handle different data dimensions
            data_size = size(data);

            if length(data_size) == 2
                % 2D data: [time_points, components] or [components, time_points]
                if data_size(1) == expected_length
                    % Format: [time_points, components] - extract each component
                    unique_name = makeUniqueSignalName(prefix, fieldnames(all_signals));
                    all_signals.(unique_name) = data;

                    if data_size(2) == 1
                        if verbose
                            fprintf('Extracted 1D signal: %s (length: %d)\n', unique_name, data_size(1));
                        end
                    elseif data_size(2) == 3
                        if verbose
                            fprintf('Extracted 3D vector signal: %s (size: %dx%d)\n', unique_name, data_size(1), data_size(2));
                        end
                    else
                        if verbose
                            fprintf('Extracted multi-dimensional signal: %s (size: %dx%d)\n', unique_name, data_size(1), data_size(2));
                        end
                    end
                elseif data_size(2) == expected_length
                    % Format: [components, time_points] - transpose to get [time_points, components]
                    data_transposed = data';
                    unique_name = makeUniqueSignalName(prefix, fieldnames(all_signals));
                    all_signals.(unique_name) = data_transposed;

                    if data_size(1) == 1
                        if verbose
                            fprintf('Extracted 1D signal: %s (length: %d, transposed)\n', unique_name, data_size(2));
                        end
                    elseif data_size(1) == 3
                        if verbose
                            fprintf('Extracted 3D vector signal: %s (size: %dx%d, transposed)\n', unique_name, data_size(2), data_size(1));
                        end
                    else
                        if verbose
                            fprintf('Extracted multi-dimensional signal: %s (size: %dx%d, transposed)\n', unique_name, data_size(2), data_size(1));
                        end
                    end
                else
                    if verbose
                        fprintf('Warning: Signal %s skipped (2D dimension mismatch: %dx%d vs expected %d)\n', ...
                            prefix, data_size(1), data_size(2), expected_length);
                    end
                end

            elseif length(data_size) == 3
                % 3D data: [rows, cols, time_points] - this is for 3x3 matrices
                if data_size(3) == expected_length
                    % Format: [rows, cols, time_points] - extract each matrix element
                    unique_name = makeUniqueSignalName(prefix, fieldnames(all_signals));
                    all_signals.(unique_name) = data;

                    if data_size(1) == 3 && data_size(2) == 3
                        if verbose
                            fprintf('Extracted 3x3 matrix signal: %s (size: %dx%dx%d) - rotation matrix or inertia tensor\n', ...
                                unique_name, data_size(1), data_size(2), data_size(3));
                        end
                    else
                        if verbose
                            fprintf('Extracted 3D matrix signal: %s (size: %dx%dx%d)\n', ...
                                unique_name, data_size(1), data_size(2), data_size(3));
                        end
                    end
                else
                    if verbose
                        fprintf('Warning: Signal %s skipped (3D dimension mismatch: %dx%dx%d vs expected %d)\n', ...
                            prefix, data_size(1), data_size(2), data_size(3), expected_length);
                    end
                end

            else
                if verbose
                    fprintf('Warning: Signal %s skipped (unsupported dimensions: %s)\n', ...
                        prefix, mat2str(data_size));
                end
            end
        end

    elseif isnumeric(obj) && ~isempty(obj)
        % Handle direct numeric data (not timeseries)
        if isempty(time_data)
            % If no time data yet, assume this might be time data
            if length(obj) > 100 && all(diff(obj) > 0)
                time_data = obj;
                expected_length = length(obj);
                if verbose
                    fprintf('Using numeric time data from %s (length: %d)\n', prefix, expected_length);
                end
            end
        end

        % Check if this numeric data matches expected length
        if ~isempty(time_data) && length(obj) == expected_length
            unique_name = makeUniqueSignalName(prefix, fieldnames(all_signals));
            all_signals.(unique_name) = obj;
            if verbose
                fprintf('Extracted numeric signal: %s (length: %d)\n', unique_name, length(obj));
            end
        end
    end
end

function unique_name = makeUniqueSignalName(base_name, existing_names)
    % Create a unique signal name, avoiding MATLAB reserved words and duplicates

    % Replace invalid characters
    unique_name = regexprep(base_name, '[^a-zA-Z0-9_]', '_');

    % Ensure it starts with a letter
    if ~isempty(unique_name) && ~isletter(unique_name(1))
        unique_name = ['Signal_' unique_name];
    end

    % Handle empty names
    if isempty(unique_name)
        unique_name = 'UnnamedSignal';
    end

    % Make unique by adding counter if needed
    counter = 1;
    original_name = unique_name;

    while ismember(unique_name, existing_names)
        unique_name = sprintf('%s_%d', original_name, counter);
        counter = counter + 1;
    end
end

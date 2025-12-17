% ENHANCED: Extract from Simscape with detailed diagnostics
function simscape_data = extractSimscapeDataRecursive(simlog)
simscape_data = table();  % Empty table if no data

try
    % DETAILED DIAGNOSTICS
    fprintf('=== SIMSCAPE DIAGNOSTIC START ===\n');

    if isempty(simlog)
        fprintf('❌ simlog is EMPTY\n');
        fprintf('=== SIMSCAPE DIAGNOSTIC END ===\n');
        return;
    end

    fprintf('✅ simlog exists, class: %s\n', class(simlog));

    if ~isa(simlog, 'simscape.logging.Node')
        fprintf('❌ simlog is not a simscape.logging.Node\n');
        fprintf('=== SIMSCAPE DIAGNOSTIC END ===\n');
        return;
    end

    fprintf('✅ simlog is valid simscape.logging.Node\n');

    % Try to inspect the simlog structure
    try
        fprintf(' Inspecting simlog properties...\n');
        props = properties(simlog);
        fprintf('   Properties: %s\n', strjoin(props, ', '));
    catch
        fprintf('❌ Could not get simlog properties\n');
    end

    % Try to get children (properties ARE the children in Multibody)
    try
        children_ids = simlog.children();
        fprintf('✅ Found %d top-level children: %s\n', length(children_ids), strjoin(children_ids, ', '));
    catch ME
        fprintf('❌ Could not get children method: %s\n', ME.message);
        fprintf(' Using properties as children (Multibody approach)\n');

        % Get properties excluding system properties (vectorized for performance)
        all_props = properties(simlog);
        children_ids = all_props(~ismember(all_props, {'id', 'savable', 'exportable'}));
        fprintf('✅ Found %d children from properties: %s\n', length(children_ids), strjoin(children_ids, ', '));
    end

    % Try to inspect first child
    if ~isempty(children_ids)
        try
            first_child_id = children_ids{1};
            first_child = simlog.(first_child_id);
            fprintf(' First child (%s) class: %s\n', first_child_id, class(first_child));

            % Try to get series from first child
            try
                series_children = first_child.series.children();
                fprintf('✅ First child has %d series: %s\n', length(series_children), strjoin(series_children, ', '));
            catch ME2
                fprintf('❌ First child series access failed: %s\n', ME2.message);
            end

        catch ME
            fprintf('❌ Could not inspect first child: %s\n', ME.message);
        end
    end

    fprintf('=== SIMSCAPE DIAGNOSTIC END ===\n');

    % Recursively collect all series data using primary traversal method
    [time_data, all_signals] = traverseSimlogNode(simlog, '');

    if isempty(time_data) || isempty(all_signals)
        fprintf('❌ Primary extraction method failed. No usable Simscape data found.\n');
        return;
    else
        fprintf('✅ Primary method found data!\n');
    end

    % Build table - pre-allocate for performance
    expected_length = length(time_data);
    num_signals = length(all_signals);
    data_cells = cell(num_signals + 1, 1);  % Pre-allocate (+1 for time)
    var_names = cell(num_signals + 1, 1);

    % Initialize with time
    data_cells{1} = time_data;
    var_names{1} = 'time';
    cell_idx = 1;

    for i = 1:num_signals
        signal = all_signals{i};
        signal_data = signal.data;
        data_size = size(signal_data);
        num_elements = numel(signal_data);

        if length(signal_data) == expected_length
            % Standard time series data
            cell_idx = cell_idx + 1;
            data_cells{cell_idx} = signal_data(:);
            var_names{cell_idx} = signal.name;
            fprintf('Debug: Added Simscape signal: %s (length: %d)\n', signal.name, expected_length);
        else
            fprintf('Debug: Skipped %s (size [%s] not supported - need time series, [3 1 N], or [3 3 N])\n', ...
                signal.name, num2str(data_size));
        end
    end

    % Trim to actual size
    data_cells = data_cells(1:cell_idx);
    var_names = var_names(1:cell_idx);

    if length(data_cells) > 1
        simscape_data = table(data_cells{:}, 'VariableNames', var_names);
        fprintf('Debug: Created Simscape table with %d columns, %d rows.\n', width(simscape_data), height(simscape_data));
    else
        fprintf('Debug: Only time data found in Simscape log.\n');
    end

catch ME
    fprintf('Error extracting Simscape data recursively: %s\n', ME.message);
end
end

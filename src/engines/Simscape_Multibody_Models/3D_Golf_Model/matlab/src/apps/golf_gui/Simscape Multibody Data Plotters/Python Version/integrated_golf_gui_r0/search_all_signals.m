%% Search for All Signals with "Signal" in the Name
% This script searches for any signals containing "Signal" in their names
% to identify the duplicate signals mentioned in the warnings

clear; clc;

fprintf('=== Searching for All Signals with "Signal" in Name ===\n\n');

%% Load the model
model_name = 'GolfSwing3D_Kinetic';

% Check if model exists
if ~exist([model_name '.slx'], 'file')
    fprintf('❌ Model %s.slx not found!\n', model_name);
    return;
end

% Load the model
if ~bdIsLoaded(model_name)
    load_system(model_name);
    fprintf('✅ Model %s loaded successfully\n', model_name);
else
    fprintf('✅ Model %s is already loaded\n', model_name);
end

%% Find all Bus Creator blocks and search for signal names
fprintf('\n--- Searching All Bus Creator Blocks for Signal Names ---\n');

bus_creators = find_system(model_name, 'FindAll', 'on', 'BlockType', 'BusCreator');
fprintf('Found %d Bus Creator blocks\n', length(bus_creators));

% Track all signal names across all Bus Creators
all_signal_names = {};
all_signal_locations = {};

% Analyze each Bus Creator
for bc_idx = 1:length(bus_creators)
    bus_creator = bus_creators(bc_idx);
    block_path = get(bus_creator, 'Parent');
    block_name = get(bus_creator, 'Name');

    fprintf('\n--- Bus Creator %d: %s at %s ---\n', bc_idx, block_name, block_path);

    try
        % Get input ports
        in_ports = get(bus_creator, 'PortHandles');

        if ~isfield(in_ports, 'Inport')
            fprintf('  No input ports found\n');
            continue;
        end

        num_inputs = length(in_ports.Inport);
        fprintf('  Number of inputs: %d\n', num_inputs);

        % Get signal names for each input port
        signal_names = cell(num_inputs, 1);

        for i = 1:num_inputs
            try
                % Get the line connected to this input port
                line_handle = get(in_ports.Inport(i), 'Line');

                if line_handle > 0
                    % Get the signal name
                    signal_name = get(line_handle, 'Name');
                    if isempty(signal_name)
                        % Try to get name from the source block
                        src_block = get(line_handle, 'SrcBlockHandle');
                        if src_block > 0
                            signal_name = get(src_block, 'Name');
                        else
                            signal_name = sprintf('Unnamed_Signal_%d', i);
                        end
                    end
                else
                    signal_name = sprintf('Unconnected_Port_%d', i);
                end

                signal_names{i} = signal_name;

                % Check if signal name contains "Signal" (case insensitive)
                if contains(lower(signal_name), 'signal')
                    fprintf('  Port %2d: %s (contains "signal")\n', i, signal_name);

                    % Store for later analysis
                    all_signal_names{end+1} = signal_name;
                    all_signal_locations{end+1} = sprintf('%s/%s:Port%d', block_path, block_name, i);
                end

            catch ME
                signal_names{i} = sprintf('Error_Port_%d', i);
                fprintf('  Port %2d: Error - %s\n', i, ME.message);
            end
        end

        % Check for duplicates within this Bus Creator
        [unique_names, ~, name_indices] = unique(signal_names);
        name_counts = accumarray(name_indices, 1);
        duplicate_names = unique_names(name_counts > 1);

        if ~isempty(duplicate_names)
            fprintf('  ⚠️  Duplicate signal names found in this Bus Creator:\n');
            for i = 1:length(duplicate_names)
                dup_name = duplicate_names{i};
                dup_ports = find(strcmp(signal_names, dup_name));
                fprintf('    "%s" at ports: %s\n', dup_name, mat2str(dup_ports));

                % Check if this includes ports 15, 17, 18
                problematic_ports = intersect(dup_ports, [15, 17, 18]);
                if ~isempty(problematic_ports)
                    fprintf('      ⚠️  This includes the problematic ports: %s\n', mat2str(problematic_ports));
                end
            end
        end

    catch ME
        fprintf('  ❌ Error analyzing this Bus Creator: %s\n', ME.message);
    end
end

%% Analyze all signal names across all Bus Creators
fprintf('\n--- Analysis of All Signal Names ---\n');

if ~isempty(all_signal_names)
    fprintf('Found %d signals with "signal" in the name:\n', length(all_signal_names));

    % Find unique signal names and their counts
    [unique_names, ~, name_indices] = unique(all_signal_names);
    name_counts = accumarray(name_indices, 1);

    for i = 1:length(unique_names)
        signal_name = unique_names{i};
        count = name_counts(i);

        if count > 1
            fprintf('  ❌ "%s" appears %d times:\n', signal_name, count);

            % Find all locations where this signal appears
            signal_locations = all_signal_locations(strcmp(all_signal_names, signal_name));
            for j = 1:length(signal_locations)
                fprintf('    - %s\n', signal_locations{j});
            end
        else
            fprintf('  ✅ "%s" appears once at: %s\n', signal_name, all_signal_locations{strcmp(all_signal_names, signal_name)});
        end
    end
else
    fprintf('No signals with "signal" in the name found\n');
end

%% Search for any signals that might be the duplicates mentioned in warnings
fprintf('\n--- Searching for Potential Duplicate Signals ---\n');

% Look for any signals that appear multiple times
all_unique_signals = unique(all_signal_names);
duplicate_signals = {};

for i = 1:length(all_unique_signals)
    signal_name = all_unique_signals{i};
    count = sum(strcmp(all_signal_names, signal_name));

    if count > 1
        duplicate_signals{end+1} = signal_name;
        fprintf('❌ Duplicate signal: "%s" appears %d times\n', signal_name, count);
    end
end

if isempty(duplicate_signals)
    fprintf('✅ No duplicate signals found across all Bus Creators\n');
    fprintf('\nNote: The warnings might be from a different model version or configuration.\n');
    fprintf('The current model appears to have unique signal names.\n');
end

%% Close the model
if bdIsLoaded(model_name)
    close_system(model_name, 0);
    fprintf('\n✅ Model closed\n');
end

fprintf('\n=== Analysis Complete ===\n');

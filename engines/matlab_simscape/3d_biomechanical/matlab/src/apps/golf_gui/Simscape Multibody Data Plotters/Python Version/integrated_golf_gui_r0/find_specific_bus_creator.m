%% Find Specific Bus Creator with Duplicate SignalBus Names
% This script searches for the specific Bus Creator mentioned in the warnings
% that has duplicate "SignalBus" names at ports 15, 17, 18

clear; clc;

fprintf('=== Finding Specific Bus Creator with Duplicate SignalBus Names ===\n\n');

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

%% Find all Bus Creator blocks and analyze them
fprintf('\n--- Searching All Bus Creator Blocks ---\n');

bus_creators = find_system(model_name, 'FindAll', 'on', 'BlockType', 'BusCreator');
fprintf('Found %d Bus Creator blocks\n', length(bus_creators));

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

                % Only print if it's "SignalBus" or if we're looking at ports 15, 17, 18
                if strcmp(signal_name, 'SignalBus') || ismember(i, [15, 17, 18])
                    fprintf('  Port %2d: %s\n', i, signal_name);
                end

            catch ME
                signal_names{i} = sprintf('Error_Port_%d', i);
                fprintf('  Port %2d: Error - %s\n', i, ME.message);
            end
        end

        % Check for "SignalBus" duplicates
        signalbus_ports = find(strcmp(signal_names, 'SignalBus'));

        if ~isempty(signalbus_ports)
            fprintf('  ❌ Found "SignalBus" at ports: %s\n', mat2str(signalbus_ports));

            % Check if this includes the problematic ports 15, 17, 18
            problematic_ports = intersect(signalbus_ports, [15, 17, 18]);
            if ~isempty(problematic_ports)
                fprintf('  ⚠️  This includes the problematic ports: %s\n', mat2str(problematic_ports));
                fprintf('  🎯 FOUND THE TARGET BUS CREATOR!\n');

                % Show all SignalBus ports
                fprintf('  All SignalBus ports: %s\n', mat2str(signalbus_ports));

                % Provide specific fix recommendations
                fprintf('\n  --- Fix Recommendations for this Bus Creator ---\n');
                fprintf('  To fix the duplicate "SignalBus" names:\n');

                for j = 1:length(signalbus_ports)
                    port_num = signalbus_ports(j);
                    suggested_name = sprintf('SignalBus_%d', port_num);
                    fprintf('    Port %d: Rename "SignalBus" to "%s"\n', port_num, suggested_name);
                end

                fprintf('\n  Steps to fix:\n');
                fprintf('  1. Open the Simulink model\n');
                fprintf('  2. Navigate to: %s\n', block_path);
                fprintf('  3. Find the Bus Creator block: %s\n', block_name);
                fprintf('  4. For each "SignalBus" signal:\n');
                fprintf('     - Right-click on the signal line\n');
                fprintf('     - Select "Signal Properties"\n');
                fprintf('     - Change the signal name from "SignalBus" to a unique name\n');
                fprintf('     - Suggested names: SignalBus_1, SignalBus_2, etc.\n');
                fprintf('  5. Save the model\n');
                fprintf('  6. Re-run your simulation\n');
            end
        else
            fprintf('  ✅ No "SignalBus" signals found\n');
        end

        % Also check for any other duplicates
        [unique_names, ~, name_indices] = unique(signal_names);
        name_counts = accumarray(name_indices, 1);
        duplicate_names = unique_names(name_counts > 1);

        if ~isempty(duplicate_names)
            fprintf('  ⚠️  Other duplicate signal names found:\n');
            for i = 1:length(duplicate_names)
                dup_name = duplicate_names{i};
                if ~strcmp(dup_name, 'SignalBus')  % Don't repeat SignalBus info
                    dup_ports = find(strcmp(signal_names, dup_name));
                    fprintf('    "%s" at ports: %s\n', dup_name, mat2str(dup_ports));
                end
            end
        end

    catch ME
        fprintf('  ❌ Error analyzing this Bus Creator: %s\n', ME.message);
    end
end

%% Close the model
if bdIsLoaded(model_name)
    close_system(model_name, 0);
    fprintf('\n✅ Model closed\n');
end

fprintf('\n=== Analysis Complete ===\n');

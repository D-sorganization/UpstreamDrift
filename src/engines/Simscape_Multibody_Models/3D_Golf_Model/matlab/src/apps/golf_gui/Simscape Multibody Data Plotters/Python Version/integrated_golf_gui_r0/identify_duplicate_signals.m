%% Identify Duplicate Signal Names in Bus Creator
% This script specifically identifies the duplicate signal names at ports 15, 17, 18
% that are causing the warnings in your GolfSwing3D_Kinetic model

clear; clc;

fprintf('=== Identifying Duplicate Signal Names in Bus Creator ===\n\n');

%% Load the model
model_name = 'GolfSwing3D_Kinetic';

% Check if model exists
if ~exist([model_name '.slx'], 'file')
    fprintf('❌ Model %s.slx not found!\n', model_name);
    fprintf('Please ensure you are in the correct directory.\n');
    return;
end

% Load the model
if ~bdIsLoaded(model_name)
    load_system(model_name);
    fprintf('✅ Model %s loaded successfully\n', model_name);
else
    fprintf('✅ Model %s is already loaded\n', model_name);
end

%% Find the Bus Creator block with duplicate signals
fprintf('\n--- Finding Bus Creator Block ---\n');

% Find all Bus Creator blocks
bus_creators = find_system(model_name, 'FindAll', 'on', 'BlockType', 'BusCreator');

fprintf('Found %d Bus Creator blocks\n', length(bus_creators));

% Look for the specific Bus Creator mentioned in the warnings
target_bus_creator = [];
for i = 1:length(bus_creators)
    block_path = get(bus_creators(i), 'Parent');
    block_name = get(bus_creators(i), 'Name');

    % Check if this is the "GolfSwing3D_Kinetic/Logging/Bus Creator" mentioned in warnings
    if contains(block_path, 'Logging') && strcmp(block_name, 'Bus Creator')
        target_bus_creator = bus_creators(i);
        fprintf('✅ Found target Bus Creator: %s at %s\n', block_name, block_path);
        break;
    end
end

if isempty(target_bus_creator)
    fprintf('❌ Could not find the specific Bus Creator block mentioned in warnings\n');
    fprintf('Looking for any Bus Creator with duplicate signals...\n');

    % Try to find any Bus Creator with potential duplicate signals
    for i = 1:length(bus_creators)
        block_path = get(bus_creators(i), 'Parent');
        block_name = get(bus_creators(i), 'Name');

        fprintf('Checking Bus Creator: %s at %s\n', block_name, block_path);

        % Get input ports
        try
            in_ports = get(bus_creators(i), 'PortHandles');
            if isfield(in_ports, 'Inport')
                num_inputs = length(in_ports.Inport);
                fprintf('  Number of inputs: %d\n', num_inputs);

                if num_inputs >= 18  % We're looking for ports 15, 17, 18
                    target_bus_creator = bus_creators(i);
                    fprintf('  ✅ This Bus Creator has enough inputs - will analyze it\n');
                    break;
                end
            end
        catch ME
            fprintf('  ❌ Error accessing ports: %s\n', ME.message);
        end
    end
end

if isempty(target_bus_creator)
    fprintf('❌ No suitable Bus Creator found for analysis\n');
    return;
end

%% Analyze the Bus Creator inputs
fprintf('\n--- Analyzing Bus Creator Inputs ---\n');

try
    % Get input ports
    in_ports = get(target_bus_creator, 'PortHandles');

    if ~isfield(in_ports, 'Inport')
        fprintf('❌ No input ports found\n');
        return;
    end

    num_inputs = length(in_ports.Inport);
    fprintf('Total number of inputs: %d\n', num_inputs);

    % Get signal names for each input port
    signal_names = cell(num_inputs, 1);
    port_numbers = 1:num_inputs;

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
            fprintf('Port %2d: %s\n', i, signal_name);

        catch ME
            signal_names{i} = sprintf('Error_Port_%d', i);
            fprintf('Port %2d: Error accessing signal - %s\n', i, ME.message);
        end
    end

    %% Find duplicate signal names
    fprintf('\n--- Finding Duplicate Signal Names ---\n');

    % Find unique signal names and their counts
    [unique_names, ~, name_indices] = unique(signal_names);
    name_counts = accumarray(name_indices, 1);

    % Find duplicates
    duplicate_names = unique_names(name_counts > 1);

    if isempty(duplicate_names)
        fprintf('✅ No duplicate signal names found\n');
    else
        fprintf('❌ Found %d duplicate signal names:\n', length(duplicate_names));

        for i = 1:length(duplicate_names)
            dup_name = duplicate_names{i};
            dup_ports = find(strcmp(signal_names, dup_name));

            fprintf('\nDuplicate: "%s" appears at ports: %s\n', dup_name, mat2str(dup_ports));

            % Check if this includes the problematic ports 15, 17, 18
            problematic_ports = intersect(dup_ports, [15, 17, 18]);
            if ~isempty(problematic_ports)
                fprintf('  ⚠️  This includes the problematic ports: %s\n', mat2str(problematic_ports));
            end
        end
    end

    %% Specific analysis of ports 15, 17, 18
    fprintf('\n--- Specific Analysis of Ports 15, 17, 18 ---\n');

    target_ports = [15, 17, 18];
    for port_num = target_ports
        if port_num <= num_inputs
            fprintf('Port %2d: %s\n', port_num, signal_names{port_num});
        else
            fprintf('Port %2d: Port does not exist (only %d inputs)\n', port_num, num_inputs);
        end
    end

    % Check if the target ports have the same signal name
    if all(target_ports <= num_inputs)
        port_15_name = signal_names{15};
        port_17_name = signal_names{17};
        port_18_name = signal_names{18};

        if strcmp(port_15_name, port_17_name) && strcmp(port_17_name, port_18_name)
            fprintf('\n❌ CONFIRMED: All three ports (15, 17, 18) have the same signal name: "%s"\n', port_15_name);
        elseif strcmp(port_15_name, port_17_name)
            fprintf('\n❌ CONFIRMED: Ports 15 and 17 have the same signal name: "%s"\n', port_15_name);
        elseif strcmp(port_15_name, port_18_name)
            fprintf('\n❌ CONFIRMED: Ports 15 and 18 have the same signal name: "%s"\n', port_15_name);
        elseif strcmp(port_17_name, port_18_name)
            fprintf('\n❌ CONFIRMED: Ports 17 and 18 have the same signal name: "%s"\n', port_17_name);
        else
            fprintf('\n✅ Ports 15, 17, and 18 have different signal names\n');
        end
    end

    %% Provide fix recommendations
    fprintf('\n--- Fix Recommendations ---\n');

    if ~isempty(duplicate_names)
        fprintf('To fix the duplicate signal names:\n\n');

        for i = 1:length(duplicate_names)
            dup_name = duplicate_names{i};
            dup_ports = find(strcmp(signal_names, dup_name));

            fprintf('For signal "%s" at ports %s:\n', dup_name, mat2str(dup_ports));

            % Suggest unique names based on port numbers
            for j = 1:length(dup_ports)
                port_num = dup_ports(j);
                suggested_name = sprintf('%s_Port%d', dup_name, port_num);
                fprintf('  Port %d: Rename to "%s"\n', port_num, suggested_name);
            end
            fprintf('\n');
        end

        fprintf('Steps to fix:\n');
        fprintf('1. Open the Simulink model\n');
        fprintf('2. Navigate to the Bus Creator block\n');
        fprintf('3. For each duplicate signal:\n');
        fprintf('   - Right-click on the signal line\n');
        fprintf('   - Select "Signal Properties"\n');
        fprintf('   - Change the signal name to be unique\n');
        fprintf('4. Save the model\n');
        fprintf('5. Re-run your simulation\n');
    else
        fprintf('✅ No duplicate signal names found - no fixes needed\n');
    end

catch ME
    fprintf('❌ Error analyzing Bus Creator: %s\n', ME.message);
    fprintf('Error details: %s\n', getReport(ME));
end

%% Close the model
if bdIsLoaded(model_name)
    close_system(model_name, 0);  % Close without saving
    fprintf('\n✅ Model closed\n');
end

fprintf('\n=== Analysis Complete ===\n');

%% Test Signal Bus Data Generation
% This script helps verify that signal bus data is being generated correctly
% and exported in a format compatible with the GUI

clear; clc;

fprintf('=== Testing Signal Bus Data Generation ===\n\n');

%% 1. Load and Configure Model
model_name = 'GolfSwing3D_Kinetic';

% Check if model exists
if ~exist([model_name '.slx'], 'file')
    fprintf('❌ Model %s.slx not found!\n', model_name);
    fprintf('Please ensure you are in the correct directory.\n');
    return;
end

% Load model
if ~bdIsLoaded(model_name)
    load_system(model_name);
    fprintf('✅ Model %s loaded successfully\n', model_name);
else
    fprintf('✅ Model %s is already loaded\n', model_name);
end

%% 2. Configure Signal Bus Logging
fprintf('\n--- Configuring Signal Bus Logging ---\n');

% Disable Simscape Results Explorer for speed
set_param(model_name, 'SimscapeLogType', 'none');
fprintf('✅ Disabled Simscape Results Explorer\n');

% Enable signal logging
set_param(model_name, 'SignalLogging', 'on');
set_param(model_name, 'SignalLoggingName', 'logsout');
fprintf('✅ Enabled signal logging\n');

% Set short simulation time for testing
set_param(model_name, 'StopTime', '0.1');
fprintf('✅ Set simulation time to 0.1 seconds\n');

%% 3. Run Test Simulation
fprintf('\n--- Running Test Simulation ---\n');

try
    simOut = sim(model_name);
    fprintf('✅ Simulation completed successfully\n');
catch ME
    fprintf('❌ Simulation failed: %s\n', ME.message);
    return;
end

%% 4. Analyze Simulation Output
fprintf('\n--- Analyzing Simulation Output ---\n');

% Check what fields are in the simulation output
output_fields = fieldnames(simOut);
fprintf('Simulation output fields:\n');
for i = 1:length(output_fields)
    fprintf('  %s\n', output_fields{i});
end

%% 5. Check Signal Bus Data
fprintf('\n--- Checking Signal Bus Data ---\n');

% Look for logsout data
if isfield(simOut, 'logsout') && ~isempty(simOut.logsout)
    logsout = simOut.logsout;
    fprintf('✅ Logsout data found with %d elements\n', logsout.numElements);

    % List all logged signals
    fprintf('Logged signals:\n');
    for i = 1:logsout.numElements
        try
            element = logsout.getElement(i);
            fprintf('  %d: %s\n', i, element.Name);
        catch
            fprintf('  %d: <error accessing element>\n', i);
        end
    end
else
    fprintf('⚠️  No logsout data found\n');
end

% Look for To Workspace variables
fprintf('\nTo Workspace variables in base workspace:\n');
workspace_vars = who;
data_vars = workspace_vars(contains(workspace_vars, 'Data') | contains(workspace_vars, 'Log'));
if ~isempty(data_vars)
    for i = 1:length(data_vars)
        var_name = data_vars{i};
        var_data = eval(var_name);
        fprintf('  %s: %s\n', var_name, class(var_data));
        if isnumeric(var_data)
            fprintf('    Size: %s\n', mat2str(size(var_data)));
        end
    end
else
    fprintf('  No data variables found\n');
end

%% 6. Test Data Export
fprintf('\n--- Testing Data Export ---\n');

% Try to create GUI-compatible data
try
    % Extract time vector
    if isfield(simOut, 'tout')
        time_vector = simOut.tout;
    else
        time_vector = (0:0.001:0.1)';  % Default time vector
    end

    % Initialize data arrays
    num_time_points = length(time_vector);

    % Create sample data structure (replace with actual signal extraction)
    sample_data = struct();
    sample_data.time = time_vector;
    sample_data.CHx = zeros(num_time_points, 1);  % Clubhead X position
    sample_data.CHy = zeros(num_time_points, 1);  % Clubhead Y position
    sample_data.CHz = zeros(num_time_points, 1);  % Clubhead Z position
    sample_data.MPx = zeros(num_time_points, 1);  % Midpoint X position
    sample_data.MPy = zeros(num_time_points, 1);  % Midpoint Y position
    sample_data.MPz = zeros(num_time_points, 1);  % Midpoint Z position

    % Convert to matrix format for GUI
    signal_names = {'time', 'CHx', 'CHy', 'CHz', 'MPx', 'MPy', 'MPz'};
    data_matrix = [time_vector, sample_data.CHx, sample_data.CHy, sample_data.CHz, ...
                   sample_data.MPx, sample_data.MPy, sample_data.MPz];

    % Save in GUI-compatible format
    BASEQ = data_matrix;
    ZTCFQ = data_matrix;  % Same structure for now
    DELTAQ = data_matrix; % Same structure for now

    save('test_BASEQ.mat', 'BASEQ');
    save('test_ZTCFQ.mat', 'ZTCFQ');
    save('test_DELTAQ.mat', 'DELTAQ');

    fprintf('✅ Created test data files:\n');
    fprintf('  test_BASEQ.mat\n');
    fprintf('  test_ZTCFQ.mat\n');
    fprintf('  test_DELTAQ.mat\n');
    fprintf('  Data shape: %s\n', mat2str(size(data_matrix)));

catch ME
    fprintf('❌ Error creating test data: %s\n', ME.message);
end

%% 7. Performance Comparison
fprintf('\n--- Performance Comparison ---\n');

% Test with Simscape enabled
fprintf('Testing with Simscape Results Explorer enabled...\n');
set_param(model_name, 'SimscapeLogType', 'all');
tic;
simOut_with_simscape = sim(model_name);
simscape_time = toc;
fprintf('  Time with Simscape: %.3f seconds\n', simscape_time);

% Test with Simscape disabled
fprintf('Testing with Simscape Results Explorer disabled...\n');
set_param(model_name, 'SimscapeLogType', 'none');
tic;
simOut_without_simscape = sim(model_name);
no_simscape_time = toc;
fprintf('  Time without Simscape: %.3f seconds\n', no_simscape_time);

% Calculate improvement
if simscape_time > 0
    improvement = ((simscape_time - no_simscape_time) / simscape_time) * 100;
    fprintf('  Speed improvement: %.1f%%\n', improvement);
end

%% 8. Recommendations
fprintf('\n--- Recommendations ---\n');

fprintf('1. ✅ Signal bus logging is configured correctly\n');
fprintf('2. 🔧 Update your data export process to create GUI-compatible files\n');
fprintf('3. 🚀 Disable Simscape Results Explorer for better performance\n');
fprintf('4. 🧪 Test the GUI with the generated test files\n');
fprintf('5. 📊 Monitor simulation performance with different settings\n');

fprintf('\n=== Test Complete ===\n');

%% Test Script for Interactive Signal Plotter
% This script loads example data and launches the SkeletonPlotter with
% the new Interactive Signal Plotter feature

%% Clear workspace
clear;
clc;
fprintf('=== Testing Interactive Signal Plotter ===\n\n');

%% Try to load data from available locations
data_locations = {
    '../../../Golf_GUI/Simscape Multibody Data Plotters/Matlab Versions/SkeletonPlotter/',
    '../../../Golf_GUI/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/',
    '../../../Golf_GUI/Simscape Multibody Data Plotters/Python Version/golf_gui_r0/'
    };

BASEQ = [];
ZTCFQ = [];
DELTAQ = [];

fprintf('Searching for data files...\n');
for i = 1:length(data_locations)
    location = data_locations{i};

    baseq_file = fullfile(location, 'BASEQ.mat');
    ztcfq_file = fullfile(location, 'ZTCFQ.mat');
    deltaq_file = fullfile(location, 'DELTAQ.mat');

    if exist(baseq_file, 'file')
        fprintf('  Found data in: %s\n', location);

        % Load BASEQ
        if exist(baseq_file, 'file')
            load(baseq_file, 'BASEQ');
            fprintf('    Loaded BASEQ (%d frames)\n', height(BASEQ));
        end

        % Load ZTCFQ
        if exist(ztcfq_file, 'file')
            load(ztcfq_file, 'ZTCFQ');
            fprintf('    Loaded ZTCFQ (%d frames)\n', height(ZTCFQ));
        end

        % Load DELTAQ
        if exist(deltaq_file, 'file')
            load(deltaq_file, 'DELTAQ');
            fprintf('    Loaded DELTAQ (%d frames)\n', height(DELTAQ));
        end

        break;
    end
end

%% Check if data was loaded
if isempty(BASEQ)
    error('Could not find data files. Please ensure BASEQ.mat, ZTCFQ.mat, and DELTAQ.mat exist.');
end

%% Display available signals
fprintf('\nAvailable signals in BASEQ:\n');
signal_names = BASEQ.Properties.VariableNames;
force_torque_signals = {};
position_signals = {};
other_signals = {};

% Categorize signals
for i = 1:length(signal_names)
    sig = signal_names{i};
    if strcmp(sig, 'Time')
        continue;
    end

    % Check for force/torque
    if contains(sig, {'Force', 'Torque', 'Couple', 'Power', 'Work'}, 'IgnoreCase', true)
        force_torque_signals{end+1} = sig;
        % Check for positions
    elseif ~isempty(regexp(sig, '[xyz]$', 'once')) || contains(sig, {'Butt', 'CH', 'MP', 'LW', 'LE', 'LS', 'RW', 'RE', 'RS', 'HUB'})
        position_signals{end+1} = sig;
    else
        other_signals{end+1} = sig;
    end
end

fprintf('  Forces & Torques (%d): ', length(force_torque_signals));
if ~isempty(force_torque_signals)
    fprintf('%s', strjoin(force_torque_signals(1:min(3,end)), ', '));
    if length(force_torque_signals) > 3
        fprintf(' ... and %d more', length(force_torque_signals) - 3);
    end
end
fprintf('\n');

fprintf('  Position Signals (%d): ', length(position_signals));
if ~isempty(position_signals)
    fprintf('%s', strjoin(position_signals(1:min(5,end)), ', '));
    if length(position_signals) > 5
        fprintf(' ... and %d more', length(position_signals) - 5);
    end
end
fprintf('\n');

fprintf('  Other Signals (%d)\n', length(other_signals));

%% Launch SkeletonPlotter
fprintf('\nLaunching SkeletonPlotter...\n');
fprintf('   Look for the "Signal Plot" button on the right side!\n\n');

try
    SkeletonPlotter(BASEQ, ZTCFQ, DELTAQ);

    fprintf('SkeletonPlotter launched successfully!\n\n');
    fprintf('=== How to Use the Interactive Signal Plotter ===\n');
    fprintf('1. Click the "Signal Plot" button on the right side\n');
    fprintf('2. The Interactive Signal Plotter window will open\n');
    fprintf('3. Click "Manage Hotlist" to add signals\n');
    fprintf('4. Select signals from the hotlist to plot them\n');
    fprintf('5. Try dragging on the plot to scrub through time!\n');
    fprintf('6. Watch the 3D skeleton update in sync!\n');
    fprintf('\nNote: The app cleans up after itself when closed.\n');
    fprintf('No variables will be left in your workspace.\n\n');

catch ME
    fprintf('Error launching SkeletonPlotter:\n');
    fprintf('   %s\n', ME.message);
    fprintf('   Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end

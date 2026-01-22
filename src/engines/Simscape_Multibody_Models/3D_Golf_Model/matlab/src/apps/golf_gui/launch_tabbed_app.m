%% LAUNCH_TABBED_APP - Quick launcher for the Integrated Golf Analysis App
%
% This script launches the new tabbed version of the Golf Analysis Application
%
% Usage:
%   >> launch_tabbed_app

%% First, close any stuck figures
fprintf('Checking for stuck figures...\n');
all_figs = findall(0, 'Type', 'figure');
if ~isempty(all_figs)
    fprintf('Found %d open figure(s). Closing them...\n', length(all_figs));
    close all force;
    pause(0.5);
    fprintf('Figures closed.\n\n');
else
    fprintf('No stuck figures found.\n\n');
end

%% Add paths
fprintf('Setting up paths...\n');
script_path = fileparts(mfilename('fullpath'));
app_path = fullfile(script_path, 'Integrated_Analysis_App');
viz_path = fullfile(script_path, '2D GUI', 'visualization');

addpath(genpath(app_path));
addpath(viz_path);
fprintf('Paths configured.\n\n');

%% Launch the tabbed application
fprintf('======================================================\n');
fprintf('Launching Integrated Golf Analysis Application\n');
fprintf('======================================================\n\n');

try
    app_handles = main_golf_analysis_app();
    fprintf('\n✓ Application launched successfully!\n');
    fprintf('\nYou now have three tabs:\n');
    fprintf('  • Tab 1: Model Setup (placeholder)\n');
    fprintf('  • Tab 2: ZTCF Calculation (placeholder)\n');
    fprintf('  • Tab 3: Visualization (FUNCTIONAL)\n\n');
    fprintf('To use Tab 3:\n');
    fprintf('  1. Click "Load from File..."\n');
    fprintf('  2. Select a MAT file with BASEQ, ZTCFQ, DELTAQ\n');
    fprintf('  3. Click "Launch Skeleton Plotter"\n\n');
    fprintf('Or run test_tabbed_app for a full test.\n');
    fprintf('======================================================\n');
catch ME
    fprintf('\n✗ Failed to launch application:\n');
    fprintf('   %s\n\n', ME.message);
    fprintf('Stack trace:\n');
    disp(ME.stack);
end

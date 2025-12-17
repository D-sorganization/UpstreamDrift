%% TEST_TABBED_APP - Test script for the Integrated Golf Analysis Application
%
% This script tests the tabbed GUI framework and verifies that:
%   1. The application launches correctly
%   2. All three tabs are created and accessible
%   3. Data passing mechanisms work
%   4. Basic UI interactions function properly
%
% Usage:
%   Run this script from the MATLAB command window:
%   >> test_tabbed_app

%% Setup
clear; clc;
fprintf('==========================================================\n');
fprintf('Testing Integrated Golf Analysis Application\n');
fprintf('==========================================================\n\n');

% Add path to the application directory
app_path = fileparts(mfilename('fullpath'));
addpath(genpath(app_path));

% Also need path to visualization tools
viz_path = fullfile(app_path, '..', '2D GUI', 'visualization');
if exist(viz_path, 'dir')
    addpath(viz_path);
else
    warning('Visualization path not found: %s', viz_path);
end

%% Test 1: Launch Application
fprintf('Test 1: Launching application...\n');
try
    app_handles = main_golf_analysis_app();
    fprintf('  ✓ Application launched successfully\n');
    fprintf('  ✓ Main figure created: %s\n', get(app_handles.main_fig, 'Name'));
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
    return;
end

pause(1); % Allow UI to render

%% Test 2: Verify Tab Structure
fprintf('\nTest 2: Verifying tab structure...\n');
try
    assert(isfield(app_handles, 'tabs'), 'tabs field missing');
    assert(isfield(app_handles.tabs, 'tab1'), 'tab1 missing');
    assert(isfield(app_handles.tabs, 'tab2'), 'tab2 missing');
    assert(isfield(app_handles.tabs, 'tab3'), 'tab3 missing');
    fprintf('  ✓ All three tabs exist\n');

    % Check tab titles
    tab1_title = get(app_handles.tabs.tab1, 'Title');
    tab2_title = get(app_handles.tabs.tab2, 'Title');
    tab3_title = get(app_handles.tabs.tab3, 'Title');
    fprintf('  ✓ Tab 1: %s\n', tab1_title);
    fprintf('  ✓ Tab 2: %s\n', tab2_title);
    fprintf('  ✓ Tab 3: %s\n', tab3_title);
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
end

%% Test 3: Verify Managers
fprintf('\nTest 3: Verifying data and config managers...\n');
try
    assert(isfield(app_handles, 'data_manager'), 'data_manager missing');
    assert(isfield(app_handles, 'config_manager'), 'config_manager missing');
    fprintf('  ✓ Data manager exists\n');
    fprintf('  ✓ Config manager exists\n');

    % Test data manager
    data_info = app_handles.data_manager.get_data_info();
    fprintf('  ✓ Data manager operational\n');
    fprintf('    - Has simulation data: %d\n', data_info.has_simulation);
    fprintf('    - Has ZTCF data: %d\n', data_info.has_ztcf);
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
end

%% Test 4: Test Tab Navigation
fprintf('\nTest 4: Testing tab navigation...\n');
try
    % Switch to each tab
    app_handles.tab_group.SelectedTab = app_handles.tabs.tab1;
    pause(0.5);
    fprintf('  ✓ Switched to Tab 1\n');

    app_handles.tab_group.SelectedTab = app_handles.tabs.tab2;
    pause(0.5);
    fprintf('  ✓ Switched to Tab 2\n');

    app_handles.tab_group.SelectedTab = app_handles.tabs.tab3;
    pause(0.5);
    fprintf('  ✓ Switched to Tab 3\n');
catch ME
    fprintf('  ✗ FAILED: %s\n', ME.message);
end

%% Test 5: Test Data Passing (Optional - requires data)
fprintf('\nTest 5: Testing data passing mechanisms...\n');
try
    % Try to load test data if available
    test_data_file = fullfile(app_path, '..', '2D GUI', 'visualization', ...
        'test_data.mat');

    if exist(test_data_file, 'file')
        fprintf('  Loading test data from: %s\n', test_data_file);
        test_data = load(test_data_file);

        % Store in data manager
        if isfield(test_data, 'datasets')
            app_handles.data_manager.set_ztcf_data(test_data.datasets);
            fprintf('  ✓ Test data stored in data manager\n');

            % Retrieve it
            retrieved_data = app_handles.data_manager.get_ztcf_data();
            assert(~isempty(retrieved_data), 'Failed to retrieve data');
            fprintf('  ✓ Test data retrieved successfully\n');

            % Switch to Tab 3 and try to use it
            app_handles.tab_group.SelectedTab = app_handles.tabs.tab3;
            pause(1);
            fprintf('  ✓ Tab 3 should now have access to test data\n');
        else
            fprintf('  ⚠ Test data file has unexpected format\n');
        end
    else
        fprintf('  ⚠ No test data file found (optional test)\n');
        fprintf('    Expected location: %s\n', test_data_file);
    end
catch ME
    fprintf('  ⚠ Data passing test failed (optional): %s\n', ME.message);
end

%% Summary
fprintf('\n==========================================================\n');
fprintf('Testing Complete!\n');
fprintf('==========================================================\n');
fprintf('\nThe application is running. You can:\n');
fprintf('  1. Navigate between tabs\n');
fprintf('  2. Try loading data in Tab 3\n');
fprintf('  3. Test the File menu options\n');
fprintf('  4. Close the application when done\n\n');
fprintf('Note: Tab 1 and Tab 2 are placeholders and will be\n');
fprintf('      implemented in future phases.\n\n');

function tab_handles = tab3_visualization(parent_tab, app_handles)
% TAB3_VISUALIZATION - Initialize Visualization Tab
%
% This tab launches the full SkeletonPlotter in a managed window.
% Auto-loads default data files on startup.
%
% Inputs:
%   parent_tab   - Handle to the tab UI container
%   app_handles  - Main application handles structure
%
% Returns:
%   tab_handles  - Handles structure for this tab

%% Initialize Tab Handles
tab_handles = struct();
tab_handles.parent = parent_tab;
tab_handles.skeleton_plotter_fig = [];
tab_handles.datasets = [];
tab_handles.data_loaded = false;

%% Default Data Files Path
default_data_path = fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'Simscape Multibody Data Plotters', 'Matlab Versions', 'SkeletonPlotter');

%% Add visualization path
viz_path = fullfile(fileparts(mfilename('fullpath')), ...
    '..', '2D GUI', 'visualization');
if ~contains(path, viz_path)
    addpath(viz_path);
end

%% Create UI Layout - Embedded Visualization Container
% Create full-size visualization panel (will contain embedded SkeletonPlotter)
tab_handles.viz_panel = uipanel('Parent', parent_tab, ...
    'Units', 'normalized', ...
    'Position', [0, 0, 1, 1], ...
    'BorderType', 'none', ...
    'BackgroundColor', [0.9, 1, 0.9]);

% Store panels and data path
tab_handles.default_data_path = default_data_path;
tab_handles.plotter_loaded = false;

%% Set up Refresh and Cleanup Callbacks
tab_handles.refresh_callback = @() refresh_tab3();
tab_handles.cleanup_callback = @() cleanup_tab3(tab_handles);

%% Auto-launch with default data on startup (embedded mode)
fprintf('Tab 3: Embedding visualization with default data...\n');
pause(0.5); % Brief pause to let UI render
embed_visualization_with_defaults(app_handles, tab_handles, default_data_path);

end

%% Callback Functions

function embed_visualization_with_defaults(~, tab_handles, default_data_path)
% Embed SkeletonPlotter directly in tab with default data

try
    % Check if already loaded
    if tab_handles.plotter_loaded
        fprintf('Visualization already embedded.\n');
        return;
    end

    % Load default data
    baseq_file = fullfile(default_data_path, 'BASEQ.mat');
    ztcfq_file = fullfile(default_data_path, 'ZTCFQ.mat');
    deltaq_file = fullfile(default_data_path, 'DELTAQ.mat');

    if ~exist(baseq_file, 'file')
        errordlg(sprintf('Default data not found at:\n%s', default_data_path), 'Data Not Found');
        return;
    end

    fprintf('Loading default data for embedded visualization...\n');
    BASEQ_data = load(baseq_file);
    ZTCFQ_data = load(ztcfq_file);
    DELTAQ_data = load(deltaq_file);

    % Extract tables
    BASEQ = extract_table(BASEQ_data, 'BASEQ');
    ZTCFQ = extract_table(ZTCFQ_data, 'ZTCFQ');
    DELTAQ = extract_table(DELTAQ_data, 'DELTAQ');

    % Embed SkeletonPlotter in the viz_panel
    fprintf('Embedding SkeletonPlotter in Tab 3...\n');
    SkeletonPlotter(BASEQ, ZTCFQ, DELTAQ, tab_handles.viz_panel);

    tab_handles.plotter_loaded = true;
    num_frames = height(BASEQ);
    fprintf('✓ SkeletonPlotter embedded successfully (%d frames)\n', num_frames);

catch ME
    errordlg(sprintf('Failed to embed visualization: %s', ME.message), 'Embedding Error');
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
end

function table_data = extract_table(loaded_data, expected_name)
% Helper to extract table from loaded structure

if istable(loaded_data)
    table_data = loaded_data;
elseif isstruct(loaded_data)
    if isfield(loaded_data, expected_name)
        table_data = loaded_data.(expected_name);
    else
        fields = fieldnames(loaded_data);
        table_data = loaded_data.(fields{1});
    end
else
    error('Unexpected data format');
end
end

function refresh_tab3()
% Refresh callback (placeholder)
end

function cleanup_tab3(tab_handles)
% Cleanup when closing

fprintf('Tab 3: Cleaning up...\n');

% Clear embedded visualization if loaded
if isfield(tab_handles, 'viz_panel') && ishandle(tab_handles.viz_panel)
    try
        % Delete all children of the viz panel
        delete(findobj(tab_handles.viz_panel, '-depth', 1));
    catch
        % Children may already be deleted
    end
end

fprintf('Tab 3: Cleanup complete\n');
end

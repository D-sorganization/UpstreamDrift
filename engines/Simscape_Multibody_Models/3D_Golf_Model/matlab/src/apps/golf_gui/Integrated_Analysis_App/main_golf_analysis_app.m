function app_handles = main_golf_analysis_app()
    % MAIN_GOLF_ANALYSIS_APP Integrated Golf Swing Analysis Application
    %
    % This is the main entry point for the integrated golf analysis application
    % featuring three tabs:
    %   Tab 1: Model Setup & Simulation
    %   Tab 2: ZTCF Calculation
    %   Tab 3: Analysis & Visualization
    %
    % Outputs:
    %   app_handles - Structure containing all application handles and managers
    %                 with fields:
    %                 .main_fig - Main figure handle
    %                 .tab_group - Tab group handle
    %                 .tabs - Structure with tab handles
    %                 .data_manager - Data manager instance
    %                 .config_manager - Configuration manager instance
    %                 .config - Configuration structure
    %
    % Usage:
    %   app_handles = main_golf_analysis_app();
    %
    % See also: tab1_model_setup, tab2_ztcf_calculation, tab3_visualization

    fprintf('Initializing Golf Swing Analysis Application...\n');

%% Initialize Managers
% Configuration manager
cfg_mgr = config_manager();
config = cfg_mgr.load_config();

% Create main figure
main_fig = create_main_figure(config);

% Data manager (for data passing between tabs)
data_mgr = data_manager(main_fig);

%% Create Tab Group
tab_group = uitabgroup('Parent', main_fig, ...
    'Units', 'normalized', ...
    'Position', [0, 0, 1, 1]);

% Tab 1: Model Setup & Simulation
tab1 = uitab(tab_group, 'Title', 'Model Setup', ...
    'Tag', 'tab1');

% Tab 2: ZTCF Calculation
tab2 = uitab(tab_group, 'Title', 'ZTCF Calculation', ...
    'Tag', 'tab2');

% Tab 3: Analysis & Visualization
tab3 = uitab(tab_group, 'Title', 'Visualization', ...
    'Tag', 'tab3');

%% Initialize Application Handles Structure
app_handles = struct();
app_handles.main_fig = main_fig;
app_handles.tab_group = tab_group;
app_handles.tabs = struct('tab1', tab1, 'tab2', tab2, 'tab3', tab3);
app_handles.data_manager = data_mgr;
app_handles.config_manager = cfg_mgr;
app_handles.config = config;

% Store handles in figure
guidata(main_fig, app_handles);

%% Initialize Each Tab
fprintf('  Initializing Tab 1: Model Setup...\n');
app_handles.tab1_handles = tab1_model_setup(tab1, app_handles);

fprintf('  Initializing Tab 2: ZTCF Calculation...\n');
app_handles.tab2_handles = tab2_ztcf_calculation(tab2, app_handles);

fprintf('  Initializing Tab 3: Visualization...\n');
app_handles.tab3_handles = tab3_visualization(tab3, app_handles);

% Update handles
guidata(main_fig, app_handles);

%% Set Up Tab Change Callback
set(tab_group, 'SelectionChangedFcn', @(src, event) on_tab_changed(src, event, app_handles));

%% Apply Saved Window State
cfg_mgr.apply_window_state(config, main_fig);

% Set active tab from config
if isfield(config.window, 'last_active_tab')
    try
        tab_group.SelectedTab = tab_group.Children(config.window.last_active_tab);
    catch
        % Invalid tab index, ignore
    end
end

%% Set Close Request Function
set(main_fig, 'CloseRequestFcn', @(src, event) on_close_request(src, event, app_handles));

fprintf('Golf Swing Analysis Application initialized successfully!\n');
fprintf('--------------------------------------------------------------\n');
fprintf('Ready to use. Start with Tab 1 (Model Setup) or Tab 3 (Visualization)\n');

end

%% Helper Functions

function main_fig = create_main_figure(config)
% Create the main application figure

% Get position from config or use default
if isfield(config, 'window') && isfield(config.window, 'position')
    pos = config.window.position;
else
    pos = [100, 100, 1400, 800];
end

main_fig = figure('Name', 'Golf Swing Analysis - Integrated Application', ...
    'NumberTitle', 'off', ...
    'MenuBar', 'figure', ...
    'ToolBar', 'figure', ...
    'Resize', 'on', ...
    'Position', pos, ...
    'Color', [0.94, 0.94, 0.94], ...
    'Tag', 'GolfAnalysisApp');

% Create custom menu
create_app_menu(main_fig);
end

function create_app_menu(main_fig)
% Create application menu bar

% File menu
file_menu = uimenu(main_fig, 'Label', 'File');
uimenu(file_menu, 'Label', 'Load Session...', ...
    'Callback', @on_load_session);
uimenu(file_menu, 'Label', 'Save Session...', ...
    'Callback', @on_save_session);
uimenu(file_menu, 'Label', 'Save Session As...', ...
    'Callback', @on_save_session_as);
uimenu(file_menu, 'Label', 'Exit', ...
    'Separator', 'on', ...
    'Callback', @(src, event) close(main_fig));

% Tools menu
tools_menu = uimenu(main_fig, 'Label', 'Tools');
uimenu(tools_menu, 'Label', 'Clear All Data', ...
    'Callback', @on_clear_data);
uimenu(tools_menu, 'Label', 'Reset Configuration', ...
    'Separator', 'on', ...
    'Callback', @on_reset_config);

% Help menu
help_menu = uimenu(main_fig, 'Label', 'Help');
uimenu(help_menu, 'Label', 'About', ...
    'Callback', @on_about);
uimenu(help_menu, 'Label', 'Documentation', ...
    'Callback', @on_documentation);
end

function on_tab_changed(~, event, app_handles)
% Callback when tab is changed

% Get selected tab
selected_tab = event.NewValue;
tab_title = get(selected_tab, 'Title');

fprintf('Switched to: %s\n', tab_title);

% Update tab enable states based on available data
update_tab_states(app_handles);
end

function update_tab_states(~)
% Enable/disable tabs based on available data

% Tab 2 requires simulation data from Tab 1 (or can load independently)
% For now, always enable all tabs (users can load data independently)

% Future: Implement smart enabling based on workflow
% if ~app_handles.data_manager.has_simulation_data()
%     app_handles.tabs.tab2.Enable = 'off';
% end
end

function on_load_session(src, ~)
% Load session callback
fig = ancestor(src, 'figure');
app_handles = guidata(fig);

[file, path] = uigetfile('*.mat', 'Load Session');
if file ~= 0
    try
        fullpath = fullfile(path, file);
        app_handles.data_manager.load_session(fullpath);

        % Refresh all tabs with new data
        refresh_all_tabs(app_handles);

        msgbox('Session loaded successfully!', 'Load Session', 'help');
    catch ME
        errordlg(sprintf('Failed to load session: %s', ME.message), ...
            'Load Error');
    end
end
end

function on_save_session(src, ~)
% Save session callback (to default location)
fig = ancestor(src, 'figure');
app_handles = guidata(fig);

% Default session filename with timestamp
timestamp = datetime('now', 'Format', "yyyyMMdd'T'HHmmss");
default_name = sprintf('golf_session_%s.mat', string(timestamp));

try
    app_handles.data_manager.save_session(default_name);
    msgbox(sprintf('Session saved to: %s', default_name), ...
        'Save Session', 'help');
catch ME
    errordlg(sprintf('Failed to save session: %s', ME.message), ...
        'Save Error');
end
end

function on_save_session_as(src, ~)
% Save session as callback (choose location)
fig = ancestor(src, 'figure');
app_handles = guidata(fig);

timestamp = datetime('now', 'Format', "yyyyMMdd'T'HHmmss");
default_name = sprintf('golf_session_%s.mat', string(timestamp));

[file, path] = uiputfile('*.mat', 'Save Session As', default_name);
if file ~= 0
    try
        fullpath = fullfile(path, file);
        app_handles.data_manager.save_session(fullpath);
        msgbox('Session saved successfully!', 'Save Session', 'help');
    catch ME
        errordlg(sprintf('Failed to save session: %s', ME.message), ...
            'Save Error');
    end
end
end

function on_clear_data(src, ~)
% Clear all data callback
fig = ancestor(src, 'figure');
app_handles = guidata(fig);

answer = questdlg('Clear all data from memory?', ...
    'Clear Data', 'Yes', 'No', 'No');

if strcmp(answer, 'Yes')
    app_handles.data_manager.clear_all_data();
    fprintf('All data cleared from memory\n');

    % Refresh all tabs
    refresh_all_tabs(app_handles);
end
end

function on_reset_config(src, ~)
% Reset configuration callback
fig = ancestor(src, 'figure');
app_handles = guidata(fig);

answer = questdlg('Reset configuration to defaults?', ...
    'Reset Configuration', 'Yes', 'No', 'No');

if strcmp(answer, 'Yes')
    app_handles.config_manager.reset_config();
    msgbox('Configuration reset. Restart application to apply.', ...
        'Reset Configuration', 'help');
end
end

function on_about(~, ~)
% About dialog
msg = sprintf(['Golf Swing Analysis Application\n\n', ...
    'Version: 1.0\n\n', ...
    'Integrated application for golf swing modeling,\n', ...
    'ZTCF analysis, and visualization.\n\n', ...
    'Developed for biomechanics research and analysis.']);
msgbox(msg, 'About', 'help');
end

function on_documentation(~, ~)
% Open documentation
doc_file = fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', '..', '..', 'docs', 'TABBED_GUI_IMPLEMENTATION_PLAN.md');

if exist(doc_file, 'file')
    open(doc_file);
else
    msgbox('Documentation not found', 'Documentation', 'warn');
end
end

function refresh_all_tabs(app_handles)
% Refresh all tabs with current data

% Refresh Tab 1
if isfield(app_handles, 'tab1_handles') && ...
        isfield(app_handles.tab1_handles, 'refresh_callback')
    app_handles.tab1_handles.refresh_callback();
end

% Refresh Tab 2
if isfield(app_handles, 'tab2_handles') && ...
        isfield(app_handles.tab2_handles, 'refresh_callback')
    app_handles.tab2_handles.refresh_callback();
end

% Refresh Tab 3
if isfield(app_handles, 'tab3_handles') && ...
        isfield(app_handles.tab3_handles, 'refresh_callback')
    app_handles.tab3_handles.refresh_callback();
end
end

function on_close_request(src, ~, app_handles)
% Handle application close request

fprintf('Closing Golf Swing Analysis Application...\n');

try
    % Ask user if they want to save session
    if isfield(app_handles, 'config') && isfield(app_handles.config, 'general') && ...
            isfield(app_handles.config.general, 'confirm_on_exit') && ...
            app_handles.config.general.confirm_on_exit

        answer = questdlg('Save session before exiting?', ...
            'Exit Application', 'Yes', 'No', 'Cancel', 'Yes');

        if strcmp(answer, 'Cancel')
            fprintf('Close cancelled by user.\n');
            return;  % Don't close
        elseif strcmp(answer, 'Yes')
            try
                on_save_session(app_handles.main_fig, []);
            catch ME
                warning('%s: %s', ME.identifier, ME.message);
            end
        end
    end

    % Save window state
    try
        if isfield(app_handles, 'config_manager') && isfield(app_handles, 'config')
            app_handles.config_manager.update_window_state(app_handles.config, src);
            app_handles.config_manager.save_config(app_handles.config);
        end
    catch ME
        warning('%s: %s', ME.identifier, ME.message);
    end

    % Clean up each tab
    try
        if isfield(app_handles, 'tab1_handles') && ...
                isfield(app_handles.tab1_handles, 'cleanup_callback')
            app_handles.tab1_handles.cleanup_callback();
        end
    catch ME
        warning('%s: %s', ME.identifier, ME.message);
    end

    try
        if isfield(app_handles, 'tab2_handles') && ...
                isfield(app_handles.tab2_handles, 'cleanup_callback')
            app_handles.tab2_handles.cleanup_callback();
        end
    catch ME
        warning('%s: %s', ME.identifier, ME.message);
    end

    try
        if isfield(app_handles, 'tab3_handles') && ...
                isfield(app_handles.tab3_handles, 'cleanup_callback')
            app_handles.tab3_handles.cleanup_callback();
        end
    catch ME
        warning('%s: %s', ME.identifier, ME.message);
    end

    % Clear app data
    try
        if isfield(app_handles, 'data_manager')
            app_handles.data_manager.clear_all_data();
        end
    catch ME
        warning('%s: %s', ME.identifier, ME.message);
    end

catch ME
    warning('%s: %s', ME.identifier, ME.message);
end

% Always delete the figure, even if there were errors
try
    if ishandle(src)
        delete(src);
    elseif isfield(app_handles, 'main_fig') && ishandle(app_handles.main_fig)
        delete(app_handles.main_fig);
    end
catch
    % Force close using closereq
    closereq;
end

fprintf('Application closed.\n');
end

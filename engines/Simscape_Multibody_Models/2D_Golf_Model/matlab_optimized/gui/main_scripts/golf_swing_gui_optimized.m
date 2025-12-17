function golf_swing_gui_optimized()
% GOLF_SWING_GUI_OPTIMIZED - Enhanced GUI for optimized analysis system
%
% This GUI provides a user-friendly interface to the optimized golf swing
% analysis pipeline, featuring:
%   - Easy parameter configuration
%   - One-click analysis execution
%   - Real-time progress monitoring
%   - Interactive results visualization
%   - Performance metrics display
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
    end

    % Window Configuration
    WINDOW_WIDTH = 1200; % [px] Window width
    WINDOW_HEIGHT = 800; % [px] Window height
    WINDOW_X = 100; % [px] Window X position
    WINDOW_Y = 100; % [px] Window Y position

    % Layout Constants
    MARGIN_SMALL = 0.02; % [ratio] Small margin relative to container
    MARGIN_MED = 0.05;   % [ratio] Medium margin relative to container
    TAB_GROUP_POS = [0.01, 0.01, 0.98, 0.98]; % [x y w h] Tab group position
    CONTROL_PANEL_WIDTH = 0.45; % [ratio] Control panel width
    CONTROL_PANEL_HEIGHT = 0.96; % [ratio] Control panel height

    % Inner Layout Constants
    LEFT_COL_X = 0.05;   % [ratio] Left column X position
    RIGHT_COL_X = 0.5;   % [ratio] Right column X position
    FULL_WIDTH_X = 0.05; % [ratio] Full width element X position
    FULL_WIDTH_W = 0.9;  % [ratio] Full width element width
    RUN_BTN_X = 0.1;     % [ratio] Run button X position
    RUN_BTN_W = 0.8;     % [ratio] Run button width

    INITIAL_Y_POS = 0.88; % [ratio] Initial Y position for controls
    V_SPACING = 0.07;     % [ratio] Standard vertical spacing
    V_SPACING_LARGE = 0.1; % [ratio] Large vertical spacing
    V_SPACING_RUN = 0.12;  % [ratio] Vertical spacing before run button
    V_SPACING_STATUS = 0.08; % [ratio] Vertical spacing after progress text

    % UI Element Dimensions
    CHECKBOX_HEIGHT = 0.04; % [ratio] Checkbox height
    CHECKBOX_WIDTH = 0.1;   % [ratio] Checkbox width
    TEXT_HEIGHT = 0.04;     % [ratio] Text label height
    TEXT_WIDTH = 0.4;       % [ratio] Text label width
    BTN_HEIGHT = 0.08;      % [ratio] Button height
    DIVIDER_HEIGHT = 0.03;  % [ratio] Divider height
    PROGRESS_HEIGHT = 0.05; % [ratio] Progress text height

    % Colors
    BTN_COLOR_RUN = [0.2, 0.8, 0.2]; % [RGB] Green color for run button
    MSG_COLOR_SUCCESS = [0, 0.6, 0]; % [RGB] Green color for success message
    MSG_COLOR_ERROR = [0.8, 0, 0];   % [RGB] Red color for error message

    %% Load configurations
    sim_config = simulation_config();
    plot_cfg = plot_config();

    %% Create main figure
    main_fig = figure('Name', 'Optimized Golf Swing Analysis', ...
                      'NumberTitle', 'off', ...
                      'Position', [WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT], ...
                      'MenuBar', 'none', ...
                      'ToolBar', 'none', ...
                      'Resize', 'on', ...
                      'CloseRequestFcn', @close_gui);

    %% Create tab group
    tab_group = uitabgroup('Parent', main_fig, ...
                          'Position', TAB_GROUP_POS);

    %% Create tabs
    setup_tab = uitab('Parent', tab_group, 'Title', '⚙️ Setup & Run');
    results_tab = uitab('Parent', tab_group, 'Title', '📊 Results');
    performance_tab = uitab('Parent', tab_group, 'Title', '📈 Performance');
    skeleton_tab = uitab('Parent', tab_group, 'Title', '🦴 3D Skeleton');

    %% Setup Tab - Left Panel (Controls)
    control_panel = uipanel('Parent', setup_tab, ...
                           'Title', 'Analysis Configuration', ...
                           'Position', [MARGIN_SMALL, MARGIN_SMALL, CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT]);

    y_pos = INITIAL_Y_POS;

    % Parallel Processing
    uicontrol('Parent', control_panel, 'Style', 'text', ...
             'String', 'Parallel Processing:', ...
             'Units', 'normalized', ...
             'Position', [LEFT_COL_X, y_pos, TEXT_WIDTH, TEXT_HEIGHT], ...
             'HorizontalAlignment', 'left', 'FontSize', 11, ...
             'TooltipString', 'Distribute computation across all available CPU cores');

    parallel_checkbox = uicontrol('Parent', control_panel, 'Style', 'checkbox', ...
                                 'Value', sim_config.use_parallel, ...
                                 'Units', 'normalized', ...
                                 'Position', [RIGHT_COL_X, y_pos, CHECKBOX_WIDTH, CHECKBOX_HEIGHT], ...
                                 'TooltipString', 'Enable parallel processing for 7-10x speedup');
    y_pos = y_pos - V_SPACING;

    % Checkpointing
    uicontrol('Parent', control_panel, 'Style', 'text', ...
             'String', 'Enable Checkpoints:', ...
             'Units', 'normalized', ...
             'Position', [LEFT_COL_X, y_pos, TEXT_WIDTH, TEXT_HEIGHT], ...
             'HorizontalAlignment', 'left', 'FontSize', 11, ...
             'TooltipString', 'Save progress automatically at each analysis stage');

    checkpoint_checkbox = uicontrol('Parent', control_panel, 'Style', 'checkbox', ...
                                   'Value', sim_config.enable_checkpoints, ...
                                   'Units', 'normalized', ...
                                   'Position', [RIGHT_COL_X, y_pos, CHECKBOX_WIDTH, CHECKBOX_HEIGHT], ...
                                   'TooltipString', 'Allow resuming analysis if interrupted');
    y_pos = y_pos - V_SPACING;

    % Generate Plots
    uicontrol('Parent', control_panel, 'Style', 'text', ...
             'String', 'Generate Plots:', ...
             'Units', 'normalized', ...
             'Position', [LEFT_COL_X, y_pos, TEXT_WIDTH, TEXT_HEIGHT], ...
             'HorizontalAlignment', 'left', 'FontSize', 11, ...
             'TooltipString', 'Automatically generate standard visualizations after analysis');

    plots_checkbox = uicontrol('Parent', control_panel, 'Style', 'checkbox', ...
                              'Value', true, ...
                              'Units', 'normalized', ...
                              'Position', [RIGHT_COL_X, y_pos, CHECKBOX_WIDTH, CHECKBOX_HEIGHT], ...
                              'TooltipString', 'Create and save visualization plots');
    y_pos = y_pos - V_SPACING_LARGE;

    % Divider
    uicontrol('Parent', control_panel, 'Style', 'text', ...
             'String', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', ...
             'Units', 'normalized', ...
             'Position', [FULL_WIDTH_X, y_pos, FULL_WIDTH_W, DIVIDER_HEIGHT], ...
             'HorizontalAlignment', 'center', 'FontSize', 10);
    y_pos = y_pos - MARGIN_MED;

    % Run Analysis Button
    run_button = uicontrol('Parent', control_panel, 'Style', 'pushbutton', ...
                          'String', '▶️ RUN COMPLETE ANALYSIS', ...
                          'Units', 'normalized', ...
                          'Position', [RUN_BTN_X, y_pos, RUN_BTN_W, BTN_HEIGHT], ...
                          'FontSize', 14, 'FontWeight', 'bold', ...
                          'BackgroundColor', BTN_COLOR_RUN, ...
                          'Callback', @run_analysis_callback, ...
                          'TooltipString', 'Execute the full golf swing analysis pipeline with current settings');
    y_pos = y_pos - V_SPACING_RUN;

    % Progress Text
    progress_text = uicontrol('Parent', control_panel, 'Style', 'text', ...
                             'String', 'Ready to run analysis', ...
                             'Units', 'normalized', ...
                             'Position', [FULL_WIDTH_X, y_pos, FULL_WIDTH_W, PROGRESS_HEIGHT], ...
                             'HorizontalAlignment', 'center', ...
                             'FontSize', 11, 'FontWeight', 'bold');
    y_pos = y_pos - V_SPACING_STATUS;

    % Status Panel
    status_panel = uipanel('Parent', control_panel, ...
                          'Title', 'Status Log', ...
                          'Position', [MARGIN_MED, MARGIN_MED, FULL_WIDTH_W, y_pos - MARGIN_MED]);

    status_listbox = uicontrol('Parent', status_panel, 'Style', 'listbox', ...
                              'Units', 'normalized', ...
                              'Position', [0.02, 0.02, 0.96, 0.96], ...
                              'FontName', 'Courier', 'FontSize', 9, ...
                              'TooltipString', 'Real-time log of analysis progress and system messages');

    % Add context menu for copying status log
    status_cmenu = uicontextmenu(main_fig);
    uimenu(status_cmenu, 'Label', '📋 Copy Log', 'Callback', @(s,e) copy_listbox_content(status_listbox));
    set(status_listbox, 'UIContextMenu', status_cmenu);

    %% Setup Tab - Right Panel (Info)
    % Calculate remaining width
    info_panel_x = MARGIN_SMALL + CONTROL_PANEL_WIDTH + MARGIN_SMALL;
    info_panel_w = 1 - info_panel_x - MARGIN_SMALL;

    info_panel = uipanel('Parent', setup_tab, ...
                        'Title', 'System Information', ...
                        'Position', [info_panel_x, MARGIN_SMALL, info_panel_w, CONTROL_PANEL_HEIGHT]);

    info_text = {
        'OPTIMIZED 2D GOLF SWING ANALYSIS SYSTEM'
        ''
        '═══════════════════════════════════════'
        ''
        'KEY FEATURES:'
        ''
        '🚀 Parallel ZTCF Generation'
        '   - 7-10x faster than original'
        '   - Utilizes all CPU cores'
        ''
        '💾 Checkpoint System'
        '   - Auto-save at each stage'
        '   - Resume from interruptions'
        ''
        '📊 Unified Plotting'
        '   - 90% code reduction'
        '   - Consistent styling'
        ''
        '🔬 Complete Analysis Pipeline:'
        '   1. Base simulation'
        '   2. ZTCF generation (parallel)'
        '   3. Data synchronization'
        '   4. Work & impulse calculations'
        '   5. ZVCF generation'
        '   6. Plot generation'
        ''
        '═══════════════════════════════════════'
        ''
        'CONFIGURATION:'
        sprintf('  Model: %s', sim_config.model_name)
        sprintf('  Stop Time: %.2f s', sim_config.stop_time)
        sprintf('  ZTCF Points: %d', sim_config.ztcf_num_points)
        sprintf('  Parallel: %s', conditional_str(sim_config.use_parallel))
        ''
        'OUTPUT DIRECTORIES:'
        sprintf('  Data: %s', sim_config.output_path)
        sprintf('  Plots: %s', sim_config.plots_path)
    };

    info_listbox = uicontrol('Parent', info_panel, 'Style', 'listbox', ...
             'String', info_text, ...
             'Units', 'normalized', ...
             'Position', [0.02, 0.02, 0.96, 0.96], ...
             'FontName', 'Courier', 'FontSize', 10, ...
             'Enable', 'on', ...
             'TooltipString', 'System configuration and feature summary');

    % Add context menu for copying info
    info_cmenu = uicontextmenu(main_fig);
    uimenu(info_cmenu, 'Label', '📋 Copy Info', 'Callback', @(s,e) copy_listbox_content(info_listbox));
    set(info_listbox, 'UIContextMenu', info_cmenu);

    %% Results Tab
    results_text = uicontrol('Parent', results_tab, 'Style', 'text', ...
                            'String', 'Run analysis to view results', ...
                            'Units', 'normalized', ...
                            'Position', [0.1, 0.4, 0.8, 0.2], ...
                            'FontSize', 16, 'HorizontalAlignment', 'center', ...
                            'TooltipString', 'Analysis results summary');

    %% Performance Tab
    perf_text = uicontrol('Parent', performance_tab, 'Style', 'text', ...
                         'String', 'Performance metrics will appear after analysis', ...
                         'Units', 'normalized', ...
                         'Position', [0.1, 0.4, 0.8, 0.2], ...
                         'FontSize', 16, 'HorizontalAlignment', 'center', ...
                         'TooltipString', 'Performance metrics data');

    %% Skeleton Tab
    skeleton_panel = uipanel('Parent', skeleton_tab, ...
                            'Position', TAB_GROUP_POS, ...
                            'BorderType', 'none');

    skeleton_placeholder = uicontrol('Parent', skeleton_panel, 'Style', 'text', ...
                                    'String', {'Run analysis to view 3D skeleton visualization', '', ...
                                              'The skeleton plotter provides:', ...
                                              '• Interactive 3D golf swing playback', ...
                                              '• Force and torque vector visualization', ...
                                              '• Multiple camera views', ...
                                              '• Animation recording capabilities'}, ...
                                    'Units', 'normalized', ...
                                    'Position', [0.1, 0.3, 0.8, 0.4], ...
                                    'FontSize', 14, 'HorizontalAlignment', 'center', ...
                                    'TooltipString', '3D visualization placeholder');

    %% Store handles in figure
    handles = struct();
    handles.parallel_checkbox = parallel_checkbox;
    handles.checkpoint_checkbox = checkpoint_checkbox;
    handles.plots_checkbox = plots_checkbox;
    handles.run_button = run_button;
    handles.progress_text = progress_text;
    handles.status_listbox = status_listbox;
    handles.results_text = results_text;
    handles.perf_text = perf_text;
    handles.skeleton_tab = skeleton_tab;
    handles.skeleton_panel = skeleton_panel;
    handles.skeleton_placeholder = skeleton_placeholder;
    handles.sim_config = sim_config;
    handles.plot_cfg = plot_cfg;

    setappdata(main_fig, 'handles', handles);

    fprintf('✅ GUI launched successfully\n');

    %% Callback Functions
    function run_analysis_callback(~, ~)
        % RUN_ANALYSIS_CALLBACK - Callback for the run button
        arguments
            ~
            ~
        end
        h = getappdata(gcbf, 'handles');

        % Update configuration from GUI
        use_parallel = get(h.parallel_checkbox, 'Value');
        use_checkpoints = get(h.checkpoint_checkbox, 'Value');
        generate_plots = get(h.plots_checkbox, 'Value');

        % Disable run button
        set(h.run_button, 'Enable', 'off', 'String', '⏳ RUNNING...');
        set(h.progress_text, 'String', 'Analysis in progress...');
        drawnow;

        % Add status message
        add_status(h.status_listbox, 'Starting analysis...');

        try
            % Run analysis
            [BASE, ZTCF, DELTA, ZVCFTable] = run_analysis(...
                'use_parallel', use_parallel, ...
                'use_checkpoints', use_checkpoints, ...
                'generate_plots', generate_plots, ...
                'verbose', true);

            % Success
            set(h.progress_text, 'String', '✅ Analysis Complete!', ...
                'ForegroundColor', MSG_COLOR_SUCCESS);
            add_status(h.status_listbox, '✅ Analysis completed successfully');

            % Update results tab
            results_str = sprintf(['Analysis Results:\n\n' ...
                'BASE: %d rows × %d columns\n' ...
                'ZTCF: %d rows × %d columns\n' ...
                'DELTA: %d rows × %d columns\n' ...
                'ZVCF: %d rows × %d columns\n\n' ...
                'Data saved to:\n%s'], ...
                height(BASE), width(BASE), ...
                height(ZTCF), width(ZTCF), ...
                height(DELTA), width(DELTA), ...
                height(ZVCFTable), width(ZVCFTable), ...
                h.sim_config.output_path);
            set(h.results_text, 'String', results_str, ...
                'HorizontalAlignment', 'left');

            % Launch SkeletonPlotter in Skeleton tab
            try
                add_status(h.status_listbox, 'Loading 3D skeleton visualization...');

                % Load Q-tables for visualization
                output_path = h.sim_config.output_path;
                baseq_file = fullfile(output_path, 'BASEQ.mat');
                ztcfq_file = fullfile(output_path, 'ZTCFQ.mat');
                deltaq_file = fullfile(output_path, 'DELTAQ.mat');

                try
                    % Load the Q-tables
                    base_data = load(baseq_file, 'BASEQ');
                    BASEQ = base_data.BASEQ;
                    ztcf_data = load(ztcfq_file, 'ZTCFQ');
                    ZTCFQ = ztcf_data.ZTCFQ;
                    delta_data = load(deltaq_file, 'DELTAQ');
                    DELTAQ = delta_data.DELTAQ;

                    % Delete placeholder text
                    delete(h.skeleton_placeholder);

                    % Launch SkeletonPlotter in embedded mode
                    % Note: Ideally paths should be managed by startup.m
                    SkeletonPlotter(BASEQ, ZTCFQ, DELTAQ, h.skeleton_panel);

                    add_status(h.status_listbox, '✅ 3D skeleton visualization loaded');
                catch
                    add_status(h.status_listbox, '⚠️ Q-tables not found or error loading, skeleton visualization unavailable');
                end
            catch ME
                add_status(h.status_listbox, sprintf('⚠️ Skeleton plotter error: %s', ME.message));
            end

        catch ME
            % Error
            set(h.progress_text, 'String', '❌ Analysis Failed', ...
                'ForegroundColor', MSG_COLOR_ERROR);
            add_status(h.status_listbox, sprintf('❌ Error: %s', ME.message));
            errordlg(sprintf('Analysis failed:\n%s', ME.message), 'Error');
        end

        % Re-enable run button
        set(h.run_button, 'Enable', 'on', 'String', '▶️ RUN COMPLETE ANALYSIS');
    end

    function close_gui(src, ~)
        % CLOSE_GUI - Close request callback
        arguments
            src
            ~
        end
        delete(src);
    end

end

function add_status(listbox, message)
    % ADD_STATUS - Add a message to the status listbox
    arguments
        listbox
        message (1,:) char
    end
    % Add message to status listbox
    current = get(listbox, 'String');
    timestamp = datestr(now, 'HH:MM:SS');
    new_message = sprintf('[%s] %s', timestamp, message);

    if ischar(current)
        current = {current};
    end

    updated = [current; {new_message}];
    set(listbox, 'String', updated, 'Value', length(updated));
    drawnow;
end

function str = conditional_str(value)
    % CONDITIONAL_STR - Return 'Enabled' or 'Disabled' based on boolean
    arguments
        value (1,1) logical
    end
    if value
        str = 'Enabled';
    else
        str = 'Disabled';
    end
end

function copy_listbox_content(listbox_handle)
    % COPY_LISTBOX_CONTENT - Copies the content of a listbox to the clipboard
    arguments
        listbox_handle
    end

    content = get(listbox_handle, 'String');

    % Handle cell array of strings (standard listbox content)
    if iscell(content)
        % Join with newlines
        clipboard_text = strjoin(content, newline);
    elseif ischar(content)
        clipboard_text = content;
    else
        clipboard_text = '';
    end

    clipboard('copy', clipboard_text);

    % Visual feedback (briefly change background color if possible,
    % but simple message is safer for now)
    fprintf('📋 Content copied to clipboard\n');
end

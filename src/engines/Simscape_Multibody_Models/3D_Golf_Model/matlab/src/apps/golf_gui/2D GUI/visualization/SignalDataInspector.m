function updated_hotlist = SignalDataInspector(data_table, current_hotlist, config)
% SIGNALDATAINSPECTOR - Modal dialog for managing signal hotlist
%
% Usage:
%   updated_hotlist = SignalDataInspector(data_table, current_hotlist, config)
%
% Inputs:
%   data_table       - Table containing all available signals
%   current_hotlist  - Cell array of current hotlist signal names
%   config           - Configuration structure from SignalPlotConfig
%
% Returns:
%   updated_hotlist  - Cell array of updated hotlist signal names
%                      (empty if user cancelled)
%
% Features:
%   - Categorized signal list (Forces/Torques first, then positions, then others)
%   - Search/filter functionality
%   - Select all/none buttons
%   - Visual indication of currently hotlisted signals

% Extract all numeric column names from the data table
all_signals = get_numeric_signals(data_table);

if isempty(all_signals)
    errordlg('No numeric signals found in data table!', 'Signal Inspector Error');
    updated_hotlist = current_hotlist;
    return;
end

% Categorize signals
[prioritized_signals, position_signals, other_signals] = categorize_signals(all_signals, config);

% Create modal dialog
fig = create_inspector_dialog();

% Create UI components
uihandles = create_ui_components(fig, prioritized_signals, position_signals, other_signals, current_hotlist);

% Wait for user to close dialog
uiwait(fig);

% Get result
if isappdata(0, 'SignalInspectorResult')
    updated_hotlist = getappdata(0, 'SignalInspectorResult');
    rmappdata(0, 'SignalInspectorResult');
else
    updated_hotlist = current_hotlist; % User cancelled
end

%% Nested Functions

    function fig = create_inspector_dialog()
        % Create the main dialog window
        fig = figure('Name', 'Signal Data Inspector', ...
            'NumberTitle', 'off', ...
            'MenuBar', 'none', ...
            'ToolBar', 'none', ...
            'Resize', 'on', ...
            'WindowStyle', 'modal', ...
            'Position', [100, 100, 800, 600], ...
            'Color', [0.94, 0.94, 0.94], ...
            'CloseRequestFcn', @on_cancel);

        % Center on screen
        movegui(fig, 'center');
    end

    function uihandles = create_ui_components(fig, prior_sigs, pos_sigs, other_sigs, hotlist)
        % Create all UI components

        uihandles = struct();

        % Title
        uicontrol('Parent', fig, ...
            'Style', 'text', ...
            'String', 'Select Signals for Hotlist', ...
            'FontSize', 14, ...
            'FontWeight', 'bold', ...
            'Units', 'normalized', ...
            'Position', [0.05, 0.92, 0.9, 0.05], ...
            'BackgroundColor', [0.94, 0.94, 0.94], ...
            'HorizontalAlignment', 'center');

        % Search box
        uicontrol('Parent', fig, ...
            'Style', 'text', ...
            'String', 'Search:', ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.05, 0.86, 0.1, 0.04], ...
            'BackgroundColor', [0.94, 0.94, 0.94], ...
            'HorizontalAlignment', 'left');

        uihandles.search_box = uicontrol('Parent', fig, ...
            'Style', 'edit', ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.16, 0.86, 0.79, 0.04], ...
            'BackgroundColor', [1, 1, 1], ...
            'Callback', @on_search);

        % Create scrollable panel for signal list
        uihandles.panel = uipanel('Parent', fig, ...
            'Title', '', ...
            'Units', 'normalized', ...
            'Position', [0.05, 0.15, 0.9, 0.7], ...
            'BackgroundColor', [1, 1, 1]);

        % Build signal list with checkboxes
        uihandles.signal_list = build_signal_list(uihandles.panel, prior_sigs, pos_sigs, other_sigs, hotlist);

        % Button panel at bottom
        button_panel = uipanel('Parent', fig, ...
            'Units', 'normalized', ...
            'Position', [0.05, 0.02, 0.9, 0.11], ...
            'BackgroundColor', [0.94, 0.94, 0.94], ...
            'BorderType', 'none');

        % Select All button
        uicontrol('Parent', button_panel, ...
            'Style', 'pushbutton', ...
            'String', 'Select All', ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.05, 0.55, 0.2, 0.35], ...
            'Callback', @on_select_all);

        % Select None button
        uicontrol('Parent', button_panel, ...
            'Style', 'pushbutton', ...
            'String', 'Select None', ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.27, 0.55, 0.2, 0.35], ...
            'Callback', @on_select_none);

        % Hotlist count display
        uihandles.count_text = uicontrol('Parent', button_panel, ...
            'Style', 'text', ...
            'String', sprintf('Hotlist: %d signals', sum([uihandles.signal_list.Value])), ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.05, 0.1, 0.42, 0.35], ...
            'BackgroundColor', [0.94, 0.94, 0.94], ...
            'HorizontalAlignment', 'left');

        % Apply button
        uicontrol('Parent', button_panel, ...
            'Style', 'pushbutton', ...
            'String', 'Apply & Close', ...
            'FontSize', 11, ...
            'FontWeight', 'bold', ...
            'Units', 'normalized', ...
            'Position', [0.55, 0.1, 0.2, 0.8], ...
            'BackgroundColor', [0.3, 0.7, 0.3], ...
            'ForegroundColor', [1, 1, 1], ...
            'Callback', @on_apply);

        % Cancel button
        uicontrol('Parent', button_panel, ...
            'Style', 'pushbutton', ...
            'String', 'Cancel', ...
            'FontSize', 11, ...
            'Units', 'normalized', ...
            'Position', [0.77, 0.1, 0.18, 0.8], ...
            'Callback', @on_cancel);

        % Store handles in figure
        guidata(fig, uihandles);
    end

    function signal_list = build_signal_list(panel, prior_sigs, pos_sigs, other_sigs, hotlist)
        % Build categorized list of checkboxes

        signal_list = [];
        y_pos = 0.98;
        item_height = 0.04;

        % Forces & Torques section
        if ~isempty(prior_sigs)
            y_pos = add_category_header(panel, 'Forces & Torques', y_pos, item_height);
            [signal_list, y_pos] = add_signal_checkboxes(panel, prior_sigs, hotlist, signal_list, y_pos, item_height, [0.7, 0.9, 1]);
        end

        % Joint Positions section
        if ~isempty(pos_sigs)
            y_pos = y_pos - item_height * 0.5; % Extra space
            y_pos = add_category_header(panel, 'Joint Positions', y_pos, item_height);
            [signal_list, y_pos] = add_signal_checkboxes(panel, pos_sigs, hotlist, signal_list, y_pos, item_height, [1, 1, 0.9]);
        end

        % Other Signals section
        if ~isempty(other_sigs)
            y_pos = y_pos - item_height * 0.5; % Extra space
            y_pos = add_category_header(panel, 'Other Signals', y_pos, item_height);
            [signal_list, ~] = add_signal_checkboxes(panel, other_sigs, hotlist, signal_list, y_pos, item_height, [0.95, 0.95, 0.95]);
        end
    end

    function y_pos = add_category_header(panel, title, y_pos, item_height)
        % Add a category header
        uicontrol('Parent', panel, ...
            'Style', 'text', ...
            'String', title, ...
            'FontSize', 11, ...
            'FontWeight', 'bold', ...
            'Units', 'normalized', ...
            'Position', [0.02, y_pos - item_height, 0.96, item_height], ...
            'BackgroundColor', [0.85, 0.85, 0.85], ...
            'HorizontalAlignment', 'left');
        y_pos = y_pos - item_height;
    end

    function [signal_list, y_pos] = add_signal_checkboxes(panel, signals, hotlist, signal_list, y_pos, item_height, bg_color)
        % Add checkboxes for a list of signals
        for i = 1:length(signals)
            is_in_hotlist = ismember(signals{i}, hotlist);

            cb = uicontrol('Parent', panel, ...
                'Style', 'checkbox', ...
                'String', signals{i}, ...
                'FontSize', 9, ...
                'Units', 'normalized', ...
                'Position', [0.05, y_pos - item_height, 0.9, item_height], ...
                'BackgroundColor', bg_color, ...
                'Value', is_in_hotlist, ...
                'Tag', signals{i}, ...
                'Callback', @on_checkbox_changed);

            signal_list = [signal_list; cb];
            y_pos = y_pos - item_height;
        end
    end

%% Callback Functions

    function on_search(src, ~)
        % Filter signal list based on search text
        search_text = lower(get(src, 'String'));
        handles = guidata(fig);

        for i = 1:length(handles.signal_list)
            signal_name = lower(get(handles.signal_list(i), 'String'));
            if isempty(search_text) || contains(signal_name, search_text)
                set(handles.signal_list(i), 'Visible', 'on');
            else
                set(handles.signal_list(i), 'Visible', 'off');
            end
        end
    end

    function on_checkbox_changed(~, ~)
        % Update hotlist count when checkbox changes
        handles = guidata(fig);
        count = sum([handles.signal_list.Value]);
        set(handles.count_text, 'String', sprintf('Hotlist: %d signals', count));
    end

    function on_select_all(~, ~)
        % Select all visible checkboxes
        handles = guidata(fig);
        for i = 1:length(handles.signal_list)
            if strcmp(get(handles.signal_list(i), 'Visible'), 'on')
                set(handles.signal_list(i), 'Value', 1);
            end
        end
        on_checkbox_changed([], []);
    end

    function on_select_none(~, ~)
        % Deselect all checkboxes
        handles = guidata(fig);
        for i = 1:length(handles.signal_list)
            set(handles.signal_list(i), 'Value', 0);
        end
        on_checkbox_changed([], []);
    end

    function on_apply(~, ~)
        % Gather selected signals and close
        handles = guidata(fig);
        selected_signals = {};

        for i = 1:length(handles.signal_list)
            if get(handles.signal_list(i), 'Value') == 1
                selected_signals{end+1} = get(handles.signal_list(i), 'Tag');
            end
        end

        setappdata(0, 'SignalInspectorResult', selected_signals);
        delete(fig);
    end

    function on_cancel(~, ~)
        % Close without saving
        if isappdata(0, 'SignalInspectorResult')
            rmappdata(0, 'SignalInspectorResult');
        end
        delete(fig);
    end
end

%% Helper Functions

function signals = get_numeric_signals(data_table)
% Extract all numeric column names from table
signals = {};

for i = 1:width(data_table)
    var_name = data_table.Properties.VariableNames{i};

    % Skip 'Time' column
    if strcmp(var_name, 'Time')
        continue;
    end

    % Check if column is numeric
    if isnumeric(data_table.(var_name))
        % For matrix columns, check if it's actually numeric data
        if ~isempty(data_table.(var_name))
            signals{end+1} = var_name;
        end
    end
end
end

function [prioritized, positions, others] = categorize_signals(signals, config)
% Categorize signals into prioritized (forces/torques), positions, and others

prioritized = {};
positions = {};
others = {};

% Patterns for prioritized signals
priority_patterns = config.prioritized_patterns;

% Patterns for position signals
position_patterns = {'x$', 'y$', 'z$', 'Butt', 'CH', 'MP', 'LW', 'LE', 'LS', 'RW', 'RE', 'RS', 'HUB'};

for i = 1:length(signals)
    signal = signals{i};

    % Check if prioritized
    is_prioritized = false;
    for j = 1:length(priority_patterns)
        if contains(signal, priority_patterns{j}, 'IgnoreCase', true)
            prioritized{end+1} = signal;
            is_prioritized = true;
            break;
        end
    end

    if is_prioritized
        continue;
    end

    % Check if position signal
    is_position = false;
    for j = 1:length(position_patterns)
        if ~isempty(regexp(signal, position_patterns{j}, 'once'))
            positions{end+1} = signal;
            is_position = true;
            break;
        end
    end

    if ~is_position
        others{end+1} = signal;
    end
end
end

function plotter_handles = InteractiveSignalPlotter(datasets, skeleton_handles, config)
% INTERACTIVESIGNALPLOTTER - Interactive time-synchronized signal plotting window
%
% Usage:
%   plotter_handles = InteractiveSignalPlotter(datasets, skeleton_handles, config)
%
% Inputs:
%   datasets          - Structure with fields: BASEQ, ZTCFQ, DELTAQ (tables)
%   skeleton_handles  - Handles structure from SkeletonPlotter (for sync)
%   config            - Configuration structure from SignalPlotConfig
%
% Returns:
%   plotter_handles   - Structure with handles for external control
%
% Features:
%   - Time-synchronized plotting with vertical line indicator
%   - Draggable timeline for scrubbing through data
%   - Value display boxes showing current signal values
%   - Toggle between single plot and subplot modes
%   - Signal selection from hotlist
%   - Data inspector for managing hotlist
%   - Bidirectional sync with SkeletonPlotter

% Validate inputs
if ~isfield(datasets, 'BASEQ') || ~isfield(datasets, 'ZTCFQ') || ~isfield(datasets, 'DELTAQ')
    error('datasets must contain BASEQ, ZTCFQ, and DELTAQ fields');
end

% Get currently selected dataset from skeleton plotter
selected_dataset_idx = get(skeleton_handles.dataset_dropdown, 'Value');
dataset_names = {'BASEQ', 'ZTCFQ', 'DELTAQ'};
current_dataset_name = dataset_names{selected_dataset_idx};
current_dataset = datasets.(current_dataset_name);

% Get time vector
if ismember('Time', current_dataset.Properties.VariableNames)
    time_vector = current_dataset.Time;
else
    time_vector = (0:height(current_dataset)-1)';
end

% Create main figure
fig = create_plotter_window(config);

% Initialize plotter handles
plotter_handles = struct();
plotter_handles.fig = fig;
plotter_handles.datasets = datasets;
plotter_handles.current_dataset_name = current_dataset_name;
plotter_handles.current_dataset = current_dataset;
plotter_handles.time_vector = time_vector;
plotter_handles.skeleton_handles = skeleton_handles;
plotter_handles.config = config;
plotter_handles.selected_signals = config.last_selected;
plotter_handles.plot_mode = config.plot_mode;
plotter_handles.dragging = false;
plotter_handles.value_displays = [];

% Create UI components
create_ui_components(fig, plotter_handles);

% Get updated handles (create_ui_components adds UI elements to handles)
plotter_handles = guidata(fig);

% Set up initial plot
update_plot(plotter_handles);

% Store handles in figure (in case update_plot modified anything)
guidata(fig, plotter_handles);

% Set close callback to save config
set(fig, 'CloseRequestFcn', @on_close);

%% Nested Functions

    function fig = create_plotter_window(cfg)
        % Create the main plotter window

        % Determine position
        if ~isempty(cfg.window_position)
            pos = cfg.window_position;
        else
            pos = [150, 150, 1200, 600];
        end

        fig = figure('Name', 'Interactive Signal Plotter', ...
            'NumberTitle', 'off', ...
            'MenuBar', 'none', ...
            'ToolBar', 'figure', ...
            'Resize', 'on', ...
            'Position', pos, ...
            'Color', [0.94, 0.94, 0.94], ...
            'DeleteFcn', @on_close);
    end

    function create_ui_components(fig, handles)
        % Create all UI components

        % Control panel at top
        control_panel = uipanel('Parent', fig, ...
            'Units', 'normalized', ...
            'Position', [0.01, 0.88, 0.98, 0.11], ...
            'BackgroundColor', [0.9, 0.95, 1], ...
            'Title', 'Signal Selection & Controls', ...
            'FontSize', 11, ...
            'FontWeight', 'bold');

        % Dataset dropdown
        uicontrol('Parent', control_panel, ...
            'Style', 'text', ...
            'String', 'Dataset:', ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.01, 0.5, 0.06, 0.35], ...
            'BackgroundColor', [0.9, 0.95, 1], ...
            'HorizontalAlignment', 'right');

        handles.dataset_selector = uicontrol('Parent', control_panel, ...
            'Style', 'popupmenu', ...
            'String', {'BASEQ', 'ZTCFQ', 'DELTAQ'}, ...
            'Value', selected_dataset_idx, ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.08, 0.52, 0.1, 0.35], ...
            'Callback', @on_dataset_changed);

        % Signal selection listbox (multi-select)
        uicontrol('Parent', control_panel, ...
            'Style', 'text', ...
            'String', 'Select Signals:', ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.19, 0.5, 0.1, 0.35], ...
            'BackgroundColor', [0.9, 0.95, 1], ...
            'HorizontalAlignment', 'left');

        handles.signal_listbox = uicontrol('Parent', control_panel, ...
            'Style', 'listbox', ...
            'String', handles.config.hotlist_signals, ...
            'FontSize', 9, ...
            'Min', 0, 'Max', 2, ...
            'Units', 'normalized', ...
            'Position', [0.19, 0.05, 0.25, 0.45], ...
            'Value', [], ...
            'Callback', @on_signal_selection_changed);

        % Pre-select last selected signals
        if ~isempty(handles.selected_signals)
            [~, selected_idx] = intersect(handles.config.hotlist_signals, handles.selected_signals, 'stable');
            set(handles.signal_listbox, 'Value', selected_idx);
        end

        % Data Inspector button
        uicontrol('Parent', control_panel, ...
            'Style', 'pushbutton', ...
            'String', 'Manage Hotlist', ...
            'FontSize', 10, ...
            'FontWeight', 'bold', ...
            'Units', 'normalized', ...
            'Position', [0.45, 0.52, 0.12, 0.38], ...
            'BackgroundColor', [0.3, 0.6, 0.9], ...
            'ForegroundColor', [1, 1, 1], ...
            'Callback', @on_open_data_inspector);

        % Plot mode toggle
        handles.plot_mode_button = uicontrol('Parent', control_panel, ...
            'Style', 'togglebutton', ...
            'String', 'Single Plot', ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.58, 0.52, 0.1, 0.38], ...
            'Value', strcmp(handles.plot_mode, 'single'), ...
            'Callback', @on_plot_mode_changed);

        if strcmp(handles.plot_mode, 'subplot')
            set(handles.plot_mode_button, 'String', 'Subplots', 'Value', 0);
        end

        % Clear selection button
        uicontrol('Parent', control_panel, ...
            'Style', 'pushbutton', ...
            'String', 'Clear Selection', ...
            'FontSize', 10, ...
            'Units', 'normalized', ...
            'Position', [0.45, 0.08, 0.12, 0.35], ...
            'Callback', @on_clear_selection);

        % Info text
        handles.info_text = uicontrol('Parent', control_panel, ...
            'Style', 'text', ...
            'String', 'Select signals from the hotlist to plot', ...
            'FontSize', 9, ...
            'Units', 'normalized', ...
            'Position', [0.69, 0.05, 0.3, 0.85], ...
            'BackgroundColor', [1, 1, 0.9], ...
            'HorizontalAlignment', 'left');

        % Plot axes panel
        handles.plot_panel = uipanel('Parent', fig, ...
            'Units', 'normalized', ...
            'Position', [0.01, 0.01, 0.98, 0.86], ...
            'BackgroundColor', [1, 1, 1], ...
            'BorderType', 'line');

        % Store handles
        guidata(fig, handles);
    end

    function update_plot(handles)
        % Update the plot based on current selections

        % Get current frame from skeleton plotter
        current_frame = round(get(handles.skeleton_handles.slider, 'Value'));
        current_time = handles.time_vector(current_frame);

        % Get selected signals
        selected_idx = get(handles.signal_listbox, 'Value');
        if isempty(selected_idx)
            % Clear plot
            delete(findall(handles.plot_panel, 'Type', 'axes'));
            handles.value_displays = [];
            set(handles.info_text, 'String', 'Select signals from the hotlist to plot');
            guidata(handles.fig, handles);
            return;
        end

        signal_names = handles.config.hotlist_signals(selected_idx);
        handles.selected_signals = signal_names;

        % Clear existing axes
        delete(findall(handles.plot_panel, 'Type', 'axes'));

        % Create plot based on mode
        if strcmp(handles.plot_mode, 'single')
            create_single_plot(handles, signal_names, current_frame, current_time);
        else
            create_subplots(handles, signal_names, current_frame, current_time);
        end

        % Update info text
        set(handles.info_text, 'String', sprintf('Plotting %d signal(s) | Frame: %d/%d | Time: %.3fs', ...
            length(signal_names), current_frame, length(handles.time_vector), current_time));

        % Store handles
        guidata(handles.fig, handles);
    end

    function create_single_plot(handles, signal_names, current_frame, current_time)
        % Create a single plot with all signals

        ax = axes('Parent', handles.plot_panel, ...
            'Position', [0.08, 0.12, 0.88, 0.83], ...
            'Box', 'on', ...
            'FontSize', 10);

        hold(ax, 'on');
        grid(ax, 'on');

        % Plot each signal
        colors = get_plot_colors(length(signal_names));
        line_handles = [];
        value_displays = [];

        for i = 1:length(signal_names)
            signal_name = signal_names{i};

            % Get signal data
            if ismember(signal_name, handles.current_dataset.Properties.VariableNames)
                signal_data = handles.current_dataset.(signal_name);

                % Handle matrix columns (plot each component separately)
                if size(signal_data, 2) > 1
                    % Plot each component separately
                    for comp = 1:size(signal_data, 2)
                        comp_data = signal_data(:, comp);
                        comp_name = sprintf('%s_%d', signal_name, comp);

                        h = plot(ax, handles.time_vector, comp_data, ...
                            'Color', colors(i,:), ...
                            'LineWidth', 1.5, ...
                            'DisplayName', comp_name);
                        line_handles = [line_handles; h];

                        current_value = comp_data(current_frame);
                        value_displays(end+1).signal = comp_name;
                        value_displays(end).value = current_value;
                        value_displays(end).color = colors(i,:);
                    end
                else
                    % Single component - plot normally
                    h = plot(ax, handles.time_vector, signal_data, ...
                        'Color', colors(i,:), ...
                        'LineWidth', 1.5, ...
                        'DisplayName', signal_name);
                    line_handles = [line_handles; h];

                    current_value = signal_data(current_frame);
                    value_displays(end+1).signal = signal_name;
                    value_displays(end).value = current_value;
                    value_displays(end).color = colors(i,:);
                end
            end
        end

        % Add vertical line at current time
        y_limits = ylim(ax);
        handles.time_line = plot(ax, [current_time, current_time], y_limits, ...
            'r-', 'LineWidth', 2.5, 'Tag', 'TimeLine');

        % Labels and legend
        xlabel(ax, 'Time (s)', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel(ax, 'Value', 'FontSize', 11, 'FontWeight', 'bold');
        title(ax, sprintf('Signal Plot - %s', handles.current_dataset_name), ...
            'FontSize', 12, 'FontWeight', 'bold');

        if ~isempty(line_handles)
            legend(ax, line_handles, 'Location', 'best', 'FontSize', 9);
        end

        % Add value display boxes
        handles.value_displays = add_value_display(handles.plot_panel, value_displays);
        handles.axes_handle = ax;

        % Set up mouse interactions
        set(handles.fig, 'WindowButtonDownFcn', @on_mouse_down);
        set(handles.fig, 'WindowButtonMotionFcn', @on_mouse_move);
        set(handles.fig, 'WindowButtonUpFcn', @on_mouse_up);
    end

    function create_subplots(handles, signal_names, current_frame, current_time)
        % Create subplots for each signal

        n_signals = length(signal_names);
        n_cols = min(3, n_signals);
        n_rows = ceil(n_signals / n_cols);

        axes_handles = [];
        value_displays = [];

        for i = 1:n_signals
            signal_name = signal_names{i};

            % Create subplot
            ax = subplot(n_rows, n_cols, i, 'Parent', handles.plot_panel);
            hold(ax, 'on');
            grid(ax, 'on');

            % Get signal data
            if ismember(signal_name, handles.current_dataset.Properties.VariableNames)
                signal_data = handles.current_dataset.(signal_name);

                % Handle matrix columns (plot each component separately)
                if size(signal_data, 2) > 1
                    % Plot each component separately
                    for comp = 1:size(signal_data, 2)
                        comp_data = signal_data(:, comp);
                        comp_name = sprintf('%s_%d', signal_name, comp);

                        plot(ax, handles.time_vector, comp_data, ...
                            'Color', [0.2, 0.4, 0.8], ...
                            'LineWidth', 1.5);

                        % Store current value for this component
                        current_value = comp_data(current_frame);
                        value_displays(end+1).signal = comp_name;
                        value_displays(end).value = current_value;
                        value_displays(end).color = [0.2, 0.4, 0.8];
                    end
                else
                    % Single component - plot normally
                    plot(ax, handles.time_vector, signal_data, ...
                        'Color', [0.2, 0.4, 0.8], ...
                        'LineWidth', 1.5);

                    % Store current value
                    current_value = signal_data(current_frame);
                    value_displays(end+1).signal = signal_name;
                    value_displays(end).value = current_value;
                    value_displays(end).color = [0.2, 0.4, 0.8];
                end

                % Add vertical line
                y_limits = ylim(ax);
                plot(ax, [current_time, current_time], y_limits, ...
                    'r-', 'LineWidth', 2, 'Tag', 'TimeLine');

                % Labels
                xlabel(ax, 'Time (s)', 'FontSize', 9);
                ylabel(ax, 'Value', 'FontSize', 9);
                title(ax, signal_name, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');

                axes_handles = [axes_handles; ax];
            end
        end

        % Add value display
        handles.value_displays = add_value_display(handles.plot_panel, value_displays);
        handles.axes_handle = axes_handles;

        % Set up mouse interactions
        set(handles.fig, 'WindowButtonDownFcn', @on_mouse_down);
        set(handles.fig, 'WindowButtonMotionFcn', @on_mouse_move);
        set(handles.fig, 'WindowButtonUpFcn', @on_mouse_up);
    end

    function value_display_handles = add_value_display(panel, value_displays)
        % Add aesthetic value display boxes

        if isempty(value_displays)
            value_display_handles = [];
            return;
        end

        % Create panel for value displays
        value_panel = uipanel('Parent', panel, ...
            'Units', 'normalized', ...
            'Position', [0.01, 0.01, 0.25, 0.09], ...
            'BackgroundColor', [0.95, 0.95, 0.95], ...
            'Title', 'Current Values', ...
            'FontSize', 9, ...
            'FontWeight', 'bold');

        % Create text displays
        n_signals = length(value_displays);
        value_display_handles = [];

        for i = 1:min(n_signals, 5)  % Limit to 5 displays
            txt = uicontrol('Parent', value_panel, ...
                'Style', 'text', ...
                'String', sprintf('%s: %.3f', value_displays(i).signal, value_displays(i).value), ...
                'FontSize', 8, ...
                'Units', 'normalized', ...
                'Position', [0.02, 1 - i*0.18, 0.96, 0.15], ...
                'BackgroundColor', [1, 1, 1], ...
                'ForegroundColor', value_displays(i).color, ...
                'HorizontalAlignment', 'left', ...
                'Tag', value_displays(i).signal);

            value_display_handles = [value_display_handles; txt];
        end

        if n_signals > 5
            % Add indicator for more signals
            uicontrol('Parent', value_panel, ...
                'Style', 'text', ...
                'String', sprintf('...and %d more', n_signals - 5), ...
                'FontSize', 7, ...
                'Units', 'normalized', ...
                'Position', [0.02, 0.02, 0.96, 0.12], ...
                'BackgroundColor', [0.95, 0.95, 0.95], ...
                'HorizontalAlignment', 'center');
        end
    end

    function colors = get_plot_colors(n)
        % Generate distinguishable colors for plotting
        base_colors = [
            0.0, 0.4, 1.0;  % Blue
            1.0, 0.0, 0.0;  % Red
            0.0, 0.7, 0.0;  % Green
            1.0, 0.5, 0.0;  % Orange
            0.7, 0.0, 0.7;  % Purple
            0.0, 0.8, 0.8;  % Cyan
            0.8, 0.8, 0.0;  % Yellow
            0.8, 0.0, 0.4;  % Magenta
            0.4, 0.4, 0.4;  % Gray
            0.0, 0.0, 0.0;  % Black
            ];

        if n <= size(base_colors, 1)
            colors = base_colors(1:n, :);
        else
            % Generate additional colors
            colors = [base_colors; hsv(n - size(base_colors, 1))];
        end
    end

%% Callback Functions

    function on_dataset_changed(src, ~)
        % Handle dataset selection change
        handles = guidata(fig);
        dataset_idx = get(src, 'Value');
        dataset_name = dataset_names{dataset_idx};

        handles.current_dataset_name = dataset_name;
        handles.current_dataset = handles.datasets.(dataset_name);

        % Update time vector
        if ismember('Time', handles.current_dataset.Properties.VariableNames)
            handles.time_vector = handles.current_dataset.Time;
        else
            handles.time_vector = (0:height(handles.current_dataset)-1)';
        end

        guidata(fig, handles);
        update_plot(handles);
    end

    function on_signal_selection_changed(~, ~)
        % Handle signal selection change
        handles = guidata(fig);
        update_plot(handles);
    end

    function on_clear_selection(~, ~)
        % Clear signal selection
        handles = guidata(fig);
        set(handles.signal_listbox, 'Value', []);
        update_plot(handles);
    end

    function on_plot_mode_changed(src, ~)
        % Toggle between single plot and subplot modes
        handles = guidata(fig);

        if get(src, 'Value') == 1
            handles.plot_mode = 'single';
            set(src, 'String', 'Single Plot');
        else
            handles.plot_mode = 'subplot';
            set(src, 'String', 'Subplots');
        end

        guidata(fig, handles);
        update_plot(handles);
    end

    function on_open_data_inspector(~, ~)
        % Open data inspector dialog
        handles = guidata(fig);

        updated_hotlist = SignalDataInspector(handles.current_dataset, ...
            handles.config.hotlist_signals, ...
            handles.config);

        if ~isempty(updated_hotlist)
            % Update config and hotlist
            handles.config.hotlist_signals = updated_hotlist;
            set(handles.signal_listbox, 'String', updated_hotlist);

            % Clear selection if current selection not in new hotlist
            current_selection = get(handles.signal_listbox, 'Value');
            if any(current_selection > length(updated_hotlist))
                set(handles.signal_listbox, 'Value', []);
            end

            guidata(fig, handles);
            update_plot(handles);
        end
    end

    function on_mouse_down(~, ~)
        % Handle mouse down for timeline dragging
        handles = guidata(fig);

        % Check if click is on axes
        if strcmp(handles.plot_mode, 'single')
            axes_to_check = handles.axes_handle;
        else
            axes_to_check = handles.axes_handle(1);  % Use first subplot
        end

        cp = get(axes_to_check, 'CurrentPoint');
        x_click = cp(1,1);

        % Check if click is within time range
        if x_click >= min(handles.time_vector) && x_click <= max(handles.time_vector)
            handles.dragging = true;
            guidata(fig, handles);
            update_time_position(handles, x_click);
        end
    end

    function on_mouse_move(~, ~)
        % Handle mouse move for timeline dragging
        handles = guidata(fig);

        if handles.dragging
            % Get mouse position
            if strcmp(handles.plot_mode, 'single')
                axes_to_check = handles.axes_handle;
            else
                axes_to_check = handles.axes_handle(1);
            end

            cp = get(axes_to_check, 'CurrentPoint');
            x_pos = cp(1,1);

            % Update time position
            if x_pos >= min(handles.time_vector) && x_pos <= max(handles.time_vector)
                update_time_position(handles, x_pos);
            end
        end
    end

    function on_mouse_up(~, ~)
        % Handle mouse up to stop dragging
        handles = guidata(fig);
        handles.dragging = false;
        guidata(fig, handles);
    end

    function update_time_position(handles, time_value)
        % Update time position and sync with skeleton plotter

        % Find nearest frame
        [~, frame_idx] = min(abs(handles.time_vector - time_value));

        % Update skeleton plotter slider value
        set(handles.skeleton_handles.slider, 'Value', frame_idx);

        % Manually trigger the slider's callback to update skeleton plotter
        % (Setting value programmatically doesn't fire callback automatically)
        callback = get(handles.skeleton_handles.slider, 'Callback');
        if ~isempty(callback)
            if isa(callback, 'function_handle')
                callback(handles.skeleton_handles.slider, []);
            elseif iscell(callback)
                feval(callback{1}, handles.skeleton_handles.slider, [], callback{2:end});
            end
        end

        % Update our own time line
        update_time_line(handles, frame_idx);
    end

    function update_time_line(handles, frame_idx)
        % Update the vertical time line indicator

        current_time = handles.time_vector(frame_idx);

        if strcmp(handles.plot_mode, 'single')
            % Update single plot time line
            time_line = findobj(handles.axes_handle, 'Tag', 'TimeLine');
            if ~isempty(time_line)
                set(time_line, 'XData', [current_time, current_time]);
            end

            % Update value displays
            if ~isempty(handles.value_displays)
                for i = 1:length(handles.value_displays)
                    display_name = get(handles.value_displays(i), 'Tag');

                    % Extract signal name and component number from display name
                    % Format is either 'SignalName' or 'SignalName_ComponentNum'
                    if contains(display_name, '_')
                        parts = strsplit(display_name, '_');
                        comp_num = str2double(parts{end});
                        if ~isnan(comp_num)
                            % Multi-component signal
                            signal_name = strjoin(parts(1:end-1), '_');
                        else
                            % Signal name contains underscore but no component number
                            signal_name = display_name;
                            comp_num = [];
                        end
                    else
                        % Single component signal
                        signal_name = display_name;
                        comp_num = [];
                    end

                    % Get signal data and update display
                    if ismember(signal_name, handles.current_dataset.Properties.VariableNames)
                        signal_data = handles.current_dataset.(signal_name);
                        if ~isempty(comp_num) && comp_num <= size(signal_data, 2)
                            current_value = signal_data(frame_idx, comp_num);
                        else
                            current_value = signal_data(frame_idx, 1);
                        end
                        set(handles.value_displays(i), 'String', ...
                            sprintf('%s: %.3f', display_name, current_value));
                    end
                end
            end
        else
            % Update subplot time lines
            for i = 1:length(handles.axes_handle)
                time_line = findobj(handles.axes_handle(i), 'Tag', 'TimeLine');
                if ~isempty(time_line)
                    set(time_line, 'XData', [current_time, current_time]);
                end
            end

            % Update value displays
            if ~isempty(handles.value_displays)
                for i = 1:length(handles.value_displays)
                    display_name = get(handles.value_displays(i), 'Tag');

                    % Extract signal name and component number from display name
                    % Format is either 'SignalName' or 'SignalName_ComponentNum'
                    if contains(display_name, '_')
                        parts = strsplit(display_name, '_');
                        comp_num = str2double(parts{end});
                        if ~isnan(comp_num)
                            % Multi-component signal
                            signal_name = strjoin(parts(1:end-1), '_');
                        else
                            % Signal name contains underscore but no component number
                            signal_name = display_name;
                            comp_num = [];
                        end
                    else
                        % Single component signal
                        signal_name = display_name;
                        comp_num = [];
                    end

                    % Get signal data and update display
                    if ismember(signal_name, handles.current_dataset.Properties.VariableNames)
                        signal_data = handles.current_dataset.(signal_name);
                        if ~isempty(comp_num) && comp_num <= size(signal_data, 2)
                            current_value = signal_data(frame_idx, comp_num);
                        else
                            current_value = signal_data(frame_idx, 1);
                        end
                        set(handles.value_displays(i), 'String', ...
                            sprintf('%s: %.3f', display_name, current_value));
                    end
                end
            end
        end

        % Update info text
        set(handles.info_text, 'String', sprintf('Plotting %d signal(s) | Frame: %d/%d | Time: %.3fs', ...
            length(get(handles.signal_listbox, 'Value')), ...
            frame_idx, length(handles.time_vector), current_time));
    end

    function on_close(~, ~)
        % Save configuration and cleanup before closing

        fprintf('Cleaning up Signal Plotter...\n');

        try
            handles = guidata(fig);

            % Update config with current state
            handles.config.last_selected = handles.selected_signals;
            handles.config.plot_mode = handles.plot_mode;
            handles.config.window_position = get(fig, 'Position');

            % Save config
            SignalPlotConfig('save', handles.config);
        catch
            % If handles don't exist, just skip config save
            fprintf('   Unable to save config (window may have been force-closed)\n');
        end

        % Clear app data from figure
        if ishandle(fig)
            try
                props = getappdata(fig);
                fields = fieldnames(props);
                for i = 1:length(fields)
                    rmappdata(fig, fields{i});
                end
            catch
                % No app data to remove
            end
        end

        % Delete figure
        if ishandle(fig)
            delete(fig);
        end

        fprintf('Signal Plotter cleanup complete.\n');
    end
end

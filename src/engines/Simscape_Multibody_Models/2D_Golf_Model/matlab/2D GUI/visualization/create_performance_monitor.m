function create_performance_monitor(parent, config)
    % CREATE_PERFORMANCE_MONITOR - Create performance monitoring panel
    %
    % This function creates a comprehensive performance monitoring interface
    % that displays real-time metrics, historical data, and performance insights
    %
    % Inputs:
    %   parent - Parent UI container
    %   config - Configuration structure
    
    % Create main performance panel
    perf_panel = uipanel('Parent', parent, ...
                        'Title', '🔍 Performance Monitor', ...
                        'FontSize', 12, ...
                        'Position', [0.02, 0.02, 0.96, 0.96]);
    
    % Create sub-panels for different monitoring aspects
    create_realtime_metrics(perf_panel, config);
    create_performance_controls(perf_panel, config);
    create_performance_charts(perf_panel, config);
    create_performance_actions(perf_panel, config);
    
end

function create_realtime_metrics(parent, config)
    % Create real-time metrics display
    
    % Real-time metrics panel
    metrics_panel = uipanel('Parent', parent, ...
                           'Title', 'Real-time Metrics', ...
                           'FontSize', 11, ...
                           'Position', [0.02, 0.65, 0.48, 0.33]);
    
    % Current session info
    uicontrol('Parent', metrics_panel, ...
              'Style', 'text', ...
              'String', 'Session Info:', ...
              'FontSize', 10, ...
              'FontWeight', 'bold', ...
              'Position', [10, 120, 150, 20], ...
              'HorizontalAlignment', 'left');
    
    % Session duration
    uicontrol('Parent', metrics_panel, ...
              'Style', 'text', ...
              'String', 'Duration:', ...
              'FontSize', 9, ...
              'Position', [10, 100, 80, 15], ...
              'HorizontalAlignment', 'left');
    
    session_duration_text = uicontrol('Parent', metrics_panel, ...
                                     'Style', 'text', ...
                                     'String', '00:00:00', ...
                                     'FontSize', 9, ...
                                     'Position', [95, 100, 100, 15], ...
                                     'HorizontalAlignment', 'left', ...
                                     'BackgroundColor', config.colors.background);
    
    % Memory usage
    uicontrol('Parent', metrics_panel, ...
              'Style', 'text', ...
              'String', 'Memory Usage:', ...
              'FontSize', 9, ...
              'Position', [10, 80, 80, 15], ...
              'HorizontalAlignment', 'left');
    
    memory_usage_text = uicontrol('Parent', metrics_panel, ...
                                 'Style', 'text', ...
                                 'String', '0 MB', ...
                                 'FontSize', 9, ...
                                 'Position', [95, 80, 100, 15], ...
                                 'HorizontalAlignment', 'left', ...
                                 'BackgroundColor', config.colors.background);
    
    % Active operations
    uicontrol('Parent', metrics_panel, ...
              'Style', 'text', ...
              'String', 'Active Operations:', ...
              'FontSize', 9, ...
              'Position', [10, 60, 80, 15], ...
              'HorizontalAlignment', 'left');
    
    active_ops_text = uicontrol('Parent', metrics_panel, ...
                               'Style', 'text', ...
                               'String', '0', ...
                               'FontSize', 9, ...
                               'Position', [95, 60, 100, 15], ...
                               'HorizontalAlignment', 'left', ...
                               'BackgroundColor', config.colors.background);
    
    % Store handles for updates
    setappdata(parent, 'session_duration_text', session_duration_text);
    setappdata(parent, 'memory_usage_text', memory_usage_text);
    setappdata(parent, 'active_ops_text', active_ops_text);
    
end

function create_performance_controls(parent, config)
    % Create performance control buttons
    
    % Controls panel
    controls_panel = uipanel('Parent', parent, ...
                            'Title', 'Performance Controls', ...
                            'FontSize', 11, ...
                            'Position', [0.52, 0.65, 0.46, 0.33]);
    
    % Enable/Disable tracking
    enable_tracking_btn = uicontrol('Parent', controls_panel, ...
                                   'Style', 'togglebutton', ...
                                   'String', 'Enable Tracking', ...
                                   'FontSize', 10, ...
                                   'Position', [10, 100, 120, 30], ...
                                   'Callback', @toggle_tracking);
    
    % Clear history
    clear_history_btn = uicontrol('Parent', controls_panel, ...
                                 'Style', 'pushbutton', ...
                                 'String', 'Clear History', ...
                                 'FontSize', 10, ...
                                 'Position', [140, 100, 100, 30], ...
                                 'Callback', @clear_performance_history);
    
    % Auto-refresh toggle
    auto_refresh_checkbox = uicontrol('Parent', controls_panel, ...
                                     'Style', 'checkbox', ...
                                     'String', 'Auto-refresh', ...
                                     'FontSize', 10, ...
                                     'Position', [10, 70, 100, 20], ...
                                     'Value', 1, ...
                                     'Callback', @toggle_auto_refresh);
    
    % Refresh interval
    uicontrol('Parent', controls_panel, ...
              'Style', 'text', ...
              'String', 'Refresh (sec):', ...
              'FontSize', 9, ...
              'Position', [10, 45, 80, 15], ...
              'HorizontalAlignment', 'left');
    
    refresh_interval_edit = uicontrol('Parent', controls_panel, ...
                                     'Style', 'edit', ...
                                     'String', '2', ...
                                     'FontSize', 9, ...
                                     'Position', [95, 45, 50, 20], ...
                                     'Callback', @update_refresh_interval);
    
    % Manual refresh button
    refresh_btn = uicontrol('Parent', controls_panel, ...
                           'Style', 'pushbutton', ...
                           'String', 'Refresh Now', ...
                           'FontSize', 10, ...
                           'Position', [10, 10, 100, 30], ...
                           'Callback', @manual_refresh);
    
    % Store handles
    setappdata(parent, 'enable_tracking_btn', enable_tracking_btn);
    setappdata(parent, 'auto_refresh_checkbox', auto_refresh_checkbox);
    setappdata(parent, 'refresh_interval_edit', refresh_interval_edit);
    
end

function create_performance_charts(parent, config)
    % Create performance visualization charts
    
    % Charts panel
    charts_panel = uipanel('Parent', parent, ...
                          'Title', 'Performance Charts', ...
                          'FontSize', 11, ...
                          'Position', [0.02, 0.35, 0.96, 0.28]);
    
    % Create subplot layout for charts
    % Execution time chart
    time_ax = subplot(1, 3, 1, 'Parent', charts_panel);
    title(time_ax, 'Execution Times');
    xlabel(time_ax, 'Operation');
    ylabel(time_ax, 'Time (seconds)');
    grid(time_ax, 'on');
    
    % Memory usage chart
    memory_ax = subplot(1, 3, 2, 'Parent', charts_panel);
    title(memory_ax, 'Memory Usage');
    xlabel(memory_ax, 'Operation');
    ylabel(memory_ax, 'Memory (MB)');
    grid(memory_ax, 'on');
    
    % Operation frequency chart
    freq_ax = subplot(1, 3, 3, 'Parent', charts_panel);
    title(freq_ax, 'Operation Frequency');
    xlabel(freq_ax, 'Operation');
    ylabel(freq_ax, 'Count');
    grid(freq_ax, 'on');
    
    % Store axes handles
    setappdata(parent, 'time_ax', time_ax);
    setappdata(parent, 'memory_ax', memory_ax);
    setappdata(parent, 'freq_ax', freq_ax);
    
end

function create_performance_actions(parent, config)
    % Create performance action buttons
    
    % Actions panel
    actions_panel = uipanel('Parent', parent, ...
                           'Title', 'Performance Actions', ...
                           'FontSize', 11, ...
                           'Position', [0.02, 0.02, 0.96, 0.31]);
    
    % Generate report button
    generate_report_btn = uicontrol('Parent', actions_panel, ...
                                   'Style', 'pushbutton', ...
                                   'String', 'Generate Report', ...
                                   'FontSize', 10, ...
                                   'Position', [10, 80, 120, 30], ...
                                   'Callback', @generate_performance_report);
    
    % Export CSV button
    export_csv_btn = uicontrol('Parent', actions_panel, ...
                               'Style', 'pushbutton', ...
                               'String', 'Export CSV', ...
                               'FontSize', 10, ...
                               'Position', [140, 80, 100, 30], ...
                               'Callback', @export_performance_csv);
    
    % Save report button
    save_report_btn = uicontrol('Parent', actions_panel, ...
                               'Style', 'pushbutton', ...
                               'String', 'Save Report', ...
                               'FontSize', 10, ...
                               'Position', [250, 80, 100, 30], ...
                               'Callback', @save_performance_report);
    
    % Performance summary text area
    uicontrol('Parent', actions_panel, ...
              'Style', 'text', ...
              'String', 'Performance Summary:', ...
              'FontSize', 10, ...
              'FontWeight', 'bold', ...
              'Position', [10, 50, 150, 20], ...
              'HorizontalAlignment', 'left');
    
    summary_text = uicontrol('Parent', actions_panel, ...
                            'Style', 'text', ...
                            'String', 'No performance data available', ...
                            'FontSize', 9, ...
                            'Position', [10, 10, 340, 35], ...
                            'HorizontalAlignment', 'left', ...
                            'BackgroundColor', config.colors.background);
    
    % Store handles
    setappdata(parent, 'summary_text', summary_text);
    
end

% Callback functions
function toggle_tracking(src, ~)
    % Toggle performance tracking on/off
    main_fig = findobj('Type', 'figure', 'Name', 'Golf Swing Analysis GUI');
    if ~isempty(main_fig)
        tracker = getappdata(main_fig, 'performance_tracker');
        if ~isempty(tracker)
            if get(src, 'Value') == 1
                tracker.enable_tracking();
                set(src, 'String', 'Disable Tracking');
            else
                tracker.disable_tracking();
                set(src, 'String', 'Enable Tracking');
            end
        end
    end
end

function clear_performance_history(src, ~)
    % Clear performance history
    main_fig = findobj('Type', 'figure', 'Name', 'Golf Swing Analysis GUI');
    if ~isempty(main_fig)
        tracker = getappdata(main_fig, 'performance_tracker');
        if ~isempty(tracker)
            tracker.clear_history();
            update_performance_display(main_fig);
        end
    end
end

function toggle_auto_refresh(src, ~)
    % Toggle auto-refresh functionality
    main_fig = findobj('Type', 'figure', 'Name', 'Golf Swing Analysis GUI');
    if ~isempty(main_fig)
        if get(src, 'Value') == 1
            % Start auto-refresh timer
            start_auto_refresh_timer(main_fig);
        else
            % Stop auto-refresh timer
            stop_auto_refresh_timer(main_fig);
        end
    end
end

function update_refresh_interval(src, ~)
    % Update refresh interval
    main_fig = findobj('Type', 'figure', 'Name', 'Golf Swing Analysis GUI');
    if ~isempty(main_fig)
        try
            interval = str2double(get(src, 'String'));
            if ~isnan(interval) && interval > 0
                stop_auto_refresh_timer(main_fig);
                start_auto_refresh_timer(main_fig, interval);
            end
        catch
            % Invalid input, ignore
        end
    end
end

function manual_refresh(src, ~)
    % Manual refresh of performance display
    main_fig = findobj('Type', 'figure', 'Name', 'Golf Swing Analysis GUI');
    if ~isempty(main_fig)
        update_performance_display(main_fig);
    end
end

function generate_performance_report(src, ~)
    % Generate and display performance report
    main_fig = findobj('Type', 'figure', 'Name', 'Golf Swing Analysis GUI');
    if ~isempty(main_fig)
        tracker = getappdata(main_fig, 'performance_tracker');
        if ~isempty(tracker)
            tracker.display_performance_report();
        end
    end
end

function export_performance_csv(src, ~)
    % Export performance data to CSV
    main_fig = findobj('Type', 'figure', 'Name', 'Golf Swing Analysis GUI');
    if ~isempty(main_fig)
        tracker = getappdata(main_fig, 'performance_tracker');
        if ~isempty(tracker)
            [filename, pathname] = uiputfile('*.csv', 'Save Performance Data');
            if filename ~= 0
                full_path = fullfile(pathname, filename);
                tracker.export_performance_csv(full_path);
            end
        end
    end
end

function save_performance_report(src, ~)
    % Save performance report to file
    main_fig = findobj('Type', 'figure', 'Name', 'Golf Swing Analysis GUI');
    if ~isempty(main_fig)
        tracker = getappdata(main_fig, 'performance_tracker');
        if ~isempty(tracker)
            [filename, pathname] = uiputfile('*.mat', 'Save Performance Report');
            if filename ~= 0
                full_path = fullfile(pathname, filename);
                tracker.save_performance_report(full_path);
            end
        end
    end
end

function start_auto_refresh_timer(main_fig, interval)
    % Start auto-refresh timer
    if nargin < 2
        interval = 2; % Default 2 seconds
    end
    
    % Stop existing timer if any
    stop_auto_refresh_timer(main_fig);
    
    % Create new timer
    timer_obj = timer('ExecutionMode', 'fixedRate', ...
                     'Period', interval, ...
                     'TimerFcn', @(~, ~) update_performance_display(main_fig));
    start(timer_obj);
    
    % Store timer in figure data
    setappdata(main_fig, 'refresh_timer', timer_obj);
end

function stop_auto_refresh_timer(main_fig)
    % Stop auto-refresh timer
    timer_obj = getappdata(main_fig, 'refresh_timer');
    if ~isempty(timer_obj) && isvalid(timer_obj)
        stop(timer_obj);
        delete(timer_obj);
        setappdata(main_fig, 'refresh_timer', []);
    end
end

function update_performance_display(main_fig)
    % Update performance display with current data
    tracker = getappdata(main_fig, 'performance_tracker');
    if isempty(tracker)
        return;
    end
    
    % Get performance report
    report = tracker.get_performance_report();
    
    % Update session duration
    session_duration_text = findobj(main_fig, 'Tag', 'session_duration_text');
    if ~isempty(session_duration_text)
        duration_str = format_duration(report.session_info.session_duration);
        set(session_duration_text, 'String', duration_str);
    end
    
    % Update memory usage
    memory_usage_text = findobj(main_fig, 'Tag', 'memory_usage_text');
    if ~isempty(memory_usage_text)
        try
            [~, systemview] = memory;
            memory_mb = (systemview.PhysicalMemory.Total - systemview.PhysicalMemory.Available) / 1024 / 1024;
            set(memory_usage_text, 'String', sprintf('%.1f MB', memory_mb));
        catch
            set(memory_usage_text, 'String', 'N/A');
        end
    end
    
    % Update active operations
    active_ops_text = findobj(main_fig, 'Tag', 'active_ops_text');
    if ~isempty(active_ops_text)
        set(active_ops_text, 'String', num2str(report.session_info.total_operations));
    end
    
    % Update performance charts
    update_performance_charts(main_fig, report);
    
    % Update summary text
    summary_text = findobj(main_fig, 'Tag', 'summary_text');
    if ~isempty(summary_text)
        summary_str = generate_summary_string(report);
        set(summary_text, 'String', summary_str);
    end
end

function update_performance_charts(main_fig, report)
    % Update performance charts with current data
    time_ax = findobj(main_fig, 'Tag', 'time_ax');
    memory_ax = findobj(main_fig, 'Tag', 'memory_ax');
    freq_ax = findobj(main_fig, 'Tag', 'freq_ax');
    
    if isempty(time_ax) || isempty(memory_ax) || isempty(freq_ax)
        return;
    end
    
    operation_names = fieldnames(report.operations);
    if isempty(operation_names)
        return;
    end
    
    % Prepare data for plotting
    times = [];
    memories = [];
    counts = [];
    labels = {};
    
    for i = 1:length(operation_names)
        op_name = operation_names{i};
        op_data = report.operations.(op_name);
        
        times = [times, op_data.average_time];
        memories = [memories, op_data.memory_delta / 1024 / 1024]; % Convert to MB
        counts = [counts, op_data.count];
        labels{i} = op_name;
    end
    
    % Update execution time chart
    cla(time_ax);
    bar(time_ax, times);
    set(time_ax, 'XTickLabel', labels);
    title(time_ax, 'Execution Times');
    xlabel(time_ax, 'Operation');
    ylabel(time_ax, 'Time (seconds)');
    grid(time_ax, 'on');
    xtickangle(time_ax, 45);
    
    % Update memory usage chart
    cla(memory_ax);
    bar(memory_ax, memories);
    set(memory_ax, 'XTickLabel', labels);
    title(memory_ax, 'Memory Usage');
    xlabel(memory_ax, 'Operation');
    ylabel(memory_ax, 'Memory (MB)');
    grid(memory_ax, 'on');
    xtickangle(memory_ax, 45);
    
    % Update operation frequency chart
    cla(freq_ax);
    bar(freq_ax, counts);
    set(freq_ax, 'XTickLabel', labels);
    title(freq_ax, 'Operation Frequency');
    xlabel(freq_ax, 'Operation');
    ylabel(freq_ax, 'Count');
    grid(freq_ax, 'on');
    xtickangle(freq_ax, 45);
    
    drawnow;
end

function duration_str = format_duration(seconds)
    % Format duration in HH:MM:SS format
    hours = floor(seconds / 3600);
    minutes = floor(mod(seconds, 3600) / 60);
    secs = floor(mod(seconds, 60));
    duration_str = sprintf('%02d:%02d:%02d', hours, minutes, secs);
end

function summary_str = generate_summary_string(report)
    % Generate summary string for display
    if isfield(report, 'message')
        summary_str = report.message;
        return;
    end
    
    operation_names = fieldnames(report.operations);
    if isempty(operation_names)
        summary_str = 'No performance data available';
        return;
    end
    
    % Calculate summary statistics
    total_ops = length(operation_names);
    total_time = report.summary.total_time;
    slowest_op = report.summary.slowest_operation;
    fastest_op = report.summary.fastest_operation;
    
    summary_str = sprintf('Total: %d operations, %.1f seconds', total_ops, total_time);
    if ~isempty(slowest_op)
        summary_str = sprintf('%s\nSlowest: %s', summary_str, slowest_op);
    end
    if ~isempty(fastest_op)
        summary_str = sprintf('%s\nFastest: %s', summary_str, fastest_op);
    end
    
    % Add bottleneck information
    if ~isempty(report.bottlenecks)
        summary_str = sprintf('%s\nBottlenecks: %d identified', summary_str, length(report.bottlenecks));
    end
end

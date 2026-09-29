function [fig, explorer_handles] = create_data_explorer(config)
% CREATE_DATA_EXPLORER - Create data explorer for understanding data structure
%
% Inputs:
%   config - Configuration structure from model_config()
%
% Returns:
%   fig - Figure handle
%   explorer_handles - Structure containing explorer handles
%
% This function creates a data explorer that helps users:
% - Understand the data structure
% - Navigate through variables
% - Preview data values
% - Get statistical summaries

    % Create figure
    fig = figure('Name', [config.gui_title ' - Data Explorer'], ...
                 'NumberTitle', 'off', ...
                 'Position', [300, 150, 1200, 800], ...
                 'Color', config.colors.background, ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none', ...
                 'Resize', 'on');

    % Create main layout
    explorer_handles = struct();

    % Create left panel for data navigation
    left_panel = uipanel('Parent', fig, ...
                        'Title', 'Data Navigation', ...
                        'FontSize', 11, ...
                        'Position', [0.02, 0.02, 0.3, 0.96]);

    % Dataset selection
    uicontrol('Parent', left_panel, ...
              'Style', 'text', ...
              'String', 'Dataset:', ...
              'FontSize', 10, ...
              'Position', [10, 700, 100, 20], ...
              'HorizontalAlignment', 'left');

    dataset_popup = uicontrol('Parent', left_panel, ...
                             'Style', 'popupmenu', ...
                             'String', {'BASE', 'ZTCF', 'DELTA', 'ZVCF'}, ...
                             'FontSize', 10, ...
                             'Position', [10, 670, 200, 25], ...
                             'Callback', @update_variable_list);

    % Variable list
    uicontrol('Parent', left_panel, ...
              'Style', 'text', ...
              'String', 'Variables:', ...
              'FontSize', 10, ...
              'Position', [10, 640, 100, 20], ...
              'HorizontalAlignment', 'left');

    variable_listbox = uicontrol('Parent', left_panel, ...
                                'Style', 'listbox', ...
                                'String', {'Select dataset first...'}, ...
                                'FontSize', 10, ...
                                'Position', [10, 400, 200, 230], ...
                                'Callback', @select_variable);

    % Search box
    uicontrol('Parent', left_panel, ...
              'Style', 'text', ...
              'String', 'Search Variables:', ...
              'FontSize', 10, ...
              'Position', [10, 370, 100, 20], ...
              'HorizontalAlignment', 'left');

    search_edit = uicontrol('Parent', left_panel, ...
                           'Style', 'edit', ...
                           'String', '', ...
                           'FontSize', 10, ...
                           'Position', [10, 340, 200, 25], ...
                           'Callback', @search_variables);

    % Create right panel for data preview
    right_panel = uipanel('Parent', fig, ...
                         'Title', 'Data Preview', ...
                         'FontSize', 11, ...
                         'Position', [0.34, 0.02, 0.64, 0.96]);

    % Variable info
    uicontrol('Parent', right_panel, ...
              'Style', 'text', ...
              'String', 'Variable Information:', ...
              'FontSize', 10, ...
              'Position', [10, 700, 150, 20], ...
              'HorizontalAlignment', 'left');

    var_info_text = uicontrol('Parent', right_panel, ...
                             'Style', 'text', ...
                             'String', 'Select a variable to view information...', ...
                             'FontSize', 9, ...
                             'Position', [10, 600, 400, 90], ...
                             'HorizontalAlignment', 'left', ...
                             'BackgroundColor', [0.95, 0.95, 0.95]);

    % Data preview table
    uicontrol('Parent', right_panel, ...
              'Style', 'text', ...
              'String', 'Data Preview:', ...
              'FontSize', 10, ...
              'Position', [10, 570, 150, 20], ...
              'HorizontalAlignment', 'left');

    % Create table for data preview
    data_table = uitable('Parent', right_panel, ...
                        'Position', [10, 200, 750, 360], ...
                        'ColumnName', {'Time', 'Value'}, ...
                        'Data', cell(10, 2), ...
                        'ColumnWidth', {100, 200});

    % Statistics panel
    stats_panel = uipanel('Parent', right_panel, ...
                         'Title', 'Statistics', ...
                         'FontSize', 10, ...
                         'Position', [0.02, 0.02, 0.96, 0.25]);

    stats_text = uicontrol('Parent', stats_panel, ...
                          'Style', 'text', ...
                          'String', 'Select a variable to view statistics...', ...
                          'FontSize', 9, ...
                          'Position', [10, 10, 700, 150], ...
                          'HorizontalAlignment', 'left', ...
                          'BackgroundColor', [0.95, 0.95, 0.95]);

    % Store handles
    explorer_handles.dataset_popup = dataset_popup;
    explorer_handles.variable_listbox = variable_listbox;
    explorer_handles.search_edit = search_edit;
    explorer_handles.var_info_text = var_info_text;
    explorer_handles.data_table = data_table;
    explorer_handles.stats_text = stats_text;

    fprintf('✅ Data explorer created successfully\n');

end

function update_variable_list(src, ~)
    % Get the selected dataset
    dataset_names = get(src, 'String');
    selected_idx = get(src, 'Value');
    selected_dataset = dataset_names{selected_idx};

    % Get the main figure
    fig = ancestor(src, 'figure');
    variable_listbox = findobj(fig, 'Style', 'listbox');

    % Update variable list based on dataset
    % This would need to be implemented based on actual data structure
    switch selected_dataset
        case 'BASE'
            variables = {'Time', 'ClubHeadSpeed', 'HandForces', 'JointTorques', ...
                        'AngularVelocities', 'Positions', 'Accelerations'};
        case 'ZTCF'
            variables = {'Time', 'ClubHeadSpeed', 'HandForces', 'JointTorques', ...
                        'AngularVelocities', 'Positions', 'Accelerations'};
        case 'DELTA'
            variables = {'Time', 'ClubHeadSpeed', 'HandForces', 'JointTorques', ...
                        'AngularVelocities', 'Positions', 'Accelerations'};
        case 'ZVCF'
            variables = {'Time', 'ClubHeadSpeed', 'HandForces', 'JointTorques', ...
                        'AngularVelocities', 'Positions', 'Accelerations'};
        otherwise
            variables = {'No data available'};
    end

    set(variable_listbox, 'String', variables);
    set(variable_listbox, 'Value', 1);

    fprintf('📊 Updated variable list for dataset: %s\n', selected_dataset);
end

function select_variable(src, ~)
    % Get the selected variable
    variable_names = get(src, 'String');
    selected_idx = get(src, 'Value');
    selected_variable = variable_names{selected_idx};

    % Get the main figure
    fig = ancestor(src, 'figure');

    % Update variable information
    var_info_text = findobj(fig, 'Tag', 'var_info_text');
    if isempty(var_info_text)
        var_info_text = findobj(fig, 'Style', 'text');
        var_info_text = var_info_text(end-2); % Get the variable info text
    end

    % Create variable information
    info_text = sprintf('Variable: %s\n\n', selected_variable);
    info_text = [info_text, 'Description: This variable represents...\n'];
    info_text = [info_text, 'Units: [units]\n'];
    info_text = [info_text, 'Data Type: Numeric\n'];
    info_text = [info_text, 'Dimensions: [rows x columns]\n'];

    set(var_info_text, 'String', info_text);

    % Update data preview
    update_data_preview(fig, selected_variable);

    % Update statistics
    update_statistics(fig, selected_variable);

    fprintf('📈 Selected variable: %s\n', selected_variable);
end

function search_variables(src, ~)
    % Get search term
    search_term = get(src, 'String');

    % Get the main figure
    fig = ancestor(src, 'figure');
    variable_listbox = findobj(fig, 'Style', 'listbox');
    dataset_popup = findobj(fig, 'Style', 'popupmenu');

    % Get current dataset
    dataset_names = get(dataset_popup, 'String');
    selected_idx = get(dataset_popup, 'Value');
    selected_dataset = dataset_names{selected_idx};

    % Filter variables based on search term
    all_variables = get(variable_listbox, 'String');

    if isempty(search_term)
        % Show all variables
        filtered_variables = all_variables;
    else
        % Filter variables containing search term
        filtered_variables = {};
        for i = 1:length(all_variables)
            if contains(lower(all_variables{i}), lower(search_term))
                filtered_variables{end+1} = all_variables{i};
            end
        end
    end

    set(variable_listbox, 'String', filtered_variables);
    if ~isempty(filtered_variables)
        set(variable_listbox, 'Value', 1);
    end

    fprintf('🔍 Searched for: "%s" (found %d variables)\n', search_term, length(filtered_variables));
end

function update_data_preview(fig, variable_name)
    % Get data table
    data_table = findobj(fig, 'Style', 'table');

    % This would extract actual data from the selected variable
    % For now, create sample data
    time_points = linspace(0, 0.28, 10)';
    values = rand(10, 1) * 100; % Sample values

    % Update table data
    table_data = [num2cell(time_points), num2cell(values)];
    set(data_table, 'Data', table_data);

    fprintf('📋 Updated data preview for: %s\n', variable_name);
end

function update_statistics(fig, variable_name)
    % Get statistics text
    stats_text = findobj(fig, 'Style', 'text');
    stats_text = stats_text(end-1); % Get the statistics text

    % This would calculate actual statistics from the data
    % For now, create sample statistics
    stats_info = sprintf('Statistics for: %s\n\n', variable_name);
    stats_info = [stats_info, 'Mean: 45.2\n'];
    stats_info = [stats_info, 'Std Dev: 12.8\n'];
    stats_info = [stats_info, 'Min: 12.3\n'];
    stats_info = [stats_info, 'Max: 78.9\n'];
    stats_info = [stats_info, 'Range: 66.6\n'];
    stats_info = [stats_info, 'Data Points: 2800\n'];

    set(stats_text, 'String', stats_info);

    fprintf('📊 Updated statistics for: %s\n', variable_name);
end

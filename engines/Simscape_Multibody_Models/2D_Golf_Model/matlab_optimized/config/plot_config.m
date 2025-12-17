function config = plot_config()
% PLOT_CONFIG - Configuration for plotting system
%
% Returns:
%   config - Structure containing all plotting parameters
%
% This function centralizes all plotting configuration including:
%   - Figure settings
%   - Line styles and colors
%   - Export settings
%   - Plot organization
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
    end

    %% General Plot Settings
    config.figure_width = 800;            % Figure width in pixels
    config.figure_height = 600;           % Figure height in pixels
    config.figure_dpi = 300;              % DPI for saved figures
    config.pause_time = 0;                % Pause time between plots (0 = no pause)

    %% Line and Marker Settings
    config.line_width = 1.5;              % Default line width
    config.marker_size = 6;               % Default marker size
    config.font_size = 11;                % Default font size
    config.title_font_size = 13;          % Title font size
    config.axis_font_size = 10;           % Axis label font size

    %% Color Scheme
    % Standard colors for joints (consistent across all plots)
    config.colors.LS = [0.0000, 0.4470, 0.7410];  % Left Shoulder - Blue
    config.colors.RS = [0.8500, 0.3250, 0.0980];  % Right Shoulder - Orange
    config.colors.LE = [0.9290, 0.6940, 0.1250];  % Left Elbow - Yellow
    config.colors.RE = [0.4940, 0.1840, 0.5560];  % Right Elbow - Purple
    config.colors.LW = [0.4660, 0.6740, 0.1880];  % Left Wrist - Green
    config.colors.RW = [0.6350, 0.0780, 0.1840];  % Right Wrist - Red

    % Dataset colors
    config.colors.BASE = [0, 0, 0];              % Black
    config.colors.ZTCF = [0, 0.5, 0];            % Dark Green
    config.colors.DELTA = [0.8, 0, 0];           % Dark Red
    config.colors.ZVCF = [0, 0, 0.8];            % Dark Blue

    %% Grid and Legend Settings
    config.show_grid = true;              % Show grid on plots
    config.show_legend = true;            % Show legend on plots
    config.legend_location = 'best';      % Default legend location

    %% Export Settings
    config.export_formats = {'fig', 'png'}; % Export formats
    config.close_after_save = true;       % Close figure after saving

    %% Plot Organization
    % Define plot categories and their subdirectories
    config.plot_categories = {
        'Work_and_Power'
        'Forces_and_Torques'
        'Kinematics'
        'Impulse'
        'Quiver_Plots'
        'Comparison'
    };

    %% Figure Number Assignments
    % Base figure numbers for each dataset
    config.fig_num.BASE = 100;
    config.fig_num.ZTCF = 300;
    config.fig_num.DELTA = 500;
    config.fig_num.ZVCF = 700;
    config.fig_num.COMPARISON = 900;

    %% Plot Type Registry
    % Maps plot types to their categories
    config.plot_registry = struct();

    % Work and Power plots
    config.plot_registry.angular_work = struct('category', 'Work_and_Power', 'name', 'Angular Work');
    config.plot_registry.angular_power = struct('category', 'Work_and_Power', 'name', 'Angular Power');
    config.plot_registry.linear_work = struct('category', 'Work_and_Power', 'name', 'Linear Work');
    config.plot_registry.linear_power = struct('category', 'Work_and_Power', 'name', 'Linear Power');
    config.plot_registry.total_work = struct('category', 'Work_and_Power', 'name', 'Total Work');
    config.plot_registry.total_power = struct('category', 'Work_and_Power', 'name', 'Total Power');

    % Force and Torque plots
    config.plot_registry.joint_torques = struct('category', 'Forces_and_Torques', 'name', 'Joint Torque Inputs');
    config.plot_registry.force_along_path = struct('category', 'Forces_and_Torques', 'name', 'Force Along Hand Path');
    config.plot_registry.equivalent_couple = struct('category', 'Forces_and_Torques', 'name', 'Equivalent Couple and MOF');
    config.plot_registry.local_hand_forces = struct('category', 'Forces_and_Torques', 'name', 'Local Hand Forces');

    % Kinematic plots
    config.plot_registry.kinematic_sequence = struct('category', 'Kinematics', 'name', 'Kinematic Sequence');
    config.plot_registry.club_hand_speed = struct('category', 'Kinematics', 'name', 'Club Head and Hand Speed');

    % Impulse plots
    config.plot_registry.linear_impulse = struct('category', 'Impulse', 'name', 'Linear Impulse');
    config.plot_registry.angular_impulse = struct('category', 'Impulse', 'name', 'Angular Impulse');

    %% Quiver Plot Settings
    config.quiver.scale_factor = 1.0;     % Scale factor for quiver arrows
    config.quiver.auto_scale = true;      % Auto-scale arrows
    config.quiver.line_width = 1.5;       % Arrow line width
    config.quiver.max_head_size = 0.5;    % Maximum arrow head size

    %% Create plot directories
    create_plot_directories(config);

end

function create_plot_directories(config)
    % Create directories for each plot category
    base_plot_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                             'data', 'plots');

    % Create category directories
    for i = 1:length(config.plot_categories)
        category_dir = fullfile(base_plot_dir, config.plot_categories{i});
        % Use try-catch to handle directory creation safely
        try
            if exist(category_dir, 'dir') ~= 7
                mkdir(category_dir);
            end
        catch
            % Ignore errors during directory creation
        end
    end
end

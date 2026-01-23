function config = model_config()
% MODEL_CONFIG - Configuration file for 2D Golf Swing Model
% Returns a structure with all model parameters and settings
%
% Usage:
%   config = model_config();
%
% Returns:
%   config - Structure containing all configuration parameters

    % Model workspace configuration
    config.model_name = 'GolfSwing';
    config.stop_time = 0.28;
    config.max_step = 0.001;
    config.killswitch_time = 1.0; % Default killswitch time (won't trigger)

    % ZTCF generation parameters
    config.ztcf_start_time = 0;
    config.ztcf_end_time = 28; % 0.28 seconds * 100
    config.ztcf_time_scale = 100; % Scale factor for time indexing

    % Data processing parameters
    config.sample_time = 0.0001;
    config.interpolation_method = 'spline';

    % File paths
    config.base_path = matlabdrive;
    config.model_path = fullfile(config.base_path, '2DModel');
    config.scripts_path = fullfile(config.model_path, 'Scripts');
    config.tables_path = fullfile(config.model_path, 'Tables');
    config.output_path = fullfile(config.model_path, 'Model Output');

    % Model parameters
    config.dampening_included = 0; % 0 = included, 1 = excluded
    config.return_workspace_outputs = 'on';
    config.fast_restart = 'on';

    % Warning settings
    config.suppress_warnings = true;

    % Animation settings
    config.animation_fps = 30;
    config.animation_quality = 'high';

    % GUI settings
    config.gui_title = '2D Golf Swing Model - ZTCF/ZVCF Analysis';
    config.gui_width = 1200;
    config.gui_height = 800;

    % Plot settings
    config.plot_line_width = 2;
    config.plot_marker_size = 8;
    config.plot_font_size = 12;

    % Colors
    config.colors.base = [0, 0.4470, 0.7410];      % Blue
    config.colors.ztcf = [0.8500, 0.3250, 0.0980]; % Orange
    config.colors.delta = [0.9290, 0.6940, 0.1250]; % Yellow
    config.colors.zvcf = [0.4940, 0.1840, 0.5560]; % Purple
    config.colors.background = [0.94, 0.94, 0.94]; % Light gray

end

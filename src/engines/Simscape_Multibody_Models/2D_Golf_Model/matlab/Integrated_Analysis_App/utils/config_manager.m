classdef config_manager < handle
    % CONFIG_MANAGER - Manages application configuration and state
    %
    % This class handles saving/loading configuration, window state,
    % and user preferences for the Golf Analysis App.
    %
    % Usage:
    %   cm = config_manager();
    %   config = cm.load_config();
    %   cm.save_config(config);

    properties (Constant)
        CONFIG_FILE = 'golf_analysis_app_config.mat';
    end

    properties (Access = private)
        config_path  % Path to configuration directory
    end

    methods
        function obj = config_manager()
            % Constructor - determine configuration path
            % Try to use user's MATLAB preferences directory
            obj.config_path = fullfile(fileparts(mfilename('fullpath')), '..', 'config');

            % Ensure config directory exists
            if ~exist(obj.config_path, 'dir')
                mkdir(obj.config_path);
            end
        end

        function config = load_config(obj)
            % Load configuration from file, or return defaults
            config_file = fullfile(obj.config_path, obj.CONFIG_FILE);

            if exist(config_file, 'file')
                try
                    loaded = load(config_file);
                    config = loaded.config;
                    fprintf('Configuration loaded from: %s\n', config_file);
                catch ME
                    warning('config_manager:LoadFailed', ...
                        'Failed to load config: %s. Using defaults.', ME.message);
                    config = obj.get_default_config();
                end
            else
                config = obj.get_default_config();
                fprintf('Using default configuration\n');
            end
        end

        function save_config(obj, config)
            % Save configuration to file
            config_file = fullfile(obj.config_path, obj.CONFIG_FILE);

            try
                save(config_file, 'config');
                fprintf('Configuration saved to: %s\n', config_file);
            catch ME
                warning('config_manager:SaveFailed', ...
                    'Failed to save config: %s', ME.message);
            end
        end

        function config = get_default_config(~)
            % Return default configuration structure
            config = struct();

            % Window settings
            config.window = struct();
            config.window.position = [100, 100, 1400, 800];  % [x, y, width, height]
            config.window.last_active_tab = 1;

            % Tab 1 settings (Model Setup)
            config.tab1 = struct();
            config.tab1.last_model_file = '';
            config.tab1.auto_run_sim = false;

            % Tab 2 settings (ZTCF Calculation)
            config.tab2 = struct();
            config.tab2.num_iterations = 10;
            config.tab2.use_parallel = true;
            config.tab2.last_output_dir = '';

            % Tab 3 settings (Visualization)
            config.tab3 = struct();
            config.tab3.skeleton_config = struct();
            config.tab3.signal_plot_config = struct();
            config.tab3.last_data_file = '';

            % General settings
            config.general = struct();
            config.general.auto_save_session = true;
            config.general.session_save_interval = 300;  % seconds
            config.general.confirm_on_exit = true;
        end

        function update_window_state(obj, config, main_fig)
            % Update configuration with current window state
            if ishandle(main_fig)
                config.window.position = get(main_fig, 'Position');
            end
        end

        function apply_window_state(~, config, main_fig)
            % Apply saved window state to figure
            if ishandle(main_fig) && isfield(config, 'window')
                if isfield(config.window, 'position')
                    try
                        set(main_fig, 'Position', config.window.position);
                    catch
                        % Window position may be invalid (off-screen), ignore
                    end
                end
            end
        end

        function reset_config(obj)
            % Reset configuration to defaults
            config = obj.get_default_config();
            obj.save_config(config);
            fprintf('Configuration reset to defaults\n');
        end

        function export_config(obj, export_file)
            % Export current configuration to specified file
            config = obj.load_config();
            save(export_file, 'config');
            fprintf('Configuration exported to: %s\n', export_file);
        end

        function import_config(obj, import_file)
            % Import configuration from specified file
            if ~exist(import_file, 'file')
                error('config_manager:FileNotFound', ...
                    'Config file not found: %s', import_file);
            end

            loaded = load(import_file);
            if ~isfield(loaded, 'config')
                error('config_manager:InvalidFile', ...
                    'File does not contain valid configuration');
            end

            obj.save_config(loaded.config);
            fprintf('Configuration imported from: %s\n', import_file);
        end
    end
end

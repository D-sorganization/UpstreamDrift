function config = simulation_config()
% SIMULATION_CONFIG - Centralized configuration for golf swing simulation
%
% Returns:
%   config - Structure containing all simulation parameters
%
% This function provides a single source of truth for all simulation
% settings, making it easy to modify parameters without editing multiple files.
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
    end

    %% Model Configuration
    config.model_name = 'GolfSwing';
    config.model_zvcf_name = 'GolfSwingZVCF';

    %% Path Configuration
    % Use relative paths from the matlab_optimized directory
    config.base_dir = fileparts(fileparts(mfilename('fullpath')));
    config.legacy_scripts_path = fullfile(config.base_dir, '..', 'matlab', 'Scripts');
    config.model_path = fullfile(config.base_dir, '..', 'matlab', '2DModel');
    config.output_path = fullfile(config.base_dir, 'data', 'output');
    config.plots_path = fullfile(config.base_dir, 'data', 'plots');
    config.cache_path = fullfile(config.base_dir, 'data', 'cache');

    %% Simulation Parameters
    config.stop_time = 0.28;              % Simulation stop time (seconds)
    config.max_step = 0.001;              % Maximum simulation step size (seconds)
    config.killswitch_time = 1.0;         % Default killswitch time (disabled)
    config.kill_damp_final_value = 0;     % 0 = dampening included, 1 = excluded

    %% ZTCF Generation Parameters
    config.ztcf_start_time = 0;           % Start time index
    config.ztcf_end_time = 28;            % End time index
    config.ztcf_time_scale = 100;         % Scale factor (j = i/100)
    config.ztcf_num_points = config.ztcf_end_time - config.ztcf_start_time + 1;

    %% Data Processing Parameters
    config.base_sample_time = 0.0001;     % Base table sample time (100 μs)
    config.q_sample_time = 0.0025;        % Q-table sample time for plotting (2.5 ms)
    config.interpolation_method = 'spline'; % Interpolation method for resampling

    %% Parallel Processing Configuration
    config.use_parallel = true;           % Enable parallel processing
    config.num_workers = [];              % Auto-detect (use [] for automatic)
    config.parallel_batch_size = [];      % Auto-optimize batch size

    %% Checkpointing Configuration
    config.enable_checkpoints = true;     % Enable checkpoint saving
    config.checkpoint_stages = {
        'base_data'
        'ztcf_data'
        'processed_tables'
        'zvcf_data'
    };

    %% Output Configuration
    config.save_tables = true;            % Save data tables to .mat files
    config.save_plots = true;             % Save plots to files
    config.save_format = 'mat';           % Format for data tables
    config.verbose = true;                % Enable verbose output
    config.show_progress = true;          % Show progress bars

    %% Validation Configuration
    config.validate_results = false;      % Compare with legacy results
    config.tolerance = 1e-10;             % Numerical tolerance for validation

    %% Create output directories if they don't exist
    create_output_directories(config);

end

function create_output_directories(config)
    arguments
        config struct
    end
    % Create all necessary output directories
    dirs_to_create = {
        config.output_path
        config.plots_path
        config.cache_path
    };

    for i = 1:length(dirs_to_create)
        try
            if ~isfolder(dirs_to_create{i})
                mkdir(dirs_to_create{i});
            end
        catch
            % Ignore errors during directory creation
        end
    end
end

function performance_optimizer()
    % PERFORMANCE_OPTIMIZER - Performance optimization utilities for the GUI
    %
    % This module provides performance optimization functions including:
    % - Preallocation for data structures
    % - Memory management and pooling
    % - Data compression and processing optimizations
    % - Model caching and configuration management
    %
    % Usage:
    %   performance_optimizer()  % Show help
    %   data_table = preallocateDataTable(num_trials, num_time_points, config)
    %   memory_info = getMemoryUsage()
    %   compressed_data = compressData(data_table, level)
    %   model_config = cacheModelConfiguration(model_path, config)

    fprintf('Performance Optimizer Module\n');
    fprintf('===========================\n');
    fprintf('Available functions:\n');
    fprintf('  preallocateDataTable(num_trials, num_time_points, config)\n');
    fprintf('  getMemoryUsage()\n');
    fprintf('  compressData(data_table, level)\n');
    fprintf('  cacheModelConfiguration(model_path, config)\n');
    fprintf('  optimizeSimulationParameters(config)\n');
    fprintf('  preallocateSignalArrays(signal_info, num_time_points)\n');
end

function data_table = preallocateDataTable(num_trials, num_time_points, config)
    % PREALLOCATEDATATABLE - Preallocate data table with estimated size
    %
    % Inputs:
    %   num_trials - Number of trials to preallocate for
    %   num_time_points - Number of time points per trial
    %   config - Configuration structure with data extraction settings
    %
    % Outputs:
    %   data_table - Preallocated table with estimated columns

    fprintf('Preallocating data table for %d trials x %d time points...\n', num_trials, num_time_points);

    % Estimate number of columns based on configuration
    estimated_columns = estimateDataColumns(config);

    % Create preallocated table
    total_rows = num_trials * num_time_points;

    % Initialize with basic columns
    data_table = table();
    data_table.trial_id = zeros(total_rows, 1);
    data_table.time = zeros(total_rows, 1);

    % Preallocate signal columns based on configuration
    if config.use_signal_bus
        % Estimate signal bus columns
        signal_columns = estimateSignalBusColumns(config);
        for i = 1:length(signal_columns)
            data_table.(signal_columns{i}) = zeros(total_rows, 1);
        end
    end

    if config.use_logsout
        % Estimate logsout columns
        logsout_columns = estimateLogsoutColumns(config);
        for i = 1:length(logsout_columns)
            data_table.(logsout_columns{i}) = zeros(total_rows, 1);
        end
    end

    if config.use_simscape
        % Estimate Simscape columns
        simscape_columns = estimateSimscapeColumns(config);
        for i = 1:length(simscape_columns)
            data_table.(simscape_columns{i}) = zeros(total_rows, 1);
        end
    end

    fprintf('Preallocated table with %d columns and %d rows\n', width(data_table), height(data_table));
end

function num_columns = estimateDataColumns(config)
    % ESTIMATEDATACOLUMNS - Estimate number of data columns based on configuration

    num_columns = 2; % trial_id and time

    if config.use_signal_bus
        num_columns = num_columns + 50; % Conservative estimate for signal bus
    end

    if config.use_logsout
        num_columns = num_columns + 30; % Conservative estimate for logsout
    end

    if config.use_simscape
        num_columns = num_columns + 40; % Conservative estimate for Simscape
    end

    if config.capture_workspace
        num_columns = num_columns + 20; % Conservative estimate for workspace variables
    end
end

function signal_columns = estimateSignalBusColumns(config)
    % ESTIMATESIGNALBUSCOLUMNS - Estimate signal bus column names

    % Common signal bus patterns
    signal_columns = {
        'HipLogs_x', 'HipLogs_y', 'HipLogs_z',
        'SpineLogs_x', 'SpineLogs_y', 'SpineLogs_z',
        'ShoulderLogs_x', 'ShoulderLogs_y', 'ShoulderLogs_z',
        'ElbowLogs_x', 'ElbowLogs_y', 'ElbowLogs_z',
        'WristLogs_x', 'WristLogs_y', 'WristLogs_z',
        'ClubLogs_x', 'ClubLogs_y', 'ClubLogs_z',
        'AngularVelocity_x', 'AngularVelocity_y', 'AngularVelocity_z',
        'LinearVelocity_x', 'LinearVelocity_y', 'LinearVelocity_z',
        'AngularAcceleration_x', 'AngularAcceleration_y', 'AngularAcceleration_z',
        'LinearAcceleration_x', 'LinearAcceleration_y', 'LinearAcceleration_z',
        'Force_x', 'Force_y', 'Force_z',
        'Torque_x', 'Torque_y', 'Torque_z',
        'Energy', 'Power', 'Work'
    };
end

function logsout_columns = estimateLogsoutColumns(config)
    % ESTIMATELOGSOUTCOLUMNS - Estimate logsout column names

    logsout_columns = {
        'Position_x', 'Position_y', 'Position_z',
        'Velocity_x', 'Velocity_y', 'Velocity_z',
        'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
        'AngularPosition_x', 'AngularPosition_y', 'AngularPosition_z',
        'AngularVelocity_x', 'AngularVelocity_y', 'AngularVelocity_z',
        'AngularAcceleration_x', 'AngularAcceleration_y', 'AngularAcceleration_z',
        'JointTorque_x', 'JointTorque_y', 'JointTorque_z',
        'JointForce_x', 'JointForce_y', 'JointForce_z',
        'ContactForce_x', 'ContactForce_y', 'ContactForce_z',
        'Momentum_x', 'Momentum_y', 'Momentum_z'
    };
end

function simscape_columns = estimateSimscapeColumns(config)
    % ESTIMATESIMSCAPECOLUMNS - Estimate Simscape column names

    simscape_columns = {
        'JointPosition', 'JointVelocity', 'JointAcceleration',
        'JointTorque', 'JointForce',
        'BodyPosition_x', 'BodyPosition_y', 'BodyPosition_z',
        'BodyVelocity_x', 'BodyVelocity_y', 'BodyVelocity_z',
        'BodyAcceleration_x', 'BodyAcceleration_y', 'BodyAcceleration_z',
        'BodyAngularVelocity_x', 'BodyAngularVelocity_y', 'BodyAngularVelocity_z',
        'BodyAngularAcceleration_x', 'BodyAngularAcceleration_y', 'BodyAngularAcceleration_z',
        'ContactForce_x', 'ContactForce_y', 'ContactForce_z',
        'ContactTorque_x', 'ContactTorque_y', 'ContactTorque_z',
        'Energy', 'KineticEnergy', 'PotentialEnergy',
        'Power', 'Work', 'Efficiency'
    };
end

function memory_info = getMemoryUsage()
    % GETMEMORYUSAGE - Get current memory usage information

    try
        % Get MATLAB memory info
        memory_info = memory;

        % Add additional memory metrics
        if isfield(memory_info, 'PhysicalMemory')
            memory_info.available_gb = memory_info.PhysicalMemory.Available / (1024^3);
            memory_info.used_gb = (memory_info.PhysicalMemory.Total - memory_info.PhysicalMemory.Available) / (1024^3);
            memory_info.total_gb = memory_info.PhysicalMemory.Total / (1024^3);
            memory_info.usage_percent = (memory_info.used_gb / memory_info.total_gb) * 100;
        else
            memory_info.available_gb = NaN;
            memory_info.used_gb = NaN;
            memory_info.total_gb = NaN;
            memory_info.usage_percent = NaN;
        end

        % Get workspace memory usage
        workspace_vars = whos;
        memory_info.workspace_mb = sum([workspace_vars.bytes]) / (1024^2);
        memory_info.num_variables = length(workspace_vars);

    catch
        % Fallback for older MATLAB versions
        memory_info = struct();
        memory_info.available_gb = NaN;
        memory_info.used_gb = NaN;
        memory_info.total_gb = NaN;
        memory_info.usage_percent = NaN;
        memory_info.workspace_mb = NaN;
        memory_info.num_variables = NaN;
    end
end

function compressed_data = compressData(data_table, level)
    % COMPRESSDATA - Compress data table to reduce memory usage
    %
    % Inputs:
    %   data_table - Input data table
    %   level - Compression level (1-9, higher = more compression)
    %
    % Outputs:
    %   compressed_data - Compressed data structure

    if nargin < 2
        level = 6; % Default compression level
    end

    fprintf('Compressing data table with level %d...\n', level);

    % Convert table to structure for compression
    data_struct = table2struct(data_table);

    % Compress using MATLAB's built-in compression
    compressed_data = struct();
    compressed_data.compressed = true;
    compressed_data.compression_level = level;
    compressed_data.original_size = whos('data_struct');
    compressed_data.original_size = compressed_data.original_size.bytes;

    % Compress each field separately for better compression ratios
    fields = fieldnames(data_struct);
    for i = 1:length(fields)
        field_name = fields{i};
        field_data = data_struct.(field_name);

        % Compress numeric data
        if isnumeric(field_data)
            compressed_data.(field_name) = compressNumericData(field_data, level);
        else
            compressed_data.(field_name) = field_data;
        end
    end

    % Calculate compression ratio
    compressed_size = whos('compressed_data');
    compressed_size = compressed_size.bytes;
    compression_ratio = (1 - compressed_size / compressed_data.original_size) * 100;

    fprintf('Compression complete: %.1f%% reduction (%.1f MB -> %.1f MB)\n', ...
        compression_ratio, compressed_data.original_size/(1024^2), compressed_size/(1024^2));
end

function compressed_numeric = compressNumericData(numeric_data, level)
    % COMPRESSNUMERICDATA - Compress numeric data using various techniques

    % Use single precision if possible to reduce memory
    if isa(numeric_data, 'double') && all(abs(numeric_data) < 3.4e38)
        compressed_numeric = single(numeric_data);
    else
        compressed_numeric = numeric_data;
    end

    % Apply additional compression techniques based on data characteristics
    if isvector(compressed_numeric) && length(compressed_numeric) > 1000
        % For large vectors, use delta encoding if beneficial
        if std(compressed_numeric) < mean(abs(compressed_numeric)) * 0.1
            compressed_numeric = [compressed_numeric(1); diff(compressed_numeric)];
        end
    end
end

function model_config = cacheModelConfiguration(model_path, config)
    % CACHEMODELCONFIGURATION - Cache model configuration for faster loading
    %
    % Inputs:
    %   model_path - Path to Simulink model
    %   config - Configuration structure
    %
    % Outputs:
    %   model_config - Cached model configuration

    fprintf('Caching model configuration for %s...\n', model_path);

    % Create cache file path
    [cache_dir, model_name, ~] = fileparts(model_path);
    cache_file = fullfile(cache_dir, [model_name '_config_cache.mat']);

    % Check if cache exists and is newer than model
    if exist(cache_file, 'file')
        model_info = dir(model_path);
        cache_info = dir(cache_file);

        if cache_info.datenum > model_info.datenum
            % Load cached configuration
            try
                cached_data = load(cache_file);
                model_config = cached_data.model_config;
                fprintf('Loaded cached configuration\n');
                return;
            catch
                fprintf('Cache file corrupted, regenerating...\n');
            end
        end
    end

    % Generate new configuration cache
    model_config = struct();
    model_config.model_path = model_path;
    model_config.model_name = model_name;
    model_config.cache_timestamp = now();

    % Cache model parameters
    try
        if ~bdIsLoaded(model_name)
            load_system(model_path);
            model_was_loaded = true;
        else
            model_was_loaded = false;
        end

        % Cache model workspace variables
        model_workspace = get_param(model_name, 'ModelWorkspace');
        try
            variables = model_workspace.getVariableNames;
            model_config.workspace_variables = variables;
        catch
            model_config.workspace_variables = {};
        end

        % Cache model configuration parameters
        model_config.solver = get_param(model_name, 'Solver');
        model_config.stop_time = get_param(model_name, 'StopTime');
        model_config.sample_time = get_param(model_name, 'SampleTime');

        % Cache signal logging configuration
        model_config.logging_config = get_param(model_name, 'DataLogging');
        model_config.logging_format = get_param(model_name, 'SaveFormat');

        if model_was_loaded
            close_system(model_name, 0);
        end

    catch ME
        fprintf('Warning: Could not cache model configuration: %s\n', ME.message);
        model_config.error = ME.message;
    end

    % Save cache
    try
        save(cache_file, 'model_config');
        fprintf('Configuration cached to %s\n', cache_file);
    catch ME
        fprintf('Warning: Could not save configuration cache: %s\n', ME.message);
    end
end

function optimized_config = optimizeSimulationParameters(config)
    % OPTIMIZESIMULATIONPARAMETERS - Optimize simulation parameters for performance
    %
    % Inputs:
    %   config - Original configuration
    %
    % Outputs:
    %   optimized_config - Optimized configuration

    fprintf('Optimizing simulation parameters...\n');

    optimized_config = config;

    % Optimize solver settings
    if isfield(config, 'solver_type')
        switch config.solver_type
            case 'ode45'
                optimized_config.solver = 'ode45';
                optimized_config.rel_tol = 1e-3; % Relaxed tolerance for speed
                optimized_config.abs_tol = 1e-6;
            case 'ode23'
                optimized_config.solver = 'ode23';
                optimized_config.rel_tol = 1e-3;
                optimized_config.abs_tol = 1e-6;
            otherwise
                optimized_config.solver = 'ode45';
                optimized_config.rel_tol = 1e-3;
                optimized_config.abs_tol = 1e-6;
        end
    end

    % Optimize data logging settings
    optimized_config.save_format = 'Structure'; % Faster than Dataset
    optimized_config.return_workspace_outputs = 'on';
    optimized_config.signal_logging = 'on';

    % Optimize for memory efficiency
    optimized_config.enable_data_compression = true;
    optimized_config.compression_level = 6;

    % Optimize parallel processing
    if isfield(config, 'enable_parallel_processing') && config.enable_parallel_processing
        % Use user's local cluster profile
        optimized_config.cluster_profile = 'Local_Cluster';

        % Use user preference if set, otherwise use all available cores
        if isfield(config, 'max_parallel_workers') && config.max_parallel_workers > 0
            optimized_config.max_parallel_workers = min(config.max_parallel_workers, feature('numcores'));
        else
            optimized_config.max_parallel_workers = feature('numcores');
        end

        % Set cluster-specific optimizations
        optimized_config.use_local_cluster = true;
        optimized_config.cluster_name = 'Local_Cluster';
    end

    fprintf('Simulation parameters optimized\n');
end

function signal_arrays = preallocateSignalArrays(signal_info, num_time_points)
    % PREALLOCATESIGNALARRAYS - Preallocate arrays for signal data
    %
    % Inputs:
    %   signal_info - Signal information structure
    %   num_time_points - Number of time points
    %
    % Outputs:
    %   signal_arrays - Preallocated signal arrays

    fprintf('Preallocating signal arrays for %d time points...\n', num_time_points);

    signal_arrays = struct();

    if isfield(signal_info, 'signal_bus_signals')
        for i = 1:length(signal_info.signal_bus_signals)
            signal_name = signal_info.signal_bus_signals{i};
            signal_arrays.(signal_name) = zeros(num_time_points, 1);
        end
    end

    if isfield(signal_info, 'logsout_signals')
        for i = 1:length(signal_info.logsout_signals)
            signal_name = signal_info.logsout_signals{i};
            signal_arrays.(signal_name) = zeros(num_time_points, 1);
        end
    end

    if isfield(signal_info, 'simscape_signals')
        for i = 1:length(signal_info.simscape_signals)
            signal_name = signal_info.simscape_signals{i};
            signal_arrays.(signal_name) = zeros(num_time_points, 1);
        end
    end

    fprintf('Preallocated %d signal arrays\n', length(fieldnames(signal_arrays)));
end

function cluster_info = initializeLocalCluster(config)
    % INITIALIZELOCALCLUSTER - Initialize and configure local cluster for parallel processing
    %
    % Inputs:
    %   config - Configuration structure with cluster settings
    %
    % Outputs:
    %   cluster_info - Cluster information and status

    fprintf('Initializing Local_Cluster for parallel processing...\n');

    cluster_info = struct();
    cluster_info.cluster_name = 'Local_Cluster';
    cluster_info.status = 'initializing';

    try
        % Check if Parallel Computing Toolbox is available
        if ~license('test', 'Distrib_Computing_Toolbox')
            error('Parallel Computing Toolbox not available');
        end

        % Get cluster profile
        cluster_profiles = parallel.clusterProfiles();
        if ~ismember('Local_Cluster', cluster_profiles)
            error('Local_Cluster profile not found. Available profiles: %s', strjoin(cluster_profiles, ', '));
        end

        % Create cluster object
        cluster_obj = parcluster('Local_Cluster');
        cluster_info.cluster_object = cluster_obj;

        % Configure cluster settings
        if isfield(config, 'max_parallel_workers') && config.max_parallel_workers > 0
            cluster_obj.NumWorkers = min(config.max_parallel_workers, feature('numcores'));
        else
            cluster_obj.NumWorkers = feature('numcores');
        end

        % Set additional cluster properties if available
        if isprop(cluster_obj, 'NumThreads')
            cluster_obj.NumThreads = 1; % Use 1 thread per worker for better performance
        end

        % Test cluster connection
        fprintf('Testing cluster connection with %d workers...\n', cluster_obj.NumWorkers);

        % Start a small test job
        test_job = batch(cluster_obj, @() 1, 1, {}, 'Pool', 1);
        wait(test_job);

        if strcmp(test_job.State, 'finished')
            cluster_info.status = 'ready';
            cluster_info.num_workers = cluster_obj.NumWorkers;
            cluster_info.test_successful = true;
            fprintf('✓ Local_Cluster initialized successfully with %d workers\n', cluster_obj.NumWorkers);
        else
            cluster_info.status = 'error';
            cluster_info.error_message = sprintf('Cluster test failed: %s', test_job.State);
            fprintf('✗ Cluster test failed: %s\n', test_job.State);
        end

        % Clean up test job
        delete(test_job);

    catch ME
        cluster_info.status = 'error';
        cluster_info.error_message = ME.message;
        fprintf('✗ Failed to initialize Local_Cluster: %s\n', ME.message);
    end
end

function pool = getOrCreateParallelPool(config)
    % GETORCREATEPARALLELPOOL - Get existing pool or create new one using Local_Cluster
    %
    % Inputs:
    %   config - Configuration structure with pool settings
    %
    % Outputs:
    %   pool - Parallel pool object

    fprintf('Setting up parallel pool using Local_Cluster...\n');

    % Check for existing pool
    pool = gcp('nocreate');

    if isempty(pool)
        % Create new pool using Local_Cluster
        try
            % Get cluster object
            cluster_obj = parcluster('Local_Cluster');

            % Configure cluster
            if isfield(config, 'max_parallel_workers') && config.max_parallel_workers > 0
                num_workers = min(config.max_parallel_workers, feature('numcores'));
            else
                num_workers = feature('numcores');
            end

            % Create pool
            pool = parpool(cluster_obj, num_workers);
            fprintf('✓ Created parallel pool with %d workers using Local_Cluster\n', num_workers);

        catch ME
            fprintf('✗ Failed to create parallel pool: %s\n', ME.message);
            fprintf('Falling back to default parallel pool...\n');

            % Fallback to default pool
            if isfield(config, 'max_parallel_workers') && config.max_parallel_workers > 0
                num_workers = min(config.max_parallel_workers, feature('numcores'));
            else
                num_workers = feature('numcores');
            end

            pool = parpool('local', num_workers);
            fprintf('✓ Created fallback parallel pool with %d workers\n', num_workers);
        end
    else
        fprintf('✓ Using existing parallel pool with %d workers\n', pool.NumWorkers);
    end
end

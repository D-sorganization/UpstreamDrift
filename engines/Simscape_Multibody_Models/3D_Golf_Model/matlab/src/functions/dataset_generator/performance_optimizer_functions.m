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
            fprintf('✓ Created parallel pool with %d workers\n', num_workers);

        catch ME
            fprintf('✗ Failed to create parallel pool: %s\n', ME.message);
            pool = [];
        end
    else
        fprintf('✓ Using existing parallel pool with %d workers\n', pool.NumWorkers);
    end
end

function memory_info = getMemoryUsage()
    % GETMEMORYUSAGE - Get current system memory usage information
    %
    % Outputs:
    %   memory_info - Structure with memory usage details

    try
        % Get memory information
        memory_data = memory;

        memory_info = struct();

        % Check if PhysicalMemory field exists before accessing
        if isfield(memory_data, 'PhysicalMemory')
            memory_info.total_gb = memory_data.PhysicalMemory.Total / (1024^3);
            memory_info.available_gb = memory_data.PhysicalMemory.Available / (1024^3);
            memory_info.used_gb = memory_info.total_gb - memory_info.available_gb;
            memory_info.usage_percent = (memory_info.used_gb / memory_info.total_gb) * 100;
        else
            memory_info.total_gb = NaN;
            memory_info.available_gb = NaN;
            memory_info.used_gb = NaN;
            memory_info.usage_percent = NaN;
        end

        % Virtual memory
        if isfield(memory_data, 'VirtualAddressSpace')
            memory_info.virtual_total_gb = memory_data.VirtualAddressSpace.Total / (1024^3);
            memory_info.virtual_available_gb = memory_data.VirtualAddressSpace.Available / (1024^3);
        else
            memory_info.virtual_total_gb = NaN;
            memory_info.virtual_available_gb = NaN;
        end

        % MATLAB workspace memory
        if isfield(memory_data, 'MATLAB')
            memory_info.matlab_used_gb = memory_data.MATLAB.Used / (1024^3);
            memory_info.matlab_peak_gb = memory_data.MATLAB.Peak / (1024^3);
        else
            memory_info.matlab_used_gb = NaN;
            memory_info.matlab_peak_gb = NaN;
        end

    catch ME
        fprintf('Error getting memory info: %s\n', ME.message);
        memory_info = struct();
        memory_info.total_gb = NaN;
        memory_info.available_gb = NaN;
        memory_info.used_gb = NaN;
        memory_info.usage_percent = NaN;
    end
end

function compressed_data = compressData(data_table, level)
    % COMPRESSDATA - Compress data table based on compression level
    %
    % Inputs:
    %   data_table - Input data table
    %   level - Compression level (1-10, higher = more compression)
    %
    % Outputs:
    %   compressed_data - Compressed data table

    if level <= 1
        compressed_data = data_table;
        return;
    end

    % Apply compression based on level
    if level <= 3
        % Light compression - remove small variations
        compressed_data = removeSmallVariations(data_table, 0.01);
    elseif level <= 6
        % Medium compression - downsample and smooth
        compressed_data = downsampleAndSmooth(data_table, 2);
    else
        % Heavy compression - significant downsampling
        compressed_data = downsampleAndSmooth(data_table, 4);
    end
end

function data_table = removeSmallVariations(data_table, threshold)
    % Remove small variations in numerical data

    % Get numeric columns
    numeric_cols = varfun(@isnumeric, data_table, 'OutputFormat', 'cell');

    for i = 1:width(data_table)
        if numeric_cols{i}
            col_data = data_table.(i);
            if isnumeric(col_data)
                % Apply threshold filtering
                col_data(abs(col_data) < threshold) = 0;
                data_table.(i) = col_data;
            end
        end
    end
end

function data_table = downsampleAndSmooth(data_table, factor)
    % Downsample data by factor and apply smoothing

    if factor <= 1
        return;
    end

    % Get original size
    original_size = height(data_table);
    new_size = ceil(original_size / factor);

    % Create indices for downsampling
    indices = 1:factor:original_size;
    if length(indices) > new_size
        indices = indices(1:new_size);
    end

    % Downsample
    data_table = data_table(indices, :);
end

function config = cacheModelConfiguration(model_path, config)
    % CACHEMODELCONFIGURATION - Cache model configuration for faster loading
    %
    % Inputs:
    %   model_path - Path to Simulink model
    %   config - Configuration structure
    %
    % Outputs:
    %   config - Updated configuration with cached settings

    try
        % Load model if not already loaded
        if ~bdIsLoaded(bdroot(model_path))
            load_system(model_path);
        end

        % Cache model parameters
        model_name = bdroot(model_path);
        config.cached_model_name = model_name;
        config.cached_solver = get_param(model_name, 'Solver');
        config.cached_stop_time = get_param(model_name, 'StopTime');
        config.cached_max_step = get_param(model_name, 'MaxStep');
        config.cached_relative_tolerance = get_param(model_name, 'RelTol');

        fprintf('✓ Cached configuration for model: %s\n', model_name);

    catch ME
        fprintf('✗ Failed to cache model configuration: %s\n', ME.message);
    end
end

function config = optimizeSimulationParameters(config)
    % OPTIMIZESIMULATIONPARAMETERS - Optimize simulation parameters for performance
    %
    % Inputs:
    %   config - Configuration structure
    %
    % Outputs:
    %   config - Updated configuration with optimized parameters

    fprintf('Optimizing simulation parameters...\n');

    % Use max_parallel_workers from config instead of hardcoded limit
    if isfield(config, 'max_parallel_workers') && config.max_parallel_workers > 0
        max_workers = min(config.max_parallel_workers, feature('numcores'));
    else
        max_workers = feature('numcores');
    end

    % Set cluster profile to Local_Cluster
    config.cluster_profile = 'Local_Cluster';

    % Optimize solver settings
    config.optimized_solver = 'ode45';  % Fast solver for most cases
    config.optimized_max_step = 'auto';
    config.optimized_relative_tolerance = 1e-3;  % Slightly relaxed for speed
    config.optimized_absolute_tolerance = 1e-6;

    % Memory optimization
    config.enable_preallocation = true;
    config.enable_data_compression = true;
    config.compression_level = 3;  % Moderate compression

    % Parallel processing
    config.enable_parallel_processing = true;
    config.max_parallel_workers = max_workers;
    config.use_local_cluster = true;

    fprintf('✓ Optimized for %d parallel workers\n', max_workers);
end

function preallocateSignalArrays(signal_info, num_time_points)
    % PREALLOCATESIGNALARRAYS - Preallocate arrays for signal data
    %
    % Inputs:
    %   signal_info - Signal information structure
    %   num_time_points - Number of time points to preallocate for

    fprintf('Preallocating signal arrays for %d time points...\n', num_time_points);

    % Create signal arrays structure
    signal_arrays = struct();

    % Preallocate for each signal type
    if isfield(signal_info, 'position_signals')
        for i = 1:length(signal_info.position_signals)
            signal_name = signal_info.position_signals{i};
            signal_arrays.(signal_name) = zeros(num_time_points, 3);  % 3D position
        end
    end

    if isfield(signal_info, 'velocity_signals')
        for i = 1:length(signal_info.velocity_signals)
            signal_name = signal_info.velocity_signals{i};
            signal_arrays.(signal_name) = zeros(num_time_points, 3);  % 3D velocity
        end
    end

    if isfield(signal_info, 'force_signals')
        for i = 1:length(signal_info.force_signals)
            signal_name = signal_info.force_signals{i};
            signal_arrays.(signal_name) = zeros(num_time_points, 3);  % 3D force
        end
    end

    if isfield(signal_info, 'torque_signals')
        for i = 1:length(signal_info.torque_signals)
            signal_name = signal_info.torque_signals{i};
            signal_arrays.(signal_name) = zeros(num_time_points, 3);  % 3D torque
        end
    end

    fprintf('Preallocated %d signal arrays\n', length(fieldnames(signal_arrays)));
end

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

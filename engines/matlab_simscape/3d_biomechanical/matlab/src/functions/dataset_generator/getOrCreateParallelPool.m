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
            fprintf('✓ Created new parallel pool with %d workers\n', num_workers);
        catch ME
            fprintf('✗ Failed to create parallel pool: %s\n', ME.message);
            pool = [];
        end
    else
        fprintf('✓ Using existing parallel pool with %d workers\n', pool.NumWorkers);
    end
end

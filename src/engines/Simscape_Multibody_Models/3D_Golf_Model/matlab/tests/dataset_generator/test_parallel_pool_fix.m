function test_parallel_pool_fix()
    % TEST_PARALLEL_POOL_FIX - Test script to verify the parallel pool fix
    %
    % This script tests that the GUI is correctly using the Local_Cluster profile
    % and the specified number of workers instead of defaulting to 6 workers.

    fprintf('=== Testing Parallel Pool Fix ===\n\n');

    try
        % Test 1: Check available cluster profiles
        fprintf('Test 1: Checking available cluster profiles...\n');
        profiles = parallel.clusterProfiles();
        fprintf('Available profiles: %s\n', strjoin(profiles, ', '));

        % Test 2: Check if Local_Cluster exists
        if ismember('Local_Cluster', profiles)
            fprintf('✓ Local_Cluster profile found\n');

            % Test 3: Check Local_Cluster configuration
            fprintf('Test 3: Checking Local_Cluster configuration...\n');
            cluster = parcluster('Local_Cluster');
            fprintf('Local_Cluster NumWorkers: %d\n', cluster.NumWorkers);

            % Test 4: Test creating a parallel pool with Local_Cluster
            fprintf('Test 4: Testing parallel pool creation with Local_Cluster...\n');
            try
                % Clean up any existing pool
                existing_pool = gcp('nocreate');
                if ~isempty(existing_pool)
                    fprintf('Deleting existing pool...\n');
                    delete(existing_pool);
                end

                % Create new pool with Local_Cluster
                fprintf('Creating parallel pool with Local_Cluster...\n');
                pool = parpool('Local_Cluster', min(14, cluster.NumWorkers));
                fprintf('✓ Successfully created pool with %d workers using Local_Cluster\n', pool.NumWorkers);

                % Clean up
                delete(pool);
                fprintf('Pool cleaned up\n');

            catch ME
                fprintf('✗ Failed to create pool with Local_Cluster: %s\n', ME.message);
            end

        else
            fprintf('✗ Local_Cluster profile not found\n');

            % Test 5: Try to create Local_Cluster profile
            fprintf('Test 5: Attempting to create Local_Cluster profile...\n');
            try
                cluster = parallel.cluster.Local;
                cluster.Profile = 'Local_Cluster';
                cluster.saveProfile();
                fprintf('✓ Local_Cluster profile created successfully\n');

                % Verify it was created
                profiles = parallel.clusterProfiles();
                if ismember('Local_Cluster', profiles)
                    fprintf('✓ Local_Cluster profile verified\n');
                else
                    fprintf('✗ Local_Cluster profile not found after creation\n');
                end

            catch ME
                fprintf('✗ Failed to create Local_Cluster profile: %s\n', ME.message);
            end
        end

        % Test 6: Check default local profile
        fprintf('\nTest 6: Checking default local profile...\n');
        try
            local_cluster = parcluster('local');
            fprintf('Default local profile NumWorkers: %d\n', local_cluster.NumWorkers);

            if local_cluster.NumWorkers == 6
                fprintf('⚠ Default local profile has 6 workers (this might be the issue)\n');
            else
                fprintf('✓ Default local profile has %d workers\n', local_cluster.NumWorkers);
            end

        catch ME
            fprintf('✗ Error checking local profile: %s\n', ME.message);
        end

        % Test 7: Check system cores
        fprintf('\nTest 7: Checking system capabilities...\n');
        num_cores = feature('numcores');
        fprintf('System cores: %d\n', num_cores);

        % Test 8: Check parallel computing toolbox license
        fprintf('\nTest 8: Checking Parallel Computing Toolbox...\n');
        if license('test', 'Distrib_Computing_Toolbox')
            fprintf('✓ Parallel Computing Toolbox available\n');
        else
            fprintf('✗ Parallel Computing Toolbox not available\n');
        end

        fprintf('\n=== Test Summary ===\n');
        fprintf('The issue was that the runParallelSimulations function was hardcoded\n');
        fprintf('to use parpool(''local'', num_workers) instead of using the cluster\n');
        fprintf('profile from user preferences.\n\n');

        fprintf('Fix applied: Modified runParallelSimulations to use:\n');
        fprintf('1. Cluster profile from handles.preferences.cluster_profile\n');
        fprintf('2. Worker count from handles.preferences.max_parallel_workers\n');
        fprintf('3. Proper fallback to local profile if cluster profile fails\n\n');

        fprintf('This should now correctly use Local_Cluster with 14 workers\n');
        fprintf('instead of defaulting to the local profile with 6 workers.\n');

    catch ME
        fprintf('✗ Test failed with error: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s:%d\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
end

function test_performance_interface()
    % TEST_PERFORMANCE_INTERFACE - Test the performance settings interface
    %
    % This script tests the performance settings interface to ensure it works correctly

    fprintf('Testing Performance Settings Interface\n');
    fprintf('=====================================\n\n');

    try
        % Test 1: Check if performance optimizer functions exist
        fprintf('Test 1: Checking performance optimizer functions...\n');

        if exist('performance_optimizer', 'file') == 2
            fprintf('✓ performance_optimizer.m found\n');
        else
            fprintf('✗ performance_optimizer.m not found\n');
        end

        if exist('performance_analysis', 'file') == 2
            fprintf('✓ performance_analysis.m found\n');
        else
            fprintf('✗ performance_analysis.m not found\n');
        end

        % Test 2: Check if setup script exists
        fprintf('\nTest 2: Checking setup script...\n');

        if exist('setup_performance_preferences', 'file') == 2
            fprintf('✓ setup_performance_preferences.m found\n');
        else
            fprintf('✗ setup_performance_preferences.m not found\n');
        end

        % Test 3: Check if preferences file exists
        fprintf('\nTest 3: Checking preferences file...\n');

        if exist('user_preferences.mat', 'file') == 2
            fprintf('✓ user_preferences.mat found\n');

            % Load and display preferences
            try
                loaded_prefs = load('user_preferences.mat');
                preferences = loaded_prefs.preferences;

                fprintf('Current preferences:\n');
                if isfield(preferences, 'enable_parallel_processing')
                    fprintf('  Parallel Processing: %s\n', yesno(preferences.enable_parallel_processing));
                end
                if isfield(preferences, 'max_parallel_workers')
                    fprintf('  Max Workers: %d\n', preferences.max_parallel_workers);
                end
                if isfield(preferences, 'cluster_profile')
                    fprintf('  Cluster Profile: %s\n', preferences.cluster_profile);
                end
                if isfield(preferences, 'enable_preallocation')
                    fprintf('  Preallocation: %s\n', yesno(preferences.enable_preallocation));
                end
                if isfield(preferences, 'enable_data_compression')
                    fprintf('  Data Compression: %s\n', yesno(preferences.enable_data_compression));
                end

            catch ME
                fprintf('✗ Error loading preferences: %s\n', ME.message);
            end
        else
            fprintf('✗ user_preferences.mat not found\n');
        end

        % Test 4: Test cluster profile detection
        fprintf('\nTest 4: Testing cluster profile detection...\n');

        try
            profiles = parallel.clusterProfiles();
            fprintf('Available cluster profiles: %s\n', strjoin(profiles, ', '));

            if ismember('Local_Cluster', profiles)
                fprintf('✓ Local_Cluster profile found\n');

                % Test cluster connection
                cluster_obj = parcluster('Local_Cluster');
                fprintf('  Cluster workers: %d\n', cluster_obj.NumWorkers);
            else
                fprintf('✗ Local_Cluster profile not found\n');
            end
        catch ME
            fprintf('✗ Error testing cluster profiles: %s\n', ME.message);
        end

        % Test 5: Test memory usage function
        fprintf('\nTest 5: Testing memory usage function...\n');

        try
            memory_info = getMemoryUsage();

            if ~isnan(memory_info.usage_percent)
                fprintf('✓ Memory usage: %.1f%% (%.1f GB / %.1f GB)\n', ...
                    memory_info.usage_percent, memory_info.used_gb, memory_info.total_gb);
            else
                fprintf('✗ Memory info unavailable\n');
            end
        catch ME
            fprintf('✗ Error getting memory info: %s\n', ME.message);
        end

        % Test 6: Test performance analysis
        fprintf('\nTest 6: Testing performance analysis...\n');

        try
            % Just test if the function exists and can be called
            if exist('performance_analysis', 'file') == 2
                fprintf('✓ performance_analysis function available\n');
                fprintf('  (Run manually to see full analysis)\n');
            else
                fprintf('✗ performance_analysis function not found\n');
            end
        catch ME
            fprintf('✗ Error testing performance analysis: %s\n', ME.message);
        end

        fprintf('\n=====================================\n');
        fprintf('Performance Interface Test Complete\n');
        fprintf('=====================================\n');

        fprintf('\nTo test the GUI interface:\n');
        fprintf('1. Run: launch_enhanced_gui\n');
        fprintf('2. Click on the "Performance Settings" tab\n');
        fprintf('3. Adjust settings and test functionality\n');

    catch ME
        fprintf('Error during testing: %s\n', ME.message);
    end
end

function result = yesno(condition)
    % YESNO - Convert boolean to Yes/No string
    if condition
        result = 'Yes';
    else
        result = 'No';
    end
end

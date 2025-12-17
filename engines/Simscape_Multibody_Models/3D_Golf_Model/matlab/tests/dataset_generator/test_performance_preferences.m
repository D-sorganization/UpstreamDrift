function test_performance_preferences()
    % TEST_PERFORMANCE_PREFERENCES - Test script to verify performance preferences
    %
    % This script tests that performance preferences are being saved and loaded
    % correctly, specifically checking the max_parallel_workers and cluster_profile
    % settings.

    fprintf('Testing Performance Preferences\n');
    fprintf('==============================\n\n');

    % Get the script directory
    script_dir = fileparts(mfilename('fullpath'));
    pref_file = fullfile(script_dir, 'user_preferences.mat');

    % Test 1: Check if preferences file exists
    fprintf('Test 1: Checking preferences file...\n');
    if exist(pref_file, 'file')
        fprintf('✓ Preferences file exists: %s\n', pref_file);
    else
        fprintf('✗ Preferences file not found: %s\n', pref_file);
        return;
    end

    % Test 2: Load and display current preferences
    fprintf('\nTest 2: Loading current preferences...\n');
    try
        loaded_prefs = load(pref_file);
        if isfield(loaded_prefs, 'preferences')
            prefs = loaded_prefs.preferences;
            fprintf('✓ Preferences loaded successfully\n');

            % Display key performance settings
            fprintf('\nCurrent Performance Settings:\n');
            fprintf('-------------------------------\n');

            if isfield(prefs, 'enable_parallel_processing')
                fprintf('Parallel Processing: %s\n', yesno(prefs.enable_parallel_processing));
            else
                fprintf('Parallel Processing: NOT SET\n');
            end

            if isfield(prefs, 'max_parallel_workers')
                fprintf('Max Parallel Workers: %d\n', prefs.max_parallel_workers);
            else
                fprintf('Max Parallel Workers: NOT SET\n');
            end

            if isfield(prefs, 'cluster_profile')
                fprintf('Cluster Profile: %s\n', prefs.cluster_profile);
            else
                fprintf('Cluster Profile: NOT SET\n');
            end

            if isfield(prefs, 'use_local_cluster')
                fprintf('Use Local Cluster: %s\n', yesno(prefs.use_local_cluster));
            else
                fprintf('Use Local Cluster: NOT SET\n');
            end

            if isfield(prefs, 'enable_preallocation')
                fprintf('Preallocation: %s\n', yesno(prefs.enable_preallocation));
            else
                fprintf('Preallocation: NOT SET\n');
            end

            if isfield(prefs, 'preallocation_buffer_size')
                fprintf('Preallocation Buffer Size: %d\n', prefs.preallocation_buffer_size);
            else
                fprintf('Preallocation Buffer Size: NOT SET\n');
            end

        else
            fprintf('✗ No preferences field found in file\n');
            return;
        end
    catch ME
        fprintf('✗ Error loading preferences: %s\n', ME.message);
        return;
    end

    % Test 3: Check if critical settings are correct
    fprintf('\nTest 3: Validating critical settings...\n');
    issues_found = 0;

    % Check max_parallel_workers
    if isfield(prefs, 'max_parallel_workers')
        if prefs.max_parallel_workers == 14
            fprintf('✓ Max parallel workers correctly set to 14\n');
        else
            fprintf('✗ Max parallel workers is %d, should be 14\n', prefs.max_parallel_workers);
            issues_found = issues_found + 1;
        end
    else
        fprintf('✗ Max parallel workers not set\n');
        issues_found = issues_found + 1;
    end

    % Check cluster_profile
    if isfield(prefs, 'cluster_profile')
        if strcmp(prefs.cluster_profile, 'Local_Cluster')
            fprintf('✓ Cluster profile correctly set to Local_Cluster\n');
        else
            fprintf('✗ Cluster profile is %s, should be Local_Cluster\n', prefs.cluster_profile);
            issues_found = issues_found + 1;
        end
    else
        fprintf('✗ Cluster profile not set\n');
        issues_found = issues_found + 1;
    end

    % Check use_local_cluster
    if isfield(prefs, 'use_local_cluster')
        if prefs.use_local_cluster
            fprintf('✓ Use local cluster correctly set to true\n');
        else
            fprintf('✗ Use local cluster is false, should be true\n');
            issues_found = issues_found + 1;
        end
    else
        fprintf('✗ Use local cluster not set\n');
        issues_found = issues_found + 1;
    end

    % Test 4: Check available cluster profiles
    fprintf('\nTest 4: Checking available cluster profiles...\n');
    try
        profiles = parallel.clusterProfiles();
        if isempty(profiles)
            fprintf('✗ No cluster profiles found\n');
            issues_found = issues_found + 1;
        else
            fprintf('Available cluster profiles: %s\n', strjoin(profiles, ', '));

            if ismember('Local_Cluster', profiles)
                fprintf('✓ Local_Cluster profile found\n');
            else
                fprintf('✗ Local_Cluster profile not found\n');
                fprintf('  You may need to run the "Setup Local Cluster" button in the GUI\n');
                issues_found = issues_found + 1;
            end
        end
    catch ME
        fprintf('✗ Error checking cluster profiles: %s\n', ME.message);
        issues_found = issues_found + 1;
    end

    % Test 5: Check number of available cores
    fprintf('\nTest 5: Checking available cores...\n');
    try
        num_cores = feature('numcores');
        fprintf('Available cores: %d\n', num_cores);

        if num_cores >= 14
            fprintf('✓ Sufficient cores available for 14 workers\n');
        else
            fprintf('⚠ Only %d cores available, 14 workers may not be optimal\n', num_cores);
        end
    catch ME
        fprintf('✗ Error checking cores: %s\n', ME.message);
        issues_found = issues_found + 1;
    end

    % Summary
    fprintf('\nTest Summary\n');
    fprintf('============\n');
    if issues_found == 0
        fprintf('✓ All tests passed! Performance preferences are correctly configured.\n');
    else
        fprintf('✗ %d issue(s) found. Please review the above output.\n', issues_found);
        fprintf('\nRecommendations:\n');
        fprintf('1. Run the "Setup Local Cluster" button in the GUI if Local_Cluster profile is missing\n');
        fprintf('2. Check that the GUI is saving preferences correctly\n');
        fprintf('3. Verify that the preferences file is not being overwritten\n');
    end

    fprintf('\nTest completed.\n');
end

function result = yesno(condition)
    % YESNO - Convert boolean to Yes/No string
    if condition
        result = 'Yes';
    else
        result = 'No';
    end
end

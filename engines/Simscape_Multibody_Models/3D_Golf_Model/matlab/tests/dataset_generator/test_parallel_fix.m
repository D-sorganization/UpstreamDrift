% Test script to verify parallel pool fixes
% This script tests that the GUI is correctly using Local_Cluster profile
% and not transferring unnecessary performance monitoring functions

fprintf('=== Testing Parallel Pool Fixes ===\n\n');

% Test 1: Check available cluster profiles
fprintf('Test 1: Checking available cluster profiles...\n');
profiles = parallel.clusterProfiles();
fprintf('Available profiles: %s\n', strjoin(profiles, ', '));

% Test 2: Check if Local_Cluster exists
fprintf('\nTest 2: Checking if Local_Cluster exists...\n');
if ismember('Local_Cluster', profiles)
    fprintf('✓ Local_Cluster profile found\n');

    % Test 3: Check Local_Cluster configuration
    fprintf('\nTest 3: Checking Local_Cluster configuration...\n');
    cluster = parcluster('Local_Cluster');
    fprintf('Local_Cluster NumWorkers: %d\n', cluster.NumWorkers);

    % Test 4: Test creating a parallel pool with Local_Cluster
    fprintf('\nTest 4: Testing parallel pool creation with Local_Cluster...\n');
    try
        % Delete any existing pool
        delete(gcp('nocreate'));

        % Create new pool with Local_Cluster
        fprintf('Creating parallel pool with Local_Cluster...\n');
        pool = parpool('Local_Cluster', min(14, cluster.NumWorkers));
        fprintf('✓ Successfully created pool with %d workers using Local_Cluster\n', pool.NumWorkers);

        % Clean up
        delete(pool);
        fprintf('✓ Pool cleaned up successfully\n');

    catch ME
        fprintf('✗ Failed to create pool with Local_Cluster: %s\n', ME.message);
    end
else
    fprintf('✗ Local_Cluster profile not found\n');

    % Test 5: Try to create Local_Cluster profile
    fprintf('\nTest 5: Attempting to create Local_Cluster profile...\n');
    try
        cluster = parallel.cluster.Local;
        cluster.Profile = 'Local_Cluster';
        fprintf('✓ Local_Cluster profile created successfully\n');

        % Verify it was created
        new_profiles = parallel.clusterProfiles();
        if ismember('Local_Cluster', new_profiles)
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
local_cluster = parcluster('local');
fprintf('Default local profile NumWorkers: %d\n', local_cluster.NumWorkers);

if local_cluster.NumWorkers == 6
    fprintf('⚠ Default local profile is limited to 6 workers\n');
else
    fprintf('✓ Default local profile has %d workers\n', local_cluster.NumWorkers);
end

% Test 7: Check if performance monitoring functions are being transferred
fprintf('\nTest 7: Checking for performance monitoring functions in Dataset_GUI...\n');
try
    % Read the Dataset_GUI.m file to check for attached files
    gui_file = 'Dataset_GUI.m';
    if exist(gui_file, 'file')
        file_content = fileread(gui_file);

        % Check for performance monitoring functions
        if contains(file_content, 'getMemoryInfo.m')
            fprintf('⚠ getMemoryInfo.m is still being transferred to workers\n');
        else
            fprintf('✓ getMemoryInfo.m has been removed from attached files\n');
        end

        if contains(file_content, 'checkHighMemoryUsage.m')
            fprintf('⚠ checkHighMemoryUsage.m is still being transferred to workers\n');
        else
            fprintf('✓ checkHighMemoryUsage.m has been removed from attached files\n');
        end

        % Check for Local_Cluster usage
        if contains(file_content, 'Local_Cluster')
            fprintf('✓ Local_Cluster profile is being used in the code\n');
        else
            fprintf('⚠ Local_Cluster profile is not being used in the code\n');
        end

    else
        fprintf('✗ Dataset_GUI.m file not found\n');
    end
catch ME
    fprintf('✗ Error checking file: %s\n', ME.message);
end

fprintf('\n=== Test Summary ===\n');
fprintf('1. Cluster profile from handles.preferences.cluster_profile\n');
fprintf('2. Local_Cluster should be used instead of Processes profile\n');
fprintf('3. Performance monitoring functions should not be transferred\n');
fprintf('4. Number of workers should not be limited to 6\n');
fprintf('\nThis should now correctly use Local_Cluster with 14 workers\n');

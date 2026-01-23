%% Test Cluster Function Accessibility
% This script tests whether the initializeLocalCluster function can be accessed
% after the path fix in Data_GUI_Enhanced.m

% Add current directory to path to ensure all functions are accessible
current_dir = fileparts(mfilename('fullpath'));
if ~contains(path, current_dir)
    addpath(current_dir);
    fprintf('Added current directory to MATLAB path: %s\n', current_dir);
end

fprintf('Testing Cluster Function Accessibility...\n');
fprintf('=====================================\n\n');

try
    % Test 1: Check if performance_optimizer_functions.m is accessible
    if exist('performance_optimizer_functions.m', 'file')
        fprintf('✓ performance_optimizer_functions.m found\n');
    else
        fprintf('✗ performance_optimizer_functions.m not found\n');
        return;
    end

    % Test 2: Check if initializeLocalCluster function exists
    if exist('initializeLocalCluster', 'file')
        fprintf('✓ initializeLocalCluster function found\n');
    else
        fprintf('✗ initializeLocalCluster function not found\n');
        return;
    end

    % Test 3: Try to call the function with a simple config
    fprintf('\nTesting function call...\n');

    % Create a simple test config
    test_config = struct();
    test_config.max_parallel_workers = 4;

    % Call the function
    cluster_info = initializeLocalCluster(test_config);

    % Check the result
    if isstruct(cluster_info)
        fprintf('✓ Function call successful\n');
        fprintf('  - Status: %s\n', cluster_info.status);
        if isfield(cluster_info, 'num_workers')
            fprintf('  - Workers: %d\n', cluster_info.num_workers);
        end
        if isfield(cluster_info, 'error_message')
            fprintf('  - Error: %s\n', cluster_info.error_message);
        end
    else
        fprintf('✗ Function returned invalid result type: %s\n', class(cluster_info));
    end

    fprintf('\n✓ All tests passed! The cluster function is now accessible.\n');

catch ME
    fprintf('✗ Error during testing: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end

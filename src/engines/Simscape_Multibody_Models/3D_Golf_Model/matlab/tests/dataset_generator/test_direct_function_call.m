%% Test Direct Function Call
% This script tests if functions in performance_optimizer_functions.m can be called directly

fprintf('Testing direct function calls...\n');

% Add current directory to path
current_dir = pwd;
addpath(current_dir);
rehash('path');

% Test 1: Check if file exists
if exist('performance_optimizer_functions.m', 'file')
    fprintf('✓ performance_optimizer_functions.m found\n');
else
    fprintf('✗ performance_optimizer_functions.m not found\n');
    return;
end

% Test 2: Try to call initializeLocalCluster directly
try
    fprintf('Testing initializeLocalCluster...\n');

    % Create a simple test config
    test_config = struct();
    test_config.max_parallel_workers = 4;

    % Call the function
    cluster_info = initializeLocalCluster(test_config);

    if isstruct(cluster_info)
        fprintf('✓ initializeLocalCluster call successful\n');
        fprintf('  Status: %s\n', cluster_info.status);
    else
        fprintf('✗ initializeLocalCluster returned invalid result\n');
    end

catch ME
    fprintf('✗ Error calling initializeLocalCluster: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end

% Test 3: Try to call getMemoryUsage
try
    fprintf('\nTesting getMemoryUsage...\n');
    memory_info = getMemoryUsage();

    if isstruct(memory_info)
        fprintf('✓ getMemoryUsage call successful\n');
    else
        fprintf('✗ getMemoryUsage returned invalid result\n');
    end

catch ME
    fprintf('✗ Error calling getMemoryUsage: %s\n', ME.message);
end

fprintf('\nTest completed.\n');

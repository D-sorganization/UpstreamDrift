%% Test All Functions
% This script tests all the critical functions to ensure they're working

fprintf('Testing all critical functions...\n');
fprintf('================================\n\n');

% Add current directory to path
current_dir = pwd;
addpath(current_dir);
rehash('path');

% Test 1: initializeLocalCluster
fprintf('1. Testing initializeLocalCluster...\n');
try
    test_config = struct();
    test_config.max_parallel_workers = 4;
    cluster_info = initializeLocalCluster(test_config);

    if isstruct(cluster_info) && isfield(cluster_info, 'status')
        fprintf('   ✓ initializeLocalCluster: %s\n', cluster_info.status);
        if isfield(cluster_info, 'num_workers')
            fprintf('     Workers: %d\n', cluster_info.num_workers);
        end
    else
        fprintf('   ✗ initializeLocalCluster: Invalid result\n');
    end
catch ME
    fprintf('   ✗ initializeLocalCluster: %s\n', ME.message);
end

% Test 2: getOrCreateParallelPool
fprintf('\n2. Testing getOrCreateParallelPool...\n');
try
    test_config = struct();
    test_config.max_parallel_workers = 2;
    pool = getOrCreateParallelPool(test_config);

    if ~isempty(pool)
        fprintf('   ✓ getOrCreateParallelPool: Pool created\n');
    else
        fprintf('   ✗ getOrCreateParallelPool: No pool created\n');
    end
catch ME
    fprintf('   ✗ getOrCreateParallelPool: %s\n', ME.message);
end

% Test 3: getMemoryUsage
fprintf('\n3. Testing getMemoryUsage...\n');
try
    memory_info = getMemoryUsage();

    if isstruct(memory_info) && isfield(memory_info, 'usage_percent')
        fprintf('   ✓ getMemoryUsage: %.1f%% used\n', memory_info.usage_percent);
    else
        fprintf('   ✗ getMemoryUsage: Invalid result\n');
    end
catch ME
    fprintf('   ✗ getMemoryUsage: %s\n', ME.message);
end

% Test 4: compressData
fprintf('\n4. Testing compressData...\n');
try
    % Create a simple test table
    test_data = table([1; 2; 3], [4; 5; 6], 'VariableNames', {'A', 'B'});
    compressed = compressData(test_data, 5);

    if isequal(test_data, compressed)
        fprintf('   ✓ compressData: Data preserved\n');
    else
        fprintf('   ✗ compressData: Data changed\n');
    end
catch ME
    fprintf('   ✗ compressData: %s\n', ME.message);
end

fprintf('\n================================\n');
fprintf('All function tests completed!\n');
fprintf('If you see all ✓ marks, the GUI should work correctly.\n');

%% Simple Test Script
% Test if the new performance functions file works

fprintf('Testing new performance functions file...\n');

% Add current directory to path
current_dir = fileparts(mfilename('fullpath'));
addpath(current_dir);

% Test if file exists
if exist('performance_optimizer_functions.m', 'file')
    fprintf('✓ performance_optimizer_functions.m found\n');
else
    fprintf('✗ performance_optimizer_functions.m not found\n');
    return;
end

% Test if function exists
if exist('initializeLocalCluster', 'file')
    fprintf('✓ initializeLocalCluster function found\n');
else
    fprintf('✗ initializeLocalCluster function not found\n');
    return;
end

% Test if getMemoryUsage exists
if exist('getMemoryUsage', 'file')
    fprintf('✓ getMemoryUsage function found\n');
else
    fprintf('✗ getMemoryUsage function not found\n');
    return;
end

fprintf('✓ All tests passed!\n');

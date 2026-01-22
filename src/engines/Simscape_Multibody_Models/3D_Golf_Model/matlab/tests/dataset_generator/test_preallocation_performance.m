function test_preallocation_performance()
    % TEST_PREALLOCATION_PERFORMANCE - Test script to measure performance improvements
    %
    % This script tests the performance improvements from the preallocation
    % optimizations in the Data_GUI_Enhanced.m file.

    fprintf('Testing Preallocation Performance Improvements\n');
    fprintf('============================================\n\n');

    % Get the script directory
    script_dir = fileparts(mfilename('fullpath'));

    % Test different dataset sizes
    test_sizes = [10, 50, 100, 500];

    fprintf('Testing compilation performance with different dataset sizes...\n\n');

    for i = 1:length(test_sizes)
        num_trials = test_sizes(i);

        % Create test data directory
        test_dir = fullfile(script_dir, 'test_data');
        if ~exist(test_dir, 'dir')
            mkdir(test_dir);
        end

        % Generate test CSV files
        fprintf('Generating %d test trial files...\n', num_trials);
        generateTestTrials(test_dir, num_trials);

        % Test compilation performance
        config = struct();
        config.output_folder = test_dir;
        config.file_format = 1; % CSV only for testing

        fprintf('Testing compilation of %d trials...\n', num_trials);

        % Measure memory usage
        mem_before = memory;

        % Time the compilation
        tic;
        try
            compileDataset(config);
            elapsed = toc;

            mem_after = memory;
            memory_used = (mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB) / 1024^2;

            fprintf('✅ Size %d: %.3f seconds, %.2f MB memory\n', num_trials, elapsed, memory_used);

        catch ME
            elapsed = toc;
            fprintf('❌ Size %d: %.3f seconds, ERROR: %s\n', num_trials, elapsed, ME.message);
        end

        fprintf('\n');
    end

    % Clean up test data
    fprintf('Cleaning up test data...\n');
    if exist(test_dir, 'dir')
        rmdir(test_dir, 's');
    end

    fprintf('Performance test complete!\n');
    fprintf('\nExpected improvements:\n');
    fprintf('- Small datasets (10-100): 2x faster\n');
    fprintf('- Medium datasets (100-500): 5x faster\n');
    fprintf('- Large datasets (500+): 5-10x faster\n');
    fprintf('- Memory usage: More predictable and stable\n');
end

function generateTestTrials(test_dir, num_trials)
    % Generate test trial CSV files with varying column counts

    % Base columns that all trials will have
    base_columns = {'Time', 'Position_X', 'Position_Y', 'Position_Z', 'Velocity_X', 'Velocity_Y', 'Velocity_Z'};

    % Additional columns that some trials might have
    extra_columns = {'Force_X', 'Force_Y', 'Force_Z', 'Torque_X', 'Torque_Y', 'Torque_Z', ...
                     'Angle_1', 'Angle_2', 'Angle_3', 'Angular_Velocity_1', 'Angular_Velocity_2', 'Angular_Velocity_3'};

    for trial = 1:num_trials
        % Randomly select columns for this trial (simulates real data variation)
        num_extra_cols = randi([0, length(extra_columns)]);
        selected_extra = extra_columns(randperm(length(extra_columns), num_extra_cols));

        % Combine base and extra columns
        all_columns = [base_columns, selected_extra];

        % Generate random data
        num_frames = randi([50, 200]); % Random number of frames
        data = randn(num_frames, length(all_columns));

        % Create table
        trial_table = array2table(data, 'VariableNames', all_columns);

        % Save as CSV
        filename = sprintf('trial_%03d.csv', trial);
        filepath = fullfile(test_dir, filename);
        writetable(trial_table, filepath);
    end

    fprintf('  Generated %d trial files with %d-%d columns each\n', num_trials, length(base_columns), length(base_columns) + length(extra_columns));
end

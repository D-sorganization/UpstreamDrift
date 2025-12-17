function results = test_1956_columns(varargin)
% TEST_1956_COLUMNS Critical regression test for data generator output
%
% This test verifies that the data generator produces exactly 1956 columns
% in the master dataset for both sequential and parallel execution modes.
%
% The 1956 column count is a critical invariant of the golf swing analysis
% system. Any deviation indicates:
%   • Missing data sources (logsout, signal bus, or Simscape)
%   • Configuration errors
%   • Data extraction failures
%   • Column alignment issues
%
% Usage:
%   % Quick test (2 trials each mode)
%   test_1956_columns()
%
%   % Thorough test (custom trial count)
%   test_1956_columns('num_trials', 10)
%
%   % Test only one mode
%   test_1956_columns('modes', {'sequential'})
%   test_1956_columns('modes', {'parallel'})
%
%   % Silent mode (no output, just returns results)
%   results = test_1956_columns('silent', true)
%
% Returns:
%   results - Struct with test results for each mode
%
% Example:
%   >> test_1956_columns()
%
%   ========================================
%   1956 COLUMN VALIDATION TEST
%   ========================================
%   Date: 2025-11-16 14:30:45
%   Purpose: Verify data generator produces 1956 columns
%
%   [1/2] Testing SEQUENTIAL mode (2 trials)...
%   ✅ PASS - Sequential mode: 1956 columns
%
%   [2/2] Testing PARALLEL mode (2 trials)...
%   ✅ PASS - Parallel mode: 1956 columns
%
%   ========================================
%   FINAL RESULT: ✅ ALL TESTS PASSED
%   ========================================
%
% See also: createSimulationConfig, validate_baseline_behavior

%% Parse Arguments

p = inputParser;
p.addParameter('num_trials', 2, @(x) isnumeric(x) && x > 0);
p.addParameter('modes', {'sequential', 'parallel'}, @iscell);
p.addParameter('silent', false, @islogical);
p.addParameter('cleanup', true, @islogical);  % Delete test output after
p.parse(varargin{:});

num_trials = p.Results.num_trials;
modes_to_test = p.Results.modes;
silent = p.Results.silent;
cleanup = p.Results.cleanup;

%% Setup

if ~silent
    fprintf('\n');
    fprintf('========================================\n');
    fprintf('1956 COLUMN VALIDATION TEST\n');
    fprintf('========================================\n');
    fprintf('Date: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
    fprintf('Purpose: Verify data generator produces 1956 columns\n');
    fprintf('\n');
end

% Add paths
addpath(genpath('matlab/src'));

% Create test output directory
test_output_base = fullfile(pwd, 'test_output', '1956_column_test');
if ~exist(test_output_base, 'dir')
    mkdir(test_output_base);
end

% Initialize results
results = struct();
results.timestamp = char(datetime('now'));
results.num_trials = num_trials;
results.modes_tested = modes_to_test;
results.all_passed = true;
results.details = {};

%% Test Each Mode

for mode_idx = 1:length(modes_to_test)
    mode = modes_to_test{mode_idx};

    if ~silent
        fprintf('[%d/%d] Testing %s mode (%d trials)...\n', ...
            mode_idx, length(modes_to_test), upper(mode), num_trials);
    end

    % Create test configuration
    test_output_dir = fullfile(test_output_base, [mode '_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))]);

    try
        % Configure test
        config = createSimulationConfig();
        config.num_simulations = num_trials;
        config.execution_mode = mode;
        config.output_folder = test_output_dir;
        config.verbosity = 'Silent';  % Suppress normal output
        config.enable_master_dataset = true;

        % Run simulation
        test_start = tic;

        % Call the actual runSimulation function
        [successful_trials, dataset_path, ~] = runSimulation(config);

        % Compile master dataset
        master_file = dataset_path;

        if exist(master_file, 'file') ~= 2
            error('Master dataset file not created or is a directory');
        end

        % Read and check column count
        master_data = readtable(master_file);
        num_columns = width(master_data);
        num_rows = height(master_data);

        test_elapsed = toc(test_start);

        % Evaluate result
        test_passed = (num_columns == 1956);

        % Store result
        result_detail = struct();
        result_detail.mode = mode;
        result_detail.num_trials = num_trials;
        result_detail.successful_trials = successful_trials;
        result_detail.num_columns = num_columns;
        result_detail.num_rows = num_rows;
        result_detail.elapsed_seconds = test_elapsed;
        result_detail.passed = test_passed;
        result_detail.output_dir = test_output_dir;

        results.details{end+1} = result_detail;

        if ~test_passed
            results.all_passed = false;
        end

        % Print result
        if ~silent
            if test_passed
                fprintf('   ✅ PASS - %s mode: %d columns (%.1f seconds)\n', ...
                    mode, num_columns, test_elapsed);
            else
                fprintf('   ❌ FAIL - %s mode: %d columns (expected 1956)\n', ...
                    mode, num_columns);
            end
        end

    catch ME
        % Test failed with error
        result_detail = struct();
        result_detail.mode = mode;
        result_detail.num_trials = num_trials;
        result_detail.passed = false;
        result_detail.error = ME.message;
        result_detail.output_dir = test_output_dir;

        results.details{end+1} = result_detail;
        results.all_passed = false;

        if ~silent
            fprintf('   ❌ ERROR - %s mode failed: %s\n', mode, ME.message);
        end
    end

    if ~silent
        fprintf('\n');
    end
end

%% Summary

if ~silent
    fprintf('========================================\n');
    if results.all_passed
        fprintf('FINAL RESULT: ✅ ALL TESTS PASSED\n');
    else
        fprintf('FINAL RESULT: ❌ SOME TESTS FAILED\n');
        fprintf('\nFailed tests:\n');
        for i = 1:length(results.details)
            detail = results.details{i};
            if ~detail.passed
                fprintf('  • %s mode', upper(detail.mode));
                if isfield(detail, 'num_columns')
                    fprintf(': %d columns (expected 1956)', detail.num_columns);
                elseif isfield(detail, 'error')
                    fprintf(': %s', detail.error);
                end
                fprintf('\n');
            end
        end
    end
    fprintf('========================================\n');
    fprintf('\n');
end

%% Cleanup

if cleanup && results.all_passed
    % Only cleanup if all tests passed
    if ~silent
        fprintf('Cleaning up test output...\n');
    end
    try
        rmdir(test_output_base, 's');
    catch
        % Ignore cleanup errors
    end
end

%% Save Results

results_file = fullfile(pwd, 'test_output', '1956_test_results.mat');
if ~exist(fileparts(results_file), 'dir')
    mkdir(fileparts(results_file));
end
save(results_file, 'results');

if ~silent
    fprintf('Results saved to: %s\n\n', results_file);
end

end

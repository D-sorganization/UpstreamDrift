function printSimulationSummary(config, successful_trials, failed_trials, elapsed_time, num_columns)
% PRINTSIMULATIONSUMMARY Print final summary with 1956 column check
%
% Displays a comprehensive summary of simulation results including:
%   • Success/failure counts and rate
%   • Elapsed time and speedup (for parallel)
%   • CRITICAL: 1956 column verification
%   • Output file locations
%   • Failed trial details (if applicable)
%
% Args:
%   config - Simulation configuration struct
%   successful_trials - Number of successful trials
%   failed_trials - Array of failed trial indices (empty if none failed)
%   elapsed_time - Total elapsed time in seconds
%   num_columns - Number of columns in master dataset
%
% Example:
%   config = createSimulationConfig('num_simulations', 100);
%   printSimulationSummary(config, 98, [23, 67], 43.5, 1956);
%
%   Output:
%   ========================================
%   ⚠ SIMULATION COMPLETE (WITH FAILURES)
%   ========================================
%   Total Trials: 100
%   Successful: 98 (98.0%)
%   Failed: 2 (trials #23, #67)
%   Elapsed Time: 43.5 seconds
%
%   MASTER DATASET: 98 rows × 1956 columns ✅
%   Target 1956 columns: ACHIEVED ✅
%
%   Output: /path/to/output/master_dataset.csv
%   ========================================
%
% See also: printSimulationHeader, printProgressBar

%% Calculate Statistics

total_trials = config.num_simulations;
success_rate = (successful_trials / total_trials) * 100;
num_failed = length(failed_trials);

%% Print Summary Header

fprintf('\n');
fprintf('========================================\n');

% Status header (with emoji indicators)
if successful_trials == total_trials && num_columns == 1956
    % Perfect success
    fprintf('✅ SIMULATION COMPLETE\n');
elseif successful_trials == 0
    % Total failure
    fprintf('❌ SIMULATION FAILED\n');
elseif num_columns ~= 1956
    % Data integrity failure
    fprintf('❌ DATA INTEGRITY FAILURE\n');
else
    % Partial success
    fprintf('⚠ SIMULATION COMPLETE (WITH FAILURES)\n');
end

fprintf('========================================\n');

%% Print Trial Statistics

fprintf('Total Trials: %d\n', total_trials);
fprintf('Successful: %d (%.1f%%)\n', successful_trials, success_rate);

if num_failed > 0
    fprintf('Failed: %d', num_failed);

    % Show failed trial numbers (if not too many)
    if num_failed <= 20
        % Show all failed trial numbers
        fprintf(' (trials #%s)', num2str(failed_trials));
    else
        % Too many to show - just show first 10
        fprintf(' (first 10: #%s, ...)', num2str(failed_trials(1:10)));
    end
    fprintf('\n');
else
    fprintf('Failed: 0\n');
end

%% Print Timing Information

fprintf('Elapsed Time: %.1f seconds', elapsed_time);

% Add speedup for parallel mode (if available)
if strcmp(config.execution_mode, 'parallel') && isfield(config, 'sequential_baseline_time')
    speedup = config.sequential_baseline_time / elapsed_time;
    fprintf(' (%.1f× speedup)', speedup);
end

fprintf('\n');

%% CRITICAL: 1956 Column Check

fprintf('\n');

if num_columns == 1956
    % SUCCESS: 1956 columns achieved
    fprintf('MASTER DATASET: %d rows × %d columns ✅\n', successful_trials, num_columns);
    fprintf('Target 1956 columns: ACHIEVED ✅\n');
else
    % FAILURE: Column count mismatch
    fprintf('MASTER DATASET: %d rows × %d columns ❌\n', successful_trials, num_columns);
    fprintf('Target 1956 columns: NOT ACHIEVED ❌\n');
    fprintf('\n');

    % Detailed diagnostic information
    column_diff = num_columns - 1956;
    if column_diff > 0
        fprintf('⚠ WARNING: %d EXTRA columns detected\n', column_diff);
    else
        fprintf('⚠ WARNING: %d MISSING columns detected\n', abs(column_diff));
    end

    fprintf('\nPossible causes:\n');

    % Check data sources
    fprintf('  Data sources enabled:\n');
    if isfield(config, 'use_logsout')
        fprintf('    • Logsout: %s\n', tf2str(config.use_logsout));
    end
    if isfield(config, 'use_signal_bus')
        fprintf('    • Signal Bus: %s\n', tf2str(config.use_signal_bus));
    end
    if isfield(config, 'use_simscape')
        fprintf('    • Simscape: %s\n', tf2str(config.use_simscape));
    end

    fprintf('\n  Troubleshooting steps:\n');
    fprintf('    1. Verify all data sources are enabled\n');
    fprintf('    2. Check model configuration\n');
    fprintf('    3. Verify Simscape Results logging settings\n');
    fprintf('    4. Review individual trial CSV files for column counts\n');
    fprintf('    5. Check for data extraction errors in logs\n');
end

%% Print Output Locations

fprintf('\n');

% Master dataset path
if isfield(config, 'output_folder')
    master_file = fullfile(config.output_folder, 'master_dataset.csv');
    fprintf('Output: %s\n', master_file);

    % Error log path (if there were failures)
    if num_failed > 0
        error_log = fullfile(config.output_folder, 'error_log.txt');
        if exist(error_log, 'file')
            fprintf('Error Log: %s\n', error_log);
        end
    end
end

fprintf('========================================\n\n');

%% Print Failed Trial Details (if applicable and available)

if num_failed > 0 && isfield(config, 'trial_error_details')
    fprintf('Failed trial details:\n');

    error_details = config.trial_error_details;

    for i = 1:min(num_failed, 10)  % Show max 10 error details
        trial_idx = failed_trials(i);

        if isfield(error_details, sprintf('trial_%d', trial_idx))
            detail = error_details.(sprintf('trial_%d', trial_idx));
            fprintf('  • Trial #%d: %s\n', trial_idx, detail.error_message);
        else
            fprintf('  • Trial #%d: Unknown error\n', trial_idx);
        end
    end

    if num_failed > 10
        fprintf('  ... and %d more (see error log)\n', num_failed - 10);
    end

    fprintf('\n');
end

end

function str = tf2str(tf)
% TF2STR Convert true/false to YES/NO string
if tf
    str = 'YES';
else
    str = 'NO';
end
end

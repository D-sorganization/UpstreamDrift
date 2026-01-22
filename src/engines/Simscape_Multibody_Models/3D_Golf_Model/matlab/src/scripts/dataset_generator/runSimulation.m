function [successful_trials, dataset_path, metadata] = runSimulation(config, options)
% RUNSIMULATION Execute golf swing simulations without GUI
%
% This is the main entry point for the data generator module. It orchestrates
% the entire simulation process from validation to execution to compilation,
% enabling counterfactual analysis and parameter sweeps without GUI interaction.
%
% Args:
%   config - Simulation configuration struct (from createSimulationConfig)
%            See DATA_GENERATOR_INTERFACE_SPEC.md for full field documentation
%   options - Optional settings struct (default: struct('verbose', true))
%     .verbose - Enable verbose output (default: true)
%     .progress_callback - Function handle for progress updates (optional)
%
% Returns:
%   successful_trials - Number of successfully completed trials (integer)
%   dataset_path - Full path to generated dataset directory (string)
%   metadata - Execution metadata struct with timing, errors, performance stats
%
% Example - Basic Usage:
%   config = createSimulationConfig('num_simulations', 100);
%   [trials, path, meta] = runSimulation(config);
%   fprintf('Completed %d trials in %.1f seconds\n', trials, meta.elapsed_seconds);
%
% Example - Parameter Sweep (Counterfactual Analysis):
%   driver_masses = linspace(0.28, 0.34, 10);
%   for i = 1:length(driver_masses)
%       config = createSimulationConfig();
%       config.driver_mass = driver_masses(i);
%       config.folder_name = sprintf('mass_%.3f', driver_masses(i));
%       [trials, path, meta] = runSimulation(config);
%       % NOTE: loadResults is a user-defined function - implement as needed
%       results{i} = loadResults(path);
%   end
%   % NOTE: analyzeParameterSweep is a user-defined function - implement as needed
%   analyzeParameterSweep(results);
%
% Example - Counterfactual Comparison:
%   % Baseline
%   config_base = createSimulationConfig('swing_speed', 100);
%   [~, path_base, ~] = runSimulation(config_base);
%
%   % Counterfactual: +10% swing speed
%   config_cf = config_base;
%   config_cf.swing_speed = 110;
%   [~, path_cf, ~] = runSimulation(config_cf);
%
%   % Compare
%   % NOTE: analyzeCounterfactual is a user-defined function - implement as needed
%   effect = analyzeCounterfactual(path_base, path_cf);
%
% Key Features:
%   - Zero GUI dependencies (pure function interface)
%   - Parallel or sequential execution
%   - Checkpoint/resume capability
%   - Comprehensive error handling
%   - Configurable verbosity levels
%   - Memory-efficient batch processing
%
% See also: CREATESIMULATIONCONFIG, VALIDATESIMULATIONCONFIG, DATASET_GUI
%
% Author: Phase 2 Refactoring - Extracted from Dataset_GUI.m
% Date: 2025-11-17
% Version: 2.0

arguments
    config struct {mustBeNonempty}
    options struct = struct('verbose', true)
end

%% Initialize Metadata

metadata = struct();
metadata.start_time = datetime('now');
metadata.matlab_version = version;
metadata.config = config;
metadata.failed_trials = [];
metadata.errors = {};

start_timer = tic;

try
    %% Validate MATLAB Path Dependencies

    logMessage(config, 'Verbose', 'Checking required functions on MATLAB path...');
    validateFunctionDependencies();
    logMessage(config, 'Verbose', '  ✓ All required functions are available');

    %% Validate Configuration

    logMessage(config, 'Verbose', 'Validating configuration...');
    validateSimulationConfig(config);
    logMessage(config, 'Verbose', '  ✓ Configuration validated');

    %% Ensure Enhanced Configuration

    % Add computed fields and defaults
    config = ensureEnhancedConfig(config);

    %% Create Output Directory

    output_dir = fullfile(config.output_folder, config.folder_name);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
        logMessage(config, 'Normal', 'Created output directory: %s', output_dir);
    end

    % CRITICAL: Update config.output_folder to point to the run-specific directory
    % This ensures all downstream functions (processSimulationOutput, compileDataset, checkpoints)
    % write to the correct per-run subdirectory instead of the parent folder
    config.full_output_path = output_dir;
    config.output_folder = output_dir;  % Update to use run-specific directory

    %% Save Configuration

    config_file = fullfile(output_dir, 'simulation_config.mat');
    save(config_file, 'config');
    logMessage(config, 'Verbose', 'Saved configuration to: %s', config_file);

    %% Run Simulations

    logMessage(config, 'Normal', '\nStarting %d simulations in %s mode...', ...
        config.num_simulations, config.execution_mode);

    if strcmp(config.execution_mode, 'parallel')
        [successful_trials, trial_metadata] = runParallelSimulations(config);
    else
        [successful_trials, trial_metadata] = runSequentialSimulations(config);
    end

    metadata.trial_metadata = trial_metadata;
    metadata.successful_trials = successful_trials;
    metadata.failed_trials = config.num_simulations - successful_trials;

    %% Compile Master Dataset

    if successful_trials > 0 && config.enable_master_dataset
        logMessage(config, 'Normal', '\nCompiling master dataset...');
        dataset_path = compileDataset(config);
        metadata.dataset_path = dataset_path;
        metadata.dataset_compiled = true;
    else
        dataset_path = output_dir;
        metadata.dataset_compiled = false;
        if successful_trials == 0
            logMessage(config, 'Normal', 'No successful trials - skipping dataset compilation');
        elseif ~config.enable_master_dataset
            logMessage(config, 'Verbose', 'Master dataset compilation disabled in config');
        end
    end

    %% Save Script and Settings

    if isfield(config, 'save_script_backup') && config.save_script_backup
        saveScriptAndSettings(config);
    end

    %% Finalize Metadata

    metadata.end_time = datetime('now');
    metadata.elapsed_seconds = toc(start_timer);
    metadata.success_rate = successful_trials / config.num_simulations;

    % Save metadata
    metadata_file = fullfile(output_dir, 'execution_metadata.mat');
    save(metadata_file, 'metadata');

    %% Print Summary

    logMessage(config, 'Normal', '\n========================================');
    logMessage(config, 'Normal', '✅ SIMULATION COMPLETE');
    logMessage(config, 'Normal', '========================================');
    logMessage(config, 'Normal', 'Successful trials: %d/%d (%.1f%%)', ...
        successful_trials, config.num_simulations, 100*metadata.success_rate);

    if metadata.failed_trials > 0
        logMessage(config, 'Normal', 'Failed trials: %d', metadata.failed_trials);
    end

    logMessage(config, 'Normal', 'Elapsed time: %.1f seconds', metadata.elapsed_seconds);

    if ~isempty(dataset_path) && metadata.dataset_compiled
        logMessage(config, 'Normal', 'Dataset: %s', dataset_path);
    end

    logMessage(config, 'Normal', 'Output directory: %s', output_dir);
    logMessage(config, 'Normal', '========================================\n');

catch ME
    % Error handling with comprehensive context
    metadata.end_time = datetime('now');
    metadata.elapsed_seconds = toc(start_timer);
    metadata.error = ME;
    metadata.error_message = ME.message;
    metadata.error_stack = ME.stack;

    % Save partial metadata
    if exist('output_dir', 'var') && exist(output_dir, 'dir')
        metadata_file = fullfile(output_dir, 'execution_metadata_FAILED.mat');
        save(metadata_file, 'metadata');
    end

    % Re-throw with context
    error('DataGenerator:SimulationFailed', ...
        'Simulation failed after %.1f seconds: %s', ...
        metadata.elapsed_seconds, ME.message);
end

end

%% ========================================================================
%  INTERNAL FUNCTIONS (Private - Called by runSimulation)
%% ========================================================================

function [successful_trials, trial_metadata] = runParallelSimulations(config)
% RUNPARALLELSIMULATIONS Execute simulations in parallel using parpool
%
% Initializes parallel pool, runs simulations in batches with parsim,
% handles checkpointing, and manages worker resources.
%
% Args:
%   config - Simulation configuration
%
% Returns:
%   successful_trials - Number of successful trials
%   trial_metadata - Execution metadata

% Initialize metadata
trial_metadata = struct();
trial_metadata.mode = 'parallel';
trial_metadata.start_time = datetime('now');

% Initialize parallel pool with error handling
try
    % Check for existing pool
    existing_pool = gcp('nocreate');

    if ~isempty(existing_pool)
        try
            % Test if pool is responsive
            pool_info = existing_pool;
            logMessage(config, 'Verbose', 'Found existing parallel pool with %d workers', ...
                pool_info.NumWorkers);

            try
                spmd
                    % Test pool responsiveness
                end
                logMessage(config, 'Verbose', 'Existing pool is healthy, using it');
            catch
                logMessage(config, 'Normal', 'Existing pool unresponsive, recreating...');
                delete(existing_pool);
                existing_pool = [];
            end
        catch
            logMessage(config, 'Verbose', 'Error checking existing pool, deleting it');
            delete(existing_pool);
            existing_pool = [];
        end
    end

    % Create new pool if needed
    if isempty(existing_pool)
        % Determine number of workers
        if isfield(config, 'num_workers') && config.num_workers > 0
            num_workers = config.num_workers;
        else
            num_workers = feature('numcores');
        end

        % Try cluster profiles
        try
            cluster_profiles = parallel.clusterProfiles();
            if ismember('Local_Cluster', cluster_profiles)
                cluster_obj = parcluster('Local_Cluster');
                num_workers = cluster_obj.NumWorkers;
                logMessage(config, 'Normal', 'Using Local_Cluster profile with %d workers', ...
                    num_workers);
                parpool(cluster_obj, num_workers);
            else
                logMessage(config, 'Normal', 'Starting parallel pool with %d workers', num_workers);
                parpool('local', num_workers);
            end
            logMessage(config, 'Normal', 'Successfully started parallel pool');
        catch ME
            logMessage(config, 'Normal', 'Using local profile with %d workers', num_workers);
            parpool('local', num_workers);
        end
    end

    trial_metadata.num_workers = gcp().NumWorkers;

catch ME
    warning('DataGenerator:ParallelPoolFailed', ...
        'Failed to start parallel pool: %s. Falling back to sequential execution.', ME.message);
    [successful_trials, trial_metadata] = runSequentialSimulations(config);
    return;
end

% Get batch processing parameters
batch_size = config.batch_size;
save_interval = config.save_interval;
total_trials = config.num_simulations;

logMessage(config, 'Debug', '[RUNTIME] Using batch size: %d, save_interval: %d, verbosity: %s', ...
    batch_size, save_interval, config.verbosity);

logMessage(config, 'Normal', 'Starting parallel batch processing:');
logMessage(config, 'Normal', '  Total trials: %d', total_trials);
logMessage(config, 'Normal', '  Batch size: %d', batch_size);
logMessage(config, 'Verbose', '  Save interval: %d batches', save_interval);

% Calculate number of batches
num_batches = ceil(total_trials / batch_size);
successful_trials = 0;

% Store initial workspace state for restoration
initial_vars = who;

% Check for existing checkpoint
checkpoint_file = fullfile(config.output_folder, 'parallel_checkpoint.mat');
start_batch = 1;

if exist(checkpoint_file, 'file') && config.enable_checkpoint_resume
    try
        checkpoint_data = load(checkpoint_file);
        if isfield(checkpoint_data, 'completed_trials')
            successful_trials = checkpoint_data.completed_trials;
            start_batch = checkpoint_data.next_batch;
            logMessage(config, 'Normal', 'Found checkpoint: %d trials completed, resuming from batch %d', ...
                successful_trials, start_batch);
        end
    catch ME
        logMessage(config, 'Verbose', 'Warning: Could not load checkpoint: %s', ME.message);
    end
elseif exist(checkpoint_file, 'file') && ~config.enable_checkpoint_resume
    logMessage(config, 'Verbose', 'Checkpoint found but resume disabled - starting fresh');
end

% Ensure model is available on all parallel workers
try
    logMessage(config, 'Verbose', 'Loading model on parallel workers...');
    spmd
        if ~bdIsLoaded(config.model_name)
            load_system(config.model_path);
        end
    end
    logMessage(config, 'Verbose', 'Model loaded on all workers');
catch ME
    logMessage(config, 'Verbose', 'Warning: Could not preload model on workers: %s', ME.message);
end

% Process batches
for batch_idx = start_batch:num_batches
    % Calculate trials for this batch
    start_trial = (batch_idx - 1) * batch_size + 1;
    end_trial = min(batch_idx * batch_size, total_trials);
    batch_trials = end_trial - start_trial + 1;

    logMessage(config, 'Verbose', '\n--- Batch %d/%d (Trials %d-%d) ---', ...
        batch_idx, num_batches, start_trial, end_trial);

    % Update progress
    logMessage(config, 'Normal', 'Batch %d/%d: Processing trials %d-%d...', ...
        batch_idx, num_batches, start_trial, end_trial);

    % Prepare simulation inputs for this batch
    try
        batch_simInputs = prepareSimulationInputsForBatch(config, start_trial, end_trial);

        if isempty(batch_simInputs)
            logMessage(config, 'Normal', 'Failed to prepare simulation inputs for batch %d', batch_idx);
            continue;
        end

        logMessage(config, 'Verbose', 'Prepared %d simulation inputs for batch %d', ...
            length(batch_simInputs), batch_idx);

    catch ME
        logMessage(config, 'Normal', 'Error preparing batch %d inputs: %s', batch_idx, ME.message);
        continue;
    end

    % Run batch simulations
    try
        logMessage(config, 'Verbose', 'Running batch %d with parsim...', batch_idx);

        % Attach all external functions needed by parallel workers
        attached_files = {
            config.model_path, ...
            'runSingleTrial.m', ...
            'processSimulationOutput.m', ...
            'setModelParameters.m', ...
            'setPolynomialCoefficients.m', ...
            'extractSignalsFromSimOut.m', ...
            'extractFromCombinedSignalBus.m', ...
            'extractFromNestedStruct.m', ...
            'extractLogsoutDataFixed.m', ...
            'extractSimscapeDataRecursive.m', ...
            'traverseSimlogNode.m', ...
            'extractDataFromField.m', ...
            'combineDataSources.m', ...
            'addModelWorkspaceData.m', ...
            'extractWorkspaceOutputs.m', ...
            'resampleDataToFrequency.m', ...
            'getPolynomialParameterInfo.m', ...
            'getShortenedJointName.m', ...
            'generateRandomCoefficients.m', ...
            'prepareSimulationInputsForBatch.m', ...
            'restoreWorkspace.m', ...
            'loadInputFile.m', ...
            'extractCoefficientsFromTable.m', ...
            'shouldShowDebug.m', ...
            'shouldShowVerbose.m', ...
            'shouldShowNormal.m', ...
            'mergeTables.m', ...
            'logical2str.m', ...
            'extractTimeSeriesData.m', ...
            'extractConstantMatrixData.m'
        };

        batch_simOuts = parsim(batch_simInputs, ...
            'AttachedFiles', attached_files, ...
            'StopOnError', 'off');  % Don't stop on individual simulation errors

        % Check if parsim succeeded
        if isempty(batch_simOuts)
            logMessage(config, 'Normal', 'Batch %d failed - no results returned', batch_idx);
            continue;
        end

        % Process batch results
        batch_successful = 0;
        for i = 1:length(batch_simOuts)
            trial_num = start_trial + i - 1;

            try
                current_simOut = batch_simOuts(i);

                % Check if we got a valid simulation output
                if isempty(current_simOut)
                    logMessage(config, 'Verbose', 'Trial %d: Empty simulation output', trial_num);
                    continue;
                end

                if ~isscalar(current_simOut)
                    logMessage(config, 'Verbose', 'Trial %d: Multiple simulation outputs returned', trial_num);
                    continue;
                end

                % Check if simulation completed successfully
                simulation_success = false;
                has_error = false;

                % Try multiple ways to check simulation status
                try
                    % Method 1: Check SimulationMetadata
                    if isprop(current_simOut, 'SimulationMetadata') && ...
                            isfield(current_simOut.SimulationMetadata, 'ExecutionInfo')

                        execInfo = current_simOut.SimulationMetadata.ExecutionInfo;

                        if isfield(execInfo, 'StopEvent') && execInfo.StopEvent == "CompletedNormally"
                            simulation_success = true;
                        else
                            has_error = true;
                            logMessage(config, 'Verbose', 'Trial %d simulation failed (metadata)', trial_num);

                            if isfield(execInfo, 'ErrorDiagnostic') && ~isempty(execInfo.ErrorDiagnostic)
                                logMessage(config, 'Verbose', '  Error: %s', execInfo.ErrorDiagnostic.message);
                            end
                        end
                    else
                        % Method 2: Check for ErrorMessage property
                        if isprop(current_simOut, 'ErrorMessage') && ~isempty(current_simOut.ErrorMessage)
                            has_error = true;
                            logMessage(config, 'Verbose', 'Trial %d simulation failed: %s', ...
                                trial_num, current_simOut.ErrorMessage);
                        else
                            % Method 3: Check if we have output data
                            has_data = false;
                            if isprop(current_simOut, 'logsout') || isfield(current_simOut, 'logsout') || ...
                                    isprop(current_simOut, 'simlog') || isfield(current_simOut, 'simlog') || ...
                                    isprop(current_simOut, 'CombinedSignalBus') || isfield(current_simOut, 'CombinedSignalBus')
                                has_data = true;
                            end

                            if has_data
                                logMessage(config, 'Debug', 'Trial %d: Assuming success (has output data)', trial_num);
                                simulation_success = true;
                            else
                                logMessage(config, 'Verbose', 'Trial %d: No metadata, no data, assuming failure', trial_num);
                                has_error = true;
                            end
                        end
                    end
                catch ME
                    logMessage(config, 'Verbose', 'Trial %d: Error checking simulation status: %s', ...
                        trial_num, ME.message);
                    has_error = true;
                end

                % Process simulation if it succeeded
                if simulation_success && ~has_error
                    try
                        result = processSimulationOutput(trial_num, config, current_simOut, config.capture_workspace);
                        if result.success
                            batch_successful = batch_successful + 1;
                            successful_trials = successful_trials + 1;
                            logMessage(config, 'Debug', 'Trial %d completed successfully', trial_num);
                        else
                            logMessage(config, 'Verbose', 'Trial %d processing failed: %s', ...
                                trial_num, result.error);
                        end
                    catch ME
                        logMessage(config, 'Verbose', 'Error processing trial %d: %s', trial_num, ME.message);
                    end
                end

            catch ME
                % Handle brace indexing errors specifically
                if contains(ME.message, 'brace indexing') || contains(ME.message, 'comma separated list')
                    logMessage(config, 'Verbose', 'Trial %d: Brace indexing error - simulation output corrupted', trial_num);
                    logMessage(config, 'Debug', '  Error: %s', ME.message);
                else
                    logMessage(config, 'Verbose', 'Trial %d: Unexpected error accessing simulation output: %s', ...
                        trial_num, ME.message);
                end
            end
        end

        logMessage(config, 'Verbose', 'Batch %d completed: %d/%d trials successful', ...
            batch_idx, batch_successful, batch_trials);

    catch ME
        logMessage(config, 'Normal', 'Batch %d failed: %s', batch_idx, ME.message);
    end

    % Memory cleanup after each batch - optimized frequency
    % Only perform aggressive cleanup every 10 batches or on final batch
    if mod(batch_idx, 10) == 0 || batch_idx == num_batches
        logMessage(config, 'Debug', 'Performing memory cleanup after batch %d...', batch_idx);
        restoreWorkspace(initial_vars);
        % Force GC every 10 batches AND on final batch to ensure clean state
        if mod(batch_idx, 10) == 0 || batch_idx == num_batches
            java.lang.System.gc();
        end
    end

    % Save checkpoint if needed
    if mod(batch_idx, save_interval) == 0 || batch_idx == num_batches
        try
            checkpoint_data = struct();
            checkpoint_data.completed_trials = successful_trials;
            checkpoint_data.next_batch = batch_idx + 1;
            checkpoint_data.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
            checkpoint_data.batch_idx = batch_idx;
            checkpoint_data.total_batches = num_batches;

            save(checkpoint_file, '-struct', 'checkpoint_data');
            logMessage(config, 'Verbose', 'Checkpoint saved after batch %d (%d trials completed)', ...
                batch_idx, successful_trials);
        catch ME
            logMessage(config, 'Verbose', 'Warning: Could not save checkpoint: %s', ME.message);
        end
    end

    % Minimal pause only if needed - reduced from 0.5s to 0.1s
    % Skip pause entirely for small batches to maximize throughput
    if batch_idx < num_batches && num_batches > 5
        pause(0.1);
    end
end

% Final summary
logMessage(config, 'Normal', '\n=== PARALLEL BATCH PROCESSING SUMMARY ===');
logMessage(config, 'Normal', 'Total trials: %d', total_trials);
logMessage(config, 'Normal', 'Successful: %d', successful_trials);
logMessage(config, 'Normal', 'Failed: %d', total_trials - successful_trials);
logMessage(config, 'Normal', 'Success rate: %.1f%%', (successful_trials / total_trials) * 100);

if successful_trials == 0
    logMessage(config, 'Normal', '\nAll parallel simulations failed. Common causes:');
    logMessage(config, 'Normal', '   • Model path not accessible on workers');
    logMessage(config, 'Normal', '   • Missing workspace variables on workers');
    logMessage(config, 'Normal', '   • Toolbox licensing issues on workers');
    logMessage(config, 'Normal', '   • Model configuration conflicts in parallel mode');
    logMessage(config, 'Normal', '\n Try sequential mode for detailed debugging');
end

% Clean up checkpoint file if completed successfully
if successful_trials == total_trials && exist(checkpoint_file, 'file')
    try
        delete(checkpoint_file);
        logMessage(config, 'Verbose', 'Checkpoint file cleaned up (all trials completed)');
    catch ME
        logMessage(config, 'Verbose', 'Warning: Could not clean up checkpoint file: %s', ME.message);
    end
end

% Note: Pool left running for potential reuse
% User can manually delete with: delete(gcp('nocreate'))
logMessage(config, 'Verbose', 'Parallel pool left running for reuse');

% Finalize metadata
trial_metadata.end_time = datetime('now');
trial_metadata.successful_trials = successful_trials;
trial_metadata.failed_trials = total_trials - successful_trials;
trial_metadata.num_batches = num_batches;

end

function [successful_trials, trial_metadata] = runSequentialSimulations(config)
% RUNSEQUENTIALSIMULATIONS Execute simulations sequentially (no parallelization)
%
% Processes trials in batches with checkpoint/resume capability and memory management.
%
% Args:
%   config - Simulation configuration
%
% Returns:
%   successful_trials - Number of successful trials
%   trial_metadata - Metadata about execution

% Get batch processing parameters
batch_size = config.batch_size;
save_interval = config.save_interval;
total_trials = config.num_simulations;

% Initialize metadata
trial_metadata = struct();
trial_metadata.mode = 'sequential';
trial_metadata.start_time = datetime('now');
trial_metadata.batch_size = batch_size;

logMessage(config, 'Debug', '[RUNTIME] Using batch size: %d, save_interval: %d, verbosity: %s', ...
    batch_size, save_interval, config.verbosity);

logMessage(config, 'Normal', 'Starting sequential batch processing:');
logMessage(config, 'Normal', '  Total trials: %d', total_trials);
logMessage(config, 'Normal', '  Batch size: %d', batch_size);
logMessage(config, 'Verbose', '  Save interval: %d batches', save_interval);

% Calculate number of batches
num_batches = ceil(total_trials / batch_size);
successful_trials = 0;

% Store initial workspace state for restoration
initial_vars = who;

% Check for existing checkpoint
checkpoint_file = fullfile(config.output_folder, 'sequential_checkpoint.mat');
start_batch = 1;

if exist(checkpoint_file, 'file') && config.enable_checkpoint_resume
    try
        checkpoint_data = load(checkpoint_file);
        if isfield(checkpoint_data, 'completed_trials')
            successful_trials = checkpoint_data.completed_trials;
            start_batch = checkpoint_data.next_batch;
            logMessage(config, 'Normal', 'Found checkpoint: %d trials completed, resuming from batch %d', ...
                successful_trials, start_batch);
        end
    catch ME
        logMessage(config, 'Verbose', 'Warning: Could not load checkpoint: %s', ME.message);
    end
elseif exist(checkpoint_file, 'file') && ~config.enable_checkpoint_resume
    logMessage(config, 'Verbose', 'Checkpoint found but resume disabled - starting fresh');
end

% Process batches
batch_start_timer = tic;

for batch_idx = start_batch:num_batches
    % Calculate trials for this batch
    start_trial = (batch_idx - 1) * batch_size + 1;
    end_trial = min(batch_idx * batch_size, total_trials);
    batch_trials = end_trial - start_trial + 1;

    logMessage(config, 'Verbose', '\n--- Batch %d/%d (Trials %d-%d) ---', ...
        batch_idx, num_batches, start_trial, end_trial);

    % Update progress
    logMessage(config, 'Normal', 'Batch %d/%d: Processing trials %d-%d...', ...
        batch_idx, num_batches, start_trial, end_trial);

    % Process trials in this batch
    batch_successful = 0;
    for trial = start_trial:end_trial
        % Update progress with percentage
        progress_pct = (trial / total_trials) * 100;
        logMessage(config, 'Verbose', '  Trial %d/%d (%.1f%%)...', ...
            trial, total_trials, progress_pct);

        try
            % Get trial coefficients
            if trial <= size(config.coefficient_values, 1)
                trial_coefficients = config.coefficient_values(trial, :);
            else
                % Generate random coefficients for additional trials
                logMessage(config, 'Verbose', 'Generating random coefficients for trial %d', trial);
                trial_coefficients = generateRandomCoefficients(size(config.coefficient_values, 2));
            end

            % Run single trial
            result = runSingleTrial(trial, config, trial_coefficients, config.capture_workspace);

            if result.success
                batch_successful = batch_successful + 1;
                successful_trials = successful_trials + 1;
                logMessage(config, 'Debug', 'Trial %d completed successfully', trial);
            else
                logMessage(config, 'Normal', 'Trial %d failed: %s', trial, result.error);
            end

        catch ME
            logMessage(config, 'Normal', 'Trial %d error: %s', trial, ME.message);
        end
    end

    logMessage(config, 'Verbose', 'Batch %d completed: %d/%d trials successful', ...
        batch_idx, batch_successful, batch_trials);

    % Memory cleanup after each batch - optimized frequency
    % Only perform aggressive cleanup every 10 batches or on final batch
    if mod(batch_idx, 10) == 0 || batch_idx == num_batches
        logMessage(config, 'Debug', 'Performing memory cleanup after batch %d...', batch_idx);
        restoreWorkspace(initial_vars);
        % Force GC every 10 batches AND on final batch to ensure clean state
        if mod(batch_idx, 10) == 0 || batch_idx == num_batches
            java.lang.System.gc();
        end
    end

    % Save checkpoint if needed
    if mod(batch_idx, save_interval) == 0 || batch_idx == num_batches
        try
            checkpoint_data = struct();
            checkpoint_data.completed_trials = successful_trials;
            checkpoint_data.next_batch = batch_idx + 1;
            checkpoint_data.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
            checkpoint_data.batch_idx = batch_idx;
            checkpoint_data.total_batches = num_batches;

            save(checkpoint_file, '-struct', 'checkpoint_data');
            logMessage(config, 'Verbose', 'Checkpoint saved after batch %d (%d trials completed)', ...
                batch_idx, successful_trials);
        catch ME
            logMessage(config, 'Verbose', 'Warning: Could not save checkpoint: %s', ME.message);
        end
    end

    % Minimal pause only if needed - reduced from 0.5s to 0.1s
    % Skip pause entirely for small batches to maximize throughput
    if batch_idx < num_batches && num_batches > 5
        pause(0.1);
    end
end

% Final batch processing summary
logMessage(config, 'Normal', '\n=== SEQUENTIAL BATCH PROCESSING SUMMARY ===');
logMessage(config, 'Normal', 'Total trials: %d', total_trials);
logMessage(config, 'Normal', 'Successful: %d', successful_trials);
logMessage(config, 'Normal', 'Failed: %d', total_trials - successful_trials);
logMessage(config, 'Normal', 'Success rate: %.1f%%', (successful_trials / total_trials) * 100);

% Clean up checkpoint file if completed successfully
if successful_trials == total_trials && exist(checkpoint_file, 'file')
    try
        delete(checkpoint_file);
        logMessage(config, 'Verbose', 'Checkpoint file cleaned up (all trials completed)');
    catch ME
        logMessage(config, 'Verbose', 'Warning: Could not clean up checkpoint file: %s', ME.message);
    end
end

% Finalize metadata
trial_metadata.end_time = datetime('now');
trial_metadata.successful_trials = successful_trials;
trial_metadata.failed_trials = total_trials - successful_trials;
trial_metadata.num_batches = num_batches;

end

function dataset_path = compileDataset(config)
% COMPILEDATASET Compile individual trial files into master dataset
%
% This function uses an optimized 3-pass algorithm with preallocation:
%   Pass 1: Discover all unique column names across all trials
%   Pass 2: Standardize each trial to have all columns (NaN for missing)
%   Pass 3: Concatenate all standardized tables efficiently
%
% Args:
%   config - Configuration with output_folder field
%
% Returns:
%   dataset_path - Path to compiled master_dataset.csv

try
    logMessage(config, 'Normal', 'Compiling dataset from trials...');

    % Find all trial CSV files
    csv_files = dir(fullfile(config.output_folder, 'trial_*.csv'));

    if isempty(csv_files)
        warning('DataGenerator:NoTrialFiles', 'No trial CSV files found in output folder');
        dataset_path = '';
        return;
    end

    % OPTIMIZED THREE-PASS ALGORITHM with proper preallocation
    logMessage(config, 'Verbose', 'Using optimized 3-pass algorithm with preallocation...');

    % PASS 1: Discover all unique column names across all files
    logMessage(config, 'Verbose', 'Pass 1: Discovering columns...');

    % Preallocate with estimated size (most trials have similar column counts)
    estimated_columns = 2000;  % Buffer for comprehensive data extraction
    all_unique_columns = cell(estimated_columns, 1);
    valid_files = cell(length(csv_files), 1);
    column_count = 0;
    valid_file_count = 0;

    for i = 1:length(csv_files)
        file_path = fullfile(config.output_folder, csv_files(i).name);
        try
            trial_data = readtable(file_path);
            if ~isempty(trial_data)
                valid_file_count = valid_file_count + 1;
                valid_files{valid_file_count} = file_path;

                trial_columns = trial_data.Properties.VariableNames;

                % Add new columns efficiently
                for j = 1:length(trial_columns)
                    col_name = trial_columns{j};
                    if ~ismember(col_name, all_unique_columns(1:column_count))
                        column_count = column_count + 1;
                        if column_count <= length(all_unique_columns)
                            all_unique_columns{column_count} = col_name;
                        else
                            % This should not happen with proper estimation
                            warning('DataGenerator:ColumnOverflow', ...
                                'Column count exceeded estimation. Consider increasing estimated_columns.');
                            break;
                        end
                    end
                end

                logMessage(config, 'Debug', '  Pass 1 - %s: %d columns found', ...
                    csv_files(i).name, length(trial_columns));
            end
        catch ME
            warning('DataGenerator:TrialReadFailed', ...
                'Failed to read %s during discovery: %s', csv_files(i).name, ME.message);
        end
    end

    % Trim arrays to actual size
    all_unique_columns = all_unique_columns(1:column_count);
    valid_files = valid_files(1:valid_file_count);

    logMessage(config, 'Verbose', '  ✓ Pass 1: %d unique columns discovered from %d valid files', ...
        length(all_unique_columns), valid_file_count);

    % PASS 2: Standardize each trial to have all columns (with NaN for missing)
    logMessage(config, 'Verbose', 'Pass 2: Standardizing trials...');

    % Preallocate standardized tables array
    standardized_tables = cell(valid_file_count, 1);

    for i = 1:valid_file_count
        file_path = valid_files{i};
        [~, filename, ~] = fileparts(file_path);

        try
            trial_data = readtable(file_path);

            % Preallocate standardized data table with known size
            num_rows = height(trial_data);
            standardized_data = table();

            % Preallocate all columns at once for efficiency
            for col = 1:length(all_unique_columns)
                col_name = all_unique_columns{col};
                if ismember(col_name, trial_data.Properties.VariableNames)
                    standardized_data.(col_name) = trial_data.(col_name);
                else
                    % Fill missing column with NaN - preallocate entire column
                    standardized_data.(col_name) = NaN(num_rows, 1);
                end
            end

            standardized_tables{i} = standardized_data;
            logMessage(config, 'Debug', '  Pass 2 - %s: standardized to %d columns', ...
                filename, width(standardized_data));

        catch ME
            warning('DataGenerator:TrialStandardizeFailed', ...
                'Failed to standardize %s: %s', filename, ME.message);
            standardized_tables{i} = [];  % Mark as failed
        end
    end

    % PASS 3: Concatenate all standardized tables efficiently
    logMessage(config, 'Verbose', 'Pass 3: Concatenating data...');

    % Remove failed trials
    valid_tables = standardized_tables(~cellfun(@isempty, standardized_tables));

    if isempty(valid_tables)
        warning('DataGenerator:NoValidTables', 'No valid tables to concatenate');
        dataset_path = '';
        return;
    end

    % Preallocate master data with known dimensions
    total_rows = sum(cellfun(@height, valid_tables));
    master_data = table();

    % Preallocate all columns in master table
    for col = 1:length(all_unique_columns)
        col_name = all_unique_columns{col};
        master_data.(col_name) = NaN(total_rows, 1);
    end

    % Fill master table efficiently
    current_row = 1;
    for i = 1:length(valid_tables)
        trial_data = valid_tables{i};
        num_rows = height(trial_data);

        % Copy data for all columns
        for col = 1:length(all_unique_columns)
            col_name = all_unique_columns{col};
            if ismember(col_name, trial_data.Properties.VariableNames)
                master_data.(col_name)(current_row:current_row+num_rows-1) = trial_data.(col_name);
            end
        end

        current_row = current_row + num_rows;
    end

    % Save master dataset
    master_file = fullfile(config.output_folder, 'master_dataset.csv');
    writetable(master_data, master_file);

    dataset_path = master_file;

    logMessage(config, 'Normal', '  ✓ Master dataset compiled: %d rows × %d columns', ...
        height(master_data), width(master_data));

    % Check for 1956 column target (golf swing standard)
    if width(master_data) >= 1956
        logMessage(config, 'Normal', '  ✅ Target 1956 columns: ACHIEVED');
    else
        logMessage(config, 'Normal', '  ⚠️  Target 1956 columns: NOT achieved (%d found)', ...
            width(master_data));
    end

catch ME
    error('DataGenerator:CompilationFailed', ...
        'Error compiling dataset: %s', ME.message);
end

end

function saveScriptAndSettings(config)
% SAVESCRIPTANDSETTINGS Save simulation script and settings for reproducibility
%
% Creates a timestamped copy of the current script with a comprehensive
% header documenting all configuration settings used for the run.
%
% Args:
%   config - Configuration struct with all settings

try
    % Create timestamped filename
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    script_filename = sprintf('Data_GUI_run_%s.m', timestamp);
    script_path = fullfile(config.output_folder, script_filename);

    % Get the current script content
    current_script_path = mfilename('fullpath');
    current_script_path = [current_script_path '.m']; % Add .m extension

    if ~exist(current_script_path, 'file')
        logMessage(config, 'Verbose', 'Warning: Could not find current script file: %s', ...
            current_script_path);
        return;
    end

    % Read current script content
    fid_in = fopen(current_script_path, 'r');
    if fid_in == -1
        logMessage(config, 'Verbose', 'Warning: Could not open current script file for reading');
        return;
    end

    script_content = fread(fid_in, '*char')';
    fclose(fid_in);

    % Create output file with settings header
    fid_out = fopen(script_path, 'w');
    if fid_out == -1
        logMessage(config, 'Verbose', 'Warning: Could not create script copy file: %s', script_path);
        return;
    end

    % Write settings header
    fprintf(fid_out, '%% GOLF SWING DATA GENERATION RUN RECORD\n');
    fprintf(fid_out, '%% Generated: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
    fprintf(fid_out, '%% This file contains the exact script and settings used for this data generation run\n');
    fprintf(fid_out, '%%\n');
    fprintf(fid_out, '%% =================================================================\n');
    fprintf(fid_out, '%% RUN CONFIGURATION SETTINGS\n');
    fprintf(fid_out, '%% =================================================================\n');
    fprintf(fid_out, '%%\n');

    % Write all configuration settings
    fprintf(fid_out, '%% SIMULATION PARAMETERS:\n');
    fprintf(fid_out, '%% Number of trials: %d\n', config.num_simulations);
    if isfield(config, 'simulation_time')
        fprintf(fid_out, '%% Simulation time: %.3f seconds\n', config.simulation_time);
    end
    if isfield(config, 'sample_rate')
        fprintf(fid_out, '%% Sample rate: %.1f Hz\n', config.sample_rate);
    end
    fprintf(fid_out, '%%\n');

    % Torque scenario
    fprintf(fid_out, '%% TORQUE CONFIGURATION:\n');
    if isfield(config, 'torque_scenario')
        scenarios = {'Variable Torque', 'Zero Torque', 'Constant Torque'};
        if config.torque_scenario >= 1 && config.torque_scenario <= length(scenarios)
            fprintf(fid_out, '%% Torque scenario: %s\n', scenarios{config.torque_scenario});
        end
    end
    if isfield(config, 'coeff_range')
        fprintf(fid_out, '%% Coefficient range: %.3f\n', config.coeff_range);
    end
    if isfield(config, 'constant_torque_value')
        fprintf(fid_out, '%% Constant torque value: %.3f\n', config.constant_torque_value);
    end
    fprintf(fid_out, '%%\n');

    % Model information
    fprintf(fid_out, '%% MODEL INFORMATION:\n');
    if isfield(config, 'model_name')
        fprintf(fid_out, '%% Model name: %s\n', config.model_name);
    end
    if isfield(config, 'model_path')
        fprintf(fid_out, '%% Model path: %s\n', config.model_path);
    end
    fprintf(fid_out, '%%\n');

    % Data sources
    fprintf(fid_out, '%% DATA SOURCES ENABLED:\n');
    if isfield(config, 'use_signal_bus')
        fprintf(fid_out, '%% CombinedSignalBus: %s\n', logical2str(config.use_signal_bus));
    end
    if isfield(config, 'use_logsout')
        fprintf(fid_out, '%% Logsout Dataset: %s\n', logical2str(config.use_logsout));
    end
    if isfield(config, 'use_simscape')
        fprintf(fid_out, '%% Simscape Results: %s\n', logical2str(config.use_simscape));
    end
    fprintf(fid_out, '%%\n');

    % Output settings
    fprintf(fid_out, '%% OUTPUT SETTINGS:\n');
    if isfield(config, 'output_folder')
        fprintf(fid_out, '%% Output folder: %s\n', config.output_folder);
    end
    if isfield(config, 'dataset_name')
        fprintf(fid_out, '%% Dataset name: %s\n', config.dataset_name);
    end
    if isfield(config, 'file_format')
        formats = {'CSV Files', 'MAT Files', 'Both CSV and MAT'};
        if config.file_format >= 1 && config.file_format <= length(formats)
            fprintf(fid_out, '%% File format: %s\n', formats{config.file_format});
        end
    end
    fprintf(fid_out, '%%\n');

    % System information
    fprintf(fid_out, '%% SYSTEM INFORMATION:\n');
    fprintf(fid_out, '%% MATLAB version: %s\n', version);
    fprintf(fid_out, '%% Computer: %s\n', computer);
    try
        [~, hostname] = system('hostname');
        fprintf(fid_out, '%% Hostname: %s', hostname); % hostname already includes newline
    catch
        fprintf(fid_out, '%% Hostname: Unknown\n');
    end
    fprintf(fid_out, '%%\n');

    % Coefficient information if available
    if isfield(config, 'coefficient_values') && ~isempty(config.coefficient_values)
        fprintf(fid_out, '%% POLYNOMIAL COEFFICIENTS:\n');
        fprintf(fid_out, '%% Coefficient matrix size: %d trials x %d coefficients\n', ...
            size(config.coefficient_values, 1), size(config.coefficient_values, 2));

        % Show first few coefficients as example
        if size(config.coefficient_values, 1) > 0
            fprintf(fid_out, '%% First trial coefficients (first 10): ');
            coeffs_to_show = min(10, size(config.coefficient_values, 2));
            for i = 1:coeffs_to_show
                fprintf(fid_out, '%.3f', config.coefficient_values(1, i));
                if i < coeffs_to_show
                    fprintf(fid_out, ', ');
                end
            end
            fprintf(fid_out, '\n');
        end
        fprintf(fid_out, '%%\n');
    end

    fprintf(fid_out, '%% =================================================================\n');
    fprintf(fid_out, '%% END OF CONFIGURATION - ORIGINAL SCRIPT FOLLOWS\n');
    fprintf(fid_out, '%% =================================================================\n');
    fprintf(fid_out, '\n');

    % Write the original script content
    fprintf(fid_out, '%s', script_content);

    fclose(fid_out);

    logMessage(config, 'Verbose', 'Script and settings saved to: %s', script_path);

catch ME
    warning('DataGenerator:SaveScriptFailed', ...
        'Error saving script and settings: %s', ME.message);
end

end

function config = ensureEnhancedConfig(config)
% ENSUREENHANCEDCONFIG Add computed fields and ensure all required fields exist
%
% This function adds derived configuration fields and ensures backward
% compatibility with older config formats. Sets defaults for maximum
% data extraction capability.
%
% Args:
%   config - Configuration struct
%
% Returns:
%   config - Enhanced configuration with all defaults set

% Set default data extraction options for maximum column count
if ~isfield(config, 'use_signal_bus')
    config.use_signal_bus = true;  % Enable CombinedSignalBus extraction
end

if ~isfield(config, 'use_logsout')
    config.use_logsout = true;     % Enable logsout extraction
end

if ~isfield(config, 'use_simscape')
    config.use_simscape = true;    % Enable simscape extraction
end

% Ensure verbosity is set (map old 'verbose' field to new 'verbosity')
if ~isfield(config, 'verbosity')
    if isfield(config, 'verbose') && config.verbose
        config.verbosity = 'Verbose';
    else
        config.verbosity = 'Normal';  % Default to Normal
    end
end

% Set other important defaults for enhanced extraction
if ~isfield(config, 'capture_workspace')
    config.capture_workspace = true;  % Capture model workspace variables
end

% Ensure enable_master_dataset is set
if ~isfield(config, 'enable_master_dataset')
    config.enable_master_dataset = true;  % Enable by default
end

% Ensure checkpoint resume is set
if ~isfield(config, 'enable_checkpoint_resume')
    config.enable_checkpoint_resume = true;  % Enable by default
end

% Ensure save_script_backup is set
if ~isfield(config, 'save_script_backup')
    config.save_script_backup = false;  % Disabled by default (optional feature)
end

% Ensure memory monitoring is set
if ~isfield(config, 'enable_memory_monitoring')
    config.enable_memory_monitoring = false;  % Disabled by default for performance
end

end

%% ========================================================================
%  UTILITY FUNCTIONS
%% ========================================================================

function validateFunctionDependencies()
% VALIDATEFUNCTIONDEPENDENCIES Ensure critical functions are on the MATLAB path
%
% Throws an error if any required dataset generator helpers are missing from
% the MATLAB path. This provides a fast-fail check before running costly
% simulations or spinning up parallel pools.

required_functions = getRequiredFunctionList();
missing_functions = {};

for i = 1:numel(required_functions)
    func_name = required_functions{i};

    if exist(func_name, 'file') == 0
        missing_functions{end+1} = func_name; %#ok<AGROW>
    end
end

if ~isempty(missing_functions)
    formatted_list = strjoin(strcat(' - ', missing_functions), '\n');
    error('DataGenerator:MissingDependencies', ...
        ['Required functions are missing from the MATLAB path:\n', formatted_list, '\n', ...
         'Add the Dataset Generator functions folder to the MATLAB path before running.']);
end

end

function required_functions = getRequiredFunctionList()
% GETREQUIREDFUNCTIONLIST List of dataset generator helper functions
required_functions = {
    'validateSimulationConfig', ...
    'runSingleTrial', ...
    'processSimulationOutput', ...
    'setModelParameters', ...
    'setPolynomialCoefficients', ...
    'extractSignalsFromSimOut', ...
    'extractFromCombinedSignalBus', ...
    'extractFromNestedStruct', ...
    'extractLogsoutDataFixed', ...
    'extractSimscapeDataRecursive', ...
    'traverseSimlogNode', ...
    'extractDataFromField', ...
    'combineDataSources', ...
    'addModelWorkspaceData', ...
    'extractWorkspaceOutputs', ...
    'resampleDataToFrequency', ...
    'getPolynomialParameterInfo', ...
    'getShortenedJointName', ...
    'generateRandomCoefficients', ...
    'prepareSimulationInputsForBatch', ...
    'restoreWorkspace', ...
    'loadInputFile', ...
    'extractCoefficientsFromTable', ...
    'shouldShowDebug', ...
    'shouldShowVerbose', ...
    'shouldShowNormal', ...
    'mergeTables', ...
    'logical2str', ...
    'extractTimeSeriesData', ...
    'extractConstantMatrixData'
};

end

function logMessage(config, level, varargin)
% LOGMESSAGE Simple logging function with verbosity control
%
% Args:
%   config - Configuration with verbosity field
%   level - Message level: 'Silent', 'Normal', 'Verbose', 'Debug'
%   varargin - Format string and arguments (like fprintf)
%
% Verbosity Levels:
%   Silent  (0) - Only errors
%   Normal  (1) - Standard progress messages
%   Verbose (2) - Detailed progress + checkpoints
%   Debug   (3) - All messages including internals

% Define verbosity hierarchy
verbosity_levels = containers.Map(...
    {'Silent', 'Normal', 'Verbose', 'Debug'}, ...
    {0, 1, 2, 3});

% Get current and message levels
if isfield(config, 'verbosity') && isKey(verbosity_levels, config.verbosity)
    current_level = verbosity_levels(config.verbosity);
else
    current_level = 1; % Default to Normal
end

if isKey(verbosity_levels, level)
    message_level = verbosity_levels(level);
else
    message_level = 1; % Default to Normal
end

% Print if message level is within current verbosity
if message_level <= current_level
    fprintf(varargin{:});
    fprintf('\n');
end

end

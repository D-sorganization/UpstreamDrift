function checkpoint_recovery(varargin)
    % CHECKPOINT_RECOVERY - Recover and resume from dataset generation checkpoints
    %
    % Usage:
    %   checkpoint_recovery()                    % List available checkpoints
    %   checkpoint_recovery('list', folder)      % List checkpoints in folder
    %   checkpoint_recovery('resume', file)      % Resume from checkpoint
    %   checkpoint_recovery('analyze', file)     % Analyze checkpoint contents
    %   checkpoint_recovery('cleanup', folder)   % Clean up old checkpoints

    if nargin == 0
        listAvailableCheckpoints();
        return;
    end

    action = varargin{1};

    switch lower(action)
        case 'list'
            if nargin > 1
                folder = varargin{2};
            else
                folder = pwd;
            end
            listCheckpointsInFolder(folder);
        case 'resume'
            if nargin < 2
                error('Please specify checkpoint file to resume from');
            end
            checkpoint_file = varargin{2};
            resumeFromCheckpoint(checkpoint_file);
        case 'analyze'
            if nargin < 2
                error('Please specify checkpoint file to analyze');
            end
            checkpoint_file = varargin{2};
            analyzeCheckpoint(checkpoint_file);
        case 'cleanup'
            if nargin > 1
                folder = varargin{2};
            else
                folder = pwd;
            end
            cleanupOldCheckpoints(folder);
        otherwise
            error('Unknown action: %s', action);
    end
end

function listAvailableCheckpoints()
    % List all available checkpoints in current directory and subdirectories
    fprintf('Searching for checkpoints...\n');

    % Search for checkpoint files
    checkpoint_files = dir('**/checkpoint_*.mat');
    backup_files = dir('**/checkpoint_*_backup.mat');

    if isempty(checkpoint_files) && isempty(backup_files)
        fprintf('No checkpoint files found.\n');
        return;
    end

    fprintf('\n=== Available Checkpoints ===\n');

    % Process main checkpoint files
    for i = 1:length(checkpoint_files)
        file_path = fullfile(checkpoint_files(i).folder, checkpoint_files(i).name);
        displayCheckpointInfo(file_path);
    end

    % Process backup files
    for i = 1:length(backup_files)
        file_path = fullfile(backup_files(i).folder, backup_files(i).name);
        displayCheckpointInfo(file_path, ' (BACKUP)');
    end
end

function listCheckpointsInFolder(folder)
    % List checkpoints in specific folder
    if ~exist(folder, 'dir')
        error('Folder does not exist: %s', folder);
    end

    fprintf('Searching for checkpoints in: %s\n', folder);

    checkpoint_files = dir(fullfile(folder, 'checkpoint_*.mat'));

    if isempty(checkpoint_files)
        fprintf('No checkpoint files found in %s\n', folder);
        return;
    end

    fprintf('\n=== Checkpoints in %s ===\n', folder);

    for i = 1:length(checkpoint_files)
        file_path = fullfile(folder, checkpoint_files(i).name);
        displayCheckpointInfo(file_path);
    end
end

function displayCheckpointInfo(file_path, suffix)
    % Display information about a checkpoint file
    if nargin < 2
        suffix = '';
    end

    try
        % Load checkpoint info
        checkpoint_info = load(file_path, 'completed_trials', 'successful_count', 'timestamp', 'config');

        [~, filename] = fileparts(file_path);

        fprintf('\n%s%s:\n', filename, suffix);
        fprintf('  Location: %s\n', file_path);
        fprintf('  Created: %s\n', checkpoint_info.timestamp);
        fprintf('  Trials completed: %d\n', length(checkpoint_info.completed_trials));
        fprintf('  Successful trials: %d\n', checkpoint_info.successful_count);

        if isfield(checkpoint_info, 'config') && isfield(checkpoint_info.config, 'num_simulations')
            total_trials = checkpoint_info.config.num_simulations;
            progress_pct = 100 * length(checkpoint_info.completed_trials) / total_trials;
            fprintf('  Progress: %.1f%% (%d/%d)\n', progress_pct, ...
                length(checkpoint_info.completed_trials), total_trials);
        end

        % Check file size
        file_info = dir(file_path);
        fprintf('  File size: %.1f MB\n', file_info.bytes / 1024^2);

    catch ME
        fprintf('\n%s%s: ERROR - %s\n', filename, suffix, ME.message);
    end
end

function resumeFromCheckpoint(checkpoint_file)
    % Resume dataset generation from checkpoint
    if ~exist(checkpoint_file, 'file')
        error('Checkpoint file not found: %s', checkpoint_file);
    end

    fprintf('Resuming from checkpoint: %s\n', checkpoint_file);

    try
        % Load checkpoint
        checkpoint = load(checkpoint_file);

        % Validate checkpoint
        if ~isfield(checkpoint, 'config') || ~isfield(checkpoint, 'completed_trials')
            error('Invalid checkpoint file - missing required fields');
        end

        % Display resume information
        fprintf('\n=== Resume Information ===\n');
        fprintf('Checkpoint created: %s\n', checkpoint.timestamp);
        fprintf('Trials completed: %d\n', length(checkpoint.completed_trials));
        fprintf('Successful trials: %d\n', checkpoint.successful_count);
        fprintf('Total trials: %d\n', checkpoint.config.num_simulations);

        progress_pct = 100 * length(checkpoint.completed_trials) / checkpoint.config.num_simulations;
        fprintf('Progress: %.1f%%\n', progress_pct);

        % Ask for confirmation
        response = input('\nDo you want to resume from this checkpoint? (y/n): ', 's');
        if ~strcmpi(response, 'y')
            fprintf('Resume cancelled.\n');
            return;
        end

        % Robust dataset generator removed - checkpoint recovery not available
        fprintf('\nCheckpoint recovery not available - robust dataset generator has been removed.\n');
        fprintf('Please restart dataset generation from the beginning.\n');

    catch ME
        fprintf('Error resuming from checkpoint: %s\n', ME.message);
        rethrow(ME);
    end
end

function analyzeCheckpoint(checkpoint_file)
    % Analyze checkpoint contents in detail
    if ~exist(checkpoint_file, 'file')
        error('Checkpoint file not found: %s', checkpoint_file);
    end

    fprintf('Analyzing checkpoint: %s\n', checkpoint_file);

    try
        % Load checkpoint
        checkpoint = load(checkpoint_file);

        fprintf('\n=== Checkpoint Analysis ===\n');
        fprintf('File: %s\n', checkpoint_file);
        fprintf('Created: %s\n', checkpoint.timestamp);

        % Basic statistics
        fprintf('\nBasic Statistics:\n');
        fprintf('  Trials completed: %d\n', length(checkpoint.completed_trials));
        fprintf('  Successful trials: %d\n', checkpoint.successful_count);
        fprintf('  Failed trials: %d\n', length(checkpoint.completed_trials) - checkpoint.successful_count);

        if isfield(checkpoint, 'config')
            fprintf('  Total trials planned: %d\n', checkpoint.config.num_simulations);
            progress_pct = 100 * length(checkpoint.completed_trials) / checkpoint.config.num_simulations;
            fprintf('  Progress: %.1f%%\n', progress_pct);
        end

        % Analyze results if available
        if isfield(checkpoint, 'all_results') && ~isempty(checkpoint.all_results)
            fprintf('\nResults Analysis:\n');

            % Count different types of errors
            error_types = {};
            for i = 1:length(checkpoint.all_results)
                if ~checkpoint.all_results{i}.success && isfield(checkpoint.all_results{i}, 'error')
                    error_msg = checkpoint.all_results{i}.error;
                    if ~ismember(error_types, error_msg)
                        error_types{end+1} = error_msg;
                    end
                end
            end

            if ~isempty(error_types)
                fprintf('  Error types found:\n');
                for i = 1:length(error_types)
                    error_count = sum(cellfun(@(x) ~x.success && isfield(x, 'error') && ...
                        strcmp(x.error, error_types{i}), checkpoint.all_results));
                    fprintf('    %s: %d occurrences\n', error_types{i}, error_count);
                end
            end

            % Success rate by batch
            if length(checkpoint.completed_trials) > 10
                fprintf('\nSuccess Rate Analysis:\n');
                batch_size = 100;
                num_batches = ceil(length(checkpoint.completed_trials) / batch_size);

                for batch = 1:num_batches
                    start_idx = (batch-1) * batch_size + 1;
                    end_idx = min(batch * batch_size, length(checkpoint.completed_trials));

                    batch_results = checkpoint.all_results(start_idx:end_idx);
                    batch_success = sum([batch_results.success]);
                    batch_rate = 100 * batch_success / length(batch_results);

                    fprintf('  Batch %d (trials %d-%d): %.1f%% success\n', ...
                        batch, start_idx, end_idx, batch_rate);
                end
            end
        end

        % File information
        file_info = dir(checkpoint_file);
        fprintf('\nFile Information:\n');
        fprintf('  Size: %.1f MB\n', file_info.bytes / 1024^2);
        fprintf('  Last modified: %s\n', datestr(file_info.datenum));

    catch ME
        fprintf('Error analyzing checkpoint: %s\n', ME.message);
        rethrow(ME);
    end
end

function cleanupOldCheckpoints(folder)
    % Clean up old checkpoint files
    if ~exist(folder, 'dir')
        error('Folder does not exist: %s', folder);
    end

    fprintf('Cleaning up old checkpoints in: %s\n', folder);

    % Find checkpoint files
    checkpoint_files = dir(fullfile(folder, 'checkpoint_*.mat'));

    if isempty(checkpoint_files)
        fprintf('No checkpoint files found to clean up.\n');
        return;
    end

    % Sort by date (oldest first)
    [~, sort_idx] = sort([checkpoint_files.datenum]);
    checkpoint_files = checkpoint_files(sort_idx);

    fprintf('\nFound %d checkpoint files:\n', length(checkpoint_files));

    for i = 1:length(checkpoint_files)
        file_path = fullfile(folder, checkpoint_files(i).name);
        file_info = dir(file_path);

        fprintf('  %s: %.1f MB, %s\n', checkpoint_files(i).name, ...
            file_info.bytes / 1024^2, datestr(file_info.datenum));
    end

    % Keep only the most recent 3 checkpoints
    if length(checkpoint_files) > 3
        files_to_delete = checkpoint_files(1:end-3);

        fprintf('\nWill delete %d old checkpoint files:\n', length(files_to_delete));
        for i = 1:length(files_to_delete)
            fprintf('  %s\n', files_to_delete(i).name);
        end

        response = input('\nProceed with deletion? (y/n): ', 's');
        if strcmpi(response, 'y')
            for i = 1:length(files_to_delete)
                file_path = fullfile(folder, files_to_delete(i).name);
                try
                    delete(file_path);
                    fprintf('Deleted: %s\n', files_to_delete(i).name);
                catch ME
                    fprintf('Failed to delete %s: %s\n', files_to_delete(i).name, ME.message);
                end
            end
            fprintf('Cleanup completed.\n');
        else
            fprintf('Cleanup cancelled.\n');
        end
    else
        fprintf('\nNo cleanup needed - only %d checkpoint files found.\n', length(checkpoint_files));
    end
end

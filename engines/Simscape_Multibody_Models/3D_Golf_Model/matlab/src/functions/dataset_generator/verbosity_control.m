function verbosity_control(varargin)
    % VERBOSITY_CONTROL - Control output verbosity levels
    %
    % Usage:
    %   verbosity_control()                    % Show current verbosity level
    %   verbosity_control('silent')            % Minimal output (performance mode)
    %   verbosity_control('normal')            % Standard output (default)
    %   verbosity_control('verbose')           % Detailed output
    %   verbosity_control('debug')             % Maximum debugging output
    %   verbosity_control('set', level)        % Set specific level
    %   verbosity_control('test')              % Test all verbosity levels

    global verbosity_level;

    % Initialize default verbosity level if not set
    if isempty(verbosity_level)
        verbosity_level = 'normal';
    end

    if nargin == 0
        displayVerbosityStatus();
        return;
    end

    action = varargin{1};

    switch lower(action)
        case 'silent'
            setVerbosityLevel('silent');
        case 'normal'
            setVerbosityLevel('normal');
        case 'verbose'
            setVerbosityLevel('verbose');
        case 'debug'
            setVerbosityLevel('debug');
        case 'set'
            if nargin > 1
                setVerbosityLevel(varargin{2});
            else
                error('Please specify verbosity level');
            end
        case 'test'
            testVerbosityLevels();
        case 'get'
            displayVerbosityStatus();
        otherwise
            error('Unknown verbosity level: %s', action);
    end
end

function setVerbosityLevel(level)
    % Set the verbosity level
    global verbosity_level;

    valid_levels = {'silent', 'normal', 'verbose', 'debug'};

    if ~ismember(lower(level), valid_levels)
        error('Invalid verbosity level: %s. Valid levels: %s', ...
            level, strjoin(valid_levels, ', '));
    end

    verbosity_level = lower(level);

    % Display confirmation based on new level
    switch verbosity_level
        case 'silent'
            fprintf('Verbosity set to SILENT - minimal output for maximum performance\n');
        case 'normal'
            fprintf('Verbosity set to NORMAL - standard output level\n');
        case 'verbose'
            fprintf('Verbosity set to VERBOSE - detailed progress information\n');
        case 'debug'
            fprintf('Verbosity set to DEBUG - maximum debugging output\n');
    end
end

function displayVerbosityStatus()
    % Display current verbosity status
    global verbosity_level;

    fprintf('\n=== Verbosity Control ===\n');
    fprintf('Current level: %s\n', upper(verbosity_level));

    fprintf('\nAvailable levels:\n');
    fprintf('  SILENT   - Minimal output (performance mode)\n');
    fprintf('  NORMAL   - Standard output (default)\n');
    fprintf('  VERBOSE  - Detailed progress information\n');
    fprintf('  DEBUG    - Maximum debugging output\n');

    fprintf('\nUsage:\n');
    fprintf('  verbosity_control(''silent'')   - Set to silent mode\n');
    fprintf('  verbosity_control(''normal'')   - Set to normal mode\n');
    fprintf('  verbosity_control(''verbose'')  - Set to verbose mode\n');
    fprintf('  verbosity_control(''debug'')    - Set to debug mode\n');
end

function testVerbosityLevels()
    % Test all verbosity levels
    fprintf('Testing verbosity levels...\n\n');

    levels = {'silent', 'normal', 'verbose', 'debug'};

    for i = 1:length(levels)
        fprintf('=== Testing %s level ===\n', upper(levels{i}));
        setVerbosityLevel(levels{i});

        % Test different types of messages
        logMessage('info', 'This is an info message');
        logMessage('progress', 'Processing trial 1/100...');
        logMessage('warning', 'This is a warning message');
        logMessage('error', 'This is an error message');
        logMessage('debug', 'This is a debug message');
        logMessage('performance', 'Trial completed in 1.23 seconds');

        fprintf('\n');
    end

    % Reset to normal
    setVerbosityLevel('normal');
    fprintf('Verbosity testing complete - reset to NORMAL\n');
end

function logMessage(message_type, message, varargin)
    % Log a message based on current verbosity level
    global verbosity_level;

    if isempty(verbosity_level)
        verbosity_level = 'normal';
    end

    % Determine if message should be displayed
    should_display = shouldDisplayMessage(message_type, verbosity_level);

    if should_display
        % Format the message
        formatted_message = formatMessage(message_type, message, varargin{:});

        % Display the message
        fprintf('%s\n', formatted_message);
    end
end

function should_display = shouldDisplayMessage(message_type, verbosity_level)
    % Determine if a message should be displayed based on verbosity level

    % Define message priorities
    message_priorities = struct();
    message_priorities.error = 1;      % Always show errors
    message_priorities.warning = 2;    % Show warnings in normal+
    message_priorities.info = 3;       % Show info in normal+
    message_priorities.progress = 4;   % Show progress in verbose+
    message_priorities.performance = 5; % Show performance in verbose+
    message_priorities.debug = 6;      % Show debug only in debug mode

    % Define verbosity level thresholds
    verbosity_thresholds = struct();
    verbosity_thresholds.silent = 1;   % Only errors
    verbosity_thresholds.normal = 3;   % Errors, warnings, info
    verbosity_thresholds.verbose = 5;  % All except debug
    verbosity_thresholds.debug = 6;    % Everything

    % Get message priority
    if isfield(message_priorities, message_type)
        message_priority = message_priorities.(message_type);
    else
        message_priority = 3; % Default to info level
    end

    % Get verbosity threshold
    threshold = verbosity_thresholds.(verbosity_level);

    % Determine if message should be displayed
    should_display = message_priority <= threshold;
end

function formatted_message = formatMessage(message_type, message, varargin)
    % Format a message with appropriate prefix and styling

    % Define message prefixes and colors
    prefixes = struct();
    prefixes.error = '❌ ERROR: ';
    prefixes.warning = '⚠️  WARNING: ';
    prefixes.info = 'ℹ️  INFO: ';
    prefixes.progress = '🔄 ';
    prefixes.performance = '⚡ ';
    prefixes.debug = '🐛 DEBUG: ';

    % Get prefix
    if isfield(prefixes, message_type)
        prefix = prefixes.(message_type);
    else
        prefix = '';
    end

    % Format the message with any additional arguments
    if ~isempty(varargin)
        formatted_message = sprintf([prefix, message], varargin{:});
    else
        formatted_message = [prefix, message];
    end
end

function logProgress(current, total, message)
    % Log progress with percentage
    global verbosity_level;

    if strcmp(verbosity_level, 'silent')
        return;
    end

    percentage = 100 * current / total;

    if strcmp(verbosity_level, 'normal')
        % Simple progress in normal mode
        fprintf('\r%s: %d/%d (%.1f%%)', message, current, total, percentage);
        if current == total
            fprintf('\n');
        end
    else
        % Detailed progress in verbose/debug mode
        logMessage('progress', sprintf('%s: %d/%d (%.1f%%)', message, current, total, percentage));
    end
end

function logPerformance(operation, duration, details)
    % Log performance metrics
    global verbosity_level;

    if strcmp(verbosity_level, 'silent')
        return;
    end

    if strcmp(verbosity_level, 'normal')
        % Only log significant operations in normal mode
        if duration > 1.0
            logMessage('performance', '%s completed in %.2f seconds', operation, duration);
        end
    else
        % Log all performance metrics in verbose/debug mode
        if nargin > 2
            logMessage('performance', '%s completed in %.3f seconds (%s)', operation, duration, details);
        else
            logMessage('performance', '%s completed in %.3f seconds', operation, duration);
        end
    end
end

function logTrialResult(trial_num, success, duration, error_msg)
    % Log trial results with appropriate verbosity
    global verbosity_level;

    if strcmp(verbosity_level, 'silent')
        return;
    end

    if success
        if strcmp(verbosity_level, 'normal')
            % Only log failures in normal mode
            return;
        else
            logMessage('info', 'Trial %d completed successfully in %.2f seconds', trial_num, duration);
        end
    else
        if strcmp(verbosity_level, 'silent')
            return;
        elseif strcmp(verbosity_level, 'normal')
            logMessage('warning', 'Trial %d failed', trial_num);
        else
            logMessage('error', 'Trial %d failed after %.2f seconds: %s', trial_num, duration, error_msg);
        end
    end
end

function logBatchResult(batch_num, batch_size, successful, failed, duration)
    % Log batch results
    global verbosity_level;

    if strcmp(verbosity_level, 'silent')
        return;
    end

    success_rate = 100 * successful / batch_size;

    if strcmp(verbosity_level, 'normal')
        logMessage('info', 'Batch %d: %d/%d successful (%.1f%%) in %.1f seconds', ...
            batch_num, successful, batch_size, success_rate, duration);
    else
        logMessage('info', 'Batch %d completed: %d successful, %d failed (%.1f%% success rate) in %.2f seconds', ...
            batch_num, successful, failed, success_rate, duration);
    end
end

function logMemoryStatus(current_memory, change)
    % Log memory status
    global verbosity_level;

    if strcmp(verbosity_level, 'silent')
        return;
    end

    if strcmp(verbosity_level, 'normal')
        % Only log significant memory changes in normal mode
        if abs(change) > 100
            if change < 0
                logMessage('warning', 'Memory usage increased by %.1f MB (%.1f MB available)', -change, current_memory);
            else
                logMessage('info', 'Memory freed: %.1f MB (%.1f MB available)', change, current_memory);
            end
        end
    else
        % Log all memory changes in verbose/debug mode
        if change < 0
            logMessage('performance', 'Memory usage: %.1f MB (change: -%.1f MB)', current_memory, -change);
        else
            logMessage('performance', 'Memory usage: %.1f MB (change: +%.1f MB)', current_memory, change);
        end
    end
end

function logCheckpoint(save_time, file_size)
    % Log checkpoint information
    global verbosity_level;

    if strcmp(verbosity_level, 'silent')
        return;
    end

    if strcmp(verbosity_level, 'normal')
        logMessage('info', 'Checkpoint saved (%.1f MB) in %.2f seconds', file_size, save_time);
    else
        logMessage('info', 'Checkpoint saved: %.1f MB in %.3f seconds', file_size, save_time);
    end
end

function logWorkspaceCapture(enabled, num_variables)
    % Log workspace capture status
    global verbosity_level;

    if isempty(verbosity_level)
        verbosity_level = 'normal';
    end

    if strcmp(verbosity_level, 'silent')
        return;
    end

    if enabled
        if strcmp(verbosity_level, 'normal')
            logMessage('info', 'Workspace capture enabled (%d variables)', num_variables);
        else
            logMessage('info', 'Model workspace capture enabled: %d variables will be included', num_variables);
        end
    else
        logMessage('info', 'Workspace capture disabled - model parameters excluded');
    end
end

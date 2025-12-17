function logMessage(message_type, message, varargin)
    % LOGMESSAGE - Log a message based on current verbosity level
    %
    % Usage:
    %   logMessage('info', 'Processing trial %d', trial_num);
    %   logMessage('warning', 'Low memory detected');
    %   logMessage('error', 'Simulation failed: %s', error_msg);
    %
    % Message types: error, warning, info, progress, performance, debug

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

function printProgressBar(current, total, rate, varargin)
% PRINTPROGRESSBAR Display progress bar with ETA
%
% Displays a visual progress bar with percentage, trial count, rate, and
% estimated time remaining. Uses ANSI escape codes to overwrite the
% previous line for a clean, updating display.
%
% Args:
%   current - Current trial number (1 to total)
%   total - Total number of trials
%   rate - Processing rate in trials/second
%   varargin - Optional name-value pairs:
%     'message' - Additional message to display below progress bar
%     'force_newline' - Force newline instead of overwrite (default: false)
%
% Example:
%   % Show progress at 50/100 trials, 2.5 trials/sec
%   printProgressBar(50, 100, 2.5);
%   % Output: [██████████░░░░░░░░░░] 50% | 50/100 trials | 2.5 trials/sec | ETA: 00:20
%
%   % With additional message
%   printProgressBar(50, 100, 2.5, 'message', '✓ Batch 5/10 complete');
%
%   % Force newline (for logging to file)
%   printProgressBar(100, 100, 2.3, 'force_newline', true);
%
% See also: printSimulationHeader, printSimulationSummary

%% Parse Arguments

p = inputParser;
p.addParameter('message', '', @ischar);
p.addParameter('force_newline', false, @islogical);
p.parse(varargin{:});

message = p.Results.message;
force_newline = p.Results.force_newline;

%% Calculate Progress Metrics

% Percentage complete
percent = (current / total) * 100;

% ETA calculation
if rate > 0
    eta_seconds = (total - current) / rate;
else
    eta_seconds = 0;
end

%% Create Progress Bar

% Bar width (20 characters for clean display)
bar_width = 20;
filled = round((current / total) * bar_width);

% Use block characters for visual appeal
% █ = filled, ░ = empty
bar = [repmat('█', 1, filled), repmat('░', 1, bar_width - filled)];

%% Format ETA String

if current == total
    eta_str = 'Done!';
elseif eta_seconds == 0
    eta_str = 'Calculating...';
elseif eta_seconds < 60
    % Under 1 minute: show seconds
    eta_str = sprintf('ETA: %02.0f sec', eta_seconds);
elseif eta_seconds < 3600
    % 1 minute to 1 hour: show MM:SS
    minutes = floor(eta_seconds / 60);
    seconds = mod(eta_seconds, 60);
    eta_str = sprintf('ETA: %02.0f:%02.0f', minutes, seconds);
else
    % Over 1 hour: show HH:MM:SS
    hours = floor(eta_seconds / 3600);
    minutes = floor(mod(eta_seconds, 3600) / 60);
    seconds = mod(eta_seconds, 60);
    eta_str = sprintf('ETA: %02.0f:%02.0f:%02.0f', hours, minutes, seconds);
end

%% Print Progress Bar

if current == total || force_newline
    % Final message or forced newline - add newline
    fprintf('[%s] 100%% | %d/%d trials | %.1f trials/sec | Done!\n', ...
        bar, current, total, rate);
else
    % Progress message - use \r to overwrite previous line
    % Note: \r (carriage return) moves cursor to start of line
    fprintf('\r[%s] %3.0f%% | %d/%d trials | %.1f trials/sec | %s', ...
        bar, percent, current, total, rate, eta_str);
end

%% Print Optional Message

if ~isempty(message)
    fprintf('\n  %s', message);
    if ~force_newline && current < total
        % If we're going to overwrite next time, add extra space
        fprintf('\n');
    else
        fprintf('\n');
    end
end

end

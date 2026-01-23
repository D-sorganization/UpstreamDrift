function timestampPrintf(varargin)
% TIMESTAMPPRINTF - Print formatted text with automatic timestamp
%
% Usage:
%   timestampPrintf(format_str, var1, var2, ...)     % Print to command window
%   timestampPrintf(fid, format_str, var1, var2, ...) % Print to file
%
% This function automatically prepends a timestamp to all output, making it
% easy to track timing and troubleshoot performance issues.
%
% Examples:
%   timestampPrintf('Starting batch processing...\n');
%   timestampPrintf('Batch %d completed\n', batch_num);
%   timestampPrintf(fid, 'Trial %d failed: %s\n', trial, error_msg);
%
% Author: Performance Monitoring Enhancement
% Date: 2025-01-22

% Handle different calling patterns
if nargin == 0
    return;
end

% Check if first argument is a file ID (numeric and not stdout/stderr)
if nargin >= 2 && isnumeric(varargin{1}) && varargin{1} ~= 1 && varargin{1} ~= 2
    % File output: fid, format_str, varargin
    file_id = varargin{1};
    format_string = varargin{2};
    args = varargin(3:end);
else
    % Command window output: format_str, varargin
    file_id = 1; % stdout
    format_string = varargin{1};
    args = varargin(2:end);
end

% Generate timestamp
timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

% Format the message
if isempty(args)
    message = format_string;
else
    message = sprintf(format_string, args{:});
end

% Add timestamp and print
timestamped_message = sprintf('[%s] %s', timestamp, message);

if file_id == 1 || file_id == 2
    % Command window output
    fprintf(timestamped_message);
else
    % File output
    fprintf(file_id, timestamped_message);
end

end

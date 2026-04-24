function config = ensureEnhancedConfig(config)
% Ensure config has enhanced settings for maximum data extraction
% This function sets default values for data extraction options if they're missing

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

% Ensure verbose logging is available for downstream extractors.
% Prefer the explicit boolean flag when present, otherwise derive it from
% the human-readable verbosity level used by the GUI and CLI.
if ~isfield(config, 'verbose') || isempty(config.verbose)
    if isfield(config, 'verbosity')
        config.verbose = ~strcmpi(config.verbosity, 'Silent');
    else
        config.verbose = true;
    end
end

% Set other important defaults for enhanced extraction
if ~isfield(config, 'capture_workspace')
    config.capture_workspace = true;  % Capture model workspace variables
end
end

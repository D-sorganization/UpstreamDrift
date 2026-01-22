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

% Ensure verbose logging is enabled for debugging
if ~isfield(config, 'verbose')
    config.verbose = true;
end

% Set other important defaults for enhanced extraction
if ~isfield(config, 'capture_workspace')
    config.capture_workspace = true;  % Capture model workspace variables
end
end

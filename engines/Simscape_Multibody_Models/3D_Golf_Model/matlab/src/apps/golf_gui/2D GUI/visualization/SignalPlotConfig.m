function config = SignalPlotConfig(action, varargin)
% SIGNALPLOTCONFIG - Configuration manager for Interactive Signal Plotter
%
% Usage:
%   config = SignalPlotConfig('load')  - Load saved configuration
%   SignalPlotConfig('save', config)   - Save configuration
%   config = SignalPlotConfig('default') - Get default configuration
%
% Configuration Structure:
%   config.hotlist_signals     - Cell array of signal names in hotlist
%   config.last_selected       - Cell array of last selected signals
%   config.plot_mode           - 'single' or 'subplot'
%   config.window_position     - [x, y, width, height] (optional)
%   config.prioritized_signals - Force/torque signals (auto-detected)
%
% The configuration is saved to:
%   matlab/Scripts/Golf_GUI/2D GUI/config/signal_plot_config.mat

% Get the path to the config directory
config_dir = fileparts(mfilename('fullpath'));
config_dir = fullfile(config_dir, '..', 'config');
config_file = fullfile(config_dir, 'signal_plot_config.mat');

switch lower(action)
    case 'load'
        config = load_config(config_file);

    case 'save'
        if nargin < 2
            error('SignalPlotConfig:save requires a config structure');
        end
        save_config(config_file, varargin{1});
        config = [];

    case 'default'
        config = get_default_config();

    otherwise
        error('Unknown action: %s. Valid actions are: load, save, default', action);
end
end

function config = load_config(config_file)
% Load configuration from file, or return default if file doesn't exist

if exist(config_file, 'file')
    try
        loaded = load(config_file, 'config');
        config = loaded.config;
        fprintf('📊 Loaded signal plot configuration from: %s\n', config_file);

        % Validate loaded config
        config = validate_config(config);

    catch ME
        warning('SignalPlotConfig:LoadFailed', 'Failed to load config file: %s. Using defaults.', ME.message);
        config = get_default_config();
    end
else
    fprintf('📊 No saved configuration found. Using defaults.\n');
    config = get_default_config();
end
end

function save_config(config_file, config)
% Save configuration to file

% Create config directory if it doesn't exist
config_dir = fileparts(config_file);
if ~exist(config_dir, 'dir')
    mkdir(config_dir);
end

try
    % Validate before saving
    config = validate_config(config);

    % Save
    save(config_file, 'config');
    fprintf('💾 Signal plot configuration saved to: %s\n', config_file);

catch ME
    warning('SignalPlotConfig:SaveFailed', 'Failed to save config file: %s', ME.message);
end
end

function config = get_default_config()
% Return default configuration

config = struct();

% Default hotlist includes common force/torque signals
config.hotlist_signals = {
    'TotalHandForceGlobal'
    'EquivalentMidpointCoupleGlobal'
    };

% No signals selected by default
config.last_selected = {};

% Default to single plot mode
config.plot_mode = 'single';

% Default window position (will be auto-positioned if empty)
config.window_position = [];

% Prioritized signal patterns (force/torque related)
config.prioritized_patterns = {
    'Force', 'Torque', 'Couple', 'Power', 'Work', 'Energy'
    };
end

function config = validate_config(config)
% Validate and repair configuration structure

default_config = get_default_config();

% Ensure all required fields exist
if ~isfield(config, 'hotlist_signals')
    config.hotlist_signals = default_config.hotlist_signals;
end

if ~isfield(config, 'last_selected')
    config.last_selected = default_config.last_selected;
end

if ~isfield(config, 'plot_mode')
    config.plot_mode = default_config.plot_mode;
end

if ~isfield(config, 'window_position')
    config.window_position = default_config.window_position;
end

if ~isfield(config, 'prioritized_patterns')
    config.prioritized_patterns = default_config.prioritized_patterns;
end

% Validate types
if ~iscell(config.hotlist_signals)
    config.hotlist_signals = default_config.hotlist_signals;
end

if ~iscell(config.last_selected)
    config.last_selected = default_config.last_selected;
end

if ~ischar(config.plot_mode) && ~isstring(config.plot_mode)
    config.plot_mode = default_config.plot_mode;
elseif ~ismember(lower(config.plot_mode), {'single', 'subplot'})
    config.plot_mode = default_config.plot_mode;
end

% Clean up last_selected - remove signals not in hotlist
if ~isempty(config.last_selected)
    config.last_selected = intersect(config.last_selected, config.hotlist_signals, 'stable');
end
end

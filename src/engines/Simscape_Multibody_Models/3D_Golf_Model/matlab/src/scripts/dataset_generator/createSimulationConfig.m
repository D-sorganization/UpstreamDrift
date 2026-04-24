function config = createSimulationConfig(varargin)
% CREATESIMULATIONCONFIG Create simulation configuration with sensible defaults
%
% This function creates a complete simulation configuration structure with
% sensible defaults for all required fields. It supports both name-value pair
% arguments and struct merging for customization.
%
% Args:
%   varargin - Name-value pairs or struct to override defaults
%
% Returns:
%   config - Complete simulation configuration struct
%
% Examples:
%   % Create default configuration
%   config = createSimulationConfig();
%
%   % Create with custom parameters
%   config = createSimulationConfig('num_simulations', 100, 'execution_mode', 'parallel');
%
%   % Merge with existing config
%   custom = struct('num_simulations', 50, 'batch_size', 20);
%   config = createSimulationConfig(custom);
%
% See also: DATA_GENERATOR, RUNSIMULATION, VALIDATESIMULATIONCONFIG

% Input validation
arguments(Repeating)
    varargin
end

%% Create default configuration

config = struct();

% === Model Configuration ===
config.model_name = 'GolfSwing3D_Kinetic';

% Try to find model path automatically
config.model_path = resolveModelPath(config.model_name);

% Optional input file
config.input_file = '';  % Empty = use generated data

% === Simulation Parameters ===
config.num_simulations = 10;           % Number of trials
config.simulation_time = 0.5;          % Simulation duration (seconds)
config.sample_rate = 1000;             % Data sampling rate (Hz)
config.modeling_mode = 3;              % Modeling mode (forward dynamics)

% === Execution Configuration ===
config.execution_mode = 'sequential';  % 'sequential' or 'parallel'
config.batch_size = 10;                % Trials per batch
config.save_interval = 5;              % Batches between checkpoint saves

% === Torque Scenario Configuration ===
config.torque_scenario = 1;            % 1=Variable, 2=Zero, 3=Constant
config.coeff_range = 10.0;             % Range for random coefficient generation
config.constant_value = 10.0;          % Value for constant torque scenario

% Optional: Pre-generated coefficient values
% If empty, will be generated during simulation
config.coefficient_values = [];

% === Data Source Configuration ===
config.use_logsout = true;             % Extract data from logsout
config.use_signal_bus = true;          % Extract data from signal bus
config.use_simscape = true;            % Extract data from Simscape

% === Output Configuration ===
% Default output to current directory
config.output_folder = fullfile(pwd, 'simulation_output');

% Default folder name with timestamp
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
config.folder_name = ['dataset_' timestamp];

% === Optional Features ===
config.enable_animation = false;           % Enable Simulink animation (slower)
config.capture_workspace = false;          % Capture workspace variables
config.enable_memory_monitoring = false;   % Monitor memory usage
config.enable_checkpoint_resume = true;    % Enable resume from checkpoint
config.enable_master_dataset = true;       % Compile master dataset CSV

% === Verbosity Configuration ===
% Options: 'Silent', 'Normal', 'Verbose', 'Debug'
config.verbosity = 'Normal';
config.verbose = deriveVerboseFlag(config.verbosity);

% === Advanced Configuration ===
config.stop_on_error = false;              % Continue on simulation errors
config.timeout_seconds = 300;              % Maximum time per simulation (seconds)

%% Process custom overrides
explicit_verbose_override = false;

if nargin > 0
    % Check if first argument is a struct (merge mode)
    if isstruct(varargin{1})
        % Merge custom struct with defaults
        custom_config = varargin{1};
        explicit_verbose_override = isfield(custom_config, 'verbose');
        custom_fields = fieldnames(custom_config);

        for i = 1:length(custom_fields)
            field_name = custom_fields{i};
            config.(field_name) = custom_config.(field_name);
        end

        % Process remaining name-value pairs if any
        if nargin > 1
            explicit_verbose_override = explicit_verbose_override || containsParameterName(varargin{2:end}, 'verbose');
            config = applyNameValuePairs(config, varargin{2:end});
        end
    else
        % Process name-value pairs
        explicit_verbose_override = containsParameterName(varargin{:}, 'verbose');
        config = applyNameValuePairs(config, varargin{:});
    end
end

% Keep the legacy boolean logging flag aligned with the human-readable level
% unless the caller explicitly supplied their own boolean override.
if ~explicit_verbose_override
    config.verbose = deriveVerboseFlag(config.verbosity);
end

%% Validate configuration

try
    validateSimulationConfig(config);
catch ME
    warning('DataGenerator:ConfigValidationFailed', ...
        'Created configuration failed validation: %s', ME.message);
    % Return config anyway for inspection/correction
end

end

%% Helper Functions

function config = applyNameValuePairs(config, varargin)
    % Apply name-value pair arguments to config
    %
    % Args:
    %   config - Base configuration struct
    %   varargin - Name-value pairs

    if mod(length(varargin), 2) ~= 0
        error('DataGenerator:InvalidArguments', ...
            'Arguments must be name-value pairs');
    end

    for i = 1:2:length(varargin)
        param_name = varargin{i};
        param_value = varargin{i+1};

        if ~ischar(param_name) && ~isstring(param_name)
            error('DataGenerator:InvalidParameterName', ...
                'Parameter names must be strings');
        end

        % Validate parameter exists in default config
        if ~isfield(config, param_name)
            warning('DataGenerator:UnknownParameter', ...
                'Unknown parameter "%s" - adding anyway', param_name);
        end

        config.(param_name) = param_value;
    end
end

function validateSimulationConfig(config)
    % VALIDATESIMULATIONCONFIG Validate simulation configuration
    %
    % Args:
    %   config - Configuration struct to validate
    %
    % Throws error if configuration is invalid
    %
    % See also: CREATESIMULATIONCONFIG

    % Validate required fields exist
    required_fields = {
        'model_name', 'model_path', 'num_simulations', 'simulation_time', ...
        'sample_rate', 'execution_mode', 'output_folder', 'folder_name', ...
        'use_logsout', 'use_signal_bus', 'use_simscape', 'verbosity'
    };

    for i = 1:length(required_fields)
        field = required_fields{i};
        assert(isfield(config, field), ...
            'DataGenerator:MissingField', 'Required field "%s" missing', field);
    end

    % Validate model exists (warning only, might be created later)
    if ~exist(config.model_path, 'file')
        warning('DataGenerator:ModelNotFound', ...
            'Model file not found: %s', config.model_path);
    end

    % Validate numeric parameters
    assert(isnumeric(config.num_simulations) && config.num_simulations > 0 && config.num_simulations <= 10000, ...
        'DataGenerator:InvalidParameter', 'num_simulations must be 1-10000');

    assert(isnumeric(config.simulation_time) && config.simulation_time > 0 && config.simulation_time <= 60, ...
        'DataGenerator:InvalidParameter', 'simulation_time must be 0.001-60 seconds');

    assert(isnumeric(config.sample_rate) && config.sample_rate > 0 && config.sample_rate <= 10000, ...
        'DataGenerator:InvalidParameter', 'sample_rate must be 1-10000 Hz');

    assert(isnumeric(config.batch_size) && config.batch_size > 0, ...
        'DataGenerator:InvalidParameter', 'batch_size must be positive');

    assert(isnumeric(config.save_interval) && config.save_interval > 0, ...
        'DataGenerator:InvalidParameter', 'save_interval must be positive');

    % Validate execution mode
    valid_modes = {'sequential', 'parallel'};
    assert(ismember(config.execution_mode, valid_modes), ...
        'DataGenerator:InvalidParameter', ...
        'execution_mode must be "sequential" or "parallel"');

    % Validate data sources (at least one must be enabled)
    assert(config.use_logsout || config.use_signal_bus || config.use_simscape, ...
        'DataGenerator:InvalidParameter', ...
        'At least one data source must be enabled (use_logsout, use_signal_bus, or use_simscape)');

    % Validate verbosity level
    valid_verbosity = {'Silent', 'Normal', 'Verbose', 'Debug'};
    assert(ismember(config.verbosity, valid_verbosity), ...
        'DataGenerator:InvalidParameter', ...
        'verbosity must be one of: %s', strjoin(valid_verbosity, ', '));

    % Validate output paths
    assert(~isempty(config.output_folder), ...
        'DataGenerator:InvalidParameter', 'output_folder cannot be empty');

    assert(~isempty(config.folder_name), ...
        'DataGenerator:InvalidParameter', 'folder_name cannot be empty');

    % Validate torque scenario
    assert(ismember(config.torque_scenario, [1, 2, 3]), ...
        'DataGenerator:InvalidParameter', ...
        'torque_scenario must be 1 (Variable), 2 (Zero), or 3 (Constant)');

    % Validate coefficient range for variable torque scenario
    if config.torque_scenario == 1
        assert(isnumeric(config.coeff_range) && config.coeff_range > 0, ...
            'DataGenerator:InvalidParameter', ...
            'coeff_range must be positive for variable torque scenario');
    end

    % Validate parallel computing toolbox if parallel mode requested
    if strcmp(config.execution_mode, 'parallel')
        if ~license('test', 'Distrib_Computing_Toolbox')
            warning('DataGenerator:ParallelToolboxMissing', ...
                'Parallel Computing Toolbox not available - will fall back to sequential');
        end
    end

    % Validate Simscape license if Simscape extraction enabled
    if config.use_simscape
        if ~license('test', 'Simscape')
            warning('DataGenerator:SimscapeMissing', ...
                'Simscape license not available - Simscape extraction may fail');
        end
    end
end

function model_path = resolveModelPath(model_name)
    % Resolve the bundled model first, then fall back to path lookup.
    model_candidates = {
        which([model_name '.slx'])
        which([model_name '.mdl'])
        fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'model', [model_name '.slx'])
        fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'model', [model_name '.mdl'])
    };

    for i = 1:length(model_candidates)
        candidate = model_candidates{i};
        if ~isempty(candidate) && exist(candidate, 'file')
            model_path = candidate;
            return;
        end
    end

    % Return the primary bundled path so validation can report a clear warning.
    model_path = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'model', [model_name '.slx']);
end

function verbose = deriveVerboseFlag(verbosity)
    % Convert the UI verbosity level into the boolean extractor flag.
    verbose = ~strcmpi(verbosity, 'Silent');
end

function has_parameter = containsParameterName(varargin, target_name)
    % Check whether a name-value argument list includes a parameter.
    has_parameter = false;

    for i = 1:2:length(varargin)
        parameter_name = varargin{i};
        if isstring(parameter_name)
            parameter_name = char(parameter_name);
        end

        if ischar(parameter_name) && strcmpi(parameter_name, target_name)
            has_parameter = true;
            return;
        end
    end
end

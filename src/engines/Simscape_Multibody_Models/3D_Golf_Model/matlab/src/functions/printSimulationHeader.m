function printSimulationHeader(config)
% PRINTSIMULATIONHEADER Print simulation start banner
%
% Displays a formatted header with simulation configuration details at the
% start of a data generation run. Shows execution mode, trial count, batch
% size, and output directory.
%
% Args:
%   config - Simulation configuration struct (from createSimulationConfig)
%
% Example:
%   config = createSimulationConfig('num_simulations', 100);
%   printSimulationHeader(config);
%
%   Output:
%   ========================================
%   GOLF SWING DATA GENERATION
%   ========================================
%   Mode: Sequential
%   Trials: 100
%   Batch Size: 10
%   Output: /path/to/output
%   ========================================
%
% See also: printSimulationSummary, printProgressBar, createSimulationConfig

fprintf('\n');
fprintf('========================================\n');
fprintf('GOLF SWING DATA GENERATION\n');
fprintf('========================================\n');

% Mode (with worker count for parallel)
fprintf('Mode: %s', config.execution_mode);
if strcmp(config.execution_mode, 'parallel')
    pool = gcp('nocreate');
    if ~isempty(pool)
        fprintf(' (%d workers)', pool.NumWorkers);
    end
end
fprintf('\n');

% Basic configuration
fprintf('Trials: %d\n', config.num_simulations);
fprintf('Batch Size: %d\n', config.batch_size);

% Output directory
if isfield(config, 'output_folder') && ~isempty(config.output_folder)
    % Shorten path if too long (keep last 50 chars)
    output_path = config.output_folder;
    if length(output_path) > 50
        output_path = ['...' output_path(end-46:end)];
    end
    fprintf('Output: %s\n', output_path);
end

% Model information (optional)
if isfield(config, 'model_name') && ~isempty(config.model_name)
    fprintf('Model: %s\n', config.model_name);
end

fprintf('========================================\n\n');

% Starting message
fprintf('Starting simulation...\n\n');

end

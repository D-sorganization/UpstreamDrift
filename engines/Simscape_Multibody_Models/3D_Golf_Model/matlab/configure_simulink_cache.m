%% CONFIGURE_SIMULINK_CACHE - Configure Simulink cache and code generation folders
% This script configures Simulink to use cache directories within the repository
% that are excluded from git tracking.
%
% Usage:
%   Run this script once in MATLAB to configure Simulink preferences:
%   >> configure_simulink_cache
%
% The cache folders will be created at:
%   - CacheFolder: matlab/cache/simulink/cache/
%   - CodeGenFolder: matlab/cache/simulink/codegen/
%
% These directories are automatically excluded from git via .gitignore

function configure_simulink_cache()
% Get the repository root directory
% This script should be in matlab/, so go up one level
script_path = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_path);

% Define cache directories relative to repo root
cache_base = fullfile(repo_root, 'matlab', 'cache', 'simulink');
cache_folder = fullfile(cache_base, 'cache');
codegen_folder = fullfile(cache_base, 'codegen');

% Create directories if they don't exist
if ~exist(cache_folder, 'dir')
    mkdir(cache_folder);
    fprintf('Created cache folder: %s\n', cache_folder);
else
    fprintf('Cache folder already exists: %s\n', cache_folder);
end

if ~exist(codegen_folder, 'dir')
    mkdir(codegen_folder);
    fprintf('Created codegen folder: %s\n', codegen_folder);
else
    fprintf('Codegen folder already exists: %s\n', codegen_folder);
end

% Configure Simulink file generation control
try
    % Set cache folder
    Simulink.fileGenControl('set', 'CacheFolder', cache_folder, 'createDir', true);
    fprintf('✓ Configured Simulink CacheFolder: %s\n', cache_folder);

    % Set code generation folder
    Simulink.fileGenControl('set', 'CodeGenFolder', codegen_folder, 'createDir', true);
    fprintf('✓ Configured Simulink CodeGenFolder: %s\n', codegen_folder);

    % Verify settings
    current_cache = Simulink.fileGenControl('get', 'CacheFolder');
    current_codegen = Simulink.fileGenControl('get', 'CodeGenFolder');

    fprintf('\nCurrent Simulink settings:\n');
    fprintf('  CacheFolder: %s\n', current_cache);
    fprintf('  CodeGenFolder: %s\n', current_codegen);

    fprintf('\n✓ Simulink cache configuration complete!\n');
    fprintf('  These settings will persist in your MATLAB preferences.\n');

catch ME
    warning('Failed to configure Simulink cache: %s', ME.message);
    fprintf('You may need to run this script with administrator privileges.\n');
    rethrow(ME);
end
end

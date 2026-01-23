function setup_golf_suite()
%SETUP_GOLF_SUITE Initialize MATLAB environment for Golf Modeling Suite
%
% This function sets up the MATLAB path and environment for all golf
% modeling engines in the suite. Call this function once per MATLAB
% session before using any golf modeling functionality.
%
% Usage:
%   setup_golf_suite()
%
% See also: GOLF_SUITE_PATHS, VALIDATE_GOLF_ENVIRONMENT

    fprintf('Setting up Golf Modeling Suite environment...\n');
    
    % Get the suite root directory
    suite_root = get_suite_root();
    
    % Add shared utilities to path
    shared_matlab = fullfile(suite_root, 'shared', 'matlab');
    addpath(genpath(shared_matlab));
    
    % Add engine-specific paths
    engines_root = fullfile(suite_root, 'engines');
    
    % MATLAB Simscape engines
    matlab_2d = fullfile(engines_root, 'matlab_simscape', '2d_model', 'matlab');
    matlab_3d = fullfile(engines_root, 'matlab_simscape', '3d_biomechanical', 'matlab');
    
    if exist(matlab_2d, 'dir')
        addpath(genpath(matlab_2d));
        fprintf('  Added 2D MATLAB model to path\n');
    end
    
    if exist(matlab_3d, 'dir')
        addpath(genpath(matlab_3d));
        fprintf('  Added 3D MATLAB model to path\n');
    end
    
    % Pendulum models
    pendulum_matlab = fullfile(engines_root, 'pendulum_models', 'matlab');
    if exist(pendulum_matlab, 'dir')
        addpath(genpath(pendulum_matlab));
        fprintf('  Added pendulum models to path\n');
    end
    
    % Set up Simulink cache directory
    setup_simulink_cache(suite_root);
    
    % Validate environment
    validate_golf_environment();
    
    fprintf('Golf Modeling Suite setup complete!\n');
    fprintf('Use golf_suite_help() for available functions.\n');
end

function suite_root = get_suite_root()
    % Get the root directory of the Golf Modeling Suite
    current_file = mfilename('fullpath');
    shared_matlab = fileparts(current_file);
    shared_dir = fileparts(shared_matlab);
    suite_root = fileparts(shared_dir);
end

function setup_simulink_cache(suite_root)
    % Set up Simulink cache directory
    cache_dir = fullfile(suite_root, 'output', 'simulink_cache');
    
    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
    
    % Set Simulink cache folder
    try
        Simulink.fileGenControl('set', 'CacheFolder', cache_dir);
        fprintf('  Simulink cache directory: %s\n', cache_dir);
    catch ME
        warning('Could not set Simulink cache directory: %s', ME.message);
    end
end

function validate_golf_environment()
    % Validate that required toolboxes are available
    required_toolboxes = {
        'Simulink', 'Simulink';
        'Simscape Multibody', 'Simscape';
        'Control System Toolbox', 'Control_Toolbox';
        'Optimization Toolbox', 'Optimization_Toolbox'
    };
    
    missing_toolboxes = {};
    
    for i = 1:size(required_toolboxes, 1)
        toolbox_name = required_toolboxes{i, 1};
        license_name = required_toolboxes{i, 2};
        
        if ~license('test', license_name)
            missing_toolboxes{end+1} = toolbox_name; %#ok<AGROW>
        end
    end
    
    if ~isempty(missing_toolboxes)
        warning('Missing toolboxes: %s', strjoin(missing_toolboxes, ', '));
        fprintf('  Some functionality may be limited.\n');
    else
        fprintf('  All required toolboxes available\n');
    end
end
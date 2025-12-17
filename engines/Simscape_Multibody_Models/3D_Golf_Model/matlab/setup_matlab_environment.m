%% SETUP_MATLAB_ENVIRONMENT - Complete MATLAB/Simulink environment setup
% This script configures your MATLAB environment for the Golf Model project:
%   1. Adds project source folders to MATLAB path
%   2. Configures Simulink cache and code generation folders
%   3. Cleans up old Backup_Scripts paths from MATLAB path
%
% Usage:
%   Run this script once in MATLAB:
%   >> setup_matlab_environment
%
% This should be run from the matlab/ directory or the repository root.

function setup_matlab_environment()
    fprintf('========================================\n');
    fprintf('MATLAB Environment Setup for Golf Model\n');
    fprintf('========================================\n\n');

    % Determine project root
    script_path = fileparts(mfilename('fullpath'));
    % script_path is 'matlab/', so repo root is parent?
    % But if run from repo root, mfilename path still points to file location.

    %% Step 1: Add Source Paths
    fprintf('Step 1: Adding source paths...\n');
    src_path = fullfile(script_path, 'src');
    if exist(src_path, 'dir')
        addpath(genpath(src_path));
        fprintf('✓ Added %s and subdirectories to path\n', src_path);
    else
        warning('Source directory not found: %s', src_path);
    end

    % Also add Data folder?
    data_path = fullfile(script_path, 'Data');
    if exist(data_path, 'dir')
         addpath(data_path);
         fprintf('✓ Added %s to path\n', data_path);
    end

    savepath;
    fprintf('✓ Saved MATLAB path\n\n');

    %% Step 2: Configure Simulink Cache
    fprintf('Step 2: Configuring Simulink cache...\n');
    try
        configure_simulink_cache();
        fprintf('✓ Simulink cache configuration complete\n\n');
    catch ME
        warning('Simulink cache configuration failed: %s', ME.message);
        fprintf('⚠ Continuing with other setup steps...\n\n');
    end

    %% Step 3: Clean up MATLAB path
    fprintf('Step 3: Cleaning up MATLAB path...\n');
    try
        % Get the path to the cleanup script
        cleanup_script = fullfile(src_path, 'functions', 'utils', 'cleanup_matlab_path.m');

        if exist(cleanup_script, 'file')
            run(cleanup_script);
            fprintf('✓ MATLAB path cleanup complete\n\n');
        else
            fprintf('⚠ Cleanup script not found at: %s\n', cleanup_script);
            fprintf('  Attempting manual cleanup...\n');

            % Manual cleanup
            current_path = path;
            paths = strsplit(current_path, ';');
            backup_paths = paths(contains(paths, 'Backup_Scripts'));

            if ~isempty(backup_paths)
                fprintf('Found %d Backup_Scripts paths to remove:\n', length(backup_paths));
                for i = 1:length(backup_paths)
                    fprintf('  Removing: %s\n', backup_paths{i});
                    rmpath(backup_paths{i});
                end
                savepath;
                fprintf('✓ Manual cleanup complete\n\n');
            else
                fprintf('✓ No Backup_Scripts paths found in MATLAB path\n\n');
            end
        end
    catch ME
        warning('MATLAB path cleanup failed: %s', ME.message);
        fprintf('⚠ Continuing...\n\n');
    end

    %% Summary
    fprintf('========================================\n');
    fprintf('Setup Complete!\n');
    fprintf('========================================\n');
    fprintf('\nNext steps:\n');
    fprintf('1. Verify Simulink settings: Simulink.fileGenControl(''get'')\n');
    fprintf('2. Check MATLAB path: path\n');
    fprintf('3. If Backup_Scripts warnings persist, check MATLAB preferences:\n');
    fprintf('   Home > Environment > Set Path\n');
    fprintf('   Remove any entries containing "Backup_Scripts"\n');
    fprintf('\n');
end

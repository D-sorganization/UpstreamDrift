function launch_gui()
% LAUNCH_GUI - Simple launcher for the Golf Swing Analysis GUI
%
% This script adds the necessary paths and launches the GUI
%
% Usage:
%   launch_gui();

    % Get the directory where this script is located
    script_dir = fileparts(mfilename('fullpath'));

    % Add all subdirectories to the MATLAB path
    addpath(genpath(script_dir));

    % Launch the GUI
    main_scripts_dir = fullfile(script_dir, 'main_scripts');
    if exist(fullfile(main_scripts_dir, 'golf_swing_analysis_gui.m'), 'file')
        % Change to main_scripts directory and launch
        current_dir = pwd;
        cd(main_scripts_dir);
        golf_swing_analysis_gui();
        cd(current_dir);
    else
        error('GUI file not found in main_scripts directory');
    end

    fprintf('🚀 GUI launched successfully!\n');
    fprintf('📁 Working directory: %s\n', script_dir);

end

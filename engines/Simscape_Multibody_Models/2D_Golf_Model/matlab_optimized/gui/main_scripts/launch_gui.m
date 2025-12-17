function launch_gui()
% LAUNCH_GUI - Launch the optimized golf swing analysis GUI
%
% This function initializes and launches the enhanced GUI for the
% optimized 2D golf swing analysis system.
%
% Usage:
%   launch_gui();
%
% The GUI provides:
%   - Simulation parameter controls
%   - Parallel processing options
%   - Real-time progress monitoring
%   - Interactive plotting
%   - Data exploration tools
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
    end

    fprintf('🚀 Launching Optimized Golf Swing Analysis GUI...\n');

    % Add paths (Note: Ideally paths should be managed by startup.m)
    base_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(genpath(base_dir));

    % Create main GUI
    golf_swing_gui_optimized();

end

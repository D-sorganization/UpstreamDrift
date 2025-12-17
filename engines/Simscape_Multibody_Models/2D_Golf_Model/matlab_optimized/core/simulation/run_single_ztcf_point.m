function ztcf_row = run_single_ztcf_point(config, mdlWks, killswitch_time, time_index)
% RUN_SINGLE_ZTCF_POINT - Run simulation for a single ZTCF time point
%
% Inputs:
%   config - Configuration structure from simulation_config()
%   mdlWks - Model workspace handle
%   killswitch_time - Time at which to trigger killswitch (seconds)
%   time_index - Index for progress tracking
%
% Returns:
%   ztcf_row - Single-row table with ZTCF data at killswitch time
%
% This function runs one simulation with the killswitch triggered at the
% specified time, extracting only the data at the moment the killswitch
% activates (when joint torques go to zero).
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
        config (1,1) struct
        mdlWks
        killswitch_time (1,1) double
        time_index (1,1) double
    end

    %% Set killswitch time
    assignin(mdlWks, 'KillswitchStepTime', Simulink.Parameter(killswitch_time));

    %% Run simulation
    current_dir = pwd;
    try
        cd(config.legacy_scripts_path);

        % Run simulation
        out = sim(config.model_name);

        % Generate table
        SCRIPT_TableGeneration;
        ZTCFData = Data;

        % Find the row where killswitch activates
        row = find(ZTCFData.KillswitchState == 0, 1);

        if isempty(row)
            warning('No killswitch activation found at time %.3f', killswitch_time);
            ztcf_row = [];
        else
            % Extract only the killswitch activation row
            ztcf_row = ZTCFData(row, :);
        end

    catch ME
        cd(current_dir);
        rethrow(ME);
    end

    cd(current_dir);

end

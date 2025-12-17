function skeleton_plotter_wrapper(BASEQ, ZTCFQ, DELTAQ)
% SKELETON_PLOTTER_WRAPPER - Wrapper function to launch skeleton plotter from GUI
%
% Inputs:
%   BASEQ - Base data table (Q-spaced)
%   ZTCFQ - ZTCF data table (Q-spaced)
%   DELTAQ - DELTA data table (Q-spaced)
%
% This function launches the skeleton plotter with the provided data
% and provides integration with the main GUI

    arguments
        BASEQ table
        ZTCFQ table
        DELTAQ table
    end

    % Check for required columns in BASEQ
    required_columns = {'Buttx', 'Butty', 'Buttz', 'CHx', 'CHy', 'CHz', ...
                       'MPx', 'MPy', 'MPz', 'LWx', 'LWy', 'LWz', ...
                       'LEx', 'LEy', 'LEz', 'LSx', 'LSy', 'LSz', ...
                       'RWx', 'RWy', 'RWz', 'REx', 'REy', 'REz', ...
                       'RSx', 'RSy', 'RSz', 'HUBx', 'HUBy', 'HUBz'};

    missing_columns = setdiff(required_columns, BASEQ.Properties.VariableNames);
    if ~isempty(missing_columns)
        warning('Missing columns in BASEQ: %s', strjoin(missing_columns, ', '));
    end

    % Check for force and torque data
    force_columns = {'TotalHandForceGlobal', 'EquivalentMidpointCoupleGlobal'};
    missing_forces = setdiff(force_columns, BASEQ.Properties.VariableNames);
    if ~isempty(missing_forces)
        warning('Missing force/torque columns: %s', strjoin(missing_forces, ', '));
    end

    fprintf('🦴 Launching Skeleton Plotter...\n');
    fprintf('   BASEQ data points: %d\n', height(BASEQ));
    fprintf('   ZTCFQ data points: %d\n', height(ZTCFQ));
    fprintf('   DELTAQ data points: %d\n', height(DELTAQ));

    try
        % Launch the skeleton plotter
        SkeletonPlotter(BASEQ, ZTCFQ, DELTAQ);

        fprintf('✅ Skeleton Plotter launched successfully\n');

    catch ME
        fprintf('❌ Error launching Skeleton Plotter: %s\n', ME.message);
        fprintf('📍 Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        rethrow(ME);
    end

end

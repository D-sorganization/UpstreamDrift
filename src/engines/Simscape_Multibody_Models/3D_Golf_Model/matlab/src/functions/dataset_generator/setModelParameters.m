function simIn = setModelParameters(simIn, config)
    % External function for setting model parameters - can be used in parallel processing
    % This function accepts config as a parameter instead of relying on handles

    % Set basic simulation parameters with careful error handling
    try
        % Set stop time
        if isfield(config, 'simulation_time') && ~isempty(config.simulation_time)
            simIn = simIn.setModelParameter('StopTime', num2str(config.simulation_time));
        end

        % Set solver carefully
        try
            simIn = simIn.setModelParameter('Solver', 'ode23t');
        catch
            fprintf('Warning: Could not set solver to ode23t\n');
        end

        % Set tolerances carefully
        try
            simIn = simIn.setModelParameter('RelTol', '1e-3');
            simIn = simIn.setModelParameter('AbsTol', '1e-5');
        catch
            fprintf('Warning: Could not set solver tolerances\n');
        end

        % CRITICAL: Set output options for data logging
        try
            simIn = simIn.setModelParameter('SaveOutput', 'on');
            simIn = simIn.setModelParameter('SaveFormat', 'Structure');
            simIn = simIn.setModelParameter('ReturnWorkspaceOutputs', 'on');
        catch ME
            fprintf('Warning: Could not set output options: %s\n', ME.message);
        end

        % Additional logging settings
        try
            simIn = simIn.setModelParameter('SignalLogging', 'on');
            simIn = simIn.setModelParameter('SaveTime', 'on');
        catch
            fprintf('Warning: Could not set logging options\n');
        end

        % To Workspace block settings
        try
            simIn = simIn.setModelParameter('LimitDataPoints', 'off');
        catch
            fprintf('Warning: Could not set LimitDataPoints\n');
        end

        % SIMSCAPE FULL-BLOCK LOGGING (home-license workaround)
        %
        % Pin SimscapeLogType='all' so the simulation surfaces every Simscape
        % block's state (angles, rates, accels, constraint forces, internal
        % moments, ...) in the `simlog` tree without spending Simscape
        % "virtual signal" markers (which the home license caps aggressively).
        %
        % This mirrors the persistent setting baked into
        % GolfSwing3D_Kinetic.slx at MDL line 4256 (see
        %   src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/mdl_reference/README.md
        % and the project spec at
        %   src/engines/Simscape_Multibody_Models/3D_Golf_Model/PROJECT_SPEC.md
        % ), but we re-pin it on every SimulationInput so that swept runs
        % (parsim with overridden parameters) cannot inadvertently drop it.
        %
        % If this is NOT set, downstream extractSimscapeDataRecursive() finds
        % an empty simlog and the per-trial parquet falls back to bus-only
        % features (~1956 cols, no per-block constraint state) -- which is
        % exactly the regression we are fixing.
        try
            simIn = simIn.setModelParameter('SimscapeLogType', 'all');
        catch ME
            fprintf('Warning: Could not set essential SimscapeLogType parameter: %s\n', ME.message);
            fprintf('Warning: Simscape data extraction may not work without this parameter\n');
        end

        % Set simulation mode (animation control removed for now to fix data capture)
        try
            simIn = simIn.setModelParameter('SimulationMode', 'normal');
        catch ME
            fprintf('Warning: Could not set simulation mode: %s\n', ME.message);
        end

        % Set other model parameters to suppress unconnected port warnings
        try
            simIn = simIn.setModelParameter('UnconnectedInputMsg', 'none');
            simIn = simIn.setModelParameter('UnconnectedOutputMsg', 'none');
        catch
            % These parameters might not exist in all model types
        end

    catch ME
        fprintf('Error setting model parameters: %s\n', ME.message);
        rethrow(ME);
    end
end

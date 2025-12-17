function result = runSingleTrial(trial_num, config, trial_coefficients, capture_workspace)
    % External function for running a single trial - can be used in parallel processing
    % This function accepts config as a parameter instead of relying on handles

    result = struct('success', false, 'filename', '', 'data_points', 0, 'columns', 0);

    try
        % Create simulation input
        simIn = Simulink.SimulationInput(config.model_path);

        % Set model parameters
        simIn = setModelParameters(simIn, config);

        % Set polynomial coefficients for this trial
        try
            simIn = setPolynomialCoefficients(simIn, trial_coefficients, config);
        catch ME
            fprintf('Warning: Could not set polynomial coefficients: %s\n', ME.message);
        end

        % Suppress specific warnings that are not critical
        warning_state = warning('off', 'Simulink:Bus:EditTimeBusPropNotAllowed');
        warning_state2 = warning('off', 'Simulink:Engine:BlockOutputNotUpdated');
        warning_state3 = warning('off', 'Simulink:Engine:OutputNotConnected');
        warning_state4 = warning('off', 'Simulink:Engine:InputNotConnected');
        warning_state5 = warning('off', 'Simulink:Blocks:UnconnectedOutputPort');
        warning_state6 = warning('off', 'Simulink:Blocks:UnconnectedInputPort');

        % Run simulation with progress indicator and visualization suppression
        fprintf('Running trial %d simulation...', trial_num);

        simOut = sim(simIn);
        fprintf(' Done.\n');

        % Restore warning state
        warning(warning_state);
        warning(warning_state2);
        warning(warning_state3);
        warning(warning_state4);
        warning(warning_state5);
        warning(warning_state6);

        % Process simulation output
        result = processSimulationOutput(trial_num, config, simOut, capture_workspace);

    catch ME
        % Restore warning state in case of error
        if exist('warning_state', 'var')
            warning(warning_state);
        end
        if exist('warning_state2', 'var')
            warning(warning_state2);
        end
        if exist('warning_state3', 'var')
            warning(warning_state3);
        end
        if exist('warning_state4', 'var')
            warning(warning_state4);
        end
        if exist('warning_state5', 'var')
            warning(warning_state5);
        end
        if exist('warning_state6', 'var')
            warning(warning_state6);
        end

        fprintf(' Failed.\n');
        result.success = false;
        result.error = ME.message;
        fprintf('Trial %d simulation failed: %s\n', trial_num, ME.message);

        % Print stack trace for debugging
        fprintf('Error details:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
end

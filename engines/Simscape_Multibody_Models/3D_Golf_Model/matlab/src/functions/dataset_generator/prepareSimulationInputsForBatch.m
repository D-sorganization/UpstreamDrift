function simInputs = prepareSimulationInputsForBatch(config, start_trial, end_trial)
    % Prepare simulation inputs for a specific batch of trials
    % Load the Simulink model
    model_name = config.model_name;
    if ~bdIsLoaded(model_name)
        try
            load_system(model_name);
        catch ME
            error('Could not load Simulink model "%s": %s', model_name, ME.message);
        end
    end

    % Create array of SimulationInput objects for this batch
    batch_size = end_trial - start_trial + 1;
    simInputs = Simulink.SimulationInput.empty(0, batch_size);

    for i = 1:batch_size
        trial = start_trial + i - 1;

        % Get coefficients for this trial
        if trial <= size(config.coefficient_values, 1)
            trial_coefficients = config.coefficient_values(trial, :);
        else
            % Generate random coefficients for additional trials
            trial_coefficients = generateRandomCoefficients(size(config.coefficient_values, 2));
        end

        % Ensure coefficients are numeric (fix for parallel execution)
        if iscell(trial_coefficients)
            trial_coefficients = cell2mat(trial_coefficients);
        end
        trial_coefficients = double(trial_coefficients);  % Ensure double precision

        % Create SimulationInput object
        simIn = Simulink.SimulationInput(model_name);

        % Set simulation parameters safely
        simIn = setModelParameters(simIn, config);

        % Set polynomial coefficients
        try
            simIn = setPolynomialCoefficients(simIn, trial_coefficients, config);
        catch ME
            fprintf('Warning: Could not set polynomial coefficients: %s\n', ME.message);
        end

        % Load input file if specified
        if ~isempty(config.input_file) && exist(config.input_file, 'file')
            simIn = loadInputFile(simIn, config.input_file);
        end

        simInputs(i) = simIn;
    end
end

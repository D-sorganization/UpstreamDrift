function [simOut, candidateState] = evaluate_candidate_step( ...
    modelName, currentState, torqueRow, targetFrame, config, controlColumns)
%EVALUATE_CANDIDATE_STEP  Single short-horizon Simscape candidate trial.
%
%   [simOut, candidateState] = frame_search.evaluate_candidate_step( ...
%       modelName, currentState, torqueRow, targetFrame, config, controlColumns)
%   builds a Simulink.SimulationInput, applies a constant candidate torque
%   row over the configured short horizon, restores any saved final state
%   from the previous frame, runs the model, and returns the raw simOut
%   plus a candidateState struct that carries the final-state vector
%   forward to the next frame.
%
%   Save/restore strategy:
%     - When currentState.xFinal is present, the simulation is configured
%       with LoadInitialState='on' and InitialState set from xFinal.
%     - When currentState.starting_state_file is non-empty and xFinal is
%       absent, the scalar variables in that MAT file are pushed to the
%       SimulationInput workspace (same convention as
%       run_ml_polynomial_input_swing).
%     - SaveFinalState='on' / FinalStateName='xFinal' is always requested
%       so the next frame can resume.
%
%   Inputs:
%     modelName       char/string  Simulink model (e.g. 'GolfSwing3D_Kinetic').
%     currentState    struct       carried-forward state from previous frame.
%     torqueRow       (1,K) double constant torque applied this frame.
%     targetFrame     struct       target row from the manifest table.
%     config          struct       parsed manifest.
%     controlColumns  cellstr/string  manifest control column names.
%
%   Outputs:
%     simOut          Simulink.SimulationOutput (raw).
%     candidateState  struct with .xFinal, .time, .frame_index for the next
%                     frame.

    arguments
        modelName (1,1) string
        currentState struct
        torqueRow (1,:) double
        targetFrame struct
        config struct
        controlColumns
    end

    modelChar = char(modelName);
    if ~bdIsLoaded(modelChar)
        try
            load_system(modelChar);
        catch ME
            error('frame_search:evaluate_candidate_step:modelLoad', ...
                'Could not load Simulink model %s: %s', modelChar, ME.message);
        end
    end

    [startTime, stopTime] = frame_search.frame_horizon( ...
        currentState, targetFrame, config);

    simIn = Simulink.SimulationInput(modelChar);
    simIn = simIn.setModelParameter('StartTime', num2str(startTime));
    simIn = simIn.setModelParameter('StopTime', num2str(stopTime));
    simIn = simIn.setModelParameter('SaveFinalState', 'on');
    simIn = simIn.setModelParameter('FinalStateName', 'xFinal');
    simIn = simIn.setModelParameter('SaveCompleteFinalSimState', 'on');
    simIn = simIn.setModelParameter('SaveOutput', 'on');
    simIn = simIn.setModelParameter('SaveTime', 'on');

    % --- 1. Restore previous-frame state -----------------------------------
    if isfield(currentState, 'xFinal') && ~isempty(currentState.xFinal)
        simIn = simIn.setModelParameter('LoadInitialState', 'on');
        simIn = simIn.setVariable('xFinal', currentState.xFinal);
        simIn = simIn.setModelParameter('InitialState', 'xFinal');
    elseif isfield(currentState, 'starting_state_file') && ...
            strlength(string(currentState.starting_state_file)) > 0
        simIn = local_apply_scalar_variables( ...
            simIn, char(currentState.starting_state_file));
    end

    % --- 2. Apply candidate torque as polynomial workspace overrides -------
    overrides = frame_search.apply_constant_torque(torqueRow, controlColumns);
    overrideNames = fieldnames(overrides);
    for idx = 1:numel(overrideNames)
        simIn = simIn.setVariable(overrideNames{idx}, overrides.(overrideNames{idx}));
    end

    % --- 3. Run -----------------------------------------------------------
    try
        simOut = sim(simIn);
    catch ME
        error('frame_search:evaluate_candidate_step:simFailed', ...
            'Simulink sim() failed at frame time %.6f: %s', stopTime, ME.message);
    end

    % --- 4. Build candidateState ------------------------------------------
    candidateState = currentState;
    if isfield(candidateState, 'frame_index')
        candidateState.frame_index = candidateState.frame_index + 1;
    else
        candidateState.frame_index = 1;
    end
    candidateState.time = stopTime;

    candidateState.xFinal = local_extract_xFinal(simOut);
end

function simIn = local_apply_scalar_variables(simIn, matFile)
    if ~isfile(matFile)
        error('frame_search:evaluate_candidate_step:missingStateFile', ...
            'Starting state file not found: %s', matFile);
    end
    data = load(matFile);
    names = fieldnames(data);
    for idx = 1:numel(names)
        v = data.(names{idx});
        if isa(v, 'Simulink.Parameter')
            v = v.Value;
        end
        if isnumeric(v) && isscalar(v)
            simIn = simIn.setVariable(names{idx}, double(v));
        end
    end
end

function xFinal = local_extract_xFinal(simOut)
    xFinal = [];
    try
        if isprop(simOut, 'xFinal') || isfield(simOut, 'xFinal')
            xFinal = simOut.xFinal;
        end
    catch
    end
    if isempty(xFinal)
        % Some models route final state through SimulationMetadata.
        try
            if isprop(simOut, 'SimulationMetadata')
                md = simOut.SimulationMetadata;
                if isprop(md, 'UserData') && isstruct(md.UserData) && ...
                        isfield(md.UserData, 'xFinal')
                    xFinal = md.UserData.xFinal;
                end
            end
        catch
        end
    end
end

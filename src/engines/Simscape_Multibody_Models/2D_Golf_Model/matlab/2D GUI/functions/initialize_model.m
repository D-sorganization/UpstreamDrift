function mdlWks = initialize_model(config)
% INITIALIZE_MODEL - Initialize the Simulink model and workspace
%
% Inputs:
%   config - Configuration structure from model_config()
%
% Returns:
%   mdlWks - Model workspace handle
%
% This function:
%   1. Changes to the model directory
%   2. Loads the Simulink model
%   3. Configures model parameters
%   4. Sets up the model workspace
%   5. Configures warnings

    % Change to model directory
    cd(config.model_path);

    % Load the model
    model_name = config.model_name;
    if ~bdIsLoaded(model_name)
        load_system(model_name);
    end

    % Get model workspace
    mdlWks = get_param(model_name, 'ModelWorkspace');

    % Configure model parameters
    set_param(model_name, 'ReturnWorkspaceOutputs', config.return_workspace_outputs);
    set_param(model_name, 'FastRestart', config.fast_restart);
    set_param(model_name, 'MaxStep', num2str(config.max_step));

    % Set model workspace variables
    assignin(mdlWks, 'StopTime', Simulink.Parameter(config.stop_time));
    assignin(mdlWks, 'KillswitchStepTime', Simulink.Parameter(config.killswitch_time));
    assignin(mdlWks, 'KillDampFinalValue', Simulink.Parameter(config.dampening_included));

    % Suppress warnings if configured
    if config.suppress_warnings
        warning('off', 'MATLAB:MKDIR:DirectoryExists');
        warning('off', 'Simulink:Masking:NonTunableParameterChangedDuringSimulation');
        warning('off', 'Simulink:Engine:NonTunableVarChangedInFastRestart');
        warning('off', 'Simulink:Engine:NonTunableVarChangedMaxWarnings');
    end

    fprintf('✅ Model initialized: %s\n', model_name);

end

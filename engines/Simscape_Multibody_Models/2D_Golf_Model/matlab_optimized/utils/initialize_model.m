function mdlWks = initialize_model(config)
% INITIALIZE_MODEL - Initialize Simulink model for golf swing analysis
%
% Inputs:
%   config - Configuration structure from simulation_config()
%
% Returns:
%   mdlWks - Model workspace handle
%
% This function:
%   1. Navigates to model directory
%   2. Opens the GolfSwing Simulink model
%   3. Generates model workspace
%   4. Configures simulation parameters
%   5. Sets up killswitch and dampening
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
        config struct
    end

    if config.verbose
        fprintf('🔧 Initializing Simulink model...\n');
    end

    %% Navigate to model directory
    current_dir = pwd;
    try
        % Change to legacy scripts directory to use existing SCRIPT_mdlWks_Generate
        cd(config.legacy_scripts_path);

        %% Open GolfSwing model
        if config.verbose
            fprintf('   Opening %s model...\n', config.model_name);
        end
        open_system(config.model_name);

        %% Generate model workspace
        if config.verbose
            fprintf('   Generating model workspace...\n');
        end
        SCRIPT_mdlWks_Generate;

        %% Configure killswitch dampening
        mdlWks = get_param(config.model_name, 'ModelWorkspace');
        assignin(mdlWks, 'KillDampFinalValue', Simulink.Parameter(config.kill_damp_final_value));

        %% Set simulation parameters
        if config.verbose
            fprintf('   Configuring simulation parameters...\n');
        end
        assignin(mdlWks, 'StopTime', Simulink.Parameter(config.stop_time));
        assignin(mdlWks, 'KillswitchStepTime', Simulink.Parameter(config.killswitch_time));

        %% Configure model settings
        set_param(config.model_name, 'ReturnWorkspaceOutputs', 'on');
        set_param(config.model_name, 'FastRestart', 'on');
        set_param(config.model_name, 'MaxStep', num2str(config.max_step));

        %% Suppress common warnings
        suppress_simulink_warnings();

        if config.verbose
            fprintf('✅ Model initialized successfully\n');
        end

    catch ME
        % Return to original directory on error
        cd(current_dir);
        rethrow(ME);
    end

    % Return to original directory
    cd(current_dir);

end

function suppress_simulink_warnings()
    % Suppress common Simulink warnings that clutter output
    arguments
    end
    warning('off', 'MATLAB:MKDIR:DirectoryExists');
    warning('off', 'Simulink:Masking:NonTunableParameterChangedDuringSimulation');
    warning('off', 'Simulink:Engine:NonTunableVarChangedInFastRestart');
    warning('off', 'Simulink:Engine:NonTunableVarChangedMaxWarnings');
end

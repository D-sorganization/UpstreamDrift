function BaseData = run_base_simulation(config, mdlWks)
% RUN_BASE_SIMULATION - Run baseline golf swing simulation
%
% Inputs:
%   config - Configuration structure from simulation_config()
%   mdlWks - Model workspace handle from initialize_model()
%
% Returns:
%   BaseData - Table containing baseline simulation data
%
% This function runs the complete golf swing simulation with all joint
% torques active (no killswitch triggered) to generate baseline data.
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
        config (1,1) struct
        mdlWks
    end

    if config.verbose
        fprintf('📊 Generating baseline simulation data...\n');
    end

    %% Ensure killswitch is disabled
    assignin(mdlWks, 'KillswitchStepTime', Simulink.Parameter(config.killswitch_time));

    %% Run simulation
    current_dir = pwd;
    try
        % Navigate to scripts directory for table generation
        cd(config.legacy_scripts_path);

        % Run simulation
        if config.verbose
            fprintf('   Running simulation (0 to %.2f seconds)...\n', config.stop_time);
        end
        out = sim(config.model_name);

        % Generate data table using legacy script
        if config.verbose
            fprintf('   Generating data table...\n');
        end
        SCRIPT_TableGeneration;

        % Store result
        BaseData = Data;

        if config.verbose
            fprintf('✅ Baseline data generated: %d rows, %d variables\n', ...
                height(BaseData), width(BaseData));
        end

    catch ME
        cd(current_dir);
        rethrow(ME);
    end

    cd(current_dir);

end

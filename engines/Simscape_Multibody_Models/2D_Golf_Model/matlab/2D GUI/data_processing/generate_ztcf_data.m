function ZTCF = generate_ztcf_data(config, mdlWks, BaseData)
% GENERATE_ZTCF_DATA - Generate ZTCF (Zero Torque Counterfactual) data
%
% Inputs:
%   config - Configuration structure from model_config()
%   mdlWks - Model workspace handle from initialize_model()
%   BaseData - Base data table from generate_base_data()
%
% Returns:
%   ZTCF - Table containing the ZTCF data
%
% This function:
%   1. Loops through time points to generate ZTCF data
%   2. Creates a table with ZTCF data for each time point
%   3. Returns the compiled ZTCF data

    % Initialize ZTCF table with same structure as BaseData
    ZTCFTable = BaseData;
    ZTCFTable(:,:) = []; % Clear all data but keep structure

    % Change to scripts directory
    cd(config.scripts_path);

    fprintf('🔄 Generating ZTCF data...\n');

    % Loop through time points
    for i = config.ztcf_start_time:config.ztcf_end_time

        % Scale counter to match desired times
        j = i / config.ztcf_time_scale;

        % Display progress
        progress = i / config.ztcf_end_time * 100;
        fprintf('   Progress: %.1f%% (Time: %.3f s)\n', progress, j);

        % Set killswitch time in model workspace
        assignin(mdlWks, 'KillswitchStepTime', Simulink.Parameter(j));

        % Run simulation
        out = sim(config.model_name);
        SCRIPT_TableGeneration;
        ZTCFData = Data;

        % Find the row where KillswitchState first becomes zero
        row = find(ZTCFData.KillswitchState == 0, 1);

        if isempty(row)
            warning('No killswitch state change found at time %.3f', j);
            continue;
        end

        % Create ZTCF row
        ZTCF = ZTCFData;
        ZTCF(1,:) = ZTCFData(row,:);

        % Remove all other rows
        H = height(ZTCF) - 1;
        for k = 1:H
            DelRow = H + 2 - k;
            ZTCF(DelRow,:) = [];
        end

        % Add to ZTCF table
        ZTCFTable = [ZTCFTable; ZTCF];

    end

    % Reset killswitch time
    assignin(mdlWks, 'KillswitchStepTime', Simulink.Parameter(config.killswitch_time));

    % Clean up and return
    ZTCF = ZTCFTable;

    fprintf('✅ ZTCF data generated successfully\n');
    fprintf('   ZTCF data points: %d\n', height(ZTCF));

end

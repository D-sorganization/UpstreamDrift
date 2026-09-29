function BaseData = generate_base_data(config, mdlWks)
% GENERATE_BASE_DATA - Generate base data by running the model simulation
%
% Inputs:
%   config - Configuration structure from model_config()
%   mdlWks - Model workspace handle from initialize_model()
%
% Returns:
%   BaseData - Table containing the base simulation data
%
% This function:
%   1. Runs the model simulation
%   2. Generates the data table
%   3. Returns the base data for further processing

    % Change to scripts directory
    cd(config.scripts_path);

    % Run the model to generate BaseData table
    fprintf('🔄 Running base model simulation...\n');
    out = sim(config.model_name);

    % Run table generation script
    SCRIPT_TableGeneration;

    % Create BaseData table
    BaseData = Data;

    fprintf('✅ Base data generated successfully\n');
    fprintf('   Data points: %d\n', height(BaseData));
    fprintf('   Time range: %.3f to %.3f seconds\n', ...
        BaseData.Time(1), BaseData.Time(end));

end

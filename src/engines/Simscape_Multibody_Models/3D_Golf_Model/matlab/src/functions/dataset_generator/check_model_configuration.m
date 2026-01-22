function check_model_configuration()
    % Diagnostic script to check Simulink model configuration
    % This script helps identify issues with To Workspace block naming

    fprintf('=== Simulink Model Configuration Check ===\n\n');

    % Try to find the model
    model_name = 'GolfSwing3D_Kinetic';
    model_path = '';

    possible_paths = {
        'Model/GolfSwing3D_Kinetic.slx',
        'GolfSwing3D_Kinetic.slx',
        fullfile(pwd, 'Model', 'GolfSwing3D_Kinetic.slx'),
        fullfile(pwd, 'GolfSwing3D_Kinetic.slx'),
        which('GolfSwing3D_Kinetic.slx'),
        which('GolfSwing3D_Kinetic')
    };

    for i = 1:length(possible_paths)
        if ~isempty(possible_paths{i}) && exist(possible_paths{i}, 'file')
            model_path = possible_paths{i};
            fprintf('Found model at: %s\n', model_path);
            break;
        end
    end

    if isempty(model_path)
        fprintf('ERROR: Could not find model file\n');
        return;
    end

    % Load the model
    try
        if ~bdIsLoaded(model_name)
            load_system(model_path);
            fprintf('Model loaded successfully\n');
        else
            fprintf('Model already loaded\n');
        end
    catch ME
        fprintf('ERROR: Could not load model: %s\n', ME.message);
        return;
    end

    % Check To Workspace blocks
    fprintf('\n--- Checking To Workspace Blocks ---\n');

    try
        % Find all To Workspace blocks
        to_workspace_blocks = find_system(model_name, 'BlockType', 'ToWorkspace');

        if isempty(to_workspace_blocks)
            fprintf('WARNING: No To Workspace blocks found in the model!\n');
            fprintf('This means no data will be saved to the workspace.\n');
        else
            fprintf('Found %d To Workspace blocks:\n', length(to_workspace_blocks));

            % Expected variable names from the script
            expected_names = {
                'HipLogs', 'SpineLogs', 'TorsoLogs', ...
                'LSLogs', 'RSLogs', 'LELogs', 'RELogs', ...
                'LWLogs', 'RWLogs', 'LScapLogs', 'RScapLogs', ...
                'LFLogs', 'RFLogs'
            };

            found_expected = {};
            found_other = {};

            for i = 1:length(to_workspace_blocks)
                block_path = to_workspace_blocks{i};

                % Get block parameters
                var_name = get_param(block_path, 'VariableName');
                save_format = get_param(block_path, 'SaveFormat');
                sample_time = get_param(block_path, 'SampleTime');

                fprintf('  Block %d: %s\n', i, block_path);
                fprintf('    Variable Name: %s\n', var_name);
                fprintf('    Save Format: %s\n', save_format);
                fprintf('    Sample Time: %s\n', sample_time);

                % Check if this is an expected name
                if ismember(var_name, expected_names)
                    found_expected{end+1} = var_name;
                    fprintf('    ✓ Expected name found\n');
                else
                    found_other{end+1} = var_name;
                    fprintf('    ⚠ Unexpected name\n');
                end
                fprintf('\n');
            end

            % Summary
            fprintf('Summary:\n');
            fprintf('  Expected names found: %d/%d\n', length(found_expected), length(expected_names));
            fprintf('  Other names found: %d\n', length(found_other));

            if ~isempty(found_expected)
                fprintf('  Expected names: %s\n', strjoin(found_expected, ', '));
            end

            if ~isempty(found_other)
                fprintf('  Other names: %s\n', strjoin(found_other, ', '));
                fprintf('\nRECOMMENDATION: Consider renaming these to match expected names\n');
            end

            % Check for missing expected names
            missing_names = setdiff(expected_names, found_expected);
            if ~isempty(missing_names)
                fprintf('\nMissing expected names: %s\n', strjoin(missing_names, ', '));
                fprintf('RECOMMENDATION: Add To Workspace blocks for these signals\n');
            end
        end

    catch ME
        fprintf('ERROR: Could not check To Workspace blocks: %s\n', ME.message);
    end

    % Check model configuration
    fprintf('\n--- Checking Model Configuration ---\n');

    try
        % Get model parameters
        save_output = get_param(model_name, 'SaveOutput');
        save_format = get_param(model_name, 'SaveFormat');
        return_workspace_outputs = get_param(model_name, 'ReturnWorkspaceOutputs');

        fprintf('SaveOutput: %s\n', save_output);
        fprintf('SaveFormat: %s\n', save_format);
        fprintf('ReturnWorkspaceOutputs: %s\n', return_workspace_outputs);

        % Recommendations
        fprintf('\nRecommendations:\n');
        if ~strcmp(save_output, 'on')
            fprintf('  ⚠ Set SaveOutput to "on"\n');
        end
        if ~strcmp(save_format, 'Structure')
            fprintf('  ⚠ Set SaveFormat to "Structure" for To Workspace blocks\n');
        end
        if ~strcmp(return_workspace_outputs, 'on')
            fprintf('  ⚠ Set ReturnWorkspaceOutputs to "on"\n');
        end

        if strcmp(save_output, 'on') && strcmp(save_format, 'Structure') && strcmp(return_workspace_outputs, 'on')
            fprintf('  ✓ Model configuration looks good for data extraction\n');
        end

    catch ME
        fprintf('ERROR: Could not check model configuration: %s\n', ME.message);
    end

    % Test simulation
    fprintf('\n--- Testing Simulation ---\n');

    try
        % Create a simple test simulation
        simIn = Simulink.SimulationInput(model_name);
        simIn = simIn.setModelParameter('StopTime', '0.1'); % Very short simulation
        simIn = simIn.setModelParameter('SaveOutput', 'on');
        simIn = simIn.setModelParameter('SaveFormat', 'Structure');
        simIn = simIn.setModelParameter('ReturnWorkspaceOutputs', 'on');

        fprintf('Running test simulation...\n');
        simOut = sim(simIn);

        fprintf('Simulation completed successfully\n');

        % Check what data was captured
        if isprop(simOut, 'out') || isfield(simOut, 'out')
            out = simOut.out;
            if isstruct(out)
                fields = fieldnames(out);
                fprintf('Data captured in "out" structure:\n');
                for i = 1:length(fields)
                    field_name = fields{i};
                    field_value = out.(field_name);
                    fprintf('  %s: %s (size: %s)\n', field_name, class(field_value), mat2str(size(field_value)));
                end
            else
                fprintf('"out" is not a struct: %s\n', class(out));
            end
        else
            fprintf('No "out" field found in simulation output\n');
        end

    catch ME
        fprintf('ERROR: Test simulation failed: %s\n', ME.message);
    end

    fprintf('\n=== Configuration Check Complete ===\n');
end

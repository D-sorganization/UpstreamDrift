%% EXTRACT_SIMSCAPE_PARAMETERS - Extract model parameters from Simscape model
% This script extracts joint frames, axes, inertias, and other parameters
% from the GolfSwing3D_Kinetic.slx Simscape Multibody model.
%
% Output: YAML or JSON specification compatible with canonical model format
%
% Usage:
%   extract_simscape_parameters

function extract_simscape_parameters()
    % Get model path
    model_name = 'GolfSwing3D_Kinetic';
    model_path = which([model_name '.slx']);
    
    if isempty(model_path)
        error('Model file not found: %s.slx', model_name);
    end
    
    fprintf('Loading Simscape model: %s
', model_path);
    
    % Load model (without starting simulation)
    try
        load_system(model_name);
        fprintf('Model loaded successfully
');
    catch ME
        error('Failed to load model: %s', ME.message);
    end
    
    % Extract parameters
    fprintf('Extracting model parameters...
');
    
    % Initialize output structure
    output = struct();
    output.metadata = struct();
    output.metadata.name = 'Golf Swing 3D Model';
    output.metadata.source = model_path;
    output.metadata.extraction_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % Extract joint information
    fprintf('  Extracting joint information...
');
    joints = extract_joint_info(model_name);
    output.joints = joints;
    
    % Extract body/inertia information
    fprintf('  Extracting body/inertia information...
');
    bodies = extract_body_info(model_name);
    output.bodies = bodies;
    
    % Extract constraint information
    fprintf('  Extracting constraint information...
');
    constraints = extract_constraint_info(model_name);
    output.constraints = constraints;
    
    % Save to YAML (requires YAML toolbox or manual conversion)
    output_file = fullfile(fileparts(model_path), 'extracted_parameters.yaml');
    fprintf('Saving extracted parameters to: %s
', output_file);
    
    % Convert to YAML-compatible format
    % Note: MATLAB doesn't have native YAML support, so we'll save as JSON
    % which can be easily converted to YAML
    json_file = strrep(output_file, '.yaml', '.json');
    json_str = jsonencode(output, 'PrettyPrint', true);
    fid = fopen(json_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    
    fprintf('Extraction complete. JSON saved to: %s
', json_file);
    fprintf('Convert to YAML using Python: python -c "import json,yaml; print(yaml.dump(json.load(open(''%s''))))"
', json_file);
    
    % Close model
    close_system(model_name, 0);
end

function joints = extract_joint_info(model_name)
    % Extract joint information from Simscape model
    joints = struct();
    
    % Find all joint blocks
    joint_blocks = find_system(model_name, 'BlockType', 'Joint');
    
    for i = 1:length(joint_blocks)
        block_path = joint_blocks{i};
        block_name = get_param(block_path, 'Name');
        
        % Get joint type
        joint_type = get_param(block_path, 'JointType');
        
        % Get axis information if available
        try
            axis = get_param(block_path, 'Axis');
        catch
            axis = '[0 0 1]';  % Default
        end
        
        % Get limits if available
        try
            limits = get_param(block_path, 'PositionLimits');
        catch
            limits = '[-pi pi]';
        end
        
        joints.(matlab.lang.makeValidName(block_name)) = struct(...
            'type', joint_type, ...
            'axis', axis, ...
            'limits', limits ...
        );
    end
end

function bodies = extract_body_info(model_name)
    % Extract body/inertia information
    bodies = struct();
    
    % Find all rigid transform blocks (bodies)
    body_blocks = find_system(model_name, 'BlockType', 'RigidTransform');
    
    for i = 1:length(body_blocks)
        block_path = body_blocks{i};
        block_name = get_param(block_path, 'Name');
        
        % Try to get mass and inertia
        try
            mass = get_param(block_path, 'Mass');
        catch
            mass = '1.0';
        end
        
        try
            inertia = get_param(block_path, 'Inertia');
        catch
            inertia = '[1 1 1 0 0 0]';  % [Ixx Iyy Izz Ixy Ixz Iyz]
        end
        
        bodies.(matlab.lang.makeValidName(block_name)) = struct(...
            'mass', mass, ...
            'inertia', inertia ...
        );
    end
end

function constraints = extract_constraint_info(model_name)
    % Extract constraint information
    constraints = struct();
    
    % Find constraint blocks
    constraint_blocks = find_system(model_name, 'BlockType', 'Constraint');
    
    for i = 1:length(constraint_blocks)
        block_path = constraint_blocks{i};
        block_name = get_param(block_path, 'Name');
        
        constraints.(matlab.lang.makeValidName(block_name)) = struct(...
            'type', get_param(block_path, 'ConstraintType'), ...
            'path', block_path ...
        );
    end
end

function [ks, schema] = build_golf_kinematics()
%BUILD_GOLF_KINEMATICS Bind the loaded golf model to qualified sensor frames.
% The snapshot uses the currently loaded geometry. Joint limits, contacts and
% actuation are not represented by KinematicsSolver: use this only for pose
% calibration/seeding, and validate final trajectories with forward dynamics.
% Caller owns targets, guesses and outputs; no variable roles are preset.
% Coordinate identities use block paths and primitives, never unstable j IDs.
    schema = jsondecode(fileread(fullfile(fileparts(mfilename('fullpath')), ...
        'golf_kinematic_schema.json')));
    assert(bdIsLoaded(schema.model_name), 'build_golf_kinematics:modelNotLoaded', ...
        'Load the golf model and its runtime helper paths before construction.');
    ks = simscape.multibody.KinematicsSolver(schema.model_name, ...
        'DefaultAngleUnit', 'rad', 'DefaultLengthUnit', 'm');
    variables = jointPositionVariables(ks);
    parts = split(variables.ID, '.');
    keys = string(variables.BlockPath) + "|" + parts(:,2);
    count = numel(schema.coordinates);
    assert(height(variables) == count, 'build_golf_kinematics:topologyChanged', ...
        'The native joint inventory differs from the qualified schema.');
    schema.q_ids = strings(count,1);
    schema.coordinate_names = strings(count,1);
    for j = 1:count
        coordinate = schema.coordinates(j);
        found = keys == string(coordinate.block_path) + "|" + string(coordinate.primitive);
        assert(nnz(found) == 1, 'build_golf_kinematics:coordinateMissing', ...
            'Native coordinate is missing or ambiguous: %s', coordinate.name);
        schema.q_ids(j) = variables.ID(found);
        schema.coordinate_names(j) = string(coordinate.name);
    end
    schema.frame_ids = strings(3*numel(schema.frames),1);
    schema.rotation_ids = strings(3*numel(schema.frames),1);
    for f = 1:numel(schema.frames)
        frame = schema.frames(f);
        addFrameVariables(ks, frame.name, 'Translation', schema.world_port, frame.port);
        addFrameVariables(ks, frame.name, 'Rotation', schema.world_port, frame.port);
        schema.frame_ids(3*f-2:3*f) = string(frame.name) + ".Translation." + ["x";"y";"z"];
        schema.rotation_ids(3*f-2:3*f) = string(frame.name) + ".Rotation." + ["x";"y";"z"];
    end
end

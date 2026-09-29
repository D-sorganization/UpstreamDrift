function specified = decode_input_file_pose(input_file)
%DECODE_INPUT_FILE_POSE  Read specified initial joint angles from a 3DModelInputs MAT.
%
%   The 3D golf model input MATs (e.g. 3DModelInputs_Impact.mat) contain
%   a struct describing the desired initial conditions. The exact layout
%   has evolved; this function tolerates several known shapes and returns
%   a normalized struct:
%       .q            - column vector of joint angles in radians
%       .joint_names  - cell array of joint name strings, same length as q
%       .r_butt       - [3x1] specified butt Cartesian position (m), or []
%       .r_clubhead   - [3x1] specified clubhead Cartesian position (m), or []
%       .raw_inputs   - the original loaded struct (passed back into the model)

    s = load(input_file);
    fns = fieldnames(s);

    % Heuristic: the input struct is typically the only top-level variable,
    % or is named GolfInputs / inputs / ICs.
    candidates = {"GolfInputs", "inputs", "ICs", "InitialConditions"};
    raw = [];
    for k = 1:numel(candidates)
        if isfield(s, candidates{k})
            raw = s.(candidates{k});
            break;
        end
    end
    if isempty(raw) && numel(fns) == 1
        raw = s.(fns{1});
    end
    if isempty(raw)
        raw = s; % fall through: treat the whole file as the inputs blob
    end

    % --- joint angles ---------------------------------------------------
    [q, joint_names] = local_collect_joint_angles(raw);

    specified = struct();
    specified.q = q(:);
    specified.joint_names = joint_names(:);
    specified.raw_inputs = raw;

    % --- optional Cartesian markers ------------------------------------
    specified.r_butt = local_get_xyz(raw, ...
        {'r_butt', 'butt_position', 'ButtPosition', 'butt_xyz'});
    specified.r_clubhead = local_get_xyz(raw, ...
        {'r_clubhead', 'clubhead_position', 'ClubheadPosition', 'clubhead_xyz', 'r_ch'});
end

function [q, names] = local_collect_joint_angles(raw)
    % Try a handful of conventions used across the model's history.
    if isfield(raw, 'q0') && isnumeric(raw.q0)
        q = raw.q0(:);
        if isfield(raw, 'joint_names')
            names = cellstr(raw.joint_names);
        else
            names = arrayfun(@(i) sprintf('q%d', i), 1:numel(q), 'UniformOutput', false);
        end
        return;
    end
    if isfield(raw, 'JointAngles') && isstruct(raw.JointAngles)
        ja = raw.JointAngles;
        names = fieldnames(ja);
        q = zeros(numel(names), 1);
        for i = 1:numel(names)
            v = ja.(names{i});
            q(i) = v(1);
        end
        return;
    end
    % Last resort: scan top-level numeric scalar fields whose names look
    % like joint angles (contain "Angle" or end with "_q").
    fns = fieldnames(raw);
    keep = false(size(fns));
    for i = 1:numel(fns)
        v = raw.(fns{i});
        if isnumeric(v) && isscalar(v) ...
                && (contains(fns{i}, 'Angle', 'IgnoreCase', true) ...
                    || endsWith(fns{i}, '_q'))
            keep(i) = true;
        end
    end
    names = fns(keep);
    q = zeros(numel(names), 1);
    for i = 1:numel(names)
        q(i) = raw.(names{i});
    end
    if isempty(q)
        warning('decode_input_file_pose:NoJointsFound', ...
            'Could not locate specified joint angles in input file; returning empty.');
        q = zeros(0, 1);
        names = cell(0, 1);
    end
end

function xyz = local_get_xyz(raw, candidates)
    xyz = [];
    for k = 1:numel(candidates)
        if isfield(raw, candidates{k})
            v = raw.(candidates{k});
            if isnumeric(v) && numel(v) == 3
                xyz = v(:);
                return;
            end
        end
    end
end

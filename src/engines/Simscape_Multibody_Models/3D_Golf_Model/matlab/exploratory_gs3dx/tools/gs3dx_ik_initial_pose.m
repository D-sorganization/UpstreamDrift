function pose_out = gs3dx_ik_initial_pose(initial_pose)
%GS3DX_IK_INITIAL_POSE  Pure contract validator and normalizer for IK initial pose (#10979).
%
%   POSE_OUT = GS3DX_IK_INITIAL_POSE(INITIAL_POSE) validates and normalizes an
%   explicit initial pose struct for warm-starting GS3DX_Human whole-body IK.
%
%   Inputs:
%     INITIAL_POSE  Empty ([] or struct([])) or a scalar struct with canonical fields:
%       .joint_keys  48-element string or cellstr vector of unique non-empty text keys.
%       .joint       48-element finite real numeric vector of native joint coordinates.
%       .units       48-element string or cellstr vector of physical units ('m', 'deg', '1').
%       .status      Finite real scalar 1 (loop-closure success indicator).
%
%   Outputs:
%     POSE_OUT  struct([]) if INITIAL_POSE is empty, or normalized scalar struct:
%       .joint_keys  48x1 string array of canonical keys.
%       .joint       48x1 double vector of joint values.
%       .units       48x1 string array of expected physical units.
%       .status      1 (double scalar).
%
%   Design Contracts & Validation Invariants:
%     - Rejects unrequested aliases (keys, q, values). Canonical exact fields required.
%     - Rejects multi-frame joint arrays (single native 48-variable pose vector required;
%       no silent first-column truncation).
%     - Rejects empty text bypass (empty string/char cannot substitute for [] or struct([])).
%     - Rejects arbitrary types coerced to string (must be explicit string vector or cellstr).
%     - Reuses gs3dx_initial_target_values to validate the 48-variable schema, physical
%       units, SO(3) spherical unit axes, and gimbal singularities.
%     - Emits uniform error 'gs3dx:ik' for all contract violations; does not catch
%       unrelated environment errors (e.g., missing helper functions).
%     - Keyed seed is restricted to GS3DX_Human kinematic layout.

    arguments
        initial_pose
    end

    % 1. Empty default handling: allow numeric [] or struct([]) only
    if isempty(initial_pose)
        if (isnumeric(initial_pose) && numel(initial_pose) == 0) || ...
           (isstruct(initial_pose) && numel(initial_pose) == 0)
            pose_out = struct([]);
            return;
        end
        error('gs3dx:ik', 'Empty initial_pose must be numeric [] or struct([]).');
    end

    % 2. Strict scalar struct contract (reject array struct and non-struct types)
    if ~isstruct(initial_pose) || ~isscalar(initial_pose)
        error('gs3dx:ik', 'initial_pose must be a scalar struct or empty []/struct([]).');
    end

    % 3. Field contract: exact canonical fields, no unrequested aliases
    req_fields = {'joint_keys', 'joint', 'units', 'status'};
    for k = 1:numel(req_fields)
        if ~isfield(initial_pose, req_fields{k})
            error('gs3dx:ik', 'initial_pose missing required canonical field "%s".', req_fields{k});
        end
    end

    % 4. Status contract: required finite real scalar 1
    st = initial_pose.status;
    if ~isnumeric(st) || ~isreal(st) || ~isscalar(st) || ~isfinite(st) || st ~= 1
        error('gs3dx:ik', 'initial_pose status must be finite real scalar 1.');
    end

    % 5. Joint values contract: single native pose real finite vector 48 (no multi-frame matrix)
    j = initial_pose.joint;
    if ~isnumeric(j) || ~isreal(j) || ~isvector(j) || numel(j) ~= 48 || any(~isfinite(j(:)))
        error('gs3dx:ik', 'initial_pose joint must be a real finite 48-element numeric vector.');
    end
    joint_vec = double(j(:));

    % 6. Joint keys contract: string or cellstr vector, 48 nonempty nonmissing unique text keys
    jk = initial_pose.joint_keys;
    if (~isstring(jk) && ~iscellstr(jk)) || ~isvector(jk) || numel(jk) ~= 48
        error('gs3dx:ik', 'initial_pose joint_keys must be a 48-element string or cellstr vector.');
    end
    jk_str = string(jk(:));
    if any(ismissing(jk_str)) || any(strlength(strtrim(jk_str)) == 0)
        error('gs3dx:ik', 'initial_pose joint_keys must contain nonempty nonmissing text values.');
    end
    if numel(unique(jk_str)) ~= 48
        error('gs3dx:ik', 'initial_pose joint_keys contains duplicate entries.');
    end

    % 7. Units contract: string or cellstr vector, 48 nonempty nonmissing text units
    u = initial_pose.units;
    if (~isstring(u) && ~iscellstr(u)) || ~isvector(u) || numel(u) ~= 48
        error('gs3dx:ik', 'initial_pose units must be a 48-element string or cellstr vector.');
    end
    u_str = string(u(:));
    if any(ismissing(u_str)) || any(strlength(strtrim(u_str)) == 0)
        error('gs3dx:ik', 'initial_pose units must contain nonempty nonmissing text values.');
    end

    % 8. Canonical schema, unit, spherical axis, and gimbal validation via public helper
    try
        gs3dx_initial_target_values(jk_str, u_str, joint_vec);
    catch me
        if startsWith(me.identifier, 'gs3dx:')
            error('gs3dx:ik', '%s', me.message);
        else
            rethrow(me);
        end
    end

    % 9. Normalized output struct
    pose_out = struct( ...
        'joint_keys', jk_str, ...
        'joint', joint_vec, ...
        'units', u_str, ...
        'status', 1);
end

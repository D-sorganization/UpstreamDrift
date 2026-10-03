function roles = gs3dx_ik_joint_roles(paths_in, ids_in)
%GS3DX_IK_JOINT_ROLES  Model-independent identification of IK joint roles (#10979, #11161).
%
%   ROLES = GS3DX_IK_JOINT_ROLES(JP) or GS3DX_IK_JOINT_ROLES(PATHS, IDS)
%   identifies the invariant roles of KinematicsSolver joint position variables
%   from their stable block-path keys rather than hardcoded IDs ("j15", etc.).
%   Works across GS3DX_Fit, GS3DX_Human, and renumbered/synthetic models.
%
%   ROLES fields:
%     .closed_mask         logical(numel(ids), 1) indicating right arm closed loop
%                          (Right Elbow Joint, Right Shoulder Joint, Right Wrist and Hand)
%     .closed_ids          string array of closed variable IDs
%     .target_ids          string array of independent target variable IDs
%     .is_trunk_id         logical(numel(ids), 1) indicating trunk variables in ids
%                          (Spine Tilt, Torso, Left Scapula, Right Scapula)
%     .trunk_ids           string array of trunk variable IDs (for local_seed)
%     .pelvis_trans_keys   string array of the 3 pelvis translation keys (e.g. "j1.Px", ...)
%     .layout              struct array with .key and .n for independent variables
%     .n_independent       scalar total independent coordinates (sum of layout.n)
%     .is_trunk_coord      logical(n_independent, 1) indicating trunk parameter coordinates
%     .pelvis_trans_indices  1x3 double indices into parameter vector p for [Px, Py, Pz]
%
%   Errors:
%     gs3dx:ik:missing_anatomy    A required joint group or translation primitive is missing.
%     gs3dx:ik:ambiguous_anatomy  A joint group matches multiple conflicting blocks.

    arguments
        paths_in = []
        ids_in = []
    end

    if istable(paths_in)
        t = paths_in;
        assert(all(ismember({'BlockPath', 'ID'}, t.Properties.VariableNames)), ...
            'gs3dx:ik:missing_anatomy', 'Table must have BlockPath and ID columns');
        paths = string(t.BlockPath);
        ids = string(t.ID);
    else
        assert(~isempty(paths_in) && ~isempty(ids_in), ...
            'gs3dx:ik:missing_anatomy', 'Must provide paths and ids');
        paths = string(paths_in);
        ids = string(ids_in);
    end

    % Normalize to column vectors
    paths = paths(:);
    ids = ids(:);

    n_vars = numel(ids);
    assert(numel(paths) == n_vars, 'gs3dx:ik:ambiguous_anatomy', ...
        'Paths (%d) and IDs (%d) length mismatch', numel(paths), n_vars);
    assert(all(~ismissing(paths)) && all(~ismissing(ids)), ...
        'gs3dx:ik:missing_anatomy', 'Paths and IDs must not contain missing values');
    assert(all(strlength(strtrim(paths)) > 0) && all(strlength(strtrim(ids)) > 0), ...
        'gs3dx:ik:missing_anatomy', 'Paths and IDs must be non-empty text');
    assert(numel(unique(ids)) == n_vars, 'gs3dx:ik:ambiguous_anatomy', ...
        'IDs must be unique');

    % 1. Closed RHS arm joints (Right Elbow Joint, Right Shoulder Joint, Right Wrist and Hand)
    % Each required role must match EXACTLY ONE distinct leaf joint block.
    rhs_arm_patterns = [
        "Right Elbow Joint", ...
        "Right Shoulder Joint", ...
        "Right Wrist and Hand"
    ];
    closed_mask = false(n_vars, 1);
    for pat = rhs_arm_patterns
        matches = contains(paths, pat);
        if ~any(matches)
            error('gs3dx:ik:missing_anatomy', 'Missing RHS arm joint: %s', pat);
        end
        unique_blks = unique(paths(matches));
        if numel(unique_blks) > 1
            error('gs3dx:ik:ambiguous_anatomy', ...
                'Ambiguous multiple leaf blocks for RHS arm joint %s: %s', pat, strjoin(unique_blks, ', '));
        end
        closed_mask = closed_mask | matches(:);
    end

    target_ids = ids(~closed_mask);
    closed_ids = ids(closed_mask);

    % 2. Trunk subsystems (Spine Tilt, Torso, Left Scapula, Right Scapula)
    % Each trunk role must match EXACTLY ONE distinct leaf joint block.
    trunk_specs = {
        'Spine Tilt',   contains(paths, "Spine Tilt"); ...
        'Torso',        contains(paths, "Torso Kinetically Driven") | (contains(paths, "Torso") & ~contains(paths, "Hips and Torso")); ...
        'Left Scapula', contains(paths, "Left Scapula"); ...
        'Right Scapula', contains(paths, "Right Scapula")
    };

    is_trunk_id = false(n_vars, 1);
    for row = 1:size(trunk_specs, 1)
        name = trunk_specs{row, 1};
        m = trunk_specs{row, 2};
        if ~any(m)
            error('gs3dx:ik:missing_anatomy', 'Missing trunk joint: %s', name);
        end
        unique_blks = unique(paths(m));
        if numel(unique_blks) > 1
            error('gs3dx:ik:ambiguous_anatomy', ...
                'Ambiguous multiple leaf blocks for trunk joint %s: %s', name, strjoin(unique_blks, ', '));
        end
        is_trunk_id = is_trunk_id | m(:);
    end
    trunk_ids = ids(is_trunk_id);

    % 3. Pelvis Translation Parameters: Hip Joint (excluding Left/Right legs) with exact Px.p, Py.p, Pz.p
    is_pelvis_blk = contains(paths, "Hip Joint") & ...
        ~contains(paths, ["Left Hip", "Right Hip", "Left Knee", "Right Knee", "L Hip", "R Hip"]);
    if ~any(is_pelvis_blk)
        error('gs3dx:ik:missing_anatomy', 'Missing pelvis root Hip Joint');
    end
    unique_pelvis = unique(paths(is_pelvis_blk));
    if numel(unique_pelvis) > 1
        error('gs3dx:ik:ambiguous_anatomy', ...
            'Ambiguous multiple pelvis Hip Joint blocks: %s', strjoin(unique_pelvis, ', '));
    end

    pelvis_px = is_pelvis_blk & endsWith(ids, ".Px.p");
    pelvis_py = is_pelvis_blk & endsWith(ids, ".Py.p");
    pelvis_pz = is_pelvis_blk & endsWith(ids, ".Pz.p");

    if ~any(pelvis_px) || ~any(pelvis_py) || ~any(pelvis_pz)
        error('gs3dx:ik:missing_anatomy', 'Missing pelvis exact .Px.p, .Py.p, or .Pz.p translation primitive');
    end
    if nnz(pelvis_px) > 1 || nnz(pelvis_py) > 1 || nnz(pelvis_pz) > 1
        error('gs3dx:ik:ambiguous_anatomy', 'Duplicate pelvis translation primitives');
    end

    % 4. Target layout and position schema validation
    parts = split(target_ids, '.');
    assert(size(parts, 2) >= 3, 'gs3dx:ik:missing_anatomy', 'Malformed joint IDs schema');

    group_keys = parts(:, 1) + "." + parts(:, 2);
    unique_keys = unique(group_keys, 'stable');
    n_groups = numel(unique_keys);

    layout = repmat(struct('key', "", 'n', 0), n_groups, 1);
    for k = 1:n_groups
        cur_key = unique_keys(k);
        rows = find(group_keys == cur_key);
        prim_type = parts(rows(1), 2);
        suffixes = sort(parts(rows, 3));

        if prim_type == "S"
            % Spherical joint requires exact set: ax_x, ax_y, ax_z, q
            expected_s = sort(["ax_x"; "ax_y"; "ax_z"; "q"]);
            if ~isequal(suffixes, expected_s)
                error('gs3dx:ik:missing_anatomy', ...
                    'Spherical joint %s has incomplete or malformed primitives', cur_key);
            end
            layout(k).key = cur_key;
            layout(k).n = 3;
        else
            % 1-DOF primitive must be unique and have supported suffix 'p' or 'q'
            if numel(rows) ~= 1
                error('gs3dx:ik:ambiguous_anatomy', ...
                    'Duplicate 1-DOF primitive variable for %s', cur_key);
            end
            suf = suffixes(1);
            if ~ismember(suf, ["p", "q"])
                error('gs3dx:ik:missing_anatomy', ...
                    'Joint %s has unsupported primitive suffix: %s', cur_key, suf);
            end
            layout(k).key = cur_key;
            layout(k).n = 1;
        end
    end

    n_independent = sum([layout.n]);
    assert(n_independent > 0, 'gs3dx:ik:missing_anatomy', 'No independent coordinates found');

    % Locate parameter indices of [Px, Py, Pz] in p
    id_px = ids(pelvis_px);
    id_py = ids(pelvis_py);
    id_pz = ids(pelvis_pz);

    key_px = extract_layout_key(id_px);
    key_py = extract_layout_key(id_py);
    key_pz = extract_layout_key(id_pz);
    pelvis_trans_keys = [key_px; key_py; key_pz];

    start_indices = [0, cumsum([layout.n])];
    layout_keys = string({layout.key});

    k_px = find(layout_keys == key_px, 1);
    k_py = find(layout_keys == key_py, 1);
    k_pz = find(layout_keys == key_pz, 1);

    assert(~isempty(k_px) && ~isempty(k_py) && ~isempty(k_pz), ...
        'gs3dx:ik:missing_anatomy', 'Pelvis translation keys not found in independent layout');
    assert(layout(k_px).n == 1 && layout(k_py).n == 1 && layout(k_pz).n == 1, ...
        'gs3dx:ik:ambiguous_anatomy', 'Pelvis translation coordinates must be 1-DOF primitives');

    pelvis_trans_indices = [start_indices(k_px) + 1, start_indices(k_py) + 1, start_indices(k_pz) + 1];

    % 5. Trunk coordinates mask over p
    is_trunk_coord = false(n_independent, 1);
    trunk_joint_prefixes = unique(extractBefore(trunk_ids, "."));
    for k = 1:numel(layout)
        j_prefix = extractBefore(layout(k).key, ".");
        if ismember(j_prefix, trunk_joint_prefixes)
            idx_range = start_indices(k) + (1:layout(k).n);
            is_trunk_coord(idx_range) = true;
        end
    end
    assert(nnz(is_trunk_coord) == 7, 'gs3dx:ik:missing_anatomy', ...
        'Expected exactly 7 trunk coordinates in independent layout, found %d', nnz(is_trunk_coord));

    % Build output struct
    roles.closed_mask = closed_mask;
    roles.closed_ids = closed_ids;
    roles.target_ids = target_ids;
    roles.is_trunk_id = is_trunk_id;
    roles.trunk_ids = trunk_ids;
    roles.pelvis_trans_keys = pelvis_trans_keys;
    roles.layout = layout;
    roles.n_independent = n_independent;
    roles.is_trunk_coord = is_trunk_coord;
    roles.pelvis_trans_indices = pelvis_trans_indices;
end

function k = extract_layout_key(id_str)
    parts = split(id_str, '.');
    k = parts(1) + "." + parts(2);
end

function audit = gs3dx_inertia_audit(mdl)
%GS3DX_INERTIA_AUDIT  Evaluate solid and segment inertias of a loaded GS3DX model (#10979).
%
%   AUDIT = GS3DX_INERTIA_AUDIT(MDL) audits the mass, centre of mass and
%   principal moments of inertia for every Solid block in the loaded Simulink
%   model MDL, and compares the body segment moments against the de Leva
%   (1996) anthropometric reference from GS3DX_SEGMENT_INERTIA.
%
%   MDL must be loaded (no simulation is performed).
%
%   Returns struct AUDIT:
%     .solids       table (block, shape, inertia_type, mass, com, moments)
%                   giving evaluated parameters in SI units (kg, m, kg*m^2)
%                   for each Solid block.
%     .segments     table comparing model segment moments against de Leva
%                   reference moments, including transverse and longitudinal
%                   ratios.
%     .total_mass   sum of masses across all solids (kg).
%     .model_name   name of the audited model (string).
%
%   Solid inertia cases supported:
%     - CalculateFromGeometry: Cylindrical (axis = z), Spherical, Brick,
%       Ellipsoidal solids with BasedOnType 'Mass' or 'Density'.
%     - Custom: evaluated Mass, CenterOfMass, MomentsOfInertia, and
%       ProductsOfInertia.
%     - PointMass: zero moments about its centre.
%
%   Multi-piece segments that share an axis (forearm halves, UpperTorsoBase +
%   UpperTorsoTop) are lumped using the parallel-axis theorem.
%
%   Precondition: MDL must be a valid, loaded Simulink model.
%   Postcondition: AUDIT.solids and AUDIT.segments must be valid tables with
%   positive masses.

    arguments
        mdl {mustBeA(mdl, ["char", "string"])}
    end

    mdl_str = char(mdl);
    assert(bdIsLoaded(mdl_str), 'gs3dx:audit', 'Model %s must be loaded.', mdl_str);

    % Find every Solid block in the model
    solids_list = find_system(mdl_str, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'Regexp', 'on', 'ReferenceBlock', 'sm_lib/Body Elements/.* Solid');
    n_solids = numel(solids_list);
    assert(n_solids > 0, 'gs3dx:audit', 'No Solid blocks found in %s.', mdl_str);

    % Allocate solid audit arrays
    block_names = strings(n_solids, 1);
    shapes = strings(n_solids, 1);
    inertia_types = strings(n_solids, 1);
    masses = zeros(n_solids, 1);
    coms = zeros(n_solids, 3);
    moments = zeros(n_solids, 3);

    for k = 1:n_solids
        blk = solids_list{k};
        rel = extractAfter(blk, [mdl_str '/']);
        block_names(k) = string(rel);

        [shape_k, itype_k, m_k, com_k, I_k] = local_evaluate_solid(blk);
        shapes(k) = shape_k;
        inertia_types(k) = itype_k;
        masses(k) = m_k;
        coms(k, :) = com_k;
        moments(k, :) = I_k;
    end

    solids_tbl = table(block_names, shapes, inertia_types, masses, coms, moments, ...
        'VariableNames', {'block', 'shape', 'inertia_type', 'mass', 'com', 'moments'});

    % Evaluate segment-level audit
    segments_tbl = local_build_segment_table(mdl_str, solids_tbl);

    audit = struct();
    audit.solids = solids_tbl;
    audit.segments = segments_tbl;
    audit.total_mass = sum(masses);
    audit.model_name = string(mdl_str);

    % Postconditions
    assert(istable(audit.solids), 'gs3dx:audit', 'Postcondition: audit.solids must be a table.');
    assert(istable(audit.segments), 'gs3dx:audit', 'Postcondition: audit.segments must be a table.');
    assert(audit.total_mass > 0, 'gs3dx:audit', 'Postcondition: total_mass must be positive.');
end

function [shape, inertia_type, m, com, I] = local_evaluate_solid(blk)
    ref = get_param(blk, 'ReferenceBlock');
    params = get_param(blk, 'DialogParameters');

    shape = "Unknown";
    if isfield(params, 'GeometryShape')
        shape = string(get_param(blk, 'GeometryShape'));
    elseif contains(ref, 'Cylindrical')
        shape = "Cylinder";
    elseif contains(ref, 'Spherical')
        shape = "Sphere";
    elseif contains(ref, 'Brick')
        shape = "Brick";
    elseif contains(ref, 'Ellipsoid')
        shape = "Ellipsoid";
    elseif contains(ref, 'Point Mass') || contains(ref, 'PointMass')
        shape = "PointMass";
    end

    inertia_type = "CalculateFromGeometry";
    if isfield(params, 'InertiaType')
        inertia_type = string(get_param(blk, 'InertiaType'));
    end

    based_on = "Mass";
    if isfield(params, 'BasedOnType')
        based_on = string(get_param(blk, 'BasedOnType'));
    end

    m = 0;
    com = [0, 0, 0];
    I = [0, 0, 0];

    if strcmp(inertia_type, "CalculateFromGeometry")
        switch shape
            case "Cylinder"
                R = local_eval_expr(blk, 'CylinderRadius') * local_length_unit(get_param(blk, 'CylinderRadiusUnits'));
                L = local_eval_expr(blk, 'CylinderLength') * local_length_unit(get_param(blk, 'CylinderLengthUnits'));
                V = pi * R^2 * L;
                if strcmp(based_on, "Mass")
                    m = local_eval_expr(blk, 'Mass') * local_mass_unit(get_param(blk, 'MassUnits'));
                else
                    rho = local_eval_expr(blk, 'Density') * local_density_unit(get_param(blk, 'DensityUnits'));
                    m = rho * V;
                end
                if m > 0
                    % Cylinder axis is z: principal moments about center
                    I = [m/12 * (3*R^2 + L^2), m/12 * (3*R^2 + L^2), 0.5 * m * R^2];
                end

            case "Sphere"
                R = local_eval_expr(blk, 'SphereRadius') * local_length_unit(get_param(blk, 'SphereRadiusUnits'));
                V = 4/3 * pi * R^3;
                if strcmp(based_on, "Mass")
                    m = local_eval_expr(blk, 'Mass') * local_mass_unit(get_param(blk, 'MassUnits'));
                else
                    rho = local_eval_expr(blk, 'Density') * local_density_unit(get_param(blk, 'DensityUnits'));
                    m = rho * V;
                end
                if m > 0
                    I = (2/5 * m * R^2) * [1, 1, 1];
                end

            case "Brick"
                D = local_eval_expr(blk, 'BrickDimensions') * local_length_unit(get_param(blk, 'BrickDimensionsUnits'));
                V = prod(D);
                if strcmp(based_on, "Mass")
                    m = local_eval_expr(blk, 'Mass') * local_mass_unit(get_param(blk, 'MassUnits'));
                else
                    rho = local_eval_expr(blk, 'Density') * local_density_unit(get_param(blk, 'DensityUnits'));
                    m = rho * V;
                end
                if m > 0
                    I = [m/12 * (D(2)^2 + D(3)^2), m/12 * (D(1)^2 + D(3)^2), m/12 * (D(1)^2 + D(2)^2)];
                end

            case "Ellipsoid"
                E = local_eval_expr(blk, 'EllipsoidRadii') * local_length_unit(get_param(blk, 'EllipsoidRadiiUnits'));
                V = 4/3 * pi * prod(E);
                if strcmp(based_on, "Mass")
                    m = local_eval_expr(blk, 'Mass') * local_mass_unit(get_param(blk, 'MassUnits'));
                else
                    rho = local_eval_expr(blk, 'Density') * local_density_unit(get_param(blk, 'DensityUnits'));
                    m = rho * V;
                end
                if m > 0
                    I = [m/5 * (E(2)^2 + E(3)^2), m/5 * (E(1)^2 + E(3)^2), m/5 * (E(1)^2 + E(2)^2)];
                end
        end

    elseif strcmp(inertia_type, "Custom")
        m = local_eval_expr(blk, 'Mass') * local_mass_unit(get_param(blk, 'MassUnits'));
        if isfield(params, 'CenterOfMass')
            com = local_eval_expr(blk, 'CenterOfMass') * local_length_unit(get_param(blk, 'CenterOfMassUnits'));
        end
        if isfield(params, 'MomentsOfInertia')
            I_diag = local_eval_expr(blk, 'MomentsOfInertia') * local_inertia_unit(get_param(blk, 'MomentsOfInertiaUnits'));
        else
            I_diag = [0, 0, 0];
        end
        if isfield(params, 'ProductsOfInertia')
            P = local_eval_expr(blk, 'ProductsOfInertia') * local_inertia_unit(get_param(blk, 'ProductsOfInertiaUnits'));
        else
            P = [0, 0, 0];
        end
        if all(P == 0)
            I = I_diag;
        else
            % Simscape convention: ProductsOfInertia is [Iyz Izx Ixy]
            I_mat = [I_diag(1), -P(3), -P(2); ...
                    -P(3), I_diag(2), -P(1); ...
                    -P(2), -P(1), I_diag(3)];
            I = sort(eig(I_mat))';
        end

    elseif strcmp(inertia_type, "PointMass")
        m = local_eval_expr(blk, 'Mass') * local_mass_unit(get_param(blk, 'MassUnits'));
        if isfield(params, 'CenterOfMass')
            com = local_eval_expr(blk, 'CenterOfMass') * local_length_unit(get_param(blk, 'CenterOfMassUnits'));
        end
        I = [0, 0, 0];
    end
end

function val = local_eval_expr(blk, param_name)
    raw = get_param(blk, param_name);
    if isnumeric(raw)
        val = double(raw);
        return;
    end
    raw_str = char(strtrim(raw));
    num = str2double(raw_str);
    if ~isnan(num)
        val = num;
        return;
    end
    try
        val = double(slResolve(raw_str, blk));
    catch
        val = str2num(raw_str); %#ok<ST2NM>
        assert(~isempty(val), 'gs3dx:audit', 'Failed to resolve parameter %s on block %s.', param_name, blk);
    end
end

function f = local_length_unit(u)
    switch lower(strtrim(char(u)))
        case 'm', f = 1;
        case 'cm', f = 0.01;
        case 'mm', f = 0.001;
        case 'in', f = 0.0254;
        case 'ft', f = 0.3048;
        otherwise, error('gs3dx:audit', 'Unknown length unit: %s', u);
    end
end

function f = local_mass_unit(u)
    switch lower(strtrim(char(u)))
        case 'kg', f = 1;
        case 'g', f = 1e-3;
        case 'lbm', f = 0.45359237;
        case 'slug', f = 14.5939029;
        case 'oz', f = 0.45359237 / 16;
        otherwise, error('gs3dx:audit', 'Unknown mass unit: %s', u);
    end
end

function f = local_density_unit(u)
    switch lower(strtrim(char(u)))
        case 'kg/m^3', f = 1;
        case 'g/cm^3', f = 1000;
        case 'lbm/in^3', f = 0.45359237 / (0.0254^3);
        case 'lbm/ft^3', f = 0.45359237 / (0.3048^3);
        otherwise, error('gs3dx:audit', 'Unknown density unit: %s', u);
    end
end

function f = local_inertia_unit(u)
    switch lower(strtrim(char(u)))
        case {'kg*m^2', 'kg*m2'}, f = 1;
        case {'g*cm^2', 'g*cm2'}, f = 1e-7;
        case {'lbm*in^2', 'lbm*in2'}, f = 0.45359237 * (0.0254^2);
        case {'lbm*ft^2', 'lbm*ft2'}, f = 0.45359237 * (0.3048^2);
        otherwise, error('gs3dx:audit', 'Unknown inertia unit: %s', u);
    end
end

function num = local_to_double(val, mdl_str)
    if isempty(val)
        num = [];
        return;
    end
    if isobject(val)
        if isprop(val, 'Value')
            val = val.Value;
        elseif ismethod(val, 'double')
            val = double(val);
        end
    end
    if ischar(val) || isstring(val)
        val_str = char(strtrim(val));
        num = str2double(val_str);
        if isnan(num) && nargin > 1 && bdIsLoaded(mdl_str)
            try
                num = double(slResolve(val_str, mdl_str));
            catch
                num = [];
            end
        end
        return;
    end
    try
        num = double(val);
    catch
        num = [];
    end
end

function [val, var_name] = local_ws_lookup(ws, names_list, unit_scale, mdl_str)
    if nargin < 3, unit_scale = 1; end
    if nargin < 4, mdl_str = ''; end
    val = [];
    var_name = "";
    for i = 1:numel(names_list)
        vn = names_list{i};
        if ws.hasVariable(vn)
            raw = ws.getVariable(vn);
            num = local_to_double(raw, mdl_str);
            if ~isempty(num) && ~isnan(num)
                val = num * unit_scale;
                var_name = string(vn);
                return;
            end
        end
    end
end

function row = local_find_solid(solids_tbl, rel_path)
    idx = find(solids_tbl.block == string(rel_path), 1);
    if isempty(idx)
        idx = find(endsWith(solids_tbl.block, string(rel_path)), 1);
    end
    assert(~isempty(idx), 'gs3dx:audit', 'Solid %s not found in audit.solids', rel_path);
    row = solids_tbl(idx, :);
end

function tbl = local_build_segment_table(mdl_str, solids_tbl)
    ws = get_param(mdl_str, 'ModelWorkspace');
    if ws.hasVariable('GolferBodyMass')
        M_body = local_to_double(ws.getVariable('GolferBodyMass'), mdl_str);
    else
        M_body = 80;
    end
    anthro = gs3dx_anthropometry(M_body);

    % Segments with a de Leva counterpart: head, upper_arm, forearm, hand,
    % thigh, shank, foot (L / R), lower_trunk, upper_trunk and the whole
    % trunk.  The neck (de Leva's head includes it) and the shoulder hubs
    % (part of de Leva's upper trunk) have none and are left out.
    entries = {};

    % --- Head ---
    r_head = local_find_solid(solids_tbl, 'Hips and Torso Inputs/Head');
    L_head = anthro.length.head;
    m_head = r_head.mass;
    I_head = r_head.moments;
    [~, ~, I_ref_head] = gs3dx_segment_inertia(m_head, L_head, anthro.com.head, anthro.gyration.head);
    entries{end+1} = local_make_seg_entry("head", m_head, L_head, "anthro.length.head",...
        I_head, I_ref_head, 3);

    % --- Upper Arm (L / R) ---
    [L_ua, v_ua] = local_ws_lookup(ws, {'FitUpperArmLength', 'UpperArmLength'}, 0.0254, mdl_str);
    if isempty(L_ua), L_ua = anthro.length.upper_arm; v_ua = "anthro.length.upper_arm"; end
    for side = ["L", "R"]
        r_ua = local_find_solid(solids_tbl, side + "UpperArm");
        m_ua = r_ua.mass;
        I_ua = r_ua.moments;
        [~, ~, I_ref_ua] = gs3dx_segment_inertia(m_ua, L_ua, anthro.com.upper_arm, anthro.gyration.upper_arm);
        entries{end+1} = local_make_seg_entry("upper_arm_" + side, m_ua, L_ua, v_ua,...
            I_ua, I_ref_ua, 3);
    end

    % --- Forearm (L / R) ---
    % Lump two forearm halves (upper + lower) using parallel-axis theorem along cylinder z-axis
    [L_fa, v_fa] = local_ws_lookup(ws, {'FitLowerArmLength', 'LowerArmLength'}, 0.0254, mdl_str);
    if isempty(L_fa), L_fa = anthro.length.forearm; v_fa = "anthro.length.forearm"; end
    for side = ["Left", "Right"]
        s_code = "L";
        if side == "Right", s_code = "R"; end
        r_uf = local_find_solid(solids_tbl, side + " Forearm/" + s_code + "UpperForearm");
        r_lf = local_find_solid(solids_tbl, side + " Forearm/" + s_code + "LowerForearm");
        m_fa = r_uf.mass + r_lf.mass;
        d_fa = L_fa / 4;
        I_fa_trans = (r_uf.moments(1) + r_uf.mass * d_fa^2) + (r_lf.moments(1) + r_lf.mass * d_fa^2);
        I_fa_long = r_uf.moments(3) + r_lf.moments(3);
        I_fa = [I_fa_trans, I_fa_trans, I_fa_long];
        [~, ~, I_ref_fa] = gs3dx_segment_inertia(m_fa, L_fa, anthro.com.forearm, anthro.gyration.forearm);
        entries{end+1} = local_make_seg_entry("forearm_" + s_code, m_fa, L_fa, v_fa,...
            I_fa, I_ref_fa, 3);
    end

    % --- Hand (L / R) ---
    L_hand = anthro.length.hand;
    for side = ["L", "R"]
        r_hand = local_find_solid(solids_tbl, "Grip/" + side + "Hand");
        m_hand = r_hand.mass;
        I_hand = r_hand.moments;
        [~, ~, I_ref_hand] = gs3dx_segment_inertia(m_hand, L_hand, anthro.com.hand, anthro.gyration.hand);
        entries{end+1} = local_make_seg_entry("hand_" + side, m_hand, L_hand, "anthro.length.hand",...
            I_hand, I_ref_hand, 3);
    end

    % --- Thigh (L / R) ---
    [L_thigh, v_thigh] = local_ws_lookup(ws, {'ThighLength'}, 1, mdl_str);
    if isempty(L_thigh), L_thigh = anthro.length.thigh; v_thigh = "anthro.length.thigh"; end
    for side = ["L", "R"]
        r_thigh = local_find_solid(solids_tbl, "Lower Body/" + side + " Thigh");
        m_thigh = r_thigh.mass;
        I_thigh = r_thigh.moments;
        [~, ~, I_ref_thigh] = gs3dx_segment_inertia(m_thigh, L_thigh, anthro.com.thigh, anthro.gyration.thigh);
        entries{end+1} = local_make_seg_entry("thigh_" + side, m_thigh, L_thigh, v_thigh,...
            I_thigh, I_ref_thigh, 3);
    end

    % --- Shank (L / R) ---
    [L_shank, v_shank] = local_ws_lookup(ws, {'ShankLength'}, 1, mdl_str);
    if isempty(L_shank), L_shank = anthro.length.shank; v_shank = "anthro.length.shank"; end
    for side = ["L", "R"]
        r_shank = local_find_solid(solids_tbl, "Lower Body/" + side + " Shank");
        m_shank = r_shank.mass;
        I_shank = r_shank.moments;
        [~, ~, I_ref_shank] = gs3dx_segment_inertia(m_shank, L_shank, anthro.com.shank, anthro.gyration.shank);
        entries{end+1} = local_make_seg_entry("shank_" + side, m_shank, L_shank, v_shank,...
            I_shank, I_ref_shank, 3);
    end

    % --- Foot (L / R) ---
    [L_foot, v_foot] = local_ws_lookup(ws, {'FootLength'}, 1, mdl_str);
    if isempty(L_foot), L_foot = anthro.length.foot; v_foot = "anthro.length.foot"; end
    for side = ["L", "R"]
        r_foot = local_find_solid(solids_tbl, "Lower Body/" + side + " Foot");
        m_foot = r_foot.mass;
        I_foot = r_foot.moments;
        [~, ~, I_ref_foot] = gs3dx_segment_inertia(m_foot, L_foot, anthro.com.foot, anthro.gyration.foot);
        % The foot brick's long axis is its largest dimension
        [~, long_foot] = max(local_eval_expr([mdl_str '/' char(r_foot.block)], 'BrickDimensions'));
        entries{end+1} = local_make_seg_entry("foot_" + side, m_foot, L_foot, v_foot,...
            I_foot, I_ref_foot, long_foot);
    end

    % --- Lower Trunk (LowerTorso) ---
    [L_lt, v_lt] = local_ws_lookup(ws, {'FitLowerTorsoLength', 'LowerTorsoLength'}, 0.0254, mdl_str);
    if isempty(L_lt), L_lt = 0.2438; v_lt = "estimated"; end
    r_lt = local_find_solid(solids_tbl, 'Hips and Torso Inputs/LowerTorso');
    m_lt = r_lt.mass;
    I_lt = r_lt.moments;
    [~, ~, I_ref_lt] = gs3dx_segment_inertia(m_lt, L_lt, anthro.com.trunk, anthro.gyration.lower_trunk);
    entries{end+1} = local_make_seg_entry("lower_trunk", m_lt, L_lt, v_lt,...
        I_lt, I_ref_lt, 3);

    % --- Upper Trunk (UpperTorsoBase + UpperTorsoTop) ---
    % Lump base (20%) and top (80%) using parallel-axis theorem along cylinder z-axis
    [L_ut, v_ut] = local_ws_lookup(ws, {'FitUpperTorsoLength', 'UpperTorsoLength'}, 0.0254, mdl_str);
    if isempty(L_ut), L_ut = 0.2438; v_ut = "estimated"; end
    r_utb = local_find_solid(solids_tbl, 'Hips and Torso Inputs/UpperTorsoBase');
    r_utt = local_find_solid(solids_tbl, 'Hips and Torso Inputs/UpperTorsoTop');
    m_ut = r_utb.mass + r_utt.mass;
    d_utb = 0.4 * L_ut;
    d_utt = 0.1 * L_ut;
    I_ut_trans = (r_utb.moments(1) + r_utb.mass * d_utb^2) + (r_utt.moments(1) + r_utt.mass * d_utt^2);
    I_ut_long = r_utb.moments(3) + r_utt.moments(3);
    I_ut = [I_ut_trans, I_ut_trans, I_ut_long];
    [~, ~, I_ref_ut] = gs3dx_segment_inertia(m_ut, L_ut, anthro.com.trunk, anthro.gyration.upper_trunk);
    entries{end+1} = local_make_seg_entry("upper_trunk", m_ut, L_ut, v_ut,...
        I_ut, I_ref_ut, 3);

    % --- Trunk (Lumped Lower + Upper Torso) ---
    % LowerTorso and UpperTorso are joined by spine tilt & twist joints, but if lumped
    % along the continuous 6-inch torso cylinder axis:
    L_trunk = L_lt + L_ut;
    m_trunk = m_lt + m_ut;
    z_com = (m_lt * (0.5 * L_lt) + m_ut * (L_lt + 0.5 * L_ut)) / m_trunk;
    d_lower = z_com - 0.5 * L_lt;
    d_upper = (L_lt + 0.5 * L_ut) - z_com;
    I_trunk_trans = (I_lt(1) + m_lt * d_lower^2) + (I_ut(1) + m_ut * d_upper^2);
    I_trunk_long = I_lt(3) + I_ut(3);
    I_trunk = [I_trunk_trans, I_trunk_trans, I_trunk_long];
    [~, ~, I_ref_trunk] = gs3dx_segment_inertia(m_trunk, L_trunk, anthro.com.trunk, anthro.gyration.trunk);
    entries{end+1} = local_make_seg_entry("trunk", m_trunk, L_trunk, v_lt + "+" + v_ut,...
        I_trunk, I_ref_trunk, 3);

    % Convert entries to table
    n_entries = numel(entries);
    s_names = strings(n_entries, 1);
    m_mods = zeros(n_entries, 1);
    lens = zeros(n_entries, 1);
    l_vars = strings(n_entries, 1);
    I_mods = zeros(n_entries, 3);
    I_refs = zeros(n_entries, 3);
    r_trans = zeros(n_entries, 1);
    r_long = zeros(n_entries, 1);

    for i = 1:n_entries
        e = entries{i};
        s_names(i) = e.segment;
        m_mods(i) = e.mass_model;
        lens(i) = e.length;
        l_vars(i) = e.length_var;
        I_mods(i, :) = e.I_model;
        I_refs(i, :) = e.I_de_leva;
        r_trans(i) = e.ratio_transverse;
        r_long(i) = e.ratio_longitudinal;
    end

    tbl = table(s_names, m_mods, lens, l_vars, I_mods, I_refs, r_trans, r_long, ...
        'VariableNames', {'segment', 'mass_model', 'length', 'length_var', ...
                         'I_model', 'I_de_leva', 'ratio_transverse', 'ratio_longitudinal'});
end

function e = local_make_seg_entry(seg, m_mod, len, len_var, I_mod, I_ref, long_axis)
    % I_REF is de Leva's [sagittal transverse longitudinal]; LONG_AXIS is the
    % model solid's long axis.  Transverse ratio: mean of the model's two
    % other moments over the mean of de Leva's sagittal and transverse.
    trans = setdiff(1:3, long_axis);
    e = struct();
    e.segment = string(seg);
    e.mass_model = m_mod;
    e.length = len;
    e.length_var = string(len_var);
    e.I_model = I_mod;
    e.I_de_leva = I_ref;
    e.ratio_transverse = mean(I_mod(trans)) / mean(I_ref(1:2));
    e.ratio_longitudinal = I_mod(long_axis) / I_ref(3);
end

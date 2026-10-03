function report = gs3dx_apply_subject_physics(mdl, subject_mass, opts)
%GS3DX_APPLY_SUBJECT_PHYSICS  Apply in-memory subject mass and capture-fitted lengths (#10979, #11011).
%
%   REPORT = GS3DX_APPLY_SUBJECT_PHYSICS(MDL, SUBJECT_MASS, OPTS) applies a
%   subject's body mass (kg) and capture-fitted segment lengths to the loaded
%   GS3DX_Human Simscape Multibody model in memory, without writing or saving
%   the SLX file.
%
%   Scope: Strictly GS3DX_Human only. Model must already be loaded in memory.
%   Fails closed on any unexpected topology, unknown fields, or unverified units.
%
%   Preconditions (DbC):
%     - MDL must be loaded and exactly 'GS3DX_Human'.
%     - SUBJECT_MASS must be a positive finite real numeric scalar (kg).
%     - OPTS.fitted_lengths whitelist: FitHubtoSLength, FitUpperArmLength,
%       FitLowerArmLength, FitLowerTorsoLength, FitUpperTorsoLength,
%       ThighLength, ShankLength. All other fields are rejected.
%     - All required mass and geometry workspace variables must already exist.
%     - Solid blocks must have InertiaType Custom, MassUnits 'kg',
%       CenterOfMassUnits 'm', MomentsOfInertiaUnits 'kg*m^2'.
%
%   Postconditions (DbC):
%     - All custom limb, head and rearfoot moments scale consistently.
%     - Transactional rollback attempts to restore workspace variables and
%       block parameters if mutation fails; any rollback failure is explicitly
%       propagated with error ID 'gs3dx:apply_physics:rollback_failed'.
%     - Equipment mass is evaluated directly from unaffected model solids.
%     - Qualification status is explicitly UNQUALIFIED.

    arguments
        mdl {mustBeA(mdl, ["char", "string"])}
        subject_mass double = []
        opts.fitted_lengths = struct()
    end

    mdl_str = char(mdl);
    assert(~isempty(mdl_str), 'gs3dx:apply_physics:invalid_model', 'Model name cannot be empty');

    % 1. Verify model is strictly GS3DX_Human and ALREADY LOADED (no auto_load)
    names = gs3dx_names();
    assert(strcmp(mdl_str, char(names.variants.human)), 'gs3dx:apply_physics:unsupported_model', ...
        'gs3dx_apply_subject_physics is strictly scoped to %s (found "%s")', char(names.variants.human), mdl_str);
    assert(bdIsLoaded(mdl_str), 'gs3dx:apply_physics:model_not_loaded', ...
        'Model "%s" must already be loaded before applying subject physics.', mdl_str);

    ws = get_param(mdl_str, 'ModelWorkspace');

    % 2. Preflight subject mass
    if isempty(subject_mass)
        assert(ws.hasVariable('GolferBodyMass'), 'gs3dx:apply_physics:missing_variable', ...
            'Model workspace lacks GolferBodyMass and no subject_mass was supplied.');
        subject_mass = local_get_ws_val(ws, 'GolferBodyMass');
    end

    assert(isnumeric(subject_mass) && isreal(subject_mass) && isscalar(subject_mass) && ...
           isfinite(subject_mass) && subject_mass > 0, ...
           'gs3dx:apply_physics:invalid_mass', ...
           'subject_mass must be a positive finite real numeric scalar (kg)');
    M_body = double(subject_mass);

    % 3. Preflight and validate fitted lengths against strict WHITELIST
    assert(isstruct(opts.fitted_lengths) && isscalar(opts.fitted_lengths), ...
        'gs3dx:apply_physics:invalid_lengths', 'fitted_lengths must be a scalar struct');

    user_lens = struct();
    if isfield(opts.fitted_lengths, 'vars')
        assert(isstruct(opts.fitted_lengths.vars) && isscalar(opts.fitted_lengths.vars), ...
            'gs3dx:apply_physics:invalid_lengths', 'fitted_lengths.vars must be a scalar struct');
        extra_wrapper_fields = setdiff(fieldnames(opts.fitted_lengths), {'vars'});
        assert(isempty(extra_wrapper_fields), 'gs3dx:apply_physics:unsupported_length_field', ...
            'fitted_lengths wrapper cannot contain extra fields: %s', strjoin(extra_wrapper_fields, ', '));
        user_lens = opts.fitted_lengths.vars;
    elseif ~isempty(fieldnames(opts.fitted_lengths))
        user_lens = opts.fitted_lengths;
    end

    allowed_fields = [ ...
        "FitHubtoSLength", ...
        "FitUpperArmLength", ...
        "FitLowerArmLength", ...
        "FitLowerTorsoLength", ...
        "FitUpperTorsoLength", ...
        "ThighLength", ...
        "ShankLength"];

    % Explicit rejection of FootLength
    if isfield(user_lens, 'FootLength')
        error('gs3dx:apply_physics:unsupported_length_field', ...
            'FootLength is explicitly unsupported: midfoot geometry mount updates are not yet verified.');
    end

    given_fields = fieldnames(user_lens);
    for i = 1:numel(given_fields)
        fn = given_fields{i};
        assert(ismember(string(fn), allowed_fields), ...
            'gs3dx:apply_physics:unsupported_length_field', ...
            'Field "%s" is not an allowed capture length variable. Allowed fields: %s', ...
            fn, strjoin(allowed_fields, ', '));
        val = user_lens.(fn);
        assert(isnumeric(val) && isreal(val) && isscalar(val) && isfinite(val) && val > 0, ...
            'gs3dx:apply_physics:invalid_lengths', ...
            'Length field "%s" must be a positive finite real numeric scalar', fn);
        assert(ws.hasVariable(fn), 'gs3dx:apply_physics:missing_variable', ...
            'Workspace lacks variable "%s" for fitted length', fn);
    end

    % 4. Preflight required workspace variables (no quiet new variables)
    req_mass_vars = [ ...
        "GolferBodyMass", "GolferHeadMass", "GolferNeckMass", ...
        "GolferLowerTrunkMass", "GolferUpperTrunkMass", "GolferShoulderMass", ...
        "GolferUpperArmMass", "GolferForearmMass", "GolferHandMass", ...
        "ThighMass", "ShankMass", "FootMass", "ForefootMass"];
    for v = req_mass_vars
        assert(ws.hasVariable(char(v)), 'gs3dx:apply_physics:missing_variable', ...
            'Required model workspace variable "%s" does not exist', v);
    end

    req_geom_vars = ["FootLength", "FootHeelOffset", "AnkleHeight", "FootWidth", "MidfootOffset"];
    for v = req_geom_vars
        assert(ws.hasVariable(char(v)), 'gs3dx:apply_physics:missing_variable', ...
            'Required foot geometry workspace variable "%s" does not exist', v);
    end

    % 5. Preflight solid blocks: verify existence, Custom inertia, mass expressions, and SI units
    solid_preflight_spec = { ...
        'Lower Body/L Thigh',          'ThighMass'; ...
        'Lower Body/R Thigh',          'ThighMass'; ...
        'Lower Body/L Shank',          'ShankMass'; ...
        'Lower Body/R Shank',          'ShankMass'; ...
        'LUpperArm',                   'GolferUpperArmMass'; ...
        'RUpperArm',                   'GolferUpperArmMass'; ...
        'Left Forearm/LUpperForearm',  '0.5*GolferForearmMass'; ...
        'Left Forearm/LLowerForearm',  '0.5*GolferForearmMass'; ...
        'Right Forearm/RUpperForearm', '0.5*GolferForearmMass'; ...
        'Right Forearm/RLowerForearm', '0.5*GolferForearmMass'; ...
        'Grip/LHand',                  'GolferHandMass'; ...
        'Grip/RHand',                  'GolferHandMass'; ...
        'Hips and Torso Inputs/Head',  'GolferHeadMass'; ...
        'Lower Body/L Foot',           'FootMass - ForefootMass'; ...
        'Lower Body/R Foot',           'FootMass - ForefootMass'; ...
        'Lower Body/L Forefoot',       'ForefootMass'; ...
        'Lower Body/R Forefoot',       'ForefootMass'};

    for i = 1:size(solid_preflight_spec, 1)
        rel = solid_preflight_spec{i, 1};
        exp_expr = solid_preflight_spec{i, 2};
        blk = [mdl_str '/' rel];
        assert(getSimulinkBlockHandle(blk) > 0, ...
            'gs3dx:apply_physics:unsupported_topology', ...
            'Required solid block not found: %s', blk);
        itype = get_param(blk, 'InertiaType');
        assert(strcmp(itype, 'Custom'), ...
            'gs3dx:apply_physics:unsupported_topology', ...
            'Block %s must have InertiaType Custom (found %s)', blk, itype);
        m_expr = get_param(blk, 'Mass');
        norm_expr = regexprep(m_expr, '\s+', '');
        norm_exp = regexprep(exp_expr, '\s+', '');
        assert(strcmp(norm_expr, norm_exp), ...
            'gs3dx:apply_physics:unexpected_solid_expression', ...
            'Block %s has unexpected Mass expression "%s" (expected "%s")', blk, m_expr, exp_expr);
        % Unit verification
        assert(strcmp(get_param(blk, 'MassUnits'), 'kg'), ...
            'gs3dx:apply_physics:unexpected_unit', 'Block %s MassUnits must be kg', blk);
        assert(strcmp(get_param(blk, 'CenterOfMassUnits'), 'm'), ...
            'gs3dx:apply_physics:unexpected_unit', 'Block %s CenterOfMassUnits must be m', blk);
        assert(strcmp(get_param(blk, 'MomentsOfInertiaUnits'), 'kg*m^2'), ...
            'gs3dx:apply_physics:unexpected_unit', 'Block %s MomentsOfInertiaUnits must be kg*m^2', blk);
        assert(strcmp(get_param(blk, 'ProductsOfInertiaUnits'), 'kg*m^2'), ...
            'gs3dx:apply_physics:unexpected_unit', 'Block %s ProductsOfInertiaUnits must be kg*m^2', blk);
    end

    % Verify massive geometry solids exist and compute from geometry
    geom_solids = [ ...
        "Hips and Torso Inputs/Neck", ...
        "Hips and Torso Inputs/LowerTorso", ...
        "Hips and Torso Inputs/UpperTorsoBase", ...
        "Hips and Torso Inputs/UpperTorsoTop", ...
        "HubtoLS", "HubtoRS"];
    for s = geom_solids
        blk = [mdl_str '/' char(s)];
        assert(getSimulinkBlockHandle(blk) > 0, ...
            'gs3dx:apply_physics:unsupported_topology', ...
            'Required geometry solid block not found: %s', blk);
        assert(strcmp(get_param(blk, 'InertiaType'), 'CalculateFromGeometry'), ...
            'gs3dx:apply_physics:unsupported_topology', ...
            'Block %s must have InertiaType CalculateFromGeometry', blk);
    end

    % 6. Resolve actual longitudinal lengths
    in_to_m = 0.0254;
    if isfield(user_lens, 'ThighLength'), L_thigh = double(user_lens.ThighLength);
    else, L_thigh = local_get_ws_val(ws, 'ThighLength'); end

    if isfield(user_lens, 'ShankLength'), L_shank = double(user_lens.ShankLength);
    else, L_shank = local_get_ws_val(ws, 'ShankLength'); end

    if isfield(user_lens, 'FitUpperArmLength'), L_ua = in_to_m * double(user_lens.FitUpperArmLength);
    else, L_ua = in_to_m * local_get_ws_val(ws, 'FitUpperArmLength'); end

    if isfield(user_lens, 'FitLowerArmLength'), L_fa = in_to_m * double(user_lens.FitLowerArmLength);
    else, L_fa = in_to_m * local_get_ws_val(ws, 'FitLowerArmLength'); end

    a = gs3dx_anthropometry(M_body);
    L_struct = struct( ...
        'thigh', L_thigh, 'shank', L_shank, ...
        'upper_arm', L_ua, 'forearm', L_fa, ...
        'hand', a.length.hand, 'head', a.length.head);
    if isfield(user_lens, 'FitLowerTorsoLength')
        L_struct.lower_torso = in_to_m * double(user_lens.FitLowerTorsoLength);
    end
    if isfield(user_lens, 'FitUpperTorsoLength')
        L_struct.upper_torso = in_to_m * double(user_lens.FitUpperTorsoLength);
    end
    if isfield(user_lens, 'FitHubtoSLength')
        L_struct.hub_to_s = in_to_m * double(user_lens.FitHubtoSLength);
    end

    % Compute custom segment inertias via shared catalog
    inert = gs3dx_custom_segment_inertias(a, L_struct);

    % Foot split calculations (gs3dx_build_human local_midfeet geometry)
    mf = local_get_ws_val(ws, 'ForefootMass');
    M_foot = a.legs.FootMass;
    assert(mf < M_foot / 2, 'gs3dx:apply_physics:forefoot_mass_too_large', ...
        'Forefoot mass (%.3f kg) exceeds half the foot mass (%.3f kg)', mf, M_foot);
    mr = M_foot - mf;
    assert(mr > 0, 'gs3dx:apply_physics:invalid_mass', 'Rearfoot mass must be positive');

    L_foot = local_get_ws_val(ws, 'FootLength');
    h_foot = local_get_ws_val(ws, 'FootHeelOffset');
    H_foot = local_get_ws_val(ws, 'AnkleHeight');
    W_foot = local_get_ws_val(ws, 'FootWidth');

    mtp = [(0.73 - h_foot) * L_foot, 0, -H_foot + 0.025];
    toe = (1 - h_foot) * L_foot;
    fore_r = [(toe - mtp(1) + 0.022) / 2, 0.048, (0.015 + H_foot + mtp(3)) / 2];
    fore_c = [-0.01 + fore_r(1), 0, 0.015 - fore_r(3)];
    brick_c = [(0.5 - h_foot) * L_foot, 0, -H_foot / 2];
    d_foot = (mtp + fore_c) - brick_c;

    Ib_rear = mr / 12 * [W_foot^2 + H_foot^2, L_foot^2 + H_foot^2, L_foot^2 + W_foot^2];
    If_fore = mf / 5 * [fore_r(2)^2 + fore_r(3)^2, fore_r(1)^2 + fore_r(3)^2, fore_r(1)^2 + fore_r(2)^2];
    com_rear = -mf / mr * d_foot;

    % 7. Evaluate actual equipment mass from unaffected model solids
    equipment_mass = local_evaluate_equipment_mass(mdl_str);

    % 8. Transactional snapshot with DEEP COPY for Simulink.Parameter containers
    ws_snapshot = containers.Map();
    vars_to_snapshot = unique([cellstr(req_mass_vars(:)); given_fields(:)], 'stable');
    for i = 1:numel(vars_to_snapshot)
        vn = vars_to_snapshot{i};
        if ws.hasVariable(vn)
            val = ws.getVariable(vn);
            if isa(val, 'Simulink.Parameter')
                ws_snapshot(vn) = val.copy(); % DEEP COPY: mutation cannot modify snapshot
            else
                ws_snapshot(vn) = val;
            end
        end
    end

    block_snapshot = containers.Map();
    blocks_to_mutate = solid_preflight_spec(:, 1);
    for i = 1:numel(blocks_to_mutate)
        rel = blocks_to_mutate{i};
        blk = [mdl_str '/' rel];
        block_snapshot(rel) = struct( ...
            'com', get_param(blk, 'CenterOfMass'), ...
            'moments', get_param(blk, 'MomentsOfInertia'), ...
            'products', get_param(blk, 'ProductsOfInertia'));
    end

    % 9. Apply In-Memory Mutations inside guarded transaction
    modified_blocks = strings(0, 1);
    try
        % 9a. Update workspace mass variables preserving containers
        mass_assign_list = { ...
            'GolferBodyMass', a.vars.GolferBodyMass; ...
            'GolferHeadMass', a.vars.GolferHeadMass; ...
            'GolferNeckMass', a.vars.GolferNeckMass; ...
            'GolferLowerTrunkMass', a.vars.GolferLowerTrunkMass; ...
            'GolferUpperTrunkMass', a.vars.GolferUpperTrunkMass; ...
            'GolferShoulderMass', a.vars.GolferShoulderMass; ...
            'GolferUpperArmMass', a.vars.GolferUpperArmMass; ...
            'GolferForearmMass', a.vars.GolferForearmMass; ...
            'GolferHandMass', a.vars.GolferHandMass; ...
            'ThighMass', a.legs.ThighMass; ...
            'ShankMass', a.legs.ShankMass; ...
            'FootMass', a.legs.FootMass};

        for i = 1:size(mass_assign_list, 1)
            local_set_ws_var(ws, mass_assign_list{i, 1}, mass_assign_list{i, 2});
        end

        % 9b. Update whitelisted fitted lengths in workspace
        for i = 1:numel(given_fields)
            fn = given_fields{i};
            local_set_ws_var(ws, fn, user_lens.(fn));
        end

        % 9c. Update custom limb and head blocks via DRY mapping loop
        limb_map = { ...
            'Lower Body/L Thigh',          'thigh_L'; ...
            'Lower Body/R Thigh',          'thigh_R'; ...
            'Lower Body/L Shank',          'shank_L'; ...
            'Lower Body/R Shank',          'shank_R'; ...
            'LUpperArm',                   'upper_arm_L'; ...
            'RUpperArm',                   'upper_arm_R'; ...
            'Left Forearm/LUpperForearm',  'forearm_L_upper'; ...
            'Left Forearm/LLowerForearm',  'forearm_L_lower'; ...
            'Right Forearm/RUpperForearm', 'forearm_R_upper'; ...
            'Right Forearm/RLowerForearm', 'forearm_R_lower'; ...
            'Grip/LHand',                  'hand_L'; ...
            'Grip/RHand',                  'hand_R'; ...
            'Hips and Torso Inputs/Head',  'head'};

        for i = 1:size(limb_map, 1)
            blk = [mdl_str '/' limb_map{i, 1}];
            key = limb_map{i, 2};
            if contains(key, '_upper')
                sub = extractBefore(key, '_upper');
                seg = inert.(sub).upper;
            elseif contains(key, '_lower')
                sub = extractBefore(key, '_lower');
                seg = inert.(sub).lower;
            else
                seg = inert.(key);
            end
            set_param(blk, ...
                'CenterOfMass', mat2str(seg.com, 17), ...
                'MomentsOfInertia', mat2str(seg.moments, 17), ...
                'ProductsOfInertia', '[0 0 0]');
            modified_blocks(end + 1, 1) = string(blk); %#ok<AGROW>
        end

        % 9d. Update split foot blocks
        for side = ["L", "R"]
            blk_foot = [mdl_str '/Lower Body/' char(side) ' Foot'];
            set_param(blk_foot, ...
                'CenterOfMass', mat2str(com_rear, 17), ...
                'MomentsOfInertia', mat2str(Ib_rear, 17), ...
                'ProductsOfInertia', '[0 0 0]');
            modified_blocks(end + 1, 1) = string(blk_foot); %#ok<AGROW>

            blk_fore = [mdl_str '/Lower Body/' char(side) ' Forefoot'];
            set_param(blk_fore, ...
                'CenterOfMass', '[0 0 0]', ...
                'MomentsOfInertia', mat2str(If_fore, 17), ...
                'ProductsOfInertia', '[0 0 0]');
            modified_blocks(end + 1, 1) = string(blk_fore); %#ok<AGROW>
        end

    catch mutation_err
        % Transactional rollback: attempt to restore snapshots on mutation failure
        try
            local_rollback(ws, ws_snapshot, mdl_str, block_snapshot);
        catch rollback_err
            % Both mutation and rollback failed; construct explicit combined exception
            comp_err = MException('gs3dx:apply_physics:rollback_failed', ...
                'Mutation failed (%s) AND subsequent rollback failed: %s', ...
                mutation_err.message, rollback_err.message);
            comp_err = comp_err.addCause(mutation_err);
            comp_err = comp_err.addCause(rollback_err);
            throw(comp_err);
        end
        rethrow(mutation_err);
    end

    % 10. Build Report Output
    report = struct();
    report.status = "applied";
    report.model_name = string(mdl_str);
    report.qualification_status = "UNQUALIFIED";
    report.subject_mass = M_body;
    report.equipment_mass = equipment_mass;
    report.equipment_mass_source = "evaluated_from_model_solids";
    report.expected_total_mass = M_body + equipment_mass;

    applied_masses = a.vars;
    applied_masses.ThighMass = a.legs.ThighMass;
    applied_masses.ShankMass = a.legs.ShankMass;
    applied_masses.FootMass = a.legs.FootMass;
    applied_masses.ForefootMass = mf;
    report.applied_masses = applied_masses;

    report.applied_inertias = inert;
    report.applied_inertias.foot_rear = struct('mass', mr, 'com', com_rear, 'moments', Ib_rear);
    report.applied_inertias.foot_fore = struct('mass', mf, 'com', [0 0 0], 'moments', If_fore);

    report.fitted_lengths = L_struct;
    report.modified_blocks = modified_blocks;

    report.remaining_assumptions = [ ...
        "Head and neck mass split (85% head, 15% neck) is a structural modeling assumption; not clinically measured.", ...
        "Torso segments (LowerTorso, UpperTorsoBase, UpperTorsoTop, HubtoLS, HubtoRS) and neck use baseline calculated-geometry cylinders; transverse dimensions are not newly measured.", ...
        "Foot dimensions (width, height, heel offset) and forefoot mass (fixed 0.25 kg structural constant) are baseline styling assumptions; transverse foot dimensions are not newly measured.", ...
        "In-memory parameter update achieves numerical parameter consistency (mass, inertia, COM) only. Physical dynamic equivalence is UNQUALIFIED until native forward-dynamics equilibrium and gate acceptance are established.", ...
        "Center-of-mass balance was NOT re-anchored and active feedback was NOT enabled."];

    report.notes = "In-memory numerical parameter consistency applied. Forward dynamics remain UNQUALIFIED until native equilibrium and full gate checks are run. No COM balance re-anchoring or automatic feedback.";
end

% -------------------------------------------------------------------------
% Helper: Read workspace variable scalar double (supports Simulink.Parameter)
% -------------------------------------------------------------------------
function val = local_get_ws_val(ws, name)
    v = ws.getVariable(char(name));
    if isa(v, 'Simulink.Parameter')
        val = double(v.Value);
    elseif isstruct(v) && isfield(v, 'Value')
        val = double(v.Value);
    else
        val = double(v);
    end
    assert(isnumeric(val) && isreal(val) && isscalar(val) && isfinite(val) && val > 0, ...
        'gs3dx:apply_physics:invalid_variable', ...
        'Workspace variable "%s" must be a positive finite real scalar', char(name));
end

% -------------------------------------------------------------------------
% Helper: Assign workspace variable preserving Simulink.Parameter container & metadata
% -------------------------------------------------------------------------
function local_set_ws_var(ws, name, new_val)
    var_name = char(name);
    if ws.hasVariable(var_name)
        v = ws.getVariable(var_name);
        if isa(v, 'Simulink.Parameter')
            v_new = v.copy();
            v_new.Value = new_val;
            assignin(ws, var_name, v_new);
            return;
        end
    end
    assignin(ws, var_name, new_val);
end

% -------------------------------------------------------------------------
% Helper: Evaluate actual equipment mass from unaffected model solids
% -------------------------------------------------------------------------
function eq_mass = local_evaluate_equipment_mass(mdl)
% Reuse the audited InertiaType/geometry/unit policy, including Custom visuals.
    audit = gs3dx_inertia_audit(mdl);
    solids = audit.solids;
    equipment = startsWith(solids.block, ["Club/", "Grip/"]) ...
        & ~endsWith(solids.block, ["/LHand", "/RHand"]);
    contact = startsWith(solids.block, "Lower Body/") & endsWith(solids.block, " Sphere");
    masses = solids.mass(equipment | contact);
    assert(~isempty(masses) && isreal(masses) && all(isfinite(masses)) && all(masses >= 0), ...
        'gs3dx:apply_physics:invalid_equipment_mass', 'Equipment masses are not fully evaluated');
    eq_mass = sum(masses);
end
% -------------------------------------------------------------------------
% Helper: Transactional rollback on mutation failure
% -------------------------------------------------------------------------
function local_rollback(ws, ws_snapshot, mdl_str, block_snapshot)
    keys_ws = ws_snapshot.keys();
    for i = 1:numel(keys_ws)
        k = keys_ws{i};
        assignin(ws, k, ws_snapshot(k));
    end

    keys_blk = block_snapshot.keys();
    for i = 1:numel(keys_blk)
        k = keys_blk{i};
        blk = [mdl_str '/' k];
        assert(getSimulinkBlockHandle(blk) > 0, ...
            'gs3dx:apply_physics:rollback_missing_block', ...
            'Block %s disappeared during rollback', blk);
        snap = block_snapshot(k);
        set_param(blk, ...
            'CenterOfMass', snap.com, ...
            'MomentsOfInertia', snap.moments, ...
            'ProductsOfInertia', snap.products);
    end
end

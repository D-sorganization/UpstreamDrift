function report = gs3dx_fit_human_visuals(mdl, personal_lengths)
%GS3DX_FIT_HUMAN_VISUALS  Adapt GS3DX_Human visual ellipsoids to subject segment lengths (#10979, #11161).
%
%   REPORT = GS3DX_FIT_HUMAN_VISUALS(MDL, PERSONAL_LENGTHS) dynamically scales
%   the visual ellipsoids and rigid placement transforms of the loaded GS3DX_Human
%   Simscape Multibody model in memory to match the fitted segment lengths from
%   GS3DX_FIT_LENGTHS, preserving mass, center of mass, inertia, and joint
%   frames.
%
%   The saved GS3DX_Human model has visual ellipsoid radii and placement offsets
%   evaluated at the baseline tour golfer's dimensions. Injecting workspace
%   length variables alone updates only the kinematic cylinder lengths and
%   joint placement transforms, leaving visual ellipsoids unadapted.
%   This adapter reads the model's baseline length variables BEFORE injection,
%   computes longitudinal scaling ratios, and adapts the visual geometry.
%
%   Rules:
%   1. Massless visuals (Mass == 0, InertiaType == 'Custom') are scaled
%      longitudinally along their segment axes (upper arms, forearm halves,
%      shoulders, torso segments) via GS3DX_SCALE_VECTOR. Transverse radii
%      remain untouched.
%   2. Nonzero-mass solids (head, thighs, shanks) are checked to verify they
%      already possess InertiaType == 'Custom'. Their longitudinal visual
%      radii are updated to match the new segment lengths via GS3DX_SCALE_VECTOR,
%      but their Mass, CenterOfMass, MomentsOfInertia, and ProductsOfInertia are
%      PRESERVED identically.
%   3. Torso visual proportions, head dimensions, and shoe widths reflect fixed
%      artistic styling and are explicitly documented as such; no complete
%      anthropometric calibration is claimed.
%   4. Any missing block, unexpected block type, non-Custom inertia on
%      nonzero-mass solids, or non-numeric/non-positive length FAILS CLOSED immediately.
%   5. Once visuals are updated, personal_lengths.vars are injected into the
%      model workspace.
%
%   REPORT fields:
%     .status            "adapted"
%     .scales            struct of longitudinal scaling factors per segment
%     .baseline_lengths  struct of baseline lengths read from model workspace
%     .fitted_lengths    personal_lengths.vars injected into model workspace
%     .adapted_blocks    string array of all modified block paths
%     .visual_geometry   struct of scaled visual ellipsoid radii and offsets
%     .notes             string documenting artistic styling & inertia preservation

    arguments
        mdl (1,:) char
        personal_lengths (1,1) struct
    end

    % 1. Fail-closed preconditions on inputs
    assert(~isempty(mdl), 'gs3dx:fit_human_visuals:ModelNotLoaded', 'Model name cannot be empty');
    assert(isfield(personal_lengths, 'vars') && isstruct(personal_lengths.vars), ...
        'gs3dx:fit_human_visuals:InvalidLengths', 'personal_lengths must contain .vars struct');

    req_fields = ["FitHubtoSLength", "FitUpperArmLength", "FitLowerArmLength", ...
        "FitLowerTorsoLength", "FitUpperTorsoLength", "ThighLength", "ShankLength"];
    for req = req_fields
        assert(isfield(personal_lengths.vars, req), 'gs3dx:fit_human_visuals:MissingField', ...
            'personal_lengths.vars lacks required field: %s', req);
        v_val = personal_lengths.vars.(req);
        assert(isnumeric(v_val) && isreal(v_val) && isscalar(v_val) && isfinite(v_val) && v_val > 0, ...
            'gs3dx:fit_human_visuals:InvalidLengths', ...
            'Field %s must be a positive finite real numeric scalar', req);
    end

    assert(bdIsLoaded(mdl), 'gs3dx:fit_human_visuals:ModelNotLoaded', ...
        'Model %s must be loaded in memory before adapting visuals', mdl);

    % 2. Read baseline lengths from ModelWorkspace BEFORE any injection
    ws = get_param(mdl, 'ModelWorkspace');
    base = struct();
    base.FitHubtoSLength = local_get_base_var(ws, ["FitHubtoSLength", "HubtoSLength"]);
    base.FitUpperArmLength = local_get_base_var(ws, ["FitUpperArmLength", "UpperArmLength"]);
    base.FitLowerArmLength = local_get_base_var(ws, ["FitLowerArmLength", "LowerArmLength"]);
    base.FitLowerTorsoLength = local_get_base_var(ws, ["FitLowerTorsoLength", "LowerTorsoLength"]);
    base.FitUpperTorsoLength = local_get_base_var(ws, ["FitUpperTorsoLength", "UpperTorsoLength"]);
    base.ThighLength = local_get_base_var(ws, "ThighLength");
    base.ShankLength = local_get_base_var(ws, "ShankLength");

    % 3. Compute longitudinal scale factors
    fit_v = personal_lengths.vars;
    scales = struct();
    scales.shoulder = fit_v.FitHubtoSLength / base.FitHubtoSLength;
    scales.upper_arm = fit_v.FitUpperArmLength / base.FitUpperArmLength;
    scales.lower_arm = fit_v.FitLowerArmLength / base.FitLowerArmLength;
    scales.lower_torso = fit_v.FitLowerTorsoLength / base.FitLowerTorsoLength;
    scales.upper_torso = fit_v.FitUpperTorsoLength / base.FitUpperTorsoLength;
    scales.thigh = fit_v.ThighLength / base.ThighLength;
    scales.shank = fit_v.ShankLength / base.ShankLength;

    adapted_blocks = string.empty;
    visual_geom = struct();

    % ---------------------------------------------------------------------
    % 4. Adapt Massless Arm and Shoulder Visuals (Z longitudinal)
    % ---------------------------------------------------------------------
    arm_visuals = { ...
        [mdl '/L Upper Arm'], scales.upper_arm, "L_Upper_Arm"; ...
        [mdl '/R Upper Arm'], scales.upper_arm, "R_Upper_Arm"; ...
        [mdl '/Left Forearm/L Forearm Upper'], scales.lower_arm, "L_Forearm_Upper"; ...
        [mdl '/Left Forearm/L Forearm Lower'], scales.lower_arm, "L_Forearm_Lower"; ...
        [mdl '/Right Forearm/R Forearm Upper'], scales.lower_arm, "R_Forearm_Upper"; ...
        [mdl '/Right Forearm/R Forearm Lower'], scales.lower_arm, "R_Forearm_Lower"; ...
        [mdl '/L Shoulder'], scales.shoulder, "L_Shoulder"; ...
        [mdl '/R Shoulder'], scales.shoulder, "R_Shoulder"};

    for i = 1:size(arm_visuals, 1)
        blk = arm_visuals{i, 1};
        s_factor = arm_visuals{i, 2};
        key = arm_visuals{i, 3};

        local_verify_massless_solid(blk);
        new_radii = local_scale_ellipsoid_radii(blk, s_factor, 3);
        visual_geom.(key) = new_radii;
        adapted_blocks(end+1) = string(blk); %#ok<AGROW>
    end

    % ---------------------------------------------------------------------
    % 5. Adapt Torso Visuals and Rigid Transform Offsets
    % ---------------------------------------------------------------------
    t_sys = [mdl '/Hips and Torso Inputs/'];

    % Pelvis (Lower Torso)
    pelvis_blk = [t_sys 'Pelvis'];
    pelvis_rt = [t_sys 'Pelvis Place'];
    local_verify_massless_solid(pelvis_blk);
    visual_geom.Pelvis = local_scale_ellipsoid_radii(pelvis_blk, scales.lower_torso, 3);
    visual_geom.Pelvis_Offset = local_scale_transform_offset(pelvis_rt, scales.lower_torso, 3);
    adapted_blocks(end+1) = string(pelvis_blk);
    adapted_blocks(end+1) = string(pelvis_rt);

    % Abdomen (Lower Torso)
    abdomen_blk = [t_sys 'Abdomen'];
    abdomen_rt = [t_sys 'Abdomen Place'];
    local_verify_massless_solid(abdomen_blk);
    visual_geom.Abdomen = local_scale_ellipsoid_radii(abdomen_blk, scales.lower_torso, 3);
    visual_geom.Abdomen_Offset = local_scale_transform_offset(abdomen_rt, scales.lower_torso, 3);
    adapted_blocks(end+1) = string(abdomen_blk);
    adapted_blocks(end+1) = string(abdomen_rt);

    % Chest (Upper Torso)
    chest_blk = [t_sys 'Chest'];
    chest_rt = [t_sys 'Chest Place'];
    local_verify_massless_solid(chest_blk);
    visual_geom.Chest = local_scale_ellipsoid_radii(chest_blk, scales.upper_torso, 3);
    visual_geom.Chest_Offset = local_scale_transform_offset(chest_rt, scales.upper_torso, 3);
    adapted_blocks(end+1) = string(chest_blk);
    adapted_blocks(end+1) = string(chest_rt);

    % Trapezius (Upper Torso)
    trap_blk = [t_sys 'Trapezius'];
    trap_rt = [t_sys 'Trapezius Place'];
    local_verify_massless_solid(trap_blk);
    visual_geom.Trapezius = local_scale_ellipsoid_radii(trap_blk, scales.upper_torso, 3);
    visual_geom.Trapezius_Offset = local_scale_transform_offset(trap_rt, scales.upper_torso, 3);
    adapted_blocks(end+1) = string(trap_blk);
    adapted_blocks(end+1) = string(trap_rt);

    % ---------------------------------------------------------------------
    % 6. Adapt Nonzero-Mass Thigh & Shank Solids (Custom Inertia Preserved)
    % ---------------------------------------------------------------------
    leg_solids = { ...
        [mdl '/Lower Body/L Thigh'], scales.thigh, "L_Thigh"; ...
        [mdl '/Lower Body/R Thigh'], scales.thigh, "R_Thigh"; ...
        [mdl '/Lower Body/L Shank'], scales.shank, "L_Shank"; ...
        [mdl '/Lower Body/R Shank'], scales.shank, "R_Shank"};

    for i = 1:size(leg_solids, 1)
        blk = leg_solids{i, 1};
        s_factor = leg_solids{i, 2};
        key = leg_solids{i, 3};

        local_verify_custom_inertia_solid(blk);
        new_radii = local_scale_ellipsoid_radii(blk, s_factor, 3);
        visual_geom.(key) = new_radii;
        adapted_blocks(end+1) = string(blk); %#ok<AGROW>
    end

    % Verify Head (Custom Inertia Preserved, Unmodified)
    head_blk = [t_sys 'Head'];
    local_verify_custom_inertia_solid(head_blk);
    visual_geom.Head = slResolve(get_param(head_blk, 'EllipsoidRadii'), head_blk);

    % ---------------------------------------------------------------------
    % 7. Inject personal fitted lengths into ModelWorkspace
    % ---------------------------------------------------------------------
    for f = fieldnames(fit_v).'
        assignin(ws, f{1}, fit_v.(f{1}));
    end

    % 8. Assemble Report
    report = struct();
    report.status = "adapted";
    report.scales = scales;
    report.baseline_lengths = base;
    report.fitted_lengths = fit_v;
    report.adapted_blocks = adapted_blocks;
    report.visual_geometry = visual_geom;
    report.notes = "Torso proportions, head dimensions and shoe widths use artistic styling; not complete anthropometric calibration. Visual edits preserve mass and Custom inertia parameters and joint-frame expressions. Fitted workspace lengths intentionally change evaluated segment geometry; dynamic equivalence remains unqualified.";
end

% -------------------------------------------------------------------------
% Helper: Read baseline variable safely from model workspace (supports Simulink.Parameter)
% -------------------------------------------------------------------------
function val = local_get_base_var(ws, candidates)
    for c = string(candidates)
        var_name = char(c);
        try
            has_var = ws.hasVariable(var_name);
        catch
            has_var = false;
        end
        if ~has_var
            continue;
        end

        v = ws.getVariable(var_name);
        if isa(v, 'Simulink.Parameter')
            raw = v.Value;
        elseif isstruct(v) && isfield(v, 'Value')
            raw = v.Value;
        else
            raw = v;
        end

        assert(isnumeric(raw) && isreal(raw) && isscalar(raw) && isfinite(raw) && raw > 0, ...
            'gs3dx:fit_human_visuals:InvalidBaselineVariable', ...
            'Baseline variable "%s" must be a positive finite real numeric scalar', var_name);
        val = double(raw);
        return;
    end
    error('gs3dx:fit_human_visuals:MissingBaselineVariable', ...
        'None of baseline variables [%s] found in model workspace', strjoin(candidates, ', '));
end

% -------------------------------------------------------------------------
% Helper: Verify block exists, is Ellipsoidal Solid, Custom inertia, Mass == 0
% -------------------------------------------------------------------------
function local_verify_massless_solid(blk)
    assert(getSimulinkBlockHandle(blk) > 0, 'gs3dx:fit_human_visuals:MissingBlock', ...
        'Required visual block not found: %s', blk);
    ref = get_param(blk, 'ReferenceBlock');
    assert(strcmp(ref, 'sm_lib/Body Elements/Ellipsoidal Solid'), 'gs3dx:fit_human_visuals:InvalidType', ...
        'Block %s is %s, expected Ellipsoidal Solid', blk, ref);
    it = get_param(blk, 'InertiaType');
    assert(strcmp(it, 'Custom'), 'gs3dx:fit_human_visuals:InertiaNotCustom', ...
        'Block %s InertiaType is %s, expected Custom', blk, it);
    m = slResolve(get_param(blk, 'Mass'), blk);
    assert(abs(double(m)) < 1e-9, 'gs3dx:fit_human_visuals:UnexpectedMass', ...
        'Block %s is expected to be massless visual, found Mass = %g', blk, double(m));
end

% -------------------------------------------------------------------------
% Helper: Verify block exists, is Ellipsoidal Solid, Custom inertia (Nonzero Mass)
% -------------------------------------------------------------------------
function local_verify_custom_inertia_solid(blk)
    assert(getSimulinkBlockHandle(blk) > 0, 'gs3dx:fit_human_visuals:MissingBlock', ...
        'Required solid block not found: %s', blk);
    ref = get_param(blk, 'ReferenceBlock');
    assert(strcmp(ref, 'sm_lib/Body Elements/Ellipsoidal Solid'), 'gs3dx:fit_human_visuals:InvalidType', ...
        'Block %s is %s, expected Ellipsoidal Solid', blk, ref);
    it = get_param(blk, 'InertiaType');
    assert(strcmp(it, 'Custom'), 'gs3dx:fit_human_visuals:InertiaNotCustom', ...
        'Block %s InertiaType is %s, expected Custom for nonzero-mass solid', blk, it);
    m = slResolve(get_param(blk, 'Mass'), blk);
    assert(double(m) > 0, 'gs3dx:fit_human_visuals:InvalidMass', ...
        'Block %s expected nonzero mass, found %g', blk, double(m));
end

% -------------------------------------------------------------------------
% Helper: Scale longitudinal axis of an Ellipsoid Solid in memory via gs3dx_scale_vector
% -------------------------------------------------------------------------
function new_radii = local_scale_ellipsoid_radii(blk, scale_factor, axis_idx)
    r_val = slResolve(get_param(blk, 'EllipsoidRadii'), blk);
    r_u = get_param(blk, 'EllipsoidRadiiUnits');
    assert(strcmp(r_u, 'm'), 'gs3dx:fit_human_visuals:InvalidUnits', ...
        'Block %s radii units are %s, expected m', blk, r_u);

    new_radii = gs3dx_scale_vector(r_val, scale_factor, axis_idx);
    set_param(blk, 'EllipsoidRadii', mat2str(new_radii, 6));
end

% -------------------------------------------------------------------------
% Helper: Scale translation offset of a Rigid Transform block in memory via gs3dx_scale_vector
% -------------------------------------------------------------------------
function new_offset = local_scale_transform_offset(blk, scale_factor, axis_idx)
    assert(getSimulinkBlockHandle(blk) > 0, 'gs3dx:fit_human_visuals:MissingBlock', ...
        'Required transform block not found: %s', blk);
    ref = regexprep(strtrim(get_param(blk, 'ReferenceBlock')), '\s+', ' ');
    assert(strcmp(ref, 'sm_lib/Frames and Transforms/Rigid Transform'), 'gs3dx:fit_human_visuals:InvalidType', ...
        'Block %s is %s, expected Rigid Transform', blk, ref);
    tm = get_param(blk, 'TranslationMethod');
    assert(strcmp(tm, 'Cartesian'), 'gs3dx:fit_human_visuals:InvalidTranslationMethod', ...
        'Block %s TranslationMethod is %s, expected Cartesian', blk, tm);
    tu = get_param(blk, 'TranslationCartesianOffsetUnits');
    assert(strcmp(tu, 'm'), 'gs3dx:fit_human_visuals:InvalidUnits', ...
        'Block %s offset units are %s, expected m', blk, tu);
    off_val = slResolve(get_param(blk, 'TranslationCartesianOffset'), blk);

    new_offset = gs3dx_scale_vector(off_val, scale_factor, axis_idx);
    set_param(blk, 'TranslationCartesianOffset', mat2str(new_offset, 6));
end

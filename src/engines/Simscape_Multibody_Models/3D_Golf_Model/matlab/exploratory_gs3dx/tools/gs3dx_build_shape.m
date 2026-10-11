function report = gs3dx_build_shape(info, ref, opts)
%GS3DX_BUILD_SHAPE  Build GS3DX_Shape: GS3DX_FitBalance with de Leva segment inertia and ellipsoid limbs (#10979).
%
%   REPORT = GS3DX_BUILD_SHAPE(INFO, REF) copies GS3DX_FitBalance to
%   GS3DX_Shape.  REF is the leg reference GS3DX_FitBalance was built from.
%
%   * Inertia.  The limb and head solids keep their masses and frames but
%     take InertiaType Custom: principal moments from de Leva's radii of
%     gyration (GS3DX_SEGMENT_INERTIA), with the sagittal and transverse
%     moments averaged so each segment is axisymmetric about its z axis, and
%     for the thighs, shanks and upper arms the centre of mass at de Leva's
%     fraction of the segment from its proximal joint.  Every limb solid has
%     its joints on its z axis, the proximal one at +z, which was measured
%     on GS3DX_FitBalance (docs/SHAPE.md).  The two forearm halves take half
%     the forearm each, with centres of mass a half forearm apart so the
%     pair lumps to de Leva's forearm.  The hands and head keep their
%     centres of mass (a hand's de Leva centre is about 3 cm further from
%     the wrist: 0.2 mm of whole-body centre of mass).  The trunk and feet
%     are left as they are (docs/INERTIA.md).
%   * Graphics.  The thighs, shanks, hands and head, whose solids carry no
%     frames other than R, become Ellipsoidal Solids with the same name,
%     connection, mass and colour.  Frames are unchanged, so the kinematics
%     are those of GS3DX_FitBalance.  An extra visual-only solid compiles to
%     7 blocks (2026-09-28), so no solid is added.
%   * Balance.  The centre-of-mass reference is rewritten by
%     GS3DX_BALANCE_REFERENCE: COM_OFFSET (from a GS3DX_Shape run, because the
%     centre of mass has moved) or COM_REF.  Without either, BalanceOn is 0.
%
%   Options: overwrite (false), com_offset ([]), com_ref ([]).
%   REPORT fields: .segments (table: solid, mass, com, moments), .on,
%   .budget.

    arguments
        info (1,1) struct
        ref (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.com_offset double = []
        opts.com_ref double = []
    end
    assert(isempty(opts.com_offset) || isempty(opts.com_ref), 'gs3dx:shape', 'Give COM_OFFSET or COM_REF, not both');
    names = gs3dx_names();
    src = char(names.variants.fit_balance);
    mdl = char(names.variants.shape);
    gs3dx_copy_models({fullfile(info.models_dir, [src '.slx']), fullfile(info.models_dir, [mdl '.slx'])}, ...
        opts.overwrite, 'gs3dx:shape');

    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    ws = get_param(mdl, 'ModelWorkspace');
    report.on = gs3dx_balance_reference(ws, ref, opts.com_offset, opts.com_ref, 'gs3dx:shape');

    segs = local_segments(mdl);
    for k = 1:numel(segs)
        s = segs(k);
        blk = [mdl '/' char(s.solid)];
        if s.ellipsoid
            blk = local_swap_to_ellipsoid(blk, s.radii);
        end
        set_param(blk, 'InertiaType', 'Custom', ...
            'CenterOfMass', mat2str(s.com, 17), 'CenterOfMassUnits', 'm', ...
            'MomentsOfInertia', mat2str(s.moments, 17), 'MomentsOfInertiaUnits', 'kg*m^2', ...
            'ProductsOfInertia', '[0 0 0]', 'ProductsOfInertiaUnits', 'kg*m^2');
    end
    report.segments = struct2table(rmfield(segs, {'radii', 'ellipsoid'}));

    report.budget = gs3dx_block_budget(mdl, compiled=true);
    reserve = 25;   % as GS3DX_BUILD_FIT_BALANCE: room for GS3DX_CONTACT_CHECK
    assert(report.budget.compiled_total <= names.license_block_limit - reserve, 'gs3dx:shape', ...
        '%s compiles to %d blocks; %d leave no room for %d validation blocks', ...
        mdl, report.budget.compiled_total, names.license_block_limit, reserve);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
end

function segs = local_segments(mdl)
% One entry per solid: its de Leva centre of mass and moments in the solid frame.
    a = gs3dx_anthropometry(local_eval(mdl, 'GolferBodyMass'));
    in = 0.0254;
    L = struct('thigh', local_eval(mdl, 'ThighLength'), 'shank', local_eval(mdl, 'ShankLength'), ...
        'upper_arm', in * local_eval(mdl, 'FitUpperArmLength'), ...
        'forearm', in * local_eval(mdl, 'FitLowerArmLength'), ...
        'hand', a.length.hand, 'head', a.length.head);

    % Read actual evaluated block masses to preserve exact model behavior
    m_actual = struct();
    for side = ["L", "R"]
        m_actual.("thigh_" + side) = local_eval(mdl, get_param([mdl '/Lower Body/' char(side) ' Thigh'], 'Mass'));
        m_actual.("shank_" + side) = local_eval(mdl, get_param([mdl '/Lower Body/' char(side) ' Shank'], 'Mass'));
        m_actual.("upper_arm_" + side) = local_eval(mdl, get_param([mdl '/' char(side) 'UpperArm'], 'Mass'));
        arm = "Left"; if side == "R", arm = "Right"; end
        m_actual.("forearm_" + side) = 2 * local_eval(mdl, get_param([mdl '/' char(arm) ' Forearm/' char(side) 'UpperForearm'], 'Mass'));
        m_actual.("hand_" + side) = local_eval(mdl, get_param([mdl '/Grip/' char(side) 'Hand'], 'Mass'));
    end
    m_actual.head = local_eval(mdl, get_param([mdl '/Hips and Torso Inputs/Head'], 'Mass'));

    inert = gs3dx_custom_segment_inertias(m_actual, L, a);
    segs = struct('solid', {}, 'mass', {}, 'com', {}, 'moments', {}, 'radii', {}, 'ellipsoid', {});
    for side = ["L", "R"]
        for key = ["thigh", "shank"]
            solid = "Lower Body/" + side + " " + upper(extractBefore(key, 2)) + extractAfter(key, 1);
            sk = key + "_" + side;
            segs(end + 1) = struct('solid', solid, 'mass', inert.(sk).mass, ...
                'com', inert.(sk).com, 'moments', inert.(sk).moments, ...
                'radii', local_limb_radii(mdl, solid, L.(key)), 'ellipsoid', true); %#ok<AGROW>
        end
        solid = side + "UpperArm";
        sk = "upper_arm_" + side;
        segs(end + 1) = struct('solid', solid, 'mass', inert.(sk).mass, ...
            'com', inert.(sk).com, 'moments', inert.(sk).moments, ...
            'radii', [], 'ellipsoid', false); %#ok<AGROW>
        arm = "Left"; if side == "R", arm = "Right"; end
        fk = "forearm_" + side;
        segs(end + 1) = struct('solid', arm + " Forearm/" + side + "UpperForearm", ...
            'mass', inert.(fk).upper.mass, 'com', inert.(fk).upper.com, ...
            'moments', inert.(fk).upper.moments, 'radii', [], 'ellipsoid', false); %#ok<AGROW>
        segs(end + 1) = struct('solid', arm + " Forearm/" + side + "LowerForearm", ...
            'mass', inert.(fk).lower.mass, 'com', inert.(fk).lower.com, ...
            'moments', inert.(fk).lower.moments, 'radii', [], 'ellipsoid', false); %#ok<AGROW>
        hand = "Grip/" + side + "Hand";
        hk = "hand_" + side;
        segs(end + 1) = struct('solid', hand, 'mass', inert.(hk).mass, ...
            'com', inert.(hk).com, 'moments', inert.(hk).moments, ...
            'radii', [0.8 0.65 1.25] * local_radius(mdl, hand), 'ellipsoid', true); %#ok<AGROW>
    end
    head = "Hips and Torso Inputs/Head";
    segs(end + 1) = struct('solid', head, 'mass', inert.head.mass, ...
        'com', inert.head.com, 'moments', inert.head.moments, ...
        'radii', [0.7 0.7 0.9] * local_radius(mdl, head), 'ellipsoid', true);
end

function r = local_limb_radii(mdl, solid, len)
% Ellipsoid through the cylinder's radius, reaching 5 % past each joint.
    blk = [mdl '/' char(solid)];
    rc = local_eval(mdl, get_param(blk, 'CylinderRadius')) * local_metres(get_param(blk, 'CylinderRadiusUnits'));
    r = [rc rc 0.55 * len];
end

function blk = local_swap_to_ellipsoid(blk, radii)
% Replace a single-port Solid with an Ellipsoidal Solid of the same name,
% position, connections (every port on its net, each rejoined through the
% new solid) and writable dialog parameters.
    sys = get_param(blk, 'Parent');
    name = get_param(blk, 'Name');
    pos = get_param(blk, 'Position');
    ph = get_param(blk, 'PortHandles');
    ports = [ph.LConn ph.RConn];
    assert(isscalar(ports), 'gs3dx:shape', '%s has %d frame ports; only R-only solids are swapped', blk, numel(ports));
    % PortConnectivity lists every port on the net; walking a branched
    % physical line's Src/Dst handles does not (Grip/RHand: 1 of 8).
    pc = get_param(blk, 'PortConnectivity');
    others = setdiff(pc.DstPort, [ports -1]);
    assert(~isempty(others), 'gs3dx:shape', '%s is not connected', blk);
    % A port with a line takes no second one, so clear the whole net first.
    for h = [ports others(:)']
        l = get_param(h, 'Line');
        if l ~= -1 && ishandle(l), delete_line(l); end
    end
    old = get_param(blk, 'DialogParameters');
    keep = {};
    tmp = add_block('sm_lib/Body Elements/Ellipsoidal Solid', [sys '/ShapeTmp'], 'MakeNameUnique', 'on');
    new = get_param(tmp, 'DialogParameters');
    for f = intersect(fieldnames(old), fieldnames(new))'
        if ~strcmp(f{1}, 'InertiaType') && ~any(strcmp(new.(f{1}).Attributes, 'read-only'))
            keep(end + 1:end + 2) = {f{1}, get_param(blk, f{1})}; %#ok<AGROW>
        end
    end
    delete_block(blk);
    set_param(tmp, 'Name', name, 'Position', pos);
    blk = [sys '/' name];
    set_param(blk, keep{:});
    set_param(blk, 'EllipsoidRadii', mat2str(radii, 17), 'EllipsoidRadiiUnits', 'm');
    nh = get_param(blk, 'PortHandles');
    np = [nh.LConn nh.RConn];
    for o = others(:)'
        add_line(sys, np, o, 'autorouting', 'on');
    end
end

function r = local_radius(mdl, solid)
    blk = [mdl '/' char(solid)];
    r = local_eval(mdl, get_param(blk, 'SphereRadius')) * local_metres(get_param(blk, 'SphereRadiusUnits'));
end

function f = local_metres(unit)
    switch unit
        case 'm', f = 1;
        case 'cm', f = 0.01;
        case 'in', f = 0.0254;
        otherwise, error('gs3dx:shape', 'Unsupported length unit %s', unit);
    end
end

function v = local_eval(mdl, expr)
    v = double(slResolve(char(expr), mdl));
end

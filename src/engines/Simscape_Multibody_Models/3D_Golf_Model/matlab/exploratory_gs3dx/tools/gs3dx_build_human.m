function report = gs3dx_build_human(info, opts)
%GS3DX_BUILD_HUMAN  Build GS3DX_Human: GS3DX_Neck with a human body shape and jointed feet (#10979).
%
%   REPORT = GS3DX_BUILD_HUMAN(INFO) copies GS3DX_Neck to GS3DX_Human and
%   changes how the golfer looks and how its feet bend (docs/HUMAN.md):
%
%   * Shape.  The trunk, neck, shoulder bars, arms and hand standoffs are
%     cylinders 30 cm deep in the trunk, and a massless bar across the
%     pelvis sits 33 deg off the hip line, so the hips read as open.  Each
%     is hidden (GraphicType 'None'; its mass and inertia stay) and a
%     massless ellipsoid is drawn in its place, the pelvis and abdomen
%     along the hip line, the chest along the shoulder line.  Ellipsoidal
%     Solid accepts no frames on its surface, so the cylinders that carry
%     frames stay as the mass carriers.  The head is made rounder
%     (graphics only: its inertia is Custom).
%   * Head.  GS3DX_Neck's neck reference is zero at address, so the neck
%     carries on along the upper trunk's axis (59 deg from vertical at
%     address) and the head centre sits 0.1 m below the capture's head
%     markers.  NECK_ADDRESS turns the neck at address to aim it at their
%     centroid: NeckReference becomes the X and Y angles of
%     Rx(x) Ry(y) Rx(ax) Ry(ay) for the stored angles x, y.
%   * Club.  At address the face pointed 20.7 deg right of the target,
%     from the grip roll.  FaceSquareRoll (deg, model workspace) rolls the
%     club about the shaft in GripStrength's transform; the head is a
%     sphere centred on the shaft, so the roll moves no mass.  The sphere
%     and the face-plane rod are hidden, and a driver-shaped head and a
%     face-normal pointer are drawn instead.
%   * Feet.  A revolute joint across each foot at the ball of the foot
%     (MidfootOffset, the metatarsophalangeal line) with a spring and
%     damper, so the heel can rise over a bending forefoot.  ForefootMass
%     moves from the foot to the forefoot; the rearfoot's centre of mass
%     moves back so the foot's is unchanged at zero angle.  The toe
%     contact spheres ride on the forefoot.  Both parts are drawn as
%     ellipsoids.
%   * Blocks.  The "Inertia Sensor" subsystem (twelve sensors whose
%     outputs nothing reads, 75 compiled blocks) is removed.
%
%   Options: overwrite (false), face_square_roll (-20.73 deg, measured at
%   address), club_axes (World X Y Z in the club frame at address, after
%   the roll), loft (10.5 deg), forefoot_mass (0.25 kg), mtp_fraction
%   (0.73 of the foot length from the heel), mtp_height (0.025 m above the
%   sole), midfoot_stiffness (100 N*m/rad), midfoot_damping (0.5
%   N*m*s/rad), head_radii ([0.09 0.09 0.105] m), neck_address ([-24.2
%   -7.0] deg, measured at address, docs/HUMAN.md).
%   REPORT fields: .budget, .hidden, .visuals, .joints.

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.face_square_roll (1,1) double = -20.73
        opts.club_axes (3,3) double = [0.685 -0.0224 -0.728; 0.728 0.021 0.685; 0 -1 0.0307]
        opts.loft (1,1) double = 10.5
        opts.forefoot_mass (1,1) double {mustBePositive} = 0.25
        opts.mtp_fraction (1,1) double {mustBeInRange(opts.mtp_fraction, 0.5, 0.95)} = 0.73
        opts.mtp_height (1,1) double {mustBeNonnegative} = 0.025
        opts.midfoot_stiffness (1,1) double {mustBePositive} = 100
        opts.midfoot_damping (1,1) double {mustBeNonnegative} = 0.5
        opts.head_radii (1,3) double {mustBePositive} = [0.09 0.09 0.105]
        opts.neck_address (1,2) double = [-24.2 -7.0]
    end
    names = gs3dx_names();
    src = char(names.variants.neck);
    mdl = char(names.variants.human);
    gs3dx_copy_models({fullfile(info.models_dir, [src '.slx']), fullfile(info.models_dir, [mdl '.slx'])}, ...
        opts.overwrite, 'gs3dx:human');

    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    mass0 = gs3dx_inertia_audit(mdl).total_mass;

    local_reference_port('');
    local_remove_sensor_subsystem(mdl);
    local_neck_address(mdl, opts.neck_address);
    report.hidden = local_hide(mdl);
    report.visuals = local_body_visuals(mdl, opts.head_radii);
    report.visuals = [report.visuals local_club(mdl, opts)];
    [report.joints, fv] = local_midfeet(mdl, opts);
    report.visuals = [report.visuals fv];

    mass = gs3dx_inertia_audit(mdl).total_mass;
    assert(abs(mass - mass0) < 1e-9, 'gs3dx:human', 'total mass changed: %.6f -> %.6f kg', mass0, mass);
    report.budget = gs3dx_block_budget(mdl, compiled=true);
    reserve = 25;   % as GS3DX_BUILD_FIT_BALANCE: room for GS3DX_CONTACT_CHECK
    assert(report.budget.compiled_total <= names.license_block_limit - reserve, 'gs3dx:human', ...
        '%s compiles to %d blocks; %d leave no room for %d validation blocks', ...
        mdl, report.budget.compiled_total, names.license_block_limit, reserve);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
end

% -------------------------------------------------------------------------
function local_remove_sensor_subsystem(mdl)
% Twelve Inertia Sensors on physical ports only: nothing reads them.
    blk = [mdl '/Inertia Sensor'];
    ph = get_param(blk, 'PortHandles');
    assert(isempty(ph.Inport) && isempty(ph.Outport) && isempty(ph.RConn), 'gs3dx:human', ...
        '%s is not the frame-only sensor subsystem', blk);
    inner = find_system(blk, 'LookUnderMasks', 'all', 'Type', 'block', 'BlockType', 'SimscapeMultibodyBlock');
    ref = string(get_param(inner, 'ReferenceBlock'));
    assert(all(ref == "sm_lib/Body Elements/Inertia Sensor"), 'gs3dx:human', ...
        '%s holds more than Inertia Sensors', blk);
    delete_block(blk);
    local_delete_dangling(mdl);
end

function local_neck_address(mdl, a)
    ws = get_param(mdl, 'ModelWorkspace');
    ref = ws.getVariable('NeckReference');
    Q0 = local_rx(a(1)) * local_ry(a(2));
    for k = 1:size(ref, 2)
        Q = local_rx(rad2deg(ref(1, k))) * local_ry(rad2deg(ref(2, k))) * Q0;
        % X-Y-Z angles of Q; the neck keeps X and Y
        ref(:, k) = [atan2(-Q(2, 3), Q(3, 3)); asin(Q(1, 3))];
    end
    assignin(ws, 'NeckReference', ref);
    assignin(ws, 'NeckAddress', deg2rad(a(:)));
end

function hidden = local_hide(mdl)
% Solids drawn by an ellipsoid instead (graphics only: mass stays).
    hidden = ["Hips and Torso Inputs/LowerTorso" "Hips and Torso Inputs/UpperTorsoBase" ...
        "Hips and Torso Inputs/UpperTorsoTop" "Hips and Torso Inputs/Neck" ...
        "Hips and Torso Inputs/ZeroMassHipReference" "Hips and Torso Inputs/ZeroMassShoulderReference" ...
        "Hips and Torso Inputs/COMRod" "HubtoLS" "HubtoRS" "LUpperArm" "RUpperArm" ...
        "Left Forearm/LUpperForearm" "Left Forearm/LLowerForearm" ...
        "Right Forearm/RUpperForearm" "Right Forearm/RLowerForearm" ...
        "Grip/LHandStandoff" "Grip/RHandStandoff" "Club/Clubhead" "Club/Clubface Vector" ...
        "Lower Body/L Foot" "Lower Body/R Foot"];
    for h = hidden
        set_param([mdl '/' char(h)], 'GraphicType', 'None');
    end
end

function vis = local_body_visuals(mdl, head_radii)
    skin = [0.93 0.76 0.63];
    shirt = [0.22 0.33 0.62];
    pants = [0.30 0.30 0.34];
    t = "Hips and Torso Inputs/";
    % The trunk frames have z down the spine (up is -z).  Lower trunk: the
    % hip joint centres are 0.213 m below its centre, their midpoint 0.038 m
    % forward of its axis, and the hip line 46.3 deg from x toward y.  Upper
    % trunk: the shoulder line 17 deg from x toward y at address, the
    % shoulder centres 0.072 m above its centre (docs/HUMAN.md).
    hip = local_rz(46.3);
    sho = local_rz(17);
    vis = [ ...
        local_visual(mdl, t + "LowerTorso", "Pelvis", [0.17 0.115 0.12], [-0.027 0.027 0.174], hip, pants), ...
        local_visual(mdl, t + "LowerTorso", "Abdomen", [0.145 0.105 0.15], [-0.0135 0.0135 -0.03], hip, shirt), ...
        local_visual(mdl, t + "UpperTorsoTop", "Chest", [0.16 0.11 0.155], [0 0 0.03], sho, shirt), ...
        local_visual(mdl, t + "Neck", "Neck Shape", [0.055 0.055 0.12], [0 0 0], eye(3), skin), ...
        local_visual(mdl, "HubtoLS", "L Shoulder", [0.05 0.05 0.095], [0 0 0], eye(3), shirt), ...
        local_visual(mdl, "HubtoRS", "R Shoulder", [0.05 0.05 0.095], [0 0 0], eye(3), shirt), ...
        local_visual(mdl, "LUpperArm", "L Upper Arm", [0.048 0.048 0.165], [0 0 0], eye(3), shirt), ...
        local_visual(mdl, "RUpperArm", "R Upper Arm", [0.048 0.048 0.165], [0 0 0], eye(3), shirt), ...
        local_visual(mdl, "Left Forearm/LUpperForearm", "L Forearm Upper", [0.042 0.042 0.08], [0 0 0], eye(3), skin), ...
        local_visual(mdl, "Left Forearm/LLowerForearm", "L Forearm Lower", [0.031 0.031 0.08], [0 0 0], eye(3), skin), ...
        local_visual(mdl, "Right Forearm/RUpperForearm", "R Forearm Upper", [0.042 0.042 0.08], [0 0 0], eye(3), skin), ...
        local_visual(mdl, "Right Forearm/RLowerForearm", "R Forearm Lower", [0.031 0.031 0.08], [0 0 0], eye(3), skin)];
    head = [mdl '/' char(t) 'Head'];
    set_param(head, 'EllipsoidRadii', mat2str(head_radii), 'EllipsoidRadiiUnits', 'm', ...
        'GraphicDiffuseColor', mat2str(skin));
    for b = ["Grip/LHand" "Grip/RHand"]
        set_param([mdl '/' char(b)], 'GraphicDiffuseColor', mat2str(skin));
    end
    for b = ["Lower Body/L Thigh" "Lower Body/R Thigh" "Lower Body/L Shank" "Lower Body/R Shank"]
        set_param([mdl '/' char(b)], 'GraphicDiffuseColor', mat2str(pants));
    end
end

function vis = local_club(mdl, opts)
% Square the face, then draw a driver head and a face-normal pointer.
    ws = get_param(mdl, 'ModelWorkspace');
    assignin(ws, 'FaceSquareRoll', opts.face_square_roll);
    grip = [mdl '/Club/GripStrength'];
    assert(strcmp(get_param(grip, 'RotationSequenceAngles'), '[-GripStrength 0 0]'), 'gs3dx:human', ...
        '%s is not the grip roll of GS3DX_Neck', grip);
    set_param(grip, 'RotationSequenceAngles', '[-GripStrength + FaceSquareRoll 0 0]');
    % World axes at address in the (rolled) club frame, tilted by the loft:
    % head x toward the toe, y the face normal, z up.
    [u, ~, v] = svd(opts.club_axes);
    M = u * v.';
    H = M * local_rx(opts.loft);
    radii = [0.06 0.055 0.03];
    centre = H * [0.045; -0.045; 0.01];       % toeward of and behind the hosel
    face = centre + H(:, 2) * radii(2);
    vis = [local_visual(mdl, "Club/Clubhead", "Driver Head", radii, centre.', H, [0.12 0.12 0.14]), ...
        local_visual(mdl, "Club/Clubhead", "Face Normal", [0.004 0.004 0.1], (face + H(:, 2) * 0.1).', ...
            H * [1 0 0; 0 0 1; 0 -1 0], [0.85 0.15 0.15])];
end

function [joints, vis] = local_midfeet(mdl, opts)
    sys = [mdl '/Lower Body'];
    ws = get_param(mdl, 'ModelWorkspace');
    val = @(e) double(slResolve(e, [sys '/L Foot']));
    L = val('FootLength');
    h = val('FootHeelOffset');
    H = val('AnkleHeight');
    W = val('FootWidth');
    M = val('FootMass');
    mf = opts.forefoot_mass;
    assert(mf < M / 2, 'gs3dx:human', 'forefoot mass %.3f kg exceeds half the foot (%.3f kg)', mf, M);
    % Ankle-frame coordinates: x forward, y to the left, z up; sole at -H.
    mtp = [(opts.mtp_fraction - h) * L, 0, -H + opts.mtp_height];
    assignin(ws, 'MidfootOffset', mtp);
    assignin(ws, 'ForefootMass', mf);
    assignin(ws, 'MidfootStiffness', opts.midfoot_stiffness);
    assignin(ws, 'MidfootDamping', opts.midfoot_damping);
    toe = (1 - h) * L;
    % forefoot: 0.01 m behind the joint to 0.012 m past the toe points,
    % sole to 0.015 m above the joint (forefoot frame)
    fore_r = [(toe - mtp(1) + 0.022) / 2, 0.048, (0.015 + H + mtp(3)) / 2];
    fore_c = [-0.01 + fore_r(1), 0, 0.015 - fore_r(3)];
    brick_c = [(0.5 - h) * L, 0, -H / 2];
    rear_c = [(-h * L - 0.005 + mtp(1) + 0.01) / 2, 0, -H / 2];
    rear_r = [(mtp(1) + 0.01 + h * L + 0.005) / 2, 0.045, H / 2 + 0.001];
    d = (mtp + fore_c) - brick_c;
    mr = M - mf;
    Ib = mr / 12 * [W^2 + H^2, L^2 + H^2, L^2 + W^2];
    If = mf / 5 * [fore_r(2)^2 + fore_r(3)^2, fore_r(1)^2 + fore_r(3)^2, fore_r(1)^2 + fore_r(2)^2];
    shoe = [0.12 0.12 0.12];
    joints = strings(1, 0);
    vis = strings(1, 0);
    for s = ["L" "R"]
        foot = [sys '/' char(s) ' Foot'];
        set_param(foot, 'InertiaType', 'Custom', 'Mass', 'FootMass - ForefootMass', 'MassUnits', 'kg', ...
            'CenterOfMass', mat2str(-mf / mr * d, 6), 'CenterOfMassUnits', 'm', ...
            'MomentsOfInertia', mat2str(Ib, 6), 'MomentsOfInertiaUnits', 'kg*m^2', ...
            'ProductsOfInertia', '[0 0 0]', 'ProductsOfInertiaUnits', 'kg*m^2');
        vis(end + 1) = local_visual(mdl, "Lower Body/" + s + " Foot", s + " Rearfoot", rear_r, rear_c - brick_c, eye(3), shoe); %#ok<AGROW>

        com = get_param([sys '/' char(s) ' Foot COM'], 'PortHandles');
        pos = get_param([sys '/' char(s) ' Toe In Point'], 'Position');
        mount = local_rt(sys, s + " Midfoot Mount", pos + [-250 -120 -250 -120], 'MidfootOffset', local_rx(-90));
        add_line(sys, com.LConn(1), mount.LConn(1), 'autorouting', 'on');
        jt = [sys '/' char(s) ' Midfoot Joint'];
        add_block('sm_lib/Joints/Revolute Joint', jt, 'Position', pos + [-150 -120 -110 -80], ...
            'SpringStiffness', 'MidfootStiffness', 'SpringStiffnessUnits', 'N*m/rad', ...
            'DampingCoefficient', 'MidfootDamping', 'DampingCoefficientUnits', 'N*m/(rad/s)');
        jp = get_param(jt, 'PortHandles');
        add_line(sys, mount.RConn(1), jp.LConn(1), 'autorouting', 'on');
        fore = local_rt(sys, s + " Forefoot Frame", pos + [-60 -120 -20 -80], '', local_rx(90));
        add_line(sys, jp.RConn(1), fore.LConn(1), 'autorouting', 'on');
        joints(end + 1) = string(jt); %#ok<AGROW>

        fm = [sys '/' char(s) ' Forefoot'];
        add_block('sm_lib/Body Elements/Ellipsoidal Solid', fm, 'Position', pos + [40 -120 80 -80], ...
            'EllipsoidRadii', mat2str(fore_r, 6), 'EllipsoidRadiiUnits', 'm', 'InertiaType', 'Custom', ...
            'Mass', 'ForefootMass', 'MassUnits', 'kg', 'CenterOfMass', '[0 0 0]', 'CenterOfMassUnits', 'm', ...
            'MomentsOfInertia', mat2str(If, 6), 'MomentsOfInertiaUnits', 'kg*m^2', ...
            'ProductsOfInertia', '[0 0 0]', 'ProductsOfInertiaUnits', 'kg*m^2', ...
            'GraphicDiffuseColor', mat2str(shoe));
        cen = local_rt(sys, s + " Forefoot Centre", pos + [-10 -60 30 -20], mat2str(fore_c, 6), eye(3));
        add_line(sys, fore.RConn(1), cen.LConn(1), 'autorouting', 'on');
        fp = get_param(fm, 'PortHandles');
        add_line(sys, cen.RConn(1), fp.RConn(1), 'autorouting', 'on');
        vis(end + 1) = string(fm); %#ok<AGROW>

        for p = ["Toe In" "Toe Out"]
            local_move_to_forefoot(sys, s + " " + p + " Point", fore.RConn(1));
        end
    end
end

function local_move_to_forefoot(sys, name, base_port)
% Re-base a toe contact point on the forefoot frame, the same point.
    old = [sys '/' char(name)];
    ph = get_param(old, 'PortHandles');
    line = get_param(ph.RConn(1), 'Line');
    ends = [get_param(line, 'SrcPortHandle') get_param(line, 'DstPortHandle')'];
    sphere = setdiff(ends(ends > 0), ph.RConn(1));
    assert(isscalar(sphere), 'gs3dx:human', '%s follower must join one contact sphere only', old);
    offset = get_param(old, 'TranslationCartesianOffset');
    tmp = [old ' new'];
    add_block(old, tmp, 'Position', get_param(old, 'Position') + [0 60 0 60]);
    delete_block(old);
    local_delete_dangling(sys);
    set_param(tmp, 'Name', char(name));
    set_param(old, 'TranslationCartesianOffset', [offset ' - MidfootOffset']);
    np = get_param(old, 'PortHandles');
    add_line(sys, base_port, np.LConn(1), 'autorouting', 'on');
    add_line(sys, np.RConn(1), sphere, 'autorouting', 'on');
end

% -------------------------------------------------------------------------
function blk = local_visual(mdl, parent, name, radii, offset, R, color)
% Massless ellipsoid at OFFSET (m) and orientation R in PARENT's reference frame.
    pblk = [mdl '/' char(parent)];
    sys = get_param(pblk, 'Parent');
    ref = local_reference_port(pblk);
    pos = get_param(pblk, 'Position');
    rt = local_rt(sys, name + " Place", pos + [0 90 0 90], mat2str(offset, 6), R);
    blk = [sys '/' char(name)];
    add_block('sm_lib/Body Elements/Ellipsoidal Solid', blk, 'Position', pos + [80 90 80 90], ...
        'EllipsoidRadii', mat2str(radii, 6), 'EllipsoidRadiiUnits', 'm', 'InertiaType', 'Custom', ...
        'Mass', '0', 'CenterOfMass', '[0 0 0]', 'MomentsOfInertia', '[0 0 0]', 'ProductsOfInertia', '[0 0 0]', ...
        'GraphicDiffuseColor', mat2str(color));
    ep = get_param(blk, 'PortHandles');
    add_line(sys, ref, rt.LConn(1), 'autorouting', 'on');
    add_line(sys, rt.RConn(1), ep.RConn(1), 'autorouting', 'on');
    blk = string(blk);
end

function port = local_reference_port(blk)
% The solid's reference frame port.  A solid with custom frames shows only
% those (end-centre frames, some with turned axes) unless its reference
% frame is exposed, which adds the port R.  Exposed here once per block.
    persistent exposed
    if isempty(blk)
        exposed = containers.Map();
        return
    end
    if isKey(exposed, blk)
        port = exposed(blk);
        return
    end
    ph = get_param(blk, 'PortHandles');
    before = [ph.LConn ph.RConn];
    if strcmp(get_param(blk, 'SerializedFrames'), '<Frames/>') || isempty(get_param(blk, 'SerializedFrames'))
        assert(isscalar(before), 'gs3dx:human', '%s: expected the single port R', blk);
        port = before;
        return
    end
    assert(strcmp(get_param(blk, 'DoExposeReferenceFrame'), 'off'), 'gs3dx:human', ...
        '%s: reference frame already exposed; cannot tell R from its custom frames', blk);
    assert(all(arrayfun(@(p) get_param(p, 'Line'), before) ~= -1), 'gs3dx:human', ...
        '%s: a custom frame port is unconnected', blk);
    set_param(blk, 'DoExposeReferenceFrame', 'on');
    % Port handles move when the port is added: R is the one unconnected.
    ph = get_param(blk, 'PortHandles');
    after = [ph.LConn ph.RConn];
    port = after(arrayfun(@(p) get_param(p, 'Line'), after) == -1);
    assert(numel(after) == numel(before) + 1 && isscalar(port), 'gs3dx:human', ...
        '%s: exposing the reference frame did not add one free port', blk);
    exposed(blk) = port;
end

function ph = local_rt(sys, name, pos, offset, R)
    blk = [sys '/' char(name)];
    add_block(['sm_lib/Frames and Transforms/Rigid' newline 'Transform'], blk, 'Position', pos);
    if isequal(R, eye(3))
        set_param(blk, 'RotationMethod', 'None');
    else
        set_param(blk, 'RotationMethod', 'RotationMatrix', 'RotationMatrix', mat2str(R, 8));
    end
    if isempty(offset)
        set_param(blk, 'TranslationMethod', 'None');
    else
        set_param(blk, 'TranslationMethod', 'Cartesian', 'TranslationCartesianOffset', offset, ...
            'TranslationCartesianOffsetUnits', 'm');
    end
    ph = get_param(blk, 'PortHandles');
end

function local_delete_dangling(sys)
    lines = find_system(sys, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line', 'Connected', 'off');
    for l = lines(:).'
        if ishandle(l)
            delete_line(l);
        end
    end
end

function R = local_rx(a)
    R = [1 0 0; 0 cosd(a) -sind(a); 0 sind(a) cosd(a)];
end

function R = local_ry(a)
    R = [cosd(a) 0 sind(a); 0 1 0; -sind(a) 0 cosd(a)];
end

function R = local_rz(a)
    R = [cosd(a) -sind(a) 0; sind(a) cosd(a) 0; 0 0 1];
end

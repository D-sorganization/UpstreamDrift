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
%     along the hip line, the chest along the shoulder line, and a
%     trapezius under C7 joins the neck to the shoulders.  Ellipsoidal
%     Solid accepts no frames on its surface, so the cylinders that carry
%     frames stay as the mass carriers.  The head is made rounder
%     (graphics only: its inertia is Custom).
%   * Head.  GS3DX_Neck's neck reference is zero at address, so the neck
%     carries on along the upper trunk's axis (59 deg from vertical at
%     address) and the head centre sits 0.1 m below the capture's head
%     markers.  A fixed rotation Rx(ax) Ry(ay) (NECK_ADDRESS, "Neck
%     Address") between the neck joint and the neck aims the head at their
%     centroid; the joint and NeckReference are unchanged, so the joint
%     starts at its reference.  (Composing the offset into NeckReference
%     instead started the joint 24 deg from its reference, and the 5 ms
%     input filter threw the head: support fell to zero, then 4.1 BW.)
%   * Neck pivot.  GS3DX_Neck's neck pivot sits 58 mm below the level of
%     C7 along the neck axis, inside the chest, so a 10 in neck showed
%     above the shoulders.  NECK_PIVOT_LIFT moves the pivot up the neck's
%     address axis and NeckLength shrinks by the same length: the head
%     keeps its place and turns about C7.
%   * Club.  At address the face pointed 20.7 deg right of the target,
%     from the grip roll.  FaceSquareRoll (deg, model workspace) rolls the
%     club about the shaft in GripStrength's transform; the head is a
%     sphere centred on the shaft, so the roll moves no mass.  The sphere
%     and the face-plane rod are hidden.  A massless File Solid draws a
%     driver head (gs3dx_driver_head.stl, from Tools'
%     rate_of_closure parametric generator: 10.5 deg loft, bulge and
%     roll), hung from its hosel point with its sole level at address,
%     and a red pointer shows the face normal at the face centre.
%   * Feet.  A revolute joint across each foot at the ball of the foot
%     (MidfootOffset, the metatarsophalangeal line) with a spring and
%     damper, so the heel can rise over a bending forefoot.  ForefootMass
%     moves from the foot to the forefoot; the rearfoot's centre of mass
%     moves back so the foot's is unchanged at zero angle.  Both parts
%     are drawn as ellipsoids.  The spring is 2000 N*m/rad: at 100 and
%     800 N*m/rad the toes folded and the golfer drifted (65 mm pelvis
%     RMS at 800); at 2000 it drifts 1.7 mm more than with the joint
%     locked, and the toes can still bend when the heel rises.
%   * Foot contacts.  Five spheres per foot instead of three: the heel on
%     the foot's axis, the first and fifth metatarsal heads on the
%     rearfoot (the ball of the foot, which carries the standing load, so
%     the midfoot spring carries none of it), and the big toe and the
%     lesser toes on the forefoot (which keep the toes on the ground as
%     the heel rises).  The outside ball and lesser toes let the foot
%     roll onto its outside edge, the inside ball and big toe onto its
%     inside edge.  The toes reach as far forward as GS3DX_Neck's toe
%     corners: with the big toe 15 mm behind the tip and the lesser toes
%     at 64% of the foot length, the pelvis drifted 26-31 mm RMS to
%     impact, against 20 mm here and 18 mm for GS3DX_Neck (docs/HUMAN.md).
%     'FootContactForces' keeps the left contacts first.
%   * Balance anchor.  Moving the head moves the body's centre of mass at
%     address (about 7 mm), and BalanceCOMRef is the capture's COM path
%     anchored at GS3DX_Neck's.  After saving, a balance-off start measures
%     this model's address COM and the reference is translated onto it
%     (as GS3DX_BUILD_SHAPE anchored it); left alone, the 7 mm error at
%     t = 0 made the balance loop kick both feet off the ground.
%   * Blocks.  The "Inertia Sensor" subsystem (twelve sensors whose
%     outputs nothing reads, 75 compiled blocks) is removed.
%
%   Options: overwrite (false), face_square_roll (-20.73 deg, measured at
%   address), club_axes (World X Y Z in the club frame at address, after
%   the roll), club_head_file ('gs3dx_driver_head.stl', on the path, mm,
%   head frame x target, y up, z toe, loft built in), club_hosel,
%   club_face_centre and club_face_normal (that file's hosel point, face
%   centre (m) and lofted face normal, models/README_DRIVER_HEAD.md),
%   forefoot_mass (0.25 kg), mtp_fraction
%   (0.73 of the foot length from the heel), mtp_height (0.025 m above the
%   sole), midfoot_stiffness (2000 N*m/rad), midfoot_damping (0.5
%   N*m*s/rad), head_radii ([0.09 0.09 0.105] m), neck_address ([-24.2
%   -7.0] deg, measured at address, docs/HUMAN.md), ball_out_fraction
%   (0.64 of the foot length from the heel: the fifth metatarsal head),
%   toe_inset (0 m: the big toe under the toe tip), toe_offset (0.05 m
%   inside the foot's axis: the big toe), lesser_toe_fraction (0.945 of
%   the foot length from the heel: the third and fourth toe pads),
%   lesser_toe_offset (0.05 m outside the foot's axis), neck_pivot_lift (0.058 m: the capture's C7 projected on
%   the address neck axis, docs/HUMAN.md).
%   REPORT fields: .budget, .hidden, .visuals, .joints, .contacts,
%   .neck_length (in), .com_shift (m, World: the balance re-anchoring).

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.face_square_roll (1,1) double = -20.73
        opts.club_axes (3,3) double = [0.685 -0.0224 -0.728; 0.728 0.021 0.685; 0 -1 0.0307]
        opts.club_head_file (1,1) string = "gs3dx_driver_head.stl"
        opts.club_hosel (3,1) double = [0.019430 0.029029 -0.052]
        opts.club_face_centre (3,1) double = [0.049897 -0.000469 0]
        opts.club_face_normal (3,1) double = [0.983255 0.182236 0]
        opts.forefoot_mass (1,1) double {mustBePositive} = 0.25
        opts.mtp_fraction (1,1) double {mustBeInRange(opts.mtp_fraction, 0.5, 0.95)} = 0.73
        opts.mtp_height (1,1) double {mustBeNonnegative} = 0.025
        opts.midfoot_stiffness (1,1) double {mustBePositive} = 2000
        opts.midfoot_damping (1,1) double {mustBeNonnegative} = 0.5
        opts.head_radii (1,3) double {mustBePositive} = [0.09 0.09 0.105]
        opts.neck_address (1,2) double = [-24.2 -7.0]
        opts.ball_out_fraction (1,1) double {mustBeInRange(opts.ball_out_fraction, 0.5, 0.95)} = 0.64
        opts.toe_inset (1,1) double {mustBeNonnegative} = 0
        opts.toe_offset (1,1) double = 0.05
        opts.lesser_toe_fraction (1,1) double {mustBeInRange(opts.lesser_toe_fraction, 0.75, 1)} = 0.945
        opts.lesser_toe_offset (1,1) double = 0.05
        opts.neck_pivot_lift (1,1) double {mustBeNonnegative} = 0.058
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
    report.neck_length = local_neck_address(mdl, opts.neck_address, opts.neck_pivot_lift);
    report.hidden = local_hide(mdl);
    report.visuals = local_body_visuals(mdl, opts.head_radii, report.neck_length * 0.0254);
    report.visuals = [report.visuals local_club(mdl, opts)];
    [report.joints, fv, forefoot] = local_midfeet(mdl, opts);
    report.visuals = [report.visuals fv];
    report.contacts = local_foot_contacts(mdl, forefoot, opts);

    mass = gs3dx_inertia_audit(mdl).total_mass;
    assert(abs(mass - mass0) < 1e-9, 'gs3dx:human', 'total mass changed: %.6f -> %.6f kg', mass0, mass);
    report.budget = gs3dx_block_budget(mdl, compiled=true);
    reserve = 25;   % as GS3DX_BUILD_FIT_BALANCE: room for GS3DX_CONTACT_CHECK
    assert(report.budget.compiled_total <= names.license_block_limit - reserve, 'gs3dx:human', ...
        '%s compiles to %d blocks; %d leave no room for %d validation blocks', ...
        mdl, report.budget.compiled_total, names.license_block_limit, reserve);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
    report.com_shift = local_anchor_balance(info, mdl);
end

% -------------------------------------------------------------------------
function shift = local_anchor_balance(info, mdl)
% The head now sits where the capture has it at address, so the body's
% centre of mass there moved (about 7 mm).  The balance reference is the
% capture's COM path anchored at GS3DX_Neck's address COM
% (GS3DX_CAPTURE_COM_REFERENCE: a pure translation); unanchored, the 7 mm
% error at t = 0 made the balance loop kick both feet off the ground.
% Re-anchor it at this model's own address COM (a balance-off start, as
% GS3DX_BUILD_SHAPE's com0), then save again.
    load_system(mdl);
    start = get_param(mdl, 'ModelWorkspace').getVariable('TrackStart');
    close_system(mdl, 0);
    start.BalanceOn = 0;
    c = gs3dx_contact_check(info, model=mdl, rest=true, stop_time=1e-3, variables=start);
    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    ws = get_param(mdl, 'ModelWorkspace');
    ref = ws.getVariable('BalanceCOMRef');
    shift = c.com(:, 1) - ref(:, 1);
    assert(norm(shift) < 0.02, 'gs3dx:human', 'address COM moved %.1f mm: not a head placement', 1000 * norm(shift));
    assignin(ws, 'BalanceCOMRef', ref + shift);
    gs3dx_save_model(mdl, info);
end

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

function len = local_neck_address(mdl, a, lift)
% "Neck Address" (rotation Rx(a1) Ry(a2)) between the neck joint and the
% neck; the joint's base moves LIFT (m) up the neck's address axis and
% NeckLength (in) shrinks by LIFT, so the head centre stays in place.
    sys = [mdl '/Hips and Torso Inputs'];
    ws = get_param(mdl, 'ModelWorkspace');
    Q0 = local_rx(a(1)) * local_ry(a(2));
    assignin(ws, 'NeckAddress', deg2rad(a(:)));
    joint = [sys '/Neck Joint'];
    jp = get_param(joint, 'PortHandles');
    neck = local_peer(joint, 'RConn');
    delete_line(get_param(jp.RConn(1), 'Line'));
    pos = get_param(joint, 'Position');
    rt = local_rt(sys, "Neck Address", pos + [70 0 70 0], '', Q0);
    add_line(sys, jp.RConn(1), rt.LConn(1), 'autorouting', 'on');
    add_line(sys, rt.RConn(1), neck, 'autorouting', 'on');
    % The neck runs from its bottom frame along that frame's -z (the head
    % is at -NeckLength), so up the neck is -Q0(:,3) in the joint's base
    % frame at zero angle.  (+Q0(:,3) moved the pivot down and the head
    % 116 mm, twice the lift; test_gs3dx_human pins the head in place.)
    % Rigid Transform5 (no offset in GS3DX_Neck) carries the joint's base.
    base = get_param(local_peer(joint, 'LConn'), 'Parent');
    assert(strcmp(get_param(base, 'TranslationMethod'), 'None'), 'gs3dx:human', ...
        '%s: expected the zero-offset transform below the neck joint', base);
    set_param(base, 'TranslationMethod', 'Cartesian', 'TranslationCartesianOffsetUnits', 'm', ...
        'TranslationCartesianOffset', mat2str(-lift * Q0(:, 3).', 6));
    p = ws.getVariable('NeckLength');
    p.Value = p.Value - lift / 0.0254;
    assignin(ws, 'NeckLength', p);
    len = p.Value;
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

function vis = local_body_visuals(mdl, head_radii, neck_len)
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
    % The neck frame sits mid-neck, the head centre at -z and the pivot (C7)
    % at +z, NECK_LEN (m) apart.  C7 is 25 mm above the chest ellipsoid's
    % top, so the shape runs from the head centre to 50 mm past C7 and the
    % trapezius (on the upper trunk, under C7, along the shoulder line)
    % fills the neck's base out toward the shoulders.
    below = 0.05;
    neck_r = [0.052 0.052 neck_len / 2 + below / 2];
    vis = [ ...
        local_visual(mdl, t + "LowerTorso", "Pelvis", [0.17 0.115 0.12], [-0.027 0.027 0.174], hip, pants), ...
        local_visual(mdl, t + "LowerTorso", "Abdomen", [0.145 0.105 0.15], [-0.0135 0.0135 -0.03], hip, shirt), ...
        local_visual(mdl, t + "UpperTorsoTop", "Chest", [0.16 0.11 0.155], [0 0 0.03], sho, shirt), ...
        local_visual(mdl, t + "UpperTorsoTop", "Trapezius", [0.12 0.075 0.045], [0.005 -0.015 -0.125], sho, shirt), ...
        local_visual(mdl, t + "Neck", "Neck Shape", neck_r, [0 0 below / 2], eye(3), skin), ...
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
% Square the face, then draw the driver head mesh and a face-normal pointer.
    ws = get_param(mdl, 'ModelWorkspace');
    assignin(ws, 'FaceSquareRoll', opts.face_square_roll);
    grip = [mdl '/Club/GripStrength'];
    assert(strcmp(get_param(grip, 'RotationSequenceAngles'), '[-GripStrength 0 0]'), 'gs3dx:human', ...
        '%s is not the grip roll of GS3DX_Neck', grip);
    set_param(grip, 'RotationSequenceAngles', '[-GripStrength + FaceSquareRoll 0 0]');
    % World axes at address in the (rolled) club frame: M's columns are
    % toe, target and up.  The mesh's axes (x target, y up, z toe) map
    % onto them, so the sole sits level at address; the loft is in the
    % mesh.  The hosel point goes to the Clubhead frame's origin, the
    % shaft's end.
    [u, ~, v] = svd(opts.club_axes);
    M = u * v.';
    H = [M(:, 2) M(:, 3) M(:, 1)];
    file = which(opts.club_head_file);
    assert(~isempty(file), 'gs3dx:human', '%s is not on the path (gs3dx_setup)', opts.club_head_file);
    n = H * opts.club_face_normal;
    toe = H(:, 3);
    face = H * (opts.club_face_centre - opts.club_hosel);
    vis = [local_mesh_visual(mdl, "Club/Clubhead", "Driver Head", opts.club_head_file, ...
            (-H * opts.club_hosel).', H, [0.16 0.17 0.19]), ...
        local_visual(mdl, "Club/Clubhead", "Face Normal", [0.004 0.004 0.1], (face + n * 0.1).', ...
            [toe cross(n, toe) n], [0.85 0.15 0.15])];
end

function blk = local_mesh_visual(mdl, parent, name, file, offset, R, color)
% Massless File Solid (STL in mm) at OFFSET (m) and orientation R in
% PARENT's reference frame.  The file is named, not pathed, so the model
% finds it on the MATLAB path wherever the repository is.
    pblk = [mdl '/' char(parent)];
    sys = get_param(pblk, 'Parent');
    ref = local_reference_port(pblk);
    pos = get_param(pblk, 'Position');
    rt = local_rt(sys, name + " Place", pos + [0 90 0 90], mat2str(offset, 6), R);
    blk = [sys '/' char(name)];
    add_block('sm_lib/Body Elements/File Solid', blk, 'Position', pos + [80 90 80 90], ...
        'ExtGeomFileName', char(file), 'UnitType', 'Custom', 'ExtGeomFileUnits', 'mm', 'InertiaType', 'Custom', ...
        'Mass', '0', 'CenterOfMass', '[0 0 0]', 'MomentsOfInertia', '[0 0 0]', 'ProductsOfInertia', '[0 0 0]', ...
        'GraphicDiffuseColor', mat2str(color));
    ep = get_param(blk, 'PortHandles');
    add_line(sys, ref, rt.LConn(1), 'autorouting', 'on');
    add_line(sys, rt.RConn(1), ep.RConn(1), 'autorouting', 'on');
    blk = string(blk);
end

function [joints, vis, forefoot] = local_midfeet(mdl, opts)
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
    forefoot = struct();
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
        forefoot.(s) = fore.RConn(1);
    end
end

function names = local_foot_contacts(mdl, forefoot, opts)
% Five contacts per foot from GS3DX_Neck's three.  Heel stays (moved onto
% the foot's axis); Toe In and Toe Out become Ball In and Ball Out (same
% parent, moved); Big Toe and Lesser Toes are copies re-based on the
% forefoot.  Then the force log is rewired, left contacts first.
    sys = [mdl '/Lower Body'];
    ws = get_param(mdl, 'ModelWorkspace');
    assignin(ws, 'FootBallOutFraction', opts.ball_out_fraction);
    assignin(ws, 'FootToeInset', opts.toe_inset);
    assignin(ws, 'FootToeOffset', opts.toe_offset);
    assignin(ws, 'FootLesserToeFraction', opts.lesser_toe_fraction);
    assignin(ws, 'FootLesserToeOffset', opts.lesser_toe_offset);
    z = '-AnkleHeight+FootContactRadius';
    % name, made from, x, y (%d = side sign: +1 left; inside is -sign), on the forefoot
    spec = { ...
        'Heel',        "keep Heel",      '-FootHeelOffset*FootLength', '0', false; ...
        'Ball In',     "rename Toe In",  'MidfootOffset(1)', '-%d*FootContactWidth/2', false; ...
        'Ball Out',    "rename Toe Out", '(FootBallOutFraction-FootHeelOffset)*FootLength', '%d*FootContactWidth/2', false; ...
        'Big Toe',     "copy Ball In",   '(1-FootHeelOffset)*FootLength-FootToeInset-MidfootOffset(1)', '-%d*FootToeOffset', true; ...
        'Lesser Toes', "copy Ball Out",  '(FootLesserToeFraction-FootHeelOffset)*FootLength-MidfootOffset(1)', '%d*FootLesserToeOffset', true};
    order = ["Heel" "Ball In" "Ball Out" "Big Toe" "Lesser Toes"];   % in the force log
    ground = local_ground_port(sys);
    % The three spheres per foot share their mass among five, so the total
    % mass stays GS3DX_Neck's.
    m0 = str2double(get_param([sys '/L Heel Sphere'], 'Mass'));
    assert(isfinite(m0), 'gs3dx:human', 'contact sphere mass is not a number');
    assignin(ws, 'FootContactSphereMass', 3 * m0 / size(spec, 1));
    names = strings(1, 0);
    forces = zeros(1, 0);
    for s = ["L" "R"]
        sgn = 1 - 2 * (s == "R");
        trio = struct();
        for k = 1:size(spec, 1)
            [name, how, x, y, fore] = spec{k, :};
            target = s + " " + name;
            src = s + " " + extractAfter(how, " ");
            zz = z;
            if startsWith(how, "keep")
                t = local_trio(sys, target);
            elseif startsWith(how, "rename")
                t = local_rename_contact(sys, src, target);
            elseif fore
                t = local_copy_contact(sys, src, target, ground, forefoot.(s));
                zz = [z '-MidfootOffset(3)'];
            else
                ph = get_param(local_trio(sys, src).point, 'PortHandles');
                t = local_copy_contact(sys, src, target, ground, ph.LConn(1));   % same frame node
            end
            if contains(y, '%d')
                y = sprintf(y, sgn);
            end
            set_param(t.point, 'TranslationCartesianOffset', sprintf('[%s, %s, %s]', x, y, zz));
            set_param(t.sphere, 'Mass', 'FootContactSphereMass');
            trio.(strrep(name, ' ', '')) = t;
        end
        for name = order
            t = trio.(strrep(char(name), ' ', ''));
            forces(end + 1) = local_force_port(t.contact); %#ok<AGROW>
            names(end + 1) = s + " " + name; %#ok<AGROW>
        end
    end
    local_rewire_force_log(sys, forces);
end

function trio = local_rename_contact(sys, from, to)
    for part = ["Point" "Sphere" "Contact"]
        set_param([sys '/' char(from + " " + part)], 'Name', char(to + " " + part));
    end
    trio = local_trio(sys, to);
end

function trio = local_copy_contact(sys, from, to, ground, parent)
% A point, sphere and contact force with FROM's parameters, on the frame
% port PARENT.
    src = local_trio(sys, from);
    dst = local_trio(sys, to);
    for part = ["point" "sphere" "contact"]
        add_block(src.(part), dst.(part), 'Position', get_param(src.(part), 'Position') + [0 400 0 400]);
    end
    pp = get_param(dst.point, 'PortHandles');
    sp = get_param(dst.sphere, 'PortHandles');
    cp = get_param(dst.contact, 'PortHandles');
    add_line(sys, parent, pp.LConn(1), 'autorouting', 'on');
    add_line(sys, pp.RConn(1), sp.RConn(1), 'autorouting', 'on');
    add_line(sys, ground, cp.LConn(1), 'autorouting', 'on');
    add_line(sys, cp.RConn(1), sp.LConn(1), 'autorouting', 'on');
    trio = dst;
end

function trio = local_trio(sys, name)
    trio = struct('point', [sys '/' char(name + " Point")], 'sphere', [sys '/' char(name + " Sphere")], ...
        'contact', [sys '/' char(name + " Contact")]);
end

function port = local_peer(blk, kind)
% The port at the far end of BLK's first KIND connection.
    ph = get_param(blk, 'PortHandles');
    mine = ph.(kind)(1);
    line = get_param(mine, 'Line');
    ends = [get_param(line, 'SrcPortHandle') get_param(line, 'DstPortHandle')'];
    port = setdiff(ends(ends > 0), mine);
    assert(isscalar(port), 'gs3dx:human', '%s: its %s port joins more than one block', blk, kind);
end

function port = local_ground_port(sys)
    ph = get_param([sys '/Ground Plane'], 'PortHandles');
    port = ph.RConn(1);
end

function port = local_force_port(contact)
% The contact's sensed total force.
    ph = get_param(contact, 'PortHandles');
    port = ph.RConn(2);
end

function local_rewire_force_log(sys, forces)
% One PS-Simulink converter per contact into the Foot Contact Mux, in the
% order of FORCES (left contacts first, as GS3DX_CONTACT_CHECK expects).
    conv = find_system(sys, 'SearchDepth', 1, 'RegExp', 'on', 'Name', '^Foot Contact Force \d+$');
    for c = conv(:).'
        delete_block(c{1});
    end
    local_delete_dangling(sys);
    mux = [sys '/Foot Contact Mux'];
    set_param(mux, 'Inputs', num2str(numel(forces)));
    mp = get_param(mux, 'PortHandles');
    pos = get_param(mux, 'Position');
    for k = 1:numel(forces)
        yy = pos(2) + 30 * (k - 1);
        c = [sys sprintf('/Foot Contact Force %d', k)];
        add_block('nesl_utility/PS-Simulink Converter', c, 'Unit', 'N', ...
            'Position', [pos(1) - 100 yy pos(1) - 70 yy + 20]);
        cp = get_param(c, 'PortHandles');
        add_line(sys, forces(k), cp.LConn(1), 'autorouting', 'on');
        add_line(sys, cp.Outport(1), mp.Inport(k), 'autorouting', 'on');
    end
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

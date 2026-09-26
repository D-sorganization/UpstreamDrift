function report = gs3dx_build_lower_body(info, opts)
%GS3DX_BUILD_LOWER_BODY  Build GS3DX_FullBody: GS3DX_Quat plus legs and grounded feet.
%
%   REPORT = GS3DX_BUILD_LOWER_BODY(INFO) copies GS3DX_Quat to GS3DX_FullBody
%   and adds a top-level 'Lower Body' subsystem between the pelvis ('Lower
%   Torso') and World frames of 'Hips and Torso Inputs' (#10957):
%
%     pelvis -> Hip Mount -> Hip Joint (GS3DX_KDS_Spherical) -> Thigh
%            -> Knee Joint (GS3DX_KDS_Revolute) -> Shank
%            -> Ankle Joint (GS3DX_KDS_Universal) -> Foot == Foot Ground (World)
%
%   for each side, from GS3DX_LEG_TABLE.  Leg geometry is laid out in a leg
%   frame measured on GS3DX_Quat at t = 0 (GS3DX_STANCE_FRAMES under OPTS.drive):
%   z opposite to gravity, y along the horizontal shoulder line (left
%   shoulder side), x = y cross z.  Each foot is rigidly framed to World
%   under its ankle at the start posture (weld stance, #10958), which closes
%   one kinematic loop per leg.  Leg position targets follow the table
%   (knee and ankle Low, hip none) and velocity targets are None, so
%   assembly keeps the pelvis on its High hip targets.
%   Leg torques come from the model-workspace vector LegTorqueCommand
%   (zeros: the legs are passive until leg inputs are fitted).
%
%   GS3DX_Quat is only read with COPYFILE (and loaded unsaved for the
%   measurement).  An existing GS3DX_FullBody is never replaced unless
%   overwrite=true.
%
%   REPORT fields: .frames (stance measurement), .leg_R (leg frame in
%   World), .budget (GS3DX_BLOCK_BUDGET of the saved model).

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.drive (1,1) string = "impact"
    end
    names = gs3dx_names();
    quat = char(names.variants.quat);
    full = char(names.variants.fullbody);
    gs3dx_copy_models({fullfile(info.models_dir, [quat '.slx']), fullfile(info.models_dir, [full '.slx'])}, ...
        opts.overwrite, 'gs3dx:fullbody');

    frames = gs3dx_stance_frames(quat, gs3dx_drive(info, opts.drive, quat));
    up = frames.up;
    lateral = frames.left_shoulder_p - frames.right_shoulder_p;
    lateral = lateral - (lateral.' * up) * up;
    assert(norm(lateral) > 0.05, 'gs3dx:fullbody', 'Shoulder line is nearly vertical at t = 0');
    lateral = lateral / norm(lateral);
    leg_R = [cross(lateral, up), lateral, up];

    leg = gs3dx_leg_table();
    p = leg.params;
    load_system(full);
    cleanup = onCleanup(@() close_system(full, 0));
    ws = get_param(full, 'ModelWorkspace');
    for f = reshape(fieldnames(p), 1, [])
        assignin(ws, f{1}, p.(f{1}));
    end
    reach = p.LegReachFraction * (p.ThighLength + p.ShankLength);
    sides = struct('prefix', {'L', 'R'}, 'name', {'Left', 'Right'}, 'sign', {1, -1});
    for s = sides
        hip_leg = [0; s.sign * p.HipJointSpacing / 2; -p.HipJointDrop];
        assignin(ws, [s.prefix 'HipMountRotation'], frames.pelvis_R.' * leg_R);
        assignin(ws, [s.prefix 'HipMountOffset'], frames.pelvis_R.' * leg_R * hip_leg);
        assignin(ws, [s.prefix 'FootGroundRotation'], leg_R);
        assignin(ws, [s.prefix 'FootGroundOffset'], frames.pelvis_p + leg_R * (hip_leg - [0; 0; reach]));
        for j = leg.joints
            assignin(ws, [s.prefix j.name 'StartPosition'], j.start);
        end
    end

    hips = [full '/Hips and Torso Inputs'];
    body = local_lower_body_subsystem(full, hips);
    add_line(full, gs3dx_pm_port(hips, 'Lower Torso'), gs3dx_pm_port(body, 'Pelvis'), 'autorouting', 'on');
    add_line(full, gs3dx_pm_port(hips, 'GlobalReferenceFrame'), gs3dx_pm_port(body, 'World'), 'autorouting', 'on');
    sys = getfullname(body);
    torque = local_torque_source(sys, leg);
    for k = 1:numel(sides)
        local_build_leg(sys, sides(k), leg, torque, (k - 1) * local_torque_count(leg));
    end

    set_param(full, 'SimulationCommand', 'update');
    gs3dx_save_model(full, info);
    close_system(full, 0);
    report.frames = frames;
    report.leg_R = leg_R;
    report.budget = gs3dx_block_budget(full);
    assert(report.budget.nonvirtual_total <= 0.9 * names.license_block_limit, 'gs3dx:fullbody', ...
        'GS3DX_FullBody has %d non-virtual blocks; the budget with 10%% margin is %d', ...
        report.budget.nonvirtual_total, 0.9 * names.license_block_limit);
end

function body = local_lower_body_subsystem(mdl, hips)
% Placed below 'Hips and Torso Inputs', whose frames it uses, at the first
% spot that overlaps no other top-level block.
    at = get_param(hips, 'Position');
    others = get_param(find_system(mdl, 'SearchDepth', 1, 'Type', 'Block'), 'Position');
    others = vertcat(others{:});
    pos = [at(1), at(4) + 60, at(1) + 160, at(4) + 160];
    while any(pos(1) < others(:, 3) & others(:, 1) < pos(3) & pos(2) < others(:, 4) & others(:, 2) < pos(4))
        pos = pos + [0 60 0 60];
    end
    body = add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Lower Body'], 'Position', pos);
    Simulink.SubSystem.deleteContents(body);
    sys = getfullname(body);
    add_block('nesl_utility/Connection Port', [sys '/Pelvis'], 'Side', 'Left', 'Position', [20 100 50 114]);
    add_block('nesl_utility/Connection Port', [sys '/World'], 'Side', 'Left', 'Position', [20 900 50 914]);
end

function torque = local_torque_source(sys, leg)
% One Constant (LegTorqueCommand) demuxed to every leg torque input.
    n = 2 * local_torque_count(leg);
    c = add_block('simulink/Sources/Constant', [sys '/Leg Torque Commands'], 'Value', 'LegTorqueCommand', ...
        'SampleTime', '0', 'Position', [20 1050 120 1080]);
    torque = add_block('simulink/Signal Routing/Demux', [sys '/Leg Torque Demux'], 'Outputs', num2str(n), ...
        'Position', [160 1000 165 1000 + 20 * n]);
    add_line(sys, local_port(c, 'Outport', 1), local_port(torque, 'Inport', 1));
end

function local_build_leg(sys, side, leg, torque, first_torque)
    P = side.prefix;
    y = 60 + (side.sign < 0) * 420;
    pelvis = local_conn_port([sys '/Pelvis']);
    world = local_conn_port([sys '/World']);
    rot_knee = '[1 0 0; 0 0 -1; 0 1 0]';   % knee axis (joint z) along -y of the leg frame
    rot_back = '[1 0 0; 0 0 1; 0 -1 0]';   % inverse: back to leg-frame axes

    mount = local_transform(sys, [P ' Hip Mount'], [P 'HipMountRotation'], [P 'HipMountOffset'], [120 y]);
    local_connect(sys, pelvis, local_port(mount, 'LConn', 1));
    proximal = local_port(mount, 'RConn', 1);
    next_torque = first_torque;
    for k = 1:numel(leg.joints)
        j = leg.joints(k);
        x = 260 + 420 * (k - 1);
        joint = local_joint(sys, side, j, [x y]);
        local_connect(sys, proximal, gs3dx_pm_port(joint, 'Proximal'));
        distal = gs3dx_pm_port(joint, 'Distal');
        labels = local_torque_labels(j);
        for a = 1:numel(labels)
            next_torque = next_torque + 1;
            local_connect(sys, local_port(torque, 'Outport', next_torque), ...
                local_port(joint, 'Inport', local_inport_index(joint, labels{a})));
        end
        seg = j.segment;
        switch j.name
            case 'Hip'   % distal frame is leg-aligned; thigh along -z
                com = local_transform(sys, [P ' Thigh COM'], '', '[0 0 -ThighLength/2]', [x + 180 y + 110]);
                next = local_transform(sys, [P ' Knee Mount'], rot_knee, '[0 0 -ThighLength]', [x + 180 y]);
                solid = local_cylinder(sys, [P ' Thigh'], 'ThighRadius', 'ThighLength', 'ThighMass', [x + 300 y + 110]);
            case 'Knee'  % distal frame has y along the leg's up axis
                com = local_transform(sys, [P ' Shank COM'], rot_back, '[0 -ShankLength/2 0]', [x + 180 y + 110]);
                next = local_transform(sys, [P ' Ankle Mount'], rot_back, '[0 -ShankLength 0]', [x + 180 y]);
                solid = local_cylinder(sys, [P ' Shank'], 'ShankRadius', 'ShankLength', 'ShankMass', [x + 300 y + 110]);
            case 'Ankle' % distal frame is the foot frame at the ankle centre
                com = local_transform(sys, [P ' Foot COM'], '', ...
                    '[(0.5 - FootHeelOffset) * FootLength, 0, -AnkleHeight/2]', [x + 180 y + 110]);
                next = local_transform(sys, [P ' Foot Ground'], [P 'FootGroundRotation'], [P 'FootGroundOffset'], [x + 180 y + 220]);
                solid = add_block('sm_lib/Body Elements/Brick Solid', [sys '/' P ' Foot'], ...
                    'BrickDimensions', '[FootLength FootWidth AnkleHeight]', 'BasedOnType', 'Mass', ...
                    'Mass', 'FootMass', 'Position', [x + 300 y + 110 x + 360 y + 160]);
        end
        local_connect(sys, distal, local_port(com, 'LConn', 1));
        local_connect(sys, local_port(com, 'RConn', 1), local_port(solid, 'RConn', 1));
        if j.name == "Ankle"
            local_connect(sys, world, local_port(next, 'LConn', 1));
            local_connect(sys, local_port(next, 'RConn', 1), distal);
        else
            local_connect(sys, distal, local_port(next, 'LConn', 1));
            proximal = local_port(next, 'RConn', 1);
        end
        local_log_bus(sys, joint, [P j.name 'Logs'], [x + 120 y + 200]);
    end
end

function joint = local_joint(sys, side, j, xy)
    joint = add_block('simulink/Ports & Subsystems/Subsystem Reference', ...
        [sys '/' side.name ' ' j.name ' Joint'], 'Position', [xy xy + [120 140]]);
    set_param(joint, 'ReferencedSubsystem', j.kds);
    var = [side.prefix j.name 'StartPosition'];
    suffix = num2cell(j.axes);
    if isempty(suffix)
        suffix = {''};
    end
    args = {};
    for a = 1:numel(suffix)
        idx = sprintf('(%d)', a);
        args = [args, {['StartPosition' suffix{a}], [var idx], ['StartVelocity' suffix{a}], '0', ...
            ['Dampening' suffix{a}], '0'}]; %#ok<AGROW>
    end
    priority = local_priority_prefixes(j);
    for a = 1:numel(priority)
        args = [args, {[priority{a} 'PositionTargetPriority'], j.priority, ...
            [priority{a} 'VelocityTargetPriority'], 'None'}]; %#ok<AGROW>
    end
    set_param(joint, args{:});   % one call: the Spherical mask checks the axes agree
end

function prefixes = local_priority_prefixes(j)
    if isempty(j.axes)
        prefixes = {'Rz'};
    else
        prefixes = arrayfun(@(c) ['R' lower(c)], j.axes, 'UniformOutput', false);
    end
end

function n = local_torque_count(leg)
% Torque inputs per leg: one per joint axis (a revolute has one).
    n = sum(arrayfun(@(j) numel(local_torque_labels(j)), leg.joints));
end

function labels = local_torque_labels(j)
    if isempty(j.axes)
        labels = {'Torque'};
    else
        labels = arrayfun(@(c) ['Torque ' c], j.axes, 'UniformOutput', false);
    end
end

function n = local_inport_index(blk, name)
    in = find_system(gs3dx_port_source(blk), 'SearchDepth', 1, 'BlockType', 'Inport', 'Name', name);
    assert(isscalar(in), 'gs3dx:fullbody', 'No inport %s in %s', name, getfullname(blk));
    n = str2double(get_param(in{1}, 'Port'));
end

function t = local_transform(sys, name, rotation, offset, xy)
    args = {'TranslationMethod', 'Cartesian', 'TranslationCartesianOffset', offset, ...
        'TranslationCartesianOffsetUnits', 'm'};
    if ~isempty(rotation)
        args = [args, {'RotationMethod', 'RotationMatrix', 'RotationMatrix', rotation}];
    end
    t = add_block('sm_lib/Frames and Transforms/Rigid Transform', [sys '/' name], ...
        'Position', [xy xy + [50 50]], args{:});
end

function s = local_cylinder(sys, name, radius, len, mass, xy)
    s = add_block('sm_lib/Body Elements/Cylindrical Solid', [sys '/' name], ...
        'CylinderRadius', radius, 'CylinderRadiusUnits', 'm', 'CylinderLength', len, ...
        'CylinderLengthUnits', 'm', 'BasedOnType', 'Mass', 'Mass', mass, 'MassUnits', 'kg', ...
        'Position', [xy xy + [60 50]]);
end

function local_log_bus(sys, joint, name, xy)
    t = add_block('simulink/Sinks/Terminator', [sys '/' name], 'Position', [xy xy + [20 20]]);
    h = add_line(sys, local_port(joint, 'Outport', 1), local_port(t, 'Inport', 1), 'autorouting', 'on');
    set_param(h, 'Name', name);
    set_param(local_port(joint, 'Outport', 1), 'DataLogging', 'on');
end

function local_connect(sys, a, b)
    add_line(sys, a, b, 'autorouting', 'on');
end

function p = local_conn_port(blk)
% The single physical port of a Connection Port block.
    ph = get_param(blk, 'PortHandles');
    p = [ph.LConn, ph.RConn];
    assert(isscalar(p), 'gs3dx:fullbody', 'Expected one physical port on %s', blk);
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end

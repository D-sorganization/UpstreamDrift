function report = gs3dx_build_contact(info, opts)
%GS3DX_BUILD_CONTACT  Build GS3DX_FullBodyContact: the golfer stands on the ground.
%
%   REPORT = GS3DX_BUILD_CONTACT(INFO) copies GS3DX_FullBody to
%   GS3DX_FullBodyContact and changes how the body is supported (#10986):
%
%   * Feet.  The weld frames ('L/R Foot Ground') are removed.  Each foot
%     rests on one Infinite Plane (the ground) at three sole spheres (heel
%     centre, toe inside, toe outside) through Spatial Contact Forces with
%     smooth stick-slip friction.  The feet sit at the stance measured from
%     the tour-average capture (GS3DX_LEG_TABLE .stance).
%   * Pelvis.  The pelvis 6-DOF joint loses its actuation (force and torque
%     inputs set to NoTorque, their four converters deleted); it keeps its
%     start targets and sensing, so the legs alone carry the pelvis.
%   * Legs.  Every leg axis has a servo torque
%       LegTorqueCommand + Kp .* (LegAngleReference - q) - Kd .* qd
%     built as the existing 'Leg Torque Commands' Constant (now
%     LegTorqueCommand + LegServoKp .* LegAngleReference) minus one matrix
%     Gain [diag(Kp) diag(Kd)] on the stacked [q; qd].  LegAngleReference
%     is the stance hold: the angles that put both feet on their stance at
%     t = 0 (GS3DX_LEG_IK).  The leg joints start on those angles, with the
%     joint rates that keep the feet still while the pelvis moves at its
%     start velocity (High targets; the loops are open now).
%   * Ground.  The plane frame is GroundRotation = the FullBody leg frame
%     [facing, lateral, up] (up = against gravity) at GroundOffset, on the
%     sole of the left stance foot.
%   * Sensing.  Each contact logs its total force (ground on foot, in the
%     ground frame axes) into 'FootContactForces' (18x1: L heel, toe-in,
%     toe-out, then R, 3 components each).
%
%   Block budget.  The Home license counts blocks after compilation, when
%   Simscape adds its own (GS3DX_FullBody: 751 before, 945 after), so the
%   design is sized against the compiled count, which REPORT.budget holds.
%
%   GS3DX_FullBody is only read with COPYFILE; GS3DX_Quat is loaded unsaved
%   to measure the pelvis start state.  An existing GS3DX_FullBodyContact
%   is never replaced unless overwrite=true.
%
%   REPORT fields: .q0, .qd0 (12x1 start angles/rates, deg and deg/s, L
%   then R, [hip X Y Z, knee, ankle X Y]), .feet (per side R/p targets),
%   .ground_R, .ground_p, .budget (GS3DX_BLOCK_BUDGET with compiled=true).

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.drive (1,1) string = "impact"
    end
    names = gs3dx_names();
    quat = char(names.variants.quat);
    full = char(names.variants.fullbody);
    mdl = char(names.variants.contact);
    gs3dx_copy_models({fullfile(info.models_dir, [full '.slx']), fullfile(info.models_dir, [mdl '.slx'])}, ...
        opts.overwrite, 'gs3dx:contact');

    leg = gs3dx_leg_table();
    st = leg.stance;
    ct = leg.contact;
    start = gs3dx_stance_frames(quat, gs3dx_drive(info, opts.drive, quat), stop_time=0.004);
    s = start.series;

    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    ws = get_param(mdl, 'ModelWorkspace');
    leg_R = ws.getVariable('LFootGroundRotation');
    p = leg.params;
    sides = struct('prefix', {'L', 'R'}, 'sign', {1, -1});
    q0 = zeros(12, 1);
    qd0 = zeros(12, 1);
    seed = [0; -leg.theta; 0; -2 * leg.theta; 0; -leg.theta];
    for k = 1:2
        P = sides(k).prefix;
        a = st.(['ankle_' P]);
        a(2) = a(2) - sides(k).sign * st.inset;
        foot_R = leg_R * local_rz(st.(['foot_yaw_' P]));
        foot_p = s.pelvis_p(:, 1) + leg_R * [a; -st.drop];
        geom = struct('mount_R', ws.getVariable([P 'HipMountRotation']), ...
            'mount_p', ws.getVariable([P 'HipMountOffset']), 'thigh', p.ThighLength, 'shank', p.ShankLength);
        q = gs3dx_leg_ik(geom, s.pelvis_R, s.pelvis_p, foot_R, foot_p, seed);
        assert(q(4, 1) < -1 && q(4, 1) > -120, 'gs3dx:contact', ...
            'Postcondition: %s knee start %.1f deg is not a forward-bent knee', P, q(4, 1));
        rows = (k - 1) * 6 + (1:6);
        q0(rows) = q(:, 1);
        qd0(rows) = local_start_rate(s.t, q);
        assignin(ws, [P 'FootGroundRotation'], foot_R);
        assignin(ws, [P 'FootGroundOffset'], foot_p);
        report.feet.(P) = struct('R', foot_R, 'p', foot_p);
    end
    up = leg_R(:, 3);
    assert(norm(up - start.up) < 1e-9, 'gs3dx:contact', 'Leg frame up axis differs from gravity');
    ground_p = (report.feet.L.p + report.feet.R.p) / 2 - p.AnkleHeight * up;
    ground_p = ground_p + ((report.feet.L.p - p.AnkleHeight * up - ground_p).' * up) * up;   % on the L sole plane
    assert(abs((report.feet.R.p - p.AnkleHeight * up - ground_p).' * up) < 0.02, 'gs3dx:contact', ...
        'Postcondition: the two stance soles are more than 2 cm apart in height');
    assignin(ws, 'GroundRotation', leg_R);
    assignin(ws, 'GroundOffset', ground_p);

    local_assign(ws, q0, qd0, ct, st);
    sys = [mdl '/Lower Body'];
    local_remove_welds(sys);
    forces = local_add_contacts(sys, ct);
    local_add_servo(sys);
    local_set_leg_targets(sys, leg);
    local_unactuate_pelvis([mdl '/Hips and Torso Inputs/Hip Kinetically Driven']);
    local_log_forces(sys, forces);

    report.budget = gs3dx_block_budget(mdl, compiled=true);
    % Keep room for the sensors GS3DX_CONTACT_CHECK adds in memory.
    reserve = 25;
    assert(report.budget.compiled_total <= names.license_block_limit - reserve, 'gs3dx:contact', ...
        'GS3DX_FullBodyContact compiles to %d blocks; %d leave no room for %d validation blocks', ...
        report.budget.compiled_total, names.license_block_limit, reserve);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
    report.q0 = q0;
    report.qd0 = qd0;
    report.ground_R = leg_R;
    report.ground_p = ground_p;
end

function qd = local_start_rate(t, q)
% Rate at t(1) from a quadratic fit to the first samples (6xN, deg).
    n = min(numel(t), 6);
    assert(n >= 3, 'gs3dx:contact', 'Need at least 3 pelvis samples for the start rate');
    tt = t(1:n) - t(1);
    qd = zeros(6, 1);
    for r = 1:6
        c = polyfit(tt, q(r, 1:n), 2);
        qd(r) = c(2);
    end
end

function local_assign(ws, q0, qd0, ct, st)
    assignin(ws, 'FootContactRadius', ct.sphere_radius);
    assignin(ws, 'FootContactStiffness', ct.stiffness);
    assignin(ws, 'FootContactDamping', ct.damping);
    assignin(ws, 'FootContactTransitionWidth', ct.transition_width);
    assignin(ws, 'FootStaticFriction', ct.mu_static);
    assignin(ws, 'FootDynamicFriction', ct.mu_dynamic);
    assignin(ws, 'FootFrictionCriticalVelocity', ct.critical_velocity);
    assignin(ws, 'FootContactWidth', st.foot_width);
    assignin(ws, 'LegServoKp', repmat(ct.kp(:), 2, 1));
    assignin(ws, 'LegServoKd', repmat(ct.kd(:), 2, 1));
    assignin(ws, 'LegAngleReference', q0);
    names = {'Hip', 'Knee', 'Ankle'};
    cols = {1:3, 4, 5:6};
    sides = 'LR';
    for k = 1:2
        for j = 1:3
            rows = (k - 1) * 6 + cols{j};
            assignin(ws, [sides(k) names{j} 'StartPosition'], q0(rows).');
            assignin(ws, [sides(k) names{j} 'StartVelocity'], qd0(rows).');
        end
    end
end

function local_remove_welds(sys)
% Physical lines are whole nets, so deleting the weld's lines also cuts the
% ankle Distal -> Foot COM connection; that one is redrawn.
    for side = struct('P', {'L', 'R'}, 'name', {'Left', 'Right'})
        blk = [sys '/' side.P ' Foot Ground'];
        lines = get_param(blk, 'LineHandles');
        delete_line([lines.LConn, lines.RConn]);
        delete_block(blk);
        com = [sys '/' side.P ' Foot COM'];
        lines = get_param(com, 'LineHandles');
        if lines.LConn(1) == -1
            local_connect(sys, local_foot_frame(sys, side.P), local_port(com, 'LConn', 1));
        end
        lines = get_param(com, 'LineHandles');
        assert(all([lines.LConn, lines.RConn] ~= -1), 'gs3dx:contact', ...
            'Postcondition: %s is not connected after the weld was removed', com);
    end
end

function forces = local_add_contacts(sys, ct)
% Ground plane plus four sole spheres and contacts per foot.
    world = local_conn_port([sys '/World']);
    frame = add_block('sm_lib/Frames and Transforms/Rigid Transform', [sys '/Ground Frame'], ...
        'RotationMethod', 'RotationMatrix', 'RotationMatrix', 'GroundRotation', ...
        'TranslationMethod', 'Cartesian', 'TranslationCartesianOffset', 'GroundOffset', ...
        'TranslationCartesianOffsetUnits', 'm', 'Position', [1500 900 1550 950]);
    plane = add_block('sm_lib/Curves and Surfaces/Infinite Plane', [sys '/Ground Plane'], ...
        'Position', [1620 900 1680 950]);
    local_connect(sys, world, local_port(frame, 'LConn', 1));
    local_connect(sys, local_port(frame, 'RConn', 1), local_port(plane, 'LConn', 1));   % frame R | geometry G
    ground = local_port(plane, 'RConn', 1);
    corners = struct('name', {'Heel', 'Toe In', 'Toe Out'}, ...
        'x', {'-FootHeelOffset*FootLength', '(1-FootHeelOffset)*FootLength', '(1-FootHeelOffset)*FootLength'}, ...
        'y', {0, -1, 1});   % -1 = inside (toward the other foot)
    forces = zeros(1, 6);
    n = 0;
    for side = struct('P', {'L', 'R'}, 'sign', {1, -1}, 'y', {60, 480})
        foot = local_foot_frame(sys, side.P);
        for c = corners
            n = n + 1;
            y_sign = side.sign * c.y;
            name = sprintf('%s %s', side.P, c.name);
            yy = side.y + 60 * n;
            rt = add_block('sm_lib/Frames and Transforms/Rigid Transform', [sys '/' name ' Point'], ...
                'TranslationMethod', 'Cartesian', 'TranslationCartesianOffsetUnits', 'm', ...
                'TranslationCartesianOffset', sprintf('[%s, %d*FootContactWidth/2, -AnkleHeight+FootContactRadius]', c.x, y_sign), ...
                'Position', [1800 yy 1840 yy + 40]);
            ball = add_block('sm_lib/Body Elements/Spherical Solid', [sys '/' name ' Sphere'], ...
                'SphereRadius', 'FootContactRadius', 'SphereRadiusUnits', 'm', 'BasedOnType', 'Mass', ...
                'Mass', '1e-3', 'MassUnits', 'kg', 'ExportEntireGeometry', 'on', ...
                'Position', [1900 yy 1940 yy + 40]);
            cf = add_block('sm_lib/Forces and Torques/Spatial Contact Force', [sys '/' name ' Contact'], ...
                'NormalStiffness', 'FootContactStiffness', 'NormalDamping', 'FootContactDamping', ...
                'NormalTransitionRegionWidth', 'FootContactTransitionWidth', ...
                'FrictionType', 'SmoothStickSlip', 'CoefficientOfStaticFriction', 'FootStaticFriction', ...
                'CoefficientOfDynamicFriction', 'FootDynamicFriction', ...
                'FrictionalCriticalVelocity', 'FootFrictionCriticalVelocity', ...
                'SenseTotalForce', 'on', 'SensingForceTorqueDirection', 'BaseOnFollower', ...
                'SensingForceTorqueResolutionFrame', 'BaseFrame', 'Position', [2000 yy 2060 yy + 40]);
            local_connect(sys, foot, local_port(rt, 'LConn', 1));
            local_connect(sys, local_port(rt, 'RConn', 1), local_port(ball, 'RConn', 1));   % sphere frame R
            local_connect(sys, ground, local_port(cf, 'LConn', 1));
            local_connect(sys, local_port(cf, 'RConn', 1), local_port(ball, 'LConn', 1));   % sphere geometry G
            forces(n) = local_port(cf, 'RConn', 2);
        end
    end
end

function p = local_foot_frame(sys, P)
% The foot frame: the ankle joint's Distal port.
    p = gs3dx_pm_port([sys '/' ternary(P == 'L', 'Left', 'Right') ' Ankle Joint'], 'Distal');
end

function local_log_forces(sys, forces)
    mux = add_block('simulink/Signal Routing/Mux', [sys '/Foot Contact Mux'], 'Inputs', num2str(numel(forces)), ...
        'Position', [2300 60 2305 60 + 30 * numel(forces)]);
    for k = 1:numel(forces)
        yy = 60 + 30 * (k - 1);
        c = add_block('nesl_utility/PS-Simulink Converter', sprintf('%s/Foot Contact Force %d', sys, k), ...
            'Unit', 'N', 'Position', [2200 yy 2230 yy + 20]);
        local_connect(sys, forces(k), local_port(c, 'LConn', 1));
        local_connect(sys, local_port(c, 'Outport', 1), local_port(mux, 'Inport', k));
    end
    t = add_block('simulink/Sinks/Terminator', [sys '/FootContactForces'], 'Position', [2400 150 2420 170]);
    h = add_line(sys, local_port(mux, 'Outport', 1), local_port(t, 'Inport', 1), 'autorouting', 'on');
    set_param(h, 'Name', 'FootContactForces');
    set_param(local_port(mux, 'Outport', 1), 'DataLogging', 'on');
end

function local_add_servo(sys)
% Leg torques = (LegTorqueCommand + Kp .* LegAngleReference) - [diag(Kp) diag(Kd)] * [q; qd].
    pos = {{'AngularPositionX', 'AngularPositionY', 'AngularPosition Z'}, {'AngularPosition'}, ...
           {'AngularPositionX', 'AngularPositionY'}};
    vel = {{'AngularVelocityX', 'AngularVelocityY', 'AngularVelocityZ'}, {'AngularVelocity'}, ...
           {'AngularVelocityX', 'AngularVelocityY'}};
    joints = {'Hip', 'Knee', 'Ankle'};
    x0 = 1100;
    state = add_block('simulink/Signal Routing/Mux', [sys '/Leg State'], 'Inputs', '24', ...
        'Position', [x0 + 200 1200 x0 + 205 1920]);
    n = 0;
    for P = {'Left', 'Right'}
        for j = 1:3
            joint = [sys '/' P{1} ' ' joints{j} ' Joint'];
            sel = add_block('simulink/Signal Routing/Bus Selector', [sys '/' P{1}(1) ' ' joints{j} ' State'], ...
                'OutputSignals', strjoin([pos{j}, vel{j}], ','), 'Position', [x0, 1200 + 60 * n, x0 + 5, 1240 + 60 * n]);
            local_connect(sys, local_port(joint, 'Outport', 1), local_port(sel, 'Inport', 1));
            m = numel(pos{j});
            for a = 1:m
                local_connect(sys, local_port(sel, 'Outport', a), local_port(state, 'Inport', n + a));
                local_connect(sys, local_port(sel, 'Outport', m + a), local_port(state, 'Inport', 12 + n + a));
            end
            n = n + m;
        end
    end
    assert(n == 12, 'gs3dx:contact', 'Expected 12 leg axes, found %d', n);
    gain = add_block('simulink/Math Operations/Gain', [sys '/Leg Servo Gain'], ...
        'Gain', '[diag(LegServoKp) diag(LegServoKd)]', 'Multiplication', 'Matrix(K*u)', ...
        'Position', [x0 + 300 1540 x0 + 380 1580]);
    local_connect(sys, local_port(state, 'Outport', 1), local_port(gain, 'Inport', 1));

    constant = [sys '/Leg Torque Commands'];
    demux = [sys '/Leg Torque Demux'];
    set_param(constant, 'Value', 'LegTorqueCommand + LegServoKp .* LegAngleReference');
    delete_line(sys, local_port(constant, 'Outport', 1), local_port(demux, 'Inport', 1));
    total = add_block('simulink/Math Operations/Sum', [sys '/Leg Servo Torque'], 'Inputs', '+-', ...
        'Position', [x0 + 450 1400 x0 + 480 1460]);
    local_connect(sys, local_port(constant, 'Outport', 1), local_port(total, 'Inport', 1));
    local_connect(sys, local_port(gain, 'Outport', 1), local_port(total, 'Inport', 2));
    local_connect(sys, local_port(total, 'Outport', 1), local_port(demux, 'Inport', 1));
end

function local_set_leg_targets(sys, leg)
% Open chains now: every leg axis starts on its angle and rate (High).
    for P = {'Left', 'Right'}
        for j = leg.joints
            joint = [sys '/' P{1} ' ' j.name ' Joint'];
            suffix = num2cell(j.axes);
            prefix = arrayfun(@(c) ['R' lower(c)], j.axes, 'UniformOutput', false);
            if isempty(suffix)
                suffix = {''};
                prefix = {'Rz'};
            end
            args = {};
            for a = 1:numel(suffix)
                idx = sprintf('(%d)', a);
                args = [args, {['StartPosition' suffix{a}], [P{1}(1) j.name 'StartPosition' idx], ...
                    ['StartVelocity' suffix{a}], [P{1}(1) j.name 'StartVelocity' idx], ...
                    [prefix{a} 'PositionTargetPriority'], 'High', ...
                    [prefix{a} 'VelocityTargetPriority'], 'High'}]; %#ok<AGROW>
            end
            set_param(joint, args{:});
        end
    end
end

function local_unactuate_pelvis(hip)
% The pelvis joint becomes a free 6-DOF joint: its drive converters go.
    for c = {'TranslateXForceHipBaseRefFrame', 'TranslateYForceHipBaseRefFrame', ...
             'TranslateZForceHipBaseRefFrame', 'FollowerTorque'}
        blk = [hip '/' c{1}];
        lines = get_param(blk, 'LineHandles');
        delete_line([lines.Inport, lines.RConn]);
        delete_block(blk);
    end
    set_param([hip '/Hip Joint'], 'PxTorqueActuationMode', 'NoTorque', 'PyTorqueActuationMode', 'NoTorque', ...
        'PzTorqueActuationMode', 'NoTorque', 'SphTorqueActuationMode', 'NoTorque');
end

function R = local_rz(a)
    R = [cosd(a) -sind(a) 0; sind(a) cosd(a) 0; 0 0 1];
end

function local_connect(sys, a, b)
    add_line(sys, a, b, 'autorouting', 'on');
end

function p = local_conn_port(blk)
    ph = get_param(blk, 'PortHandles');
    p = [ph.LConn, ph.RConn];
    assert(isscalar(p), 'gs3dx:contact', 'Expected one physical port on %s', blk);
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end

function new_joint = gs3dx_quaternion_swap(sys, spec)
%GS3DX_QUATERNION_SWAP  Replace an X-Y-Z revolute triple by a quaternion joint.
%
%   NEW_JOINT = GS3DX_QUATERNION_SWAP(SYS, SPEC) edits the loaded diagram
%   level SYS.  It replaces the joint SPEC.joint, whose rotation is three
%   torque-driven revolutes Rx-Ry-Rz (a Gimbal or Bushing Joint), with
%   SPEC.new_lib, whose rotation is one quaternion primitive, and keeps
%   every Simulink signal's meaning (#10955, #10956):
%     - the per-axis torque commands feed 'XYZ Torque', which applies
%       T = E^-T (tau - damping .* rates) in the follower frame;
%     - 'XYZ Kinematics' turns the quaternion, angular velocity and
%       acceleration back into X-Y-Z angles, rates and accelerations (deg)
%       for every consumer of the old per-axis sensors; 'Angle Reference'
%       integrates the rates only to pick the 360-degree branch;
%     - consumers of the old sensed actuator torques get the commands;
%     - start targets, priorities and damping come from the old joint.
%   The new joint takes the old joint's name.  Returns its handle.  The
%   caller saves.
%
%   SPEC fields (port indices are into PortHandles.LConn / .RConn):
%     joint, old_lib, new_lib   old block path and library references
%     prefix                    new joint's rotation parameter prefix ('' | 'Sph')
%     ports, new_ports          [numel(LConn) numel(RConn)] of the old and new joint
%     keep_left, keep_right     (k,2) [old new] physical ports kept as is
%     rot_left                  (1,3) old torque ports of Rx, Ry, Rz
%     rot_right                 (3,4) old [q w b t] sensor ports per axis
%     new_torque                new joint's torque-vector port (LConn)
%     new_qwb                   (1,3) new joint's [Q w b] sensor ports (RConn)
%     new_params                cell of extra name/value pairs for the new joint
%
%   Preconditions (all checked before editing): the joint is SPEC.old_lib
%   with SPEC.ports ports; each rotation axis is InputTorque-driven with
%   degree targets, no spring and no limits; each rotation sensor port
%   feeds exactly one PS-Simulink converter and each torque port is fed by
%   exactly one Simulink-PS converter.  Postconditions: no SPEC.old_lib
%   block remains in SYS; every consumer of a removed converter is fed
%   again; the physical nets satisfy GS3DX_REDRAW_NETS.

    arguments
        sys (1,:) char
        spec (1,1) struct
    end
    joint = spec.joint;
    assert(strcmp(local_ref(joint), spec.old_lib), 'gs3dx:quatswap', ...
        'Precondition: %s is not a %s', joint, spec.old_lib);
    jp = get_param(joint, 'PortHandles');
    assert(isequal([numel(jp.LConn) numel(jp.RConn)], spec.ports), 'gs3dx:quatswap', ...
        'Precondition: %s has %d/%d ports, expected %d/%d', joint, numel(jp.LConn), ...
        numel(jp.RConn), spec.ports);
    joint_name = get_param(joint, 'Name');
    rot = local_rotation_settings(joint);
    g = gs3dx_physical_graph(sys);

    % Plan (read only).  consumers{a,k}: Simulink input ports fed by the old
    % sensor k (q w b t) of axis a; names{a,k}: that signal's line name.
    removed = get_param(joint, 'Handle');
    consumers = cell(3, 4); names = cell(3, 4); old_lines = [];
    for a = 1:3
        for k = 1:4
            conv = local_only_converter(g, jp.RConn(spec.rot_right(a, k)), joint);
            cp = get_param(conv, 'PortHandles');
            assert(isscalar(cp.Outport), 'gs3dx:quatswap', 'Precondition: %s must have one output', ...
                getfullname(conv));
            line = get_param(cp.Outport, 'Line');
            assert(line > 0, 'gs3dx:quatswap', 'Precondition: %s output is unconnected', getfullname(conv));
            dst = get_param(line, 'DstPortHandle');
            consumers{a, k} = dst(dst > 0).';
            names{a, k} = get_param(line, 'Name');
            old_lines(end+1) = line; %#ok<AGROW>
            removed(end+1) = conv; %#ok<AGROW>
        end
    end
    command = zeros(1, 3);                     % Simulink source of each torque command
    for a = 1:3
        conv = local_only_converter(g, jp.LConn(spec.rot_left(a)), joint);
        cp = get_param(conv, 'PortHandles');
        line = get_param(cp.Inport, 'Line');
        assert(line > 0, 'gs3dx:quatswap', 'Precondition: %s input is unconnected', getfullname(conv));
        command(a) = get_param(line, 'SrcPortHandle');
        old_lines(end+1) = line; %#ok<AGROW>
        removed(end+1) = conv; %#ok<AGROW>
    end
    keep_left  = local_kept(g, jp.LConn, spec.keep_left);    % [new index, partner port]
    keep_right = local_kept(g, jp.RConn, spec.keep_right);
    composite = {'CompositeWrenchDir', 'CompositeWrenchFrame', 'SenseConstraintForce', ...
        'SenseConstraintTorque', 'SenseTotalForce', 'SenseTotalTorque'};
    composite(2, :) = cellfun(@(p) get_param(joint, p), composite, 'UniformOutput', false);

    % New blocks.
    at = get_param(joint, 'Position');
    place = @(dx, dy, w, h) [at(1) + dx, at(2) + dy, at(1) + dx + w, at(2) + dy + h];
    pre = spec.prefix;
    new_joint = add_block(spec.new_lib, [sys '/Quaternion Joint'], 'Position', place(0, 0, 60, 90));
    set_param(new_joint, composite{:}, spec.new_params{:}, ...
        [pre 'TorqueActuationMode'], 'InputTorque', [pre 'ActuateTorque'], 'on', ...
        [pre 'ActuationFrame'], 'FollowerFrame', [pre 'SensingFrame'], 'FollowerFrame', ...
        [pre 'SensePosition'], 'on', [pre 'SenseVelocity'], 'on', [pre 'SenseAcceleration'], 'on', ...
        [pre 'PositionTargetSpecify'], rot.position_specify, ...
        [pre 'PositionTargetRotationMethod'], 'RotationSequence', ...
        [pre 'PositionTargetRotationSequenceAxes'], 'FollowerAxes', ...
        [pre 'PositionTargetRotationSequence'], 'XYZ', ...
        [pre 'PositionTargetRotationSequenceAngles'], rot.angles, ...
        [pre 'PositionTargetRotationSequenceAnglesUnits'], 'deg', ...
        [pre 'VelocityTargetSpecify'], rot.velocity_specify, ...
        [pre 'VelocityTargetValue'], rot.velocity, ...
        [pre 'VelocityTargetValueUnits'], rot.velocity_units, ...
        [pre 'VelocityTargetInFollowerFrame'], 'on');
    if strcmp(rot.position_specify, 'on')
        set_param(new_joint, [pre 'PositionTargetPriority'], rot.position_priority);
    end
    if strcmp(rot.velocity_specify, 'on')
        set_param(new_joint, [pre 'VelocityTargetPriority'], rot.velocity_priority);
    end
    np = get_param(new_joint, 'PortHandles');
    assert(isequal([numel(np.LConn) numel(np.RConn)], spec.new_ports), 'gs3dx:quatswap', ...
        '%s has %d/%d ports, expected %d/%d', spec.new_lib, numel(np.LConn), numel(np.RConn), spec.new_ports);

    sense = {'Quaternion', '1'; 'FollowerAngularVelocity', 'rad/s'; 'FollowerAngularAcceleration', 'rad/s^2'};
    sense_port = zeros(3, 2);                  % [physical in, Simulink out]
    for k = 1:3
        h = add_block('nesl_utility/PS-Simulink Converter', [sys '/' sense{k, 1}], 'Unit', sense{k, 2}, ...
            'Position', place(120, 120 + 50 * k, 40, 30));
        ph = get_param(h, 'PortHandles');
        sense_port(k, :) = [ph.LConn(1), ph.Outport(1)];
    end
    % The torque converter copies an old one's input handling.
    tconv = add_block(getfullname(removed(end)), [sys '/FollowerTorque'], 'Unit', 'N*m', ...
        'Position', place(-120, 120, 40, 30));
    tp = get_param(tconv, 'PortHandles');
    mux = add_block('simulink/Signal Routing/Mux', [sys '/Axis Torques'], 'Inputs', '3', ...
        'Position', place(-320, 100, 5, 60));
    % Named actuator-torque signals come off the mux through a (virtual)
    % Demux: a name on the input line would be drawn on every branch.
    actuator = add_block('simulink/Signal Routing/Demux', [sys '/Actuator Torque'], 'Outputs', '3', ...
        'Position', place(-260, 260, 5, 60));
    damping = add_block('simulink/Sources/Constant', [sys '/Axis Damping'], 'Value', rot.damping, ...
        'SampleTime', '0', ...                 % constant time would break input-function loops
        'Position', place(-320, 200, 60, 30));
    reference = add_block('simulink/Continuous/Integrator', [sys '/Angle Reference'], ...
        'InitialCondition', rot.angle_ic, 'Position', place(300, 330, 30, 30));
    torque_fn = local_matlab_function(sys, 'XYZ Torque', place(-220, 100, 80, 90), { ...
        'function T = fcn(Q, w, tau, damping)', ...
        '% Axis torques -> follower-frame torque (gs3dx_xyz_map).', ...
        'T = gs3dx_xyz_map(Q, w, zeros(3, 1), tau, damping, zeros(3, 1));'});
    kin_fn = local_matlab_function(sys, 'XYZ Kinematics', place(280, 180, 80, 90), { ...
        'function [ang, qd, qdd] = fcn(Q, w, b, ang_ref)', ...
        '% Quaternion state -> X-Y-Z angles, rates, accelerations (deg).', ...
        '[~, ang, qd, qdd] = gs3dx_xyz_map(Q, w, b, zeros(3, 1), zeros(3, 1), ang_ref);'});
    quantity = {'Angle', 'Rate', 'Acceleration'};
    demux = zeros(1, 3);
    for k = 1:3
        demux(k) = add_block('simulink/Signal Routing/Demux', sprintf('%s/Axis %s', sys, quantity{k}), ...
            'Outputs', '3', 'Position', place(460, 60 + 110 * k, 5, 60));
    end

    % Simulink wiring.
    arrayfun(@delete_line, old_lines);
    for a = 1:3
        add_line(sys, command(a), local_port(mux, 'Inport', a), 'autorouting', 'on');
    end
    add_line(sys, local_port(mux, 'Outport', 1), local_port(actuator, 'Inport', 1), 'autorouting', 'on');
    for a = 1:3
        local_feed(sys, local_port(actuator, 'Outport', a), consumers{a, 4}, names{a, 4});
    end
    add_line(sys, sense_port(1, 2), local_port(torque_fn, 'Inport', 1), 'autorouting', 'on');
    add_line(sys, sense_port(2, 2), local_port(torque_fn, 'Inport', 2), 'autorouting', 'on');
    add_line(sys, local_port(mux, 'Outport', 1), local_port(torque_fn, 'Inport', 3), 'autorouting', 'on');
    add_line(sys, local_port(damping, 'Outport', 1), local_port(torque_fn, 'Inport', 4), 'autorouting', 'on');
    add_line(sys, local_port(torque_fn, 'Outport', 1), tp.Inport(1), 'autorouting', 'on');
    for k = 1:3
        add_line(sys, sense_port(k, 2), local_port(kin_fn, 'Inport', k), 'autorouting', 'on');
    end
    add_line(sys, local_port(reference, 'Outport', 1), local_port(kin_fn, 'Inport', 4), 'autorouting', 'on');
    add_line(sys, local_port(kin_fn, 'Outport', 2), local_port(reference, 'Inport', 1), 'autorouting', 'on');
    for k = 1:3
        add_line(sys, local_port(kin_fn, 'Outport', k), local_port(demux(k), 'Inport', 1), 'autorouting', 'on');
        for a = 1:3
            local_feed(sys, local_port(demux(k), 'Outport', a), consumers{a, k}, names{a, k});
        end
    end

    % Physical wiring.
    joins = [reshape(np.LConn(keep_left(:, 1)), [], 1), keep_left(:, 2); ...
             reshape(np.RConn(keep_right(:, 1)), [], 1), keep_right(:, 2); ...
             tp.RConn(1), np.LConn(spec.new_torque); ...
             np.RConn(spec.new_qwb(1)), sense_port(1, 1); ...
             np.RConn(spec.new_qwb(2)), sense_port(2, 1); ...
             np.RConn(spec.new_qwb(3)), sense_port(3, 1)];
    gs3dx_redraw_nets(sys, removed, joins);
    set_param(new_joint, 'Name', joint_name);

    assert(isempty(find_system(sys, 'SearchDepth', 1, 'ReferenceBlock', spec.old_lib)), ...
        'gs3dx:quatswap', 'Postcondition: %s still contains a %s', sys, spec.old_lib);
    fed = [consumers{:}];
    assert(all(arrayfun(@(p) get_param(p, 'Line') > 0 && ...
        get_param(get_param(p, 'Line'), 'SrcPortHandle') > 0, fed)), ...
        'gs3dx:quatswap', 'Postcondition: a consumer of a removed sensor in %s is unfed', sys);
end

function rot = local_rotation_settings(joint)
% Targets, priorities and damping of the old joint's Rx, Ry, Rz axes.
    axes = {'Rx', 'Ry', 'Rz'};
    p = @(a, name) get_param(joint, [a name]);
    for a = axes
        assert(strcmp(p(a{1}, 'TorqueActuationMode'), 'InputTorque') && ...
            strcmp(p(a{1}, 'MotionActuationMode'), 'ComputedMotion'), 'gs3dx:quatswap', ...
            'Precondition: %s %s must be InputTorque with ComputedMotion', joint, a{1});
        assert(str2double(p(a{1}, 'SpringStiffness')) == 0, 'gs3dx:quatswap', ...
            'Precondition: %s %s has a spring', joint, a{1});
        assert(strcmp(p(a{1}, 'LowerLimitSpecify'), 'off') && strcmp(p(a{1}, 'UpperLimitSpecify'), 'off'), ...
            'gs3dx:quatswap', 'Precondition: %s %s has limits', joint, a{1});
        assert(strcmp(p(a{1}, 'PositionTargetValueUnits'), 'deg') && ...
            strcmp(p(a{1}, 'DampingCoefficientUnits'), 'N*m/(deg/s)'), 'gs3dx:quatswap', ...
            'Precondition: %s %s must use deg targets and N*m/(deg/s) damping', joint, a{1});
    end
    same = @(name) local_same(joint, axes, name);
    rot.position_specify  = same('PositionTargetSpecify');
    rot.position_priority = same('PositionTargetPriority');
    rot.velocity_specify  = same('VelocityTargetSpecify');
    rot.velocity_priority = same('VelocityTargetPriority');
    rot.velocity_units    = same('VelocityTargetValueUnits');
    val = @(name) cellfun(@(a) ['(' p(a, name) ')'], axes, 'UniformOutput', false);
    ang = val('PositionTargetValue'); vel = val('VelocityTargetValue'); damp = val('DampingCoefficient');
    rot.angles   = ['[' strjoin(ang, ' ') ']'];
    rot.angle_ic = ['[' strjoin(ang, '; ') ']'];
    rot.velocity = sprintf('gs3dx_xyz_rate_matrix(%s*pi/180, %s*pi/180)*[%s]', ang{2}, ang{3}, strjoin(vel, '; '));
    rot.damping  = ['[' strjoin(damp, '; ') ']'];
end

function v = local_same(joint, axes, name)
    v = get_param(joint, [axes{1} name]);
    for a = axes(2:end)
        assert(strcmp(get_param(joint, [a{1} name]), v), 'gs3dx:quatswap', ...
            'Precondition: %s needs one %s for Rx, Ry, Rz (a quaternion joint has one)', joint, name);
    end
end

function conv = local_only_converter(g, port, joint)
% The single converter block on the far side of physical port PORT.
    far = local_partners(g, port);
    assert(isscalar(far), 'gs3dx:quatswap', ...
        'Precondition: %s port must connect to exactly one block', joint);
    conv = get_param(get_param(far, 'Parent'), 'Handle');
    ref = strrep(get_param(conv, 'ReferenceBlock'), newline, ' ');
    assert(any(strcmp(ref, {'nesl_utility/PS-Simulink Converter', 'nesl_utility/Simulink-PS Converter'})), ...
        'gs3dx:quatswap', 'Precondition: %s is not a converter', getfullname(conv));
end

function keep = local_kept(g, ports, pairs)
% Rows [new index, partner port] for each kept [old new] pair that is connected.
    keep = zeros(0, 2);
    for r = 1:size(pairs, 1)
        far = local_partners(g, ports(pairs(r, 1)));
        if ~isempty(far)
            keep(end+1, :) = [pairs(r, 2), far(1)]; %#ok<AGROW>
        end
    end
end

function local_feed(sys, src, dst, name)
% Branches share one signal, so the name is set once (on the first branch);
% naming every branch prints the label once per branch, on top of itself.
    for k = 1:numel(dst)
        line = add_line(sys, src, dst(k), 'autorouting', 'on');
        if k == 1 && ~isempty(name)
            set_param(line, 'Name', name);
        end
    end
end

function h = local_matlab_function(sys, name, position, lines)
    h = add_block('simulink/User-Defined Functions/MATLAB Function', [sys '/' name], 'Position', position);
    chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', [sys '/' name]);
    chart.Script = strjoin(lines, newline);
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end

function far = local_partners(g, p)
    n = g.net(g.port == p);
    far = [];
    if n > 0
        far = g.port(g.net == n & g.port ~= p);
    end
end

function ref = local_ref(blk)
    ref = strrep(get_param(blk, 'ReferenceBlock'), newline, ' ');
end

function gs3dx_gimbal_to_spherical(mdl)
%GS3DX_GIMBAL_TO_SPHERICAL  Swap a KD subsystem's Gimbal Joint for a Spherical Joint.
%
%   GS3DX_GIMBAL_TO_SPHERICAL(MDL) edits the loaded kinetically driven
%   subsystem MDL (a copy of GS3DX_KDS_Gimbal) so that its joint is a
%   quaternion-based Spherical Joint while its interface stays the same
%   (#10955):
%     - the Torque X/Y/Z inports keep their Gimbal-axis meaning; the
%       'XYZ Torque' block maps them to a follower-frame torque vector
%       T = E^-T (tau - damping .* rates)  (GS3DX_XYZ_MAP);
%     - the bus keeps every element and name: Euler angles, rates and
%       accelerations come from 'XYZ Kinematics' (quaternion, angular
%       velocity and acceleration -> X-Y-Z angles, with 'Angle Reference'
%       integrating the rates only to pick the 360-degree branch), and the
%       actuator torques are the inputs themselves;
%     - the start targets become a follower-axes X-Y-Z rotation sequence and
%       the equivalent follower-frame angular velocity; the mask's per-axis
%       priorities must agree, because a Spherical Joint has one target.
%   The caller saves.
%
%   Preconditions (checked before anything is edited): MDL/Kinetically
%   Driven is a Gimbal Joint with torque inputs and full per-axis sensing,
%   each per-axis converter and torque converter is wired where the
%   Gimbal layout says, and the composite sensing converters are attached.
%   Postconditions: no Gimbal Joint remains, every Bus Creator input is
%   connected, and the physical nets satisfy GS3DX_REDRAW_NETS.

    arguments
        mdl (1,:) char
    end
    lib.gimbal    = 'sm_lib/Joints/Gimbal Joint';
    lib.spherical = 'sm_lib/Joints/Spherical Joint';
    lib.ps_out    = 'nesl_utility/PS-Simulink Converter';
    joint = [mdl '/Kinetically Driven'];
    assert(strcmp(local_ref(joint), lib.gimbal), 'gs3dx:spherical', ...
        'Precondition: %s is not a Gimbal Joint', joint);
    jp = get_param(joint, 'PortHandles');
    assert(numel(jp.LConn) == 4 && numel(jp.RConn) == 17, 'gs3dx:spherical', ...
        'Precondition: %s needs 3 torque inputs and q/w/b/t sensing per axis plus 4 composite outputs', joint);
    g = gs3dx_physical_graph(mdl);
    bus = find_system(mdl, 'SearchDepth', 1, 'BlockType', 'BusCreator');
    assert(isscalar(bus), 'gs3dx:spherical', 'Precondition: %s needs exactly one Bus Creator', mdl);
    bus = bus{1};                              % its name contains a newline
    xyz = 'XYZ';

    % Plan (read only).  Gimbal RConn: F, then [q w b t] per axis, then
    % constraint force/torque and total force/torque.
    per_axis = {'AngularPosition', 'AngularVelocity', 'AngularAcceleration', 'ActuatorTorque'};
    bus_port = containers.Map();               % converter name -> Bus Creator input
    bus_line = containers.Map();               % converter name -> its line to the bus
    line_name = containers.Map();              % converter name -> that line's name
    removed = get_param(joint, 'Handle');
    for a = 1:3
        for k = 1:4
            name = [per_axis{k} xyz(a)];
            cp = get_param([mdl '/' name], 'PortHandles');
            assert(isequal(local_partners(g, jp.RConn(1 + 4 * (a - 1) + k)), cp.LConn(1)), ...
                'gs3dx:spherical', 'Precondition: %s is not on Gimbal port R%d', name, 1 + 4 * (a - 1) + k);
            line = get_param(cp.Outport(1), 'Line');
            dst = get_param(line, 'DstPortHandle');
            assert(isscalar(dst) && strcmp(get_param(dst, 'Parent'), bus), 'gs3dx:spherical', ...
                'Precondition: %s must feed only the Bus Creator', name);
            bus_port(name) = get_param(dst, 'PortNumber');
            bus_line(name) = line;
            line_name(name) = get_param(line, 'Name');
            removed(end+1) = get_param([mdl '/' name], 'Handle'); %#ok<AGROW>
        end
    end
    torque_lines = zeros(1, 3);
    for a = 1:3
        ip = get_param([mdl '/Torque ' xyz(a)], 'PortHandles');
        torque_lines(a) = get_param(ip.Outport(1), 'Line');
        conv = get_param(torque_lines(a), 'DstBlockHandle');
        cp = get_param(conv, 'PortHandles');
        assert(isscalar(conv) && isequal(local_partners(g, cp.RConn(1)), jp.LConn(1 + a)), ...
            'gs3dx:spherical', 'Precondition: Torque %s must drive Gimbal torque port %d', xyz(a), a);
        removed(end+1) = conv; %#ok<AGROW>
    end
    composite = {'ConstraintForce', 'ConstraintTorque', 'ForceLocal', 'TorqueLocal'};
    composite_port = zeros(1, 4);
    for k = 1:4
        cp = get_param([mdl '/' composite{k}], 'PortHandles');
        assert(isequal(local_partners(g, jp.RConn(13 + k)), cp.LConn(1)), 'gs3dx:spherical', ...
            'Precondition: %s is not on Gimbal port R%d', composite{k}, 13 + k);
        composite_port(k) = cp.LConn(1);
    end
    % Composite force/torque sensing keeps its measured direction and frame.
    wrench = {'CompositeWrenchDir', get_param(joint, 'CompositeWrenchDir'), ...
              'CompositeWrenchFrame', get_param(joint, 'CompositeWrenchFrame')};
    base = local_partners(g, jp.LConn(1));
    follower = local_partners(g, jp.RConn(1));
    assert(~isempty(base) && ~isempty(follower), 'gs3dx:spherical', ...
        'Precondition: %s must be connected on both frames', joint);

    % New blocks.
    at = get_param(joint, 'Position');
    place = @(dx, dy, w, h) [at(1) + dx, at(2) + dy, at(1) + dx + w, at(2) + dy + h];
    sph = add_block(lib.spherical, [mdl '/Spherical Joint'], 'Position', place(0, 0, 60, 90));
    set_param(sph, 'TorqueActuationMode', 'InputTorque', 'ActuateTorque', 'on', ...
        'ActuationFrame', 'FollowerFrame', 'SensingFrame', 'FollowerFrame', ...
        'SensePosition', 'on', 'SenseVelocity', 'on', 'SenseAcceleration', 'on', ...
        'SenseConstraintForce', 'on', 'SenseConstraintTorque', 'on', ...
        'SenseTotalForce', 'on', 'SenseTotalTorque', 'on', wrench{:}, ...
        'PositionTargetRotationMethod', 'RotationSequence', ...
        'PositionTargetRotationSequenceAxes', 'FollowerAxes', ...
        'PositionTargetRotationSequence', 'XYZ', ...
        'PositionTargetRotationSequenceAngles', '[StartPositionX StartPositionY StartPositionZ]', ...
        'PositionTargetRotationSequenceAnglesUnits', 'deg', ...
        'VelocityTargetValue', ['gs3dx_xyz_rate_matrix(StartPositionY*pi/180, StartPositionZ*pi/180)' ...
                                '*[StartVelocityX; StartVelocityY; StartVelocityZ]'], ...
        'VelocityTargetValueUnits', 'deg/s', 'VelocityTargetInFollowerFrame', 'on');
    sp = get_param(sph, 'PortHandles');
    assert(numel(sp.LConn) == 2 && numel(sp.RConn) == 8, 'gs3dx:spherical', ...
        'Spherical Joint ports are not [B t] and [F Q w b fc tc ft tt]');

    sense = {'Quaternion', '1'; 'FollowerAngularVelocity', 'rad/s'; 'FollowerAngularAcceleration', 'rad/s^2'};
    sense_port = zeros(3, 2);                  % [physical in, Simulink out]
    for k = 1:3
        h = add_block(lib.ps_out, [mdl '/' sense{k, 1}], 'Unit', sense{k, 2}, ...
            'Position', place(120, 120 + 50 * k, 40, 30));
        ph = get_param(h, 'PortHandles');
        sense_port(k, :) = [ph.LConn(1), ph.Outport(1)];
    end
    % The new torque converter copies an old one's input handling.
    tconv = add_block(getfullname(removed(end - 2)), [mdl '/FollowerTorque'], ...
        'Unit', 'N*m', 'Position', place(-120, 120, 40, 30));
    tp = get_param(tconv, 'PortHandles');
    mux = add_block('simulink/Signal Routing/Mux', [mdl '/Axis Torques'], 'Inputs', '3', ...
        'Position', place(-320, 100, 5, 60));
    damping = add_block('simulink/Sources/Constant', [mdl '/Axis Damping'], 'Value', ...
        '[DampeningX; DampeningY; DampeningZ]*LocalDampeningEnable*DampeningGlobalGain', ...
        'SampleTime', '0', ...                 % constant time would break the input-function loop
        'Position', place(-320, 200, 60, 30));
    reference = add_block('simulink/Continuous/Integrator', [mdl '/Angle Reference'], ...
        'InitialCondition', '[StartPositionX; StartPositionY; StartPositionZ]', ...
        'Position', place(420, 340, 30, 30));
    torque_fn = local_matlab_function(mdl, 'XYZ Torque', place(-220, 100, 80, 90), { ...
        'function T = fcn(Q, w, tau, damping)', ...
        '% Gimbal-axis torques -> follower-frame torque (gs3dx_xyz_map).', ...
        'T = gs3dx_xyz_map(Q, w, zeros(3, 1), tau, damping, zeros(3, 1));'});
    kin_fn = local_matlab_function(mdl, 'XYZ Kinematics', place(280, 180, 80, 90), { ...
        'function [ang, qd, qdd] = fcn(Q, w, b, ang_ref)', ...
        '% Quaternion state -> Gimbal X-Y-Z angles, rates, accelerations (deg).', ...
        '[~, ang, qd, qdd] = gs3dx_xyz_map(Q, w, b, zeros(3, 1), zeros(3, 1), ang_ref);'});
    demux = zeros(1, 3);
    for k = 1:3
        demux(k) = add_block('simulink/Signal Routing/Demux', ...
            sprintf('%s/%s Demux', mdl, per_axis{k}), 'Outputs', '3', 'Position', place(460, 60 + 110 * k, 5, 60));
    end

    % Simulink wiring: delete the old signal lines, then draw the new ones.
    cellfun(@(name) delete_line(bus_line(name)), keys(bus_line));
    arrayfun(@delete_line, torque_lines);
    port = @(h, kind, n) local_port(h, kind, n);
    for a = 1:3
        ip = port(get_param([mdl '/Torque ' xyz(a)], 'Handle'), 'Outport', 1);
        add_line(mdl, ip, port(mux, 'Inport', a), 'autorouting', 'on');
        line = add_line(mdl, ip, port(get_param(bus, 'Handle'), 'Inport', bus_port(['ActuatorTorque' xyz(a)])), ...
            'autorouting', 'on');
        set_param(line, 'Name', ['ActuatorTorque' xyz(a)]);
    end
    add_line(mdl, sense_port(1, 2), port(torque_fn, 'Inport', 1), 'autorouting', 'on');
    add_line(mdl, sense_port(2, 2), port(torque_fn, 'Inport', 2), 'autorouting', 'on');
    add_line(mdl, port(mux, 'Outport', 1), port(torque_fn, 'Inport', 3), 'autorouting', 'on');
    add_line(mdl, port(damping, 'Outport', 1), port(torque_fn, 'Inport', 4), 'autorouting', 'on');
    add_line(mdl, port(torque_fn, 'Outport', 1), tp.Inport(1), 'autorouting', 'on');
    for k = 1:3
        add_line(mdl, sense_port(k, 2), port(kin_fn, 'Inport', k), 'autorouting', 'on');
    end
    add_line(mdl, port(reference, 'Outport', 1), port(kin_fn, 'Inport', 4), 'autorouting', 'on');
    add_line(mdl, port(kin_fn, 'Outport', 2), port(reference, 'Inport', 1), 'autorouting', 'on');
    for k = 1:3
        add_line(mdl, port(kin_fn, 'Outport', k), port(demux(k), 'Inport', 1), 'autorouting', 'on');
        for a = 1:3
            name = [per_axis{k} xyz(a)];
            line = add_line(mdl, port(demux(k), 'Outport', a), ...
                port(get_param(bus, 'Handle'), 'Inport', bus_port(name)), 'autorouting', 'on');
            set_param(line, 'Name', line_name(name));
        end
    end

    % Physical wiring: remove the Gimbal and its axis converters, attach the
    % Spherical Joint to the same frames and composite sensors.
    joins = [sp.LConn(1), base(1); sp.RConn(1), follower(1); tp.RConn(1), sp.LConn(2); ...
             sp.RConn(2), sense_port(1, 1); sp.RConn(3), sense_port(2, 1); sp.RConn(4), sense_port(3, 1); ...
             sp.RConn(5:8).', composite_port.'];
    gs3dx_redraw_nets(mdl, removed, joins);
    set_param(sph, 'Name', 'Kinetically Driven');

    mask = Simulink.Mask.get(mdl);
    mask.Initialization = strjoin({ ...
        '% Spherical joint target binding (#10955): one priority for all axes.', ...
        'sphericalJoint = [gcb ''/Kinetically Driven''];', ...
        'for jointQuantity = {''Position'', ''Velocity''}', ...
        '  jointPriority = cellfun(@(a) get_param(gcb, [a jointQuantity{1} ''TargetPriority'']), ...', ...
        '    {''Rx'', ''Ry'', ''Rz''}, ''UniformOutput'', false);', ...
        '  assert(numel(unique(jointPriority)) == 1, ''Spherical joint needs one %s target priority'', jointQuantity{1});', ...
        '  if strcmp(jointPriority{1}, ''None'')', ...
        '    set_param(sphericalJoint, [jointQuantity{1} ''TargetSpecify''], ''off'');', ...
        '  else', ...
        '    set_param(sphericalJoint, [jointQuantity{1} ''TargetSpecify''], ''on'', ...', ...
        '      [jointQuantity{1} ''TargetPriority''], jointPriority{1});', ...
        '  end', ...
        'end'}, newline);

    assert(isempty(find_system(mdl, 'SearchDepth', 1, 'ReferenceBlock', lib.gimbal)), ...
        'gs3dx:spherical', 'Postcondition: %s still contains a Gimbal Joint', mdl);
    bp = get_param(bus, 'PortHandles');
    assert(all(arrayfun(@(p) get_param(p, 'Line') > 0 && ...
        get_param(get_param(p, 'Line'), 'SrcPortHandle') > 0, bp.Inport)), ...
        'gs3dx:spherical', 'Postcondition: a Bus Creator input of %s is unconnected', mdl);
end

function h = local_matlab_function(mdl, name, position, lines)
    h = add_block('simulink/User-Defined Functions/MATLAB Function', [mdl '/' name], 'Position', position);
    chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', [mdl '/' name]);
    chart.Script = strjoin(lines, newline);
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end

function far = local_partners(g, p)
% Other ports on the net of physical port P.
    n = g.net(g.port == p);
    far = [];
    if n > 0
        far = g.port(g.net == n & g.port ~= p);
    end
end

function ref = local_ref(blk)
    ref = strrep(get_param(blk, 'ReferenceBlock'), newline, ' ');
end

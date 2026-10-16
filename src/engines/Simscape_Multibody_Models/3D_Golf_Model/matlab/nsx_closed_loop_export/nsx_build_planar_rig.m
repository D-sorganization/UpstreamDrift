function modelFile = nsx_build_planar_rig(specFile, outDir)
%NSX_BUILD_PLANAR_RIG  Build the NSX-55 planar two-arm closed-loop test rig.
%
%   MODELFILE = NSX_BUILD_PLANAR_RIG(SPECFILE, OUTDIR) builds the Simscape
%   Multibody model NSX_PlanarRig.slx in OUTDIR from the parameters in the JSON
%   SPECFILE (field "rig", written by src/shared/python/loop_nullspace/
%   simscape_rig_spec.py) and returns its path.
%
%   The rig is the NSX-00 planar golfer (docs/research/loop_nullspace) as an
%   OPEN TREE: two serial arms hanging from the World frame, with the club
%   attached to the lead forearm.  The trail hand is NOT welded to the club
%   grip point; instead a Transform Sensor reads the grip-minus-hand offset
%   Phi(q) = H_R - G, which is the closure the exporter differentiates.
%   Coordinates (revolute joints about +z, relative angles) are
%       q = [lead_shoulder lead_elbow club_angle trail_shoulder trail_elbow].
%
%   Probe contract used by NSX_SIMSCAPE_PROBE (base-workspace variables set per
%   run through Simulink.SimulationInput):
%       nsx_q (5x1)    joint position targets, rad (priority High)
%       nsx_v (5x1)    joint velocity targets, rad/s (priority High)
%       nsx_tau (5x1)  joint input torques, N*m
%   Outports (in order): acc_1..acc_5 joint accelerations (rad/s^2), phi,
%   phidot, phiddot (3x1 closure offset and its derivatives in World, m),
%   grip (3x1 World position of the grip point G, m).
%
%   This is a test rig for the exporter and the loader, NOT the golf model.

    arguments
        specFile (1,:) char
        outDir   (1,:) char
    end
    spec = jsondecode(fileread(specFile));
    p = spec.rig;
    n = 5;
    mdl = 'NSX_PlanarRig';
    if bdIsLoaded(mdl), close_system(mdl, 0); end
    new_system(mdl);
    cleaner = onCleanup(@() close_system_quiet(mdl));
    set_param(mdl, 'Solver', 'ode23t', 'StopTime', '0', 'SaveOutput', 'on', ...
        'ReturnWorkspaceOutputs', 'on');

    lib = 'sm_lib/';
    world = add_block([lib 'Frames and Transforms/World Frame'], [mdl '/World']);
    mech = add_block([lib 'Utilities/Mechanism Configuration'], [mdl '/Mech']);
    set_param(mech, 'UniformGravity', 'Constant', ...
        'GravityVector', sprintf('[0 %.17g 0]', -p.gravity), ...
        'GravityVectorUnits', 'm/s^2');
    solver = add_block('nesl_utility/Solver Configuration', [mdl '/Solver']);
    wnode = port_of(world, 'RConn', 1);
    link(mdl, wnode, port_of(mech, 'RConn', 1));
    link(mdl, wnode, port_of(solver, 'RConn', 1));

    sep = p.shoulder_separation;
    lu = p.upper_arm_length;  lf = p.forearm_length;
    % ---- lead arm and club -------------------------------------------------
    [j1, n1] = shoulder(mdl, 'LeadShoulder', wnode, [-sep/2 0 0], 1);
    attach_rod(mdl, 'LeadUpper', n1, lu/2, p.upper_arm_mass, p.upper_arm_mass*lu^2/12);
    [j2, n2] = joint_after(mdl, 'LeadElbow', n1, [0 -lu 0], 2); %#ok<ASGLU>
    attach_rod(mdl, 'LeadFore', n2, lf/2, p.forearm_mass, p.forearm_mass*lf^2/12);
    [j3, n3] = joint_after(mdl, 'ClubJoint', n2, [0 -lf 0], 3); %#ok<ASGLU>
    attach_rod(mdl, 'Club', n3, p.club_com_from_lead_hand, p.club_mass, ...
        p.club_mass*p.club_length^2/12);
    gnode = frame_after(mdl, 'GripPoint', n3, [0 -p.trail_grip_from_lead_hand 0]);
    % ---- trail arm ---------------------------------------------------------
    [j4, n4] = shoulder(mdl, 'TrailShoulder', wnode, [sep/2 0 0], 4); %#ok<ASGLU>
    attach_rod(mdl, 'TrailUpper', n4, lu/2, p.upper_arm_mass, p.upper_arm_mass*lu^2/12);
    [j5, n5] = joint_after(mdl, 'TrailElbow', n4, [0 -lu 0], 5); %#ok<ASGLU>
    attach_rod(mdl, 'TrailFore', n5, lf/2, p.forearm_mass, p.forearm_mass*lf^2/12);
    hnode = frame_after(mdl, 'TrailHand', n5, [0 -lf 0]);

    % ---- sensors: closure offset H_R - G, its rates, and G in World ---------
    sensor(mdl, 'Phi',     'SenseXYZ',      'm',     gnode, hnode, n + 1);
    sensor(mdl, 'PhiDot',  'SenseXYZDot',   'm/s',   gnode, hnode, n + 2);
    sensor(mdl, 'PhiDDot', 'SenseXYZDDot',  'm/s^2', gnode, hnode, n + 3);
    sensor(mdl, 'Grip',    'SenseXYZ',      'm',     wnode, gnode, n + 4);

    if ~isfolder(outDir), mkdir(outDir); end
    modelFile = fullfile(outDir, [mdl '.slx']);
    save_system(mdl, modelFile);
end

% ----------------------------------------------------------------- helpers --
function close_system_quiet(mdl)
    if bdIsLoaded(mdl), close_system(mdl, 0); end
end

function h = port_of(block, kind, idx)
    ph = get_param(block, 'PortHandles');
    h = ph.(kind)(idx);
end

function link(mdl, a, b)
    add_line(mdl, a, b, 'autorouting', 'on');
end

function node = frame_after(mdl, name, parentNode, offset)
% A Rigid Transform whose follower frame is the returned node.
    rt = add_block('sm_lib/Frames and Transforms/Rigid Transform', [mdl '/' name]);
    set_param(rt, 'TranslationMethod', 'Cartesian', ...
        'TranslationCartesianOffset', mat2str(offset, 17), ...
        'TranslationCartesianOffsetUnits', 'm');
    link(mdl, parentNode, port_of(rt, 'LConn', 1));
    node = port_of(rt, 'RConn', 1);
end

function attach_rod(mdl, name, parentNode, comDist, mass, izz)
% Rod body: an Inertia block at the centre of mass (planar motion: only the
% z moment matters; the other two are set equal so the inertia is valid).
    com = frame_after(mdl, [name 'COM'], parentNode, [0 -comDist 0]);
    ib = add_block('sm_lib/Body Elements/Inertia', [mdl '/' name 'Inertia']);
    set_param(ib, 'InertiaType', 'Custom', 'Mass', sprintf('%.17g', mass), ...
        'MassUnits', 'kg', 'CenterOfMass', '[0 0 0]', 'CenterOfMassUnits', 'm', ...
        'MomentsOfInertia', sprintf('[%.17g %.17g %.17g]', izz, izz, izz), ...
        'MomentsOfInertiaUnits', 'kg*m^2', 'ProductsOfInertia', '[0 0 0]', ...
        'ProductsOfInertiaUnits', 'kg*m^2');
    link(mdl, com, port_of(ib, 'RConn', 1));
end

function [jblk, node] = shoulder(mdl, name, parentNode, offset, k)
    [jblk, node] = joint_after(mdl, name, parentNode, offset, k);
end

function [jblk, node] = joint_after(mdl, name, parentNode, offset, k)
% Rigid Transform (offset) -> revolute joint k (torque in, acceleration out).
    base = frame_after(mdl, [name 'Offset'], parentNode, offset);
    jblk = add_block('sm_lib/Joints/Revolute Joint', [mdl '/' name]);
    set_param(jblk, 'TorqueActuationMode', 'InputTorque', ...
        'SenseAcceleration', 'on', ...
        'PositionTargetSpecify', 'on', 'PositionTargetPriority', 'High', ...
        'PositionTargetValue', sprintf('nsx_q(%d)', k), ...
        'PositionTargetValueUnits', 'rad', ...
        'VelocityTargetSpecify', 'on', 'VelocityTargetPriority', 'High', ...
        'VelocityTargetValue', sprintf('nsx_v(%d)', k), ...
        'VelocityTargetValueUnits', 'rad/s');
    link(mdl, base, port_of(jblk, 'LConn', 1));
    node = port_of(jblk, 'RConn', 1);
    % torque input
    const = add_block('simulink/Sources/Constant', [mdl '/' name 'Tau']);
    set_param(const, 'Value', sprintf('nsx_tau(%d)', k));
    s2ps = add_block('nesl_utility/Simulink-PS Converter', [mdl '/' name 'TauPS']);
    set_param(s2ps, 'Unit', 'N*m');
    add_line(mdl, port_of(const, 'Outport', 1), port_of(s2ps, 'Inport', 1));
    link(mdl, port_of(s2ps, 'RConn', 1), port_of(jblk, 'LConn', 2));
    % acceleration out
    out_port(mdl, [name 'Acc'], port_of(jblk, 'RConn', 2), 'rad/s^2', k);
end

function out_port(mdl, name, psPort, unit, idx)
    ps2s = add_block('nesl_utility/PS-Simulink Converter', [mdl '/' name 'Conv']);
    set_param(ps2s, 'Unit', unit);
    link(mdl, psPort, port_of(ps2s, 'LConn', 1));
    ob = add_block('simulink/Sinks/Out1', [mdl '/' name]);
    set_param(ob, 'Port', num2str(idx));
    add_line(mdl, port_of(ps2s, 'Outport', 1), port_of(ob, 'Inport', 1));
end

function sensor(mdl, name, flag, unit, baseNode, followNode, idx)
    ts = add_block('sm_lib/Frames and Transforms/Transform Sensor', [mdl '/' name 'TS']);
    set_param(ts, 'MeasurementFrame', 'World', flag, 'on');
    link(mdl, baseNode, port_of(ts, 'LConn', 1));
    link(mdl, followNode, port_of(ts, 'RConn', 1));
    out_port(mdl, name, port_of(ts, 'RConn', 2), unit, idx);
end

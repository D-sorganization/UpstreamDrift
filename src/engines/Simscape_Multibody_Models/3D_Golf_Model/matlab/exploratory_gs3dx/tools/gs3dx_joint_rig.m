function out = gs3dx_joint_rig(subsys, opts)
%GS3DX_JOINT_RIG  Drive one KD subsystem in isolation and return its bus.
%
%   OUT = GS3DX_JOINT_RIG(SUBSYS) builds an in-memory model (never saved)
%   in which the referenced subsystem SUBSYS carries an offset brick under
%   gravity from the World frame, with a sine torque on each of its Torque
%   X/Y/Z inputs, simulates it and returns:
%     .t        (N,1) output times (opts.output_times)
%     .signals  struct of (N,k) arrays, one field per SignalBus element
%     .wall_s   wall-clock seconds of the simulation
%   This is the torqued, well-conditioned drive that compares a Gimbal
%   subsystem with its Spherical stand-in (#10955): the full model has no
%   such drive (docs/SENSITIVITY_FINDINGS.md).
%
%   Options: start_position, start_velocity, damping (1x3, deg, deg/s,
%   N*m/(deg/s)), priority ('High'|'Low'|'None' for both targets),
%   amplitude (1x3 N*m), frequency (1x3 Hz), stop_time, rel_tol,
%   output_times.

    arguments
        subsys (1,:) char
        opts.start_position (1,3) double = [20 -35 50]
        opts.start_velocity (1,3) double = [15 -10 20]
        opts.damping (1,3) double = [0.002 0.003 0.001]
        opts.priority (1,:) char {mustBeMember(opts.priority, {'High', 'Low', 'None'})} = 'High'
        opts.amplitude (1,3) double = [0.6 -0.4 0.3]
        opts.frequency (1,3) double = [1.3 2.1 0.7]
        opts.stop_time (1,1) double {mustBePositive} = 1
        opts.rel_tol (1,1) double {mustBePositive} = 1e-8
        opts.output_times (:,1) double = (0:0.005:1).'
    end
    rig = sprintf('gs3dx_rig_%s', subsys);
    if bdIsLoaded(rig)
        close_system(rig, 0);
    end
    new_system(rig);
    cleanup = onCleanup(@() close_system(rig, 0));
    add = @(lib, name, pos) add_block(lib, [rig '/' name], 'Position', pos);
    joint = add('simulink/Ports & Subsystems/Subsystem Reference', 'Joint', [200 180 300 280]);
    set_param(joint, 'ReferencedSubsystem', subsys);
    jp = get_param(joint, 'PortHandles');   % LConn: Proximal, RConn: Distal
    gs3dx_rig_scaffold(rig, jp.LConn(1), jp.RConn(1));

    xyz = 'XYZ';
    for a = 1:3
        axis = xyz(a);
        set_param(joint, ['StartPosition' axis], num2str(opts.start_position(a), 17), ...
            ['StartVelocity' axis], num2str(opts.start_velocity(a), 17), ...
            ['Dampening' axis], num2str(opts.damping(a), 17));
        src = add('simulink/Sources/Sine Wave', ['Torque ' axis], [100 60 * a 130 60 * a + 30]);
        set_param(src, 'Amplitude', num2str(opts.amplitude(a), 17), ...
            'Frequency', num2str(2 * pi * opts.frequency(a), 17), 'SampleTime', '0');
        add_line(rig, local_port(src, 'Outport', 1), jp.Inport(a), 'autorouting', 'on');
    end
    % All priorities in one call: a Spherical stand-in requires them equal.
    priorities = cellfun(@(r) {[r 'PositionTargetPriority'], opts.priority, ...
        [r 'VelocityTargetPriority'], opts.priority}, {'Rx', 'Ry', 'Rz'}, 'UniformOutput', false);
    priorities = [priorities{:}];
    set_param(joint, priorities{:});
    bus = add('simulink/Sinks/Terminator', 'SignalBus', [360 60 380 80]);
    line = add_line(rig, jp.Outport(1), local_port(bus, 'Inport', 1), 'autorouting', 'on');
    set_param(line, 'Name', 'SignalBus');

    mw = get_param(rig, 'ModelWorkspace');
    assignin(mw, 'LocalDampeningEnable', 1);
    assignin(mw, 'DampeningGlobalGain', 1);
    out = gs3dx_rig_simulate(rig, jp.Outport(1), 'SignalBus', opts);
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end

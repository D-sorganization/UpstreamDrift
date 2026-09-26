function out = gs3dx_hip_rig(mdl, opts)
%GS3DX_HIP_RIG  Drive a model's hip subsystem in isolation and return HipLogs.
%
%   OUT = GS3DX_HIP_RIG(MDL) copies 'Hips and Torso Inputs/Hip Kinetically
%   Driven' from the loaded GS3DX model MDL into an in-memory rig (never
%   saved): Base on the World frame, an offset brick on Hips under gravity,
%   sine commands on the three hip torques and three translation forces,
%   and fixed values on the other signals the subsystem reads by Goto tag.
%   Returns GS3DX_RIG_SIMULATE's struct for the HipLogs bus.  This compares
%   the Bushing hip with its 6-DOF stand-in (#10956).
%
%   Options: start_position (1x3 deg), start_velocity (1x3 deg/s),
%   translation_position (1x3 m), translation_velocity (1x3 m/s),
%   torque_amplitude (1x3 N*m), force_amplitude (1x3 N), frequency
%   (1x3 Hz), stop_time, rel_tol, output_times.

    arguments
        mdl (1,:) char
        opts.start_position (1,3) double = [15 -25 40]
        opts.start_velocity (1,3) double = [10 -15 20]
        opts.translation_position (1,3) double = [0.02 -0.01 0.03]
        opts.translation_velocity (1,3) double = [0.1 0.05 -0.1]
        opts.torque_amplitude (1,3) double = [0.1 -0.06 0.05]  % keeps the Bushing's Y angle inside +-75 deg
        opts.force_amplitude (1,3) double = [2 -1.5 1]
        opts.frequency (1,3) double = [1.3 2.1 0.7]
        opts.stop_time (1,1) double {mustBePositive} = 1
        opts.rel_tol (1,1) double {mustBePositive} = 1e-8
        opts.output_times (:,1) double = (0:0.005:1).'
    end
    source = [mdl '/Hips and Torso Inputs/Hip Kinetically Driven'];
    rig = sprintf('gs3dx_hiprig_%s', mdl);
    if bdIsLoaded(rig)
        close_system(rig, 0);
    end
    new_system(rig);
    cleanup = onCleanup(@() close_system(rig, 0));
    hip = add_block(source, [rig '/Hip'], 'Position', [200 180 300 280]);
    gs3dx_rig_scaffold(rig, gs3dx_pm_port(hip, 'Base'), gs3dx_pm_port(hip, 'Hips'));

    xyz = 'XYZ';
    feeds = {'RVectorHipJointBasetoGlobal', 'eye(3)'; 'HipGlobalPosition', 'zeros(3, 1)'; ...
             'HipGlobalVelocity', 'zeros(3, 1)'; 'HUBGlobalPosition', 'zeros(3, 1)'; ...
             'BaseonHipForceHipBase', 'zeros(3, 1)'};
    for k = 1:size(feeds, 1)
        c = add_block('simulink/Sources/Constant', [rig '/' feeds{k, 1}], 'Value', feeds{k, 2}, ...
            'Position', [20 500 + 50 * k 60 530 + 50 * k]);
        local_goto(rig, c, feeds{k, 1});
    end
    for a = 1:3
        local_sine(rig, ['FcnOutHipTorque' xyz(a)], opts.torque_amplitude(a), opts.frequency(a), a);
        local_sine(rig, ['FcnOutTranslationForce' xyz(a)], opts.force_amplitude(a), opts.frequency(a), a + 3);
    end

    mw = get_param(rig, 'ModelWorkspace');
    for a = 1:3
        assignin(mw, ['HipStartPosition' xyz(a)], opts.start_position(a));
        assignin(mw, ['HipStartVelocity' xyz(a)], opts.start_velocity(a));
        assignin(mw, ['TranslationStartPosition' xyz(a)], opts.translation_position(a));
        assignin(mw, ['TranslationStartVelocity' xyz(a)], opts.translation_velocity(a));
    end
    bus = find_system(rig, 'SearchDepth', 2, 'BlockType', 'BusCreator');
    assert(isscalar(bus), 'gs3dx:rig', 'Expected one Bus Creator in the hip subsystem');
    out = gs3dx_rig_simulate(rig, local_port(bus{1}, 'Outport', 1), 'HipLogs', opts);
end

function local_sine(rig, tag, amplitude, frequency, row)
    src = add_block('simulink/Sources/Sine Wave', [rig '/' tag], 'Amplitude', num2str(amplitude, 17), ...
        'Frequency', num2str(2 * pi * frequency, 17), 'SampleTime', '0', ...
        'Position', [20 40 * row 50 40 * row + 30]);
    local_goto(rig, src, tag);
end

function local_goto(rig, src, tag)
    pos = get_param(src, 'Position');
    g = add_block('simulink/Signal Routing/Goto', [rig '/Goto ' tag], 'GotoTag', tag, ...
        'TagVisibility', 'global', 'Position', pos + [80 0 120 0]);
    add_line(rig, local_port(src, 'Outport', 1), local_port(g, 'Inport', 1), 'autorouting', 'on');
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end

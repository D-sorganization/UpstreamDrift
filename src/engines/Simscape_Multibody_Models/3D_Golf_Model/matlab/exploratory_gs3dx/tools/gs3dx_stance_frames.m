function frames = gs3dx_stance_frames(mdl, vars, opts)
%GS3DX_STANCE_FRAMES  World pose of the pelvis and shoulders at t = 0 (and over time).
%
%   FRAMES = GS3DX_STANCE_FRAMES(MDL, VARS) adds Transform Sensors to the
%   loaded model MDL (World -> pelvis, World -> each shoulder), assembles it
%   with the model-workspace overrides VARS (StopTime 0) and returns:
%     .pelvis_R, .pelvis_p   pelvis frame ('Lower Torso') in World
%     .left_shoulder_p, .right_shoulder_p   shoulder joint base origins
%     .up                    unit vector opposite to gravity
%   MDL is closed without saving afterwards, so the sensors never persist.
%   GS3DX_BUILD_LOWER_BODY uses these to place the legs (#10957).
%
%   Options:
%     stop_time  (default 0) simulate this long; FRAMES.series then holds
%                .t (1xN), .pelvis_R (3x3xN), .pelvis_p (3xN) and, with
%                mass=true, .mass and .com (3xN, whole mechanism, World),
%                and FRAMES.out is the SimulationOutput (#10986).
%     mass       (default false) add a whole-mechanism Inertia Sensor.

    arguments
        mdl (1,:) char
        vars (1,1) struct = struct()
        opts.stop_time (1,1) double {mustBeNonnegative} = 0
        opts.mass (1,1) logical = false
    end
    if ~bdIsLoaded(mdl)
        load_system(mdl);
    end
    cleanup = onCleanup(@() close_system(mdl, 0));
    hips = [mdl '/Hips and Torso Inputs'];
    world = gs3dx_pm_port(hips, 'GlobalReferenceFrame');
    targets = {'pelvis_R', gs3dx_pm_port(hips, 'Lower Torso'), 'SenseR', '1'; ...
               'pelvis_p', gs3dx_pm_port(hips, 'Lower Torso'), 'SenseXYZ', 'm'; ...
               'left_shoulder_p', gs3dx_pm_port([mdl '/Left Shoulder Joint'], 'Left Shoulder'), 'SenseXYZ', 'm'; ...
               'right_shoulder_p', gs3dx_pm_port([mdl '/Right Shoulder Joint'], 'Right Shoulder'), 'SenseXYZ', 'm'};
    for k = 1:size(targets, 1)
        y = 2000 + 80 * k;
        s = add_block('sm_lib/Frames and Transforms/Transform Sensor', sprintf('%s/GS3DX Stance Sensor %d', mdl, k), ...
            'MeasurementFrame', 'World', targets{k, 3}, 'on', 'Position', [100 y 160 y + 50]);
        c = add_block('nesl_utility/PS-Simulink Converter', sprintf('%s/GS3DX Stance Converter %d', mdl, k), ...
            'Unit', targets{k, 4}, 'Position', [220 y 250 y + 30]);
        t = add_block('simulink/Sinks/Terminator', sprintf('%s/GS3DX Stance Terminator %d', mdl, k), ...
            'Position', [300 y 320 y + 20]);
        sp = get_param(s, 'PortHandles'); cp = get_param(c, 'PortHandles'); tp = get_param(t, 'PortHandles');
        add_line(mdl, world, sp.LConn(1));
        add_line(mdl, targets{k, 2}, sp.RConn(1));
        add_line(mdl, sp.RConn(2), cp.LConn(1));
        h = add_line(mdl, cp.Outport(1), tp.Inport(1));
        set_param(h, 'Name', targets{k, 1});
        set_param(cp.Outport(1), 'DataLogging', 'on');
    end
    if opts.mass
        local_add_inertia_sensor(mdl, world);
    end

    in = Simulink.SimulationInput(mdl);
    for f = reshape(fieldnames(vars), 1, [])
        in = in.setVariable(f{1}, vars.(f{1}), 'Workspace', mdl);
    end
    in = in.setModelParameter('StopTime', num2str(opts.stop_time), 'SignalLogging', 'on', ...
        'SignalLoggingName', 'logsout', 'ReturnWorkspaceOutputs', 'on');
    out = sim(in);
    assert(isempty(out.ErrorMessage), 'gs3dx:stance', '%s', out.ErrorMessage);
    for k = 1:size(targets, 1)
        v = out.logsout.get(targets{k, 1}).Values;
        frames.(targets{k, 1}) = local_first_sample(v.Data, numel(v.Time));
    end
    g = str2num(get_param([hips '/Mechanism Configuration'], 'GravityVector')); %#ok<ST2NM> vector literal
    frames.up = -g(:) / norm(g);
    if opts.stop_time > 0
        frames.series = local_series(out.logsout, opts.mass);
        frames.out = out;
    end
end

function local_add_inertia_sensor(mdl, world)
% Whole-mechanism mass and centre of mass, resolved in World.
    s = add_block('sm_lib/Body Elements/Inertia Sensor', [mdl '/GS3DX Mass Sensor'], ...
        'SensorExtent', 'Mechanism', 'SenseMass', 'on', 'SenseCenterOfMass', 'on', ...
        'SenseInertiaMatrix', 'off', 'Position', [100 2500 160 2550]);
    sp = get_param(s, 'PortHandles');
    add_line(mdl, world, sp.LConn(1));
    names = {'total_mass', 'com_p'};
    units = {'kg', 'm'};
    assert(numel(sp.RConn) == numel(names), 'gs3dx:stance', 'Unexpected Inertia Sensor outputs');
    for k = 1:numel(names)
        y = 2500 + 60 * k;
        c = add_block('nesl_utility/PS-Simulink Converter', sprintf('%s/GS3DX Mass Converter %d', mdl, k), ...
            'Unit', units{k}, 'Position', [220 y 250 y + 30]);
        t = add_block('simulink/Sinks/Terminator', sprintf('%s/GS3DX Mass Terminator %d', mdl, k), ...
            'Position', [300 y 320 y + 20]);
        cp = get_param(c, 'PortHandles'); tp = get_param(t, 'PortHandles');
        add_line(mdl, sp.RConn(k), cp.LConn(1));
        h = add_line(mdl, cp.Outport(1), tp.Inport(1));
        set_param(h, 'Name', names{k});
        set_param(cp.Outport(1), 'DataLogging', 'on');
    end
end

function s = local_series(logsout, with_mass)
    R = logsout.get('pelvis_R').Values;
    s.t = reshape(R.Time, 1, []);
    s.pelvis_R = local_stack(R.Data, numel(s.t), [3 3]);
    s.pelvis_p = local_stack(logsout.get('pelvis_p').Values.Data, numel(s.t), 3);
    if with_mass
        m = local_stack(logsout.get('total_mass').Values.Data, numel(s.t), 1);
        s.mass = m(1);
        s.com = local_stack(logsout.get('com_p').Values.Data, numel(s.t), 3);
    end
end

function x = local_stack(d, n, shape)
% Logged data D with N samples as shape-by-N (3xN) or 3x3xN.
    if numel(shape) == 2
        x = reshape(d, 3, 3, n);
    elseif size(d, 1) == n && n > 1
        x = reshape(d, n, []).';
    else
        x = reshape(d, [], n);
    end
end

function x = local_first_sample(d, n)
% First time sample of logged data D with N samples (time on the first
% axis for vectors logged as N-by-3, on the last axis otherwise).
    if n > 1 && size(d, 1) == n && ismatrix(d)
        x = reshape(d(1, :), [], 1);
    elseif ndims(d) == 3
        x = d(:, :, 1);
    else
        x = d;
    end
    if numel(x) == 3
        x = reshape(x, 3, 1);
    end
end

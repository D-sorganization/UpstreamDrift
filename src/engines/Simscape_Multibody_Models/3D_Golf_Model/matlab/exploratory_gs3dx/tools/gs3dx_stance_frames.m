function frames = gs3dx_stance_frames(mdl, vars)
%GS3DX_STANCE_FRAMES  World pose of the pelvis and shoulders at t = 0.
%
%   FRAMES = GS3DX_STANCE_FRAMES(MDL, VARS) adds Transform Sensors to the
%   loaded model MDL (World -> pelvis, World -> each shoulder), assembles it
%   with the model-workspace overrides VARS (StopTime 0) and returns:
%     .pelvis_R, .pelvis_p   pelvis frame ('Lower Torso') in World
%     .left_shoulder_p, .right_shoulder_p   shoulder joint base origins
%     .up                    unit vector opposite to gravity
%   MDL is closed without saving afterwards, so the sensors never persist.
%   GS3DX_BUILD_LOWER_BODY uses these to place the legs (#10957).

    arguments
        mdl (1,:) char
        vars (1,1) struct = struct()
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

    in = Simulink.SimulationInput(mdl);
    for f = reshape(fieldnames(vars), 1, [])
        in = in.setVariable(f{1}, vars.(f{1}), 'Workspace', mdl);
    end
    in = in.setModelParameter('StopTime', '0', 'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    out = sim(in);
    assert(isempty(out.ErrorMessage), 'gs3dx:stance', '%s', out.ErrorMessage);
    for k = 1:size(targets, 1)
        v = out.logsout.get(targets{k, 1}).Values;
        frames.(targets{k, 1}) = local_first_sample(v.Data, numel(v.Time));
    end
    g = str2num(get_param([hips '/Mechanism Configuration'], 'GravityVector')); %#ok<ST2NM> vector literal
    frames.up = -g(:) / norm(g);
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

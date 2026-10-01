function report = gs3dx_build_fit_track(info, ref, opts)
%GS3DX_BUILD_FIT_TRACK  Build GS3DX_FitTrack: GS3DX_FitLegs with the upper body tracking the capture.
%
%   REPORT = GS3DX_BUILD_FIT_TRACK(INFO, REF) copies GS3DX_FitLegs to
%   GS3DX_FitTrack and makes its twelve upper-body joints track REF
%   (GS3DX_UPPER_BODY_REFERENCE of the IK that GS3DX_FitLegs' leg reference
%   came from, #10979):
%
%   * Charts.  Each upper-body '<J> Input Function' MATLAB Function gains
%     seven parameters and, when UpperBodyTracking is nonzero, replaces its
%     ModelingMode torque by GS3DX_TRACK_TORQUE:
%       <J>TrackTorque(t) + <J>TrackKp (<J>TrackAngle(t) - q) + <J>TrackKd (<J>TrackRate(t) - qd)
%     on UpperBodyTrackTime, before anything else in the chart reads the
%     torque.  No block is added (asserted).  The Hip and Translation charts
%     are not edited: with the drive's ModelingMode 0 the pelvis joint
%     carries no drive, so the legs hold the body up.
%   * Chart inputs.  Every angle and rate input <J>Position/Velocity[axis]
%     is read From its joint's global Goto <J>AngularPosition/Velocity[axis];
%     the original wiring of LE, LF, LW, RF and RW is retargeted.
%   * Feedforward.  <J>TrackTorque is zero unless FEEDFORWARD (struct of
%     prefix -> axes x frames, N*m) is given; GS3DX_TRACK_LEARN learns it.
%   * Start state.  The upper-body <J>StartPosition/Velocity* variables are
%     REF's first frame; TrackStart holds them with every LegReferenceStart
%     variable, for the caller to pass after the drive.
%
%   Options: overwrite (false), feedforward (struct()), gains (struct of
%   prefix -> [Kp Kd] per axis in N*m/deg and N*m/(deg/s); defaults below).
%   REPORT fields: .start (TrackStart), .gains, .budget.

    arguments
        info (1,1) struct
        ref (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.feedforward (1,1) struct = struct()
        opts.gains (1,1) struct = gs3dx_track_gains()
    end
    names = gs3dx_names();
    src = char(names.variants.fit_legs);
    mdl = char(names.variants.fit_track);
    assert(strcmp(ref.model, char(names.variants.fit)), 'gs3dx:fittrack', ...
        'REF was computed on %s, not %s', ref.model, names.variants.fit);
    gs3dx_copy_models({fullfile(info.models_dir, [src '.slx']), fullfile(info.models_dir, [mdl '.slx'])}, ...
        opts.overwrite, 'gs3dx:fittrack');

    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    before = local_nonvirtual(mdl);
    ws = get_param(mdl, 'ModelWorkspace');
    legs_t = ws.getVariable('LegReferenceTime');
    assert(numel(legs_t) == numel(ref.t) && max(abs(legs_t(:).' - ref.t)) < 1e-9, 'gs3dx:fittrack', ...
        'REF is not on the time base of the leg reference in %s', src);
    assignin(ws, 'UpperBodyTracking', 1);
    assignin(ws, 'UpperBodyTrackTime', ref.t);
    for j = ref.joints
        local_track_chart(sprintf('%s/%s Input Function', mdl, j.prefix), j);
        ff = zeros(size(j.angle));
        if isfield(opts.feedforward, j.prefix)
            ff = opts.feedforward.(j.prefix);
            assert(isequal(size(ff), size(j.angle)), 'gs3dx:fittrack', ...
                'Feedforward of %s is %s, expected %s', j.prefix, mat2str(size(ff)), mat2str(size(j.angle)));
        end
        g = opts.gains.(j.prefix);
        assert(isequal(size(g), [size(j.angle, 1) 2]) && all(g(:) > 0), 'gs3dx:fittrack', ...
            'Gains of %s must be positive [Kp Kd] per axis', j.prefix);
        assignin(ws, [j.prefix 'TrackAngle'], j.angle);
        assignin(ws, [j.prefix 'TrackRate'], j.rate);
        assignin(ws, [j.prefix 'TrackTorque'], ff);
        assignin(ws, [j.prefix 'TrackKp'], g(:, 1));
        assignin(ws, [j.prefix 'TrackKd'], g(:, 2));
    end
    report.start = ws.getVariable('LegReferenceStart');
    for f = fieldnames(ref.start).'
        assignin(ws, f{1}, ref.start.(f{1}));
        report.start.(f{1}) = ref.start.(f{1});
    end
    assignin(ws, 'TrackStart', report.start);
    report.gains = opts.gains;

    after = local_nonvirtual(mdl);
    assert(after == before, 'gs3dx:fittrack', ...
        'Postcondition: the chart edits changed the block count (%d -> %d)', before, after);
    report.budget = gs3dx_block_budget(mdl, compiled=true);
    reserve = 25;   % the sensors GS3DX_CONTACT_CHECK adds in memory
    assert(report.budget.compiled_total <= names.license_block_limit - reserve, 'gs3dx:fittrack', ...
        '%s compiles to %d blocks; %d leave no room for %d validation blocks', ...
        mdl, report.budget.compiled_total, names.license_block_limit, reserve);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
end

function local_track_chart(sub, j)
% Add the tracking parameters and override to the chart of subsystem SUB.
    P = j.prefix;
    ch = find_system(sub, 'SearchDepth', 1, 'SFBlockType', 'MATLAB Function');
    assert(isscalar(ch), 'gs3dx:fittrack', '%s has no single MATLAB Function', sub);
    chart = sfroot().find('-isa', 'Stateflow.EMChart', 'Path', ch{1});
    s = chart.Script;
    params = [{'UpperBodyTracking', 'UpperBodyTrackTime'}, strcat(P, {'TrackAngle', 'TrackRate', 'TrackTorque', 'TrackKp', 'TrackKd'})];
    assert(~contains(s, 'UpperBodyTracking'), 'gs3dx:fittrack', '%s is already edited', sub);

    open = strfind(s, 'fcn(');
    assert(isscalar(open), 'gs3dx:fittrack', '%s: expected one fcn( signature', sub);
    close = open + find(s(open:end) == ')', 1) - 1;
    s = [s(1:close - 1) ', ' strjoin(params, ', ') s(close:end)];

    k0 = strfind(s, 'if ModelingMode==0');
    assert(isscalar(k0), 'gs3dx:fittrack', '%s: expected one ModelingMode selection', sub);
    e = regexp(s(k0:end), '\n[ \t]*end[ \t]*\r?\n', 'once', 'end');
    block = s(k0:k0 + e - 1);
    assert(isempty(regexp(block(3:end), '\n[ \t]*(if|for|while|switch)\s', 'once')), 'gs3dx:fittrack', ...
        '%s: the ModelingMode selection is not a flat if-block', sub);
    s = [s(1:k0 + e - 1) local_override(P, j.axes) s(k0 + e:end)];
    chart.Script = s;
    for p = params
        d = chart.find('-isa', 'Stateflow.Data', 'Name', p{1});
        assert(isscalar(d), 'gs3dx:fittrack', '%s: no data %s after the edit', sub, p{1});
        d.Scope = 'Parameter';
    end
    local_rewire_inputs(ch{1}, chart, P, j.axes);
end

function local_rewire_inputs(ch, chart, P, axes)
% Point each angle/rate input of the chart at its own joint's Goto.  In the
% original model the LW chart reads the left scapula's angles and the LE, LF,
% RF and RW charts read tags no Goto writes, so tracking those joints feeds
% back another joint's state (or none) and runs away.  Only From GotoTags
% change; no block or line is added.
    ph = get_param(ch, 'PortHandles');
    ax = cellstr(axes(:)).';
    if isempty(ax)
        ax = {''};
    end
    for kind = {'Position', 'Velocity'}
        for a = ax
            name = [P kind{1} a{1}];
            tag = [P 'Angular' kind{1} a{1}];
            d = chart.find('-isa', 'Stateflow.Data', 'Name', name, 'Scope', 'Input');
            assert(isscalar(d), 'gs3dx:fittrack', '%s: no input %s', ch, name);
            src = get_param(get_param(get_param(ph.Inport(d.Port), 'Line'), 'SrcPortHandle'), 'Parent');
            assert(strcmp(get_param(src, 'BlockType'), 'From'), 'gs3dx:fittrack', ...
                '%s: input %s is not fed by a From block', ch, name);
            goto = find_system(bdroot(ch), 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                'BlockType', 'Goto', 'GotoTag', tag, 'TagVisibility', 'global');
            assert(isscalar(goto), 'gs3dx:fittrack', '%s: no single global Goto %s', ch, tag);
            set_param(src, 'GotoTag', tag);
        end
    end
end

function txt = local_override(P, axes)
    sig = gs3dx_track_signals(P, axes);
    q = sig.q; qd = sig.qd; out = sig.tau;
    lines = {'', ...
        '% Capture tracking (GS3DX_BUILD_FIT_TRACK, #10979): learned feedforward plus PD.', ...
        'if UpperBodyTracking', ...
        sprintf(['    TrackTau=gs3dx_track_torque(Time,UpperBodyTrackTime,%sTrackAngle,%sTrackRate,' ...
            '%sTrackTorque,%sTrackKp,%sTrackKd,[%s],[%s]);'], P, P, P, P, P, strjoin(q, ';'), strjoin(qd, ';'))};
    for k = 1:numel(out)
        lines{end + 1} = sprintf('    %s=TrackTau(%d);', out{k}, k); %#ok<AGROW> one per axis
    end
    lines{end + 1} = 'end';
    txt = [strjoin(lines, newline) newline];
end

function n = local_nonvirtual(mdl)
    n = numel(find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'LookInsideSubsystemReference', 'on', 'Virtual', 'off'));
end

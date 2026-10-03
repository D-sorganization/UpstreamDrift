function out = gs3dx_track_learn(info, opts)
%GS3DX_TRACK_LEARN  Learn the upper-body feedforward of GS3DX_FitTrack (#10979).
%
%   OUT = GS3DX_TRACK_LEARN(INFO) learns, by iterative learning control, the
%   joint torques that make GS3DX_FitTrack's upper body follow its capture
%   reference with the PD at rest: the inverse-dynamics torques of the
%   tracked motion on the contact-supported model.  Each iteration
%   simulates the model from the reference start (the drive, every other
%   start velocity zeroed, then TrackStart), reads each joint's angles and
%   rates from the Simscape log (signal logging would add blocks past the
%   license limit; nothing is saved), and adds the PD torque it needed,
%   zero-phase low-passed, to the feedforward:
%       F <- lowpass(F + LEARNING_GAIN * (Kp (A - q) + Kd (R - qd)))
%   Filtering all of F, not only the increment, keeps content above
%   CUTOFF_HZ from accumulating over the iterations (the robust Q-filter
%   form of learning control); with the increment alone filtered, the full
%   swing's PD torque fell for two iterations and then grew.
%   At a fixed point the PD torque vanishes, so F alone drives the tracked
%   motion.  The pelvis joint is not driven: the legs and the ground carry
%   the body throughout.
%
%   The angles are the charts' own: the revolute and universal joint
%   angles, and the shoulders' intrinsic X-Y-Z angles and rates from their
%   quaternion and angular velocity (GS3DX_XYZ_MAP, on the branch of the
%   reference).
%
%   Options: iterations (8), stop_time (the reference end, s), learning_gain
%   (1), cutoff_hz (12), drive ("impact"), feedforward (struct of prefix ->
%   start feedforward; default the model's <J>TrackTorque).
%   OUT fields:
%     .feedforward  struct prefix -> axes x frames (N*m), the feedforward
%                   of the iteration with the least RMS PD torque (.best)
%     .error        iterations x 1: RMS over every axis and sample of the
%                   angle error (deg) of each iteration's simulation
%     .joint_error  iterations x joints: the RMS per joint (deg)
%     .pd           iterations x 1: RMS PD torque (N*m)
%     .joint_pd     iterations x joints: the RMS per joint (N*m)
%     .best         that iteration (NaN if none completed)
%     .prefixes, .stop_time, .status (error message of the simulation that
%                   failed, which ends the learning; '' if all ran)
%   MODEL defaults to historical FitTrack only in INITIALIZATION="legacy".
%   INITIALIZATION="configured" requires an explicit already-loaded model;
%   it preserves the caller's fitted workspace and initial states, bypasses
%   drive/TrackStart overrides and never closes a caller-owned model.
%   Both modes validate enabled tracking, finite uniformly spaced references,
%   filter sample support, cutoff, gains and feedforward shapes before dynamics.
%   Outputs also identify model, initialization and upper-only qualification.
%   Learned upper-body feedforward does not qualify prescribed neck motion,
%   feedback leg torques, floating-base support or independent forward replay.
%   Pass OUT.feedforward to GS3DX_BUILD_FIT_TRACK only for a legacy model save;
%   configured callers may replay it in memory through explicit workspace overrides.

    arguments
        info (1,1) struct
        opts.model (1,1) string = ""
        opts.initialization (1,1) string {mustBeMember(opts.initialization,["legacy","configured"])} = "legacy"
        opts.iterations (1,1) double {mustBeInteger, mustBePositive} = 8
        opts.stop_time (1,1) double = NaN
        opts.learning_gain (1,1) double = 1
        opts.cutoff_hz (1,1) double = 12
        opts.drive (1,1) string = "impact"
        opts.feedforward (1,1) struct = struct()
    end
    legacy = char(gs3dx_names().variants.fit_track);
    assert(~ismissing(opts.model), 'gs3dx:tracklearn', 'MODEL must not be missing');
    if opts.initialization == "configured"
        assert(strlength(opts.model)>0, 'gs3dx:tracklearn', 'Configured mode requires an explicit MODEL');
        mdl = char(opts.model);
        assert(bdIsLoaded(mdl), 'gs3dx:tracklearn', 'Configured MODEL must already be loaded');
        owned = false;
    else
        mdl = legacy;
        if strlength(opts.model)>0
            assert(opts.model==string(legacy), 'gs3dx:tracklearn', 'Legacy initialization requires the historical MODEL');
        end
        owned = ~bdIsLoaded(mdl);
        if owned, load_system(mdl); end
    end
    if owned, cleanup = onCleanup(@() close_system(mdl,0)); end %#ok<NASGU>
    ws = get_param(mdl, 'ModelWorkspace');
    tracking = local_value(ws,'UpperBodyTracking');
    assert((isnumeric(tracking)||islogical(tracking)) && isreal(tracking) && ...
        isscalar(tracking) && isfinite(tracking) && tracking==1, 'gs3dx:tracklearn', ...
        'UpperBodyTracking must be enabled (1)');
    assert(isreal(opts.learning_gain) && isfinite(opts.learning_gain) && opts.learning_gain>0, ...
        'gs3dx:tracklearn', 'LEARNING_GAIN must be finite, real and positive');
    T = local_value(ws,'UpperBodyTrackTime');
    assert(isnumeric(T) && isreal(T) && isvector(T) && numel(T)>=2 && all(isfinite(T(:))), ...
        'gs3dx:tracklearn', 'UpperBodyTrackTime must be a real finite vector');
    dt = diff(T(:));
    assert(all(dt>0), 'gs3dx:tracklearn', 'UpperBodyTrackTime must be strictly increasing');
    assert(max(abs(dt-mean(dt)))<=1e-6*mean(dt), 'gs3dx:tracklearn', ...
        'UpperBodyTrackTime must be uniformly spaced');
    stop = opts.stop_time;
    if isnan(stop), stop = T(end); end
    assert(isreal(stop) && isfinite(stop) && stop>T(1) && stop<=T(end), ...
        'gs3dx:tracklearn', 'STOP_TIME outside the reference');
    use = T<=stop;
    assert(sum(use)>12, 'gs3dx:tracklearn', ...
        'Insufficient selected samples for fourth-order FILTFILT (requires more than 12)');
    rate = 1/mean(dt);
    assert(isreal(opts.cutoff_hz) && isfinite(opts.cutoff_hz) && ...
        opts.cutoff_hz>0 && opts.cutoff_hz<rate/2, 'gs3dx:tracklearn', ...
        'CUTOFF_HZ must be finite, positive and below Nyquist');
    [b,a] = butter(4,opts.cutoff_hz/(rate/2));

    spec = gs3dx_upper_body_joints();
    unknown = setdiff(fieldnames(opts.feedforward),{spec.prefix});
    assert(isempty(unknown),'gs3dx:tracklearn','Unknown feedforward override prefix');
    for k=1:numel(spec)
        P = spec(k).prefix;
        axes_count = max(1,numel(spec(k).axes));
        for suffix = {'TrackAngle','TrackRate','TrackTorque'}
            name = [P suffix{1}];
            local_matrix(local_value(ws,name),[axes_count,numel(T)],name,false);
        end
        for suffix = {'TrackKp','TrackKd'}
            name = [P suffix{1}];
            local_matrix(local_value(ws,name),[axes_count,1],name,true);
        end
        if isfield(opts.feedforward,P)
            local_matrix(opts.feedforward.(P),[axes_count,numel(T)],['feedforward override ' P],false);
        end
    end
    blocks = gs3dx_track_blocks(simscape.multibody.KinematicsSolver(mdl).jointPositionVariables, spec);
    for k = 1:numel(spec)
        P = spec(k).prefix;
        j = struct('A', ws.getVariable([P 'TrackAngle']), 'R', ws.getVariable([P 'TrackRate']), ...
            'Kp', ws.getVariable([P 'TrackKp']), 'Kd', ws.getVariable([P 'TrackKd']), ...
            'F', ws.getVariable([P 'TrackTorque']), 'axes', spec(k).axes, 'block', blocks{k});
        if isfield(opts.feedforward, P)
            j.F = opts.feedforward.(P);
        end
        J.(P) = j;
    end
    prefixes = {spec.prefix};

    vars = struct();
    if opts.initialization=="legacy"
        vars = gs3dx_drive(info, opts.drive, mdl);
        known = {ws.whos.name};
        for n = known(contains(known, 'StartVelocity'))
            vars.(n{1}) = zeros(size(ws.getVariable(n{1})));
        end
        start = ws.getVariable('TrackStart');
        for f = fieldnames(start).'
            vars.(f{1}) = start.(f{1});
        end
    end

    nj = numel(prefixes);
    out = struct('error', nan(opts.iterations, 1), 'joint_error', nan(opts.iterations, nj), ...
        'joint_pd', nan(opts.iterations, nj), ...
        'pd', nan(opts.iterations, 1), 'prefixes', {prefixes}, 'stop_time', stop, 'status', '', 'best', NaN);
    out.model = mdl;
    out.initialization = char(opts.initialization);
    out.qualification = 'UPPER_BODY_LEARNING_WITH_EXISTING_FEEDBACK_NOT_INDEPENDENT_FORWARD_REPLAY';
    out.initial_joint_states = struct();
    tl = T(use);
    ran = J;   % the feedforward the next simulation runs with
    for it = 1:opts.iterations
        in = Simulink.SimulationInput(mdl);
        for f = fieldnames(vars).'
            in = in.setVariable(f{1}, vars.(f{1}), 'Workspace', mdl);
        end
        for P = prefixes
            in = in.setVariable([P{1} 'TrackTorque'], J.(P{1}).F, 'Workspace', mdl);
        end
        in = in.setModelParameter('StopTime', num2str(stop), 'SimscapeLogType', 'all', ...
            'ReturnWorkspaceOutputs', 'on', 'CaptureErrors', 'on');
        sim_out = sim(in);
        msg = sim_out.ErrorMessage;
        if ~isempty(msg)   % partial logs: no update
            out.status = msg;
            fprintf('track learn %d:%s\n', it, local_note(msg));
            break
        end
        log = sim_out.get(get_param(mdl, 'SimscapeLogName'));
        err = []; pd = [];
        for k = 1:nj
            P = prefixes{k};
            j = J.(P);
            [q, qd] = gs3dx_track_state(log, j, tl, T);
            out.initial_joint_states.(P).angle_deg(:,it) = q(:,1);
            out.initial_joint_states.(P).rate_deg_s(:,it) = qd(:,1);
            e = j.A(:, use) - q;
            u = j.Kp .* e + j.Kd .* (j.R(:, use) - qd);
            out.joint_error(it, k) = sqrt(mean(e .^ 2, 'all'));
            out.joint_pd(it, k) = sqrt(mean(u .^ 2, 'all'));
            err = [err; e(:)]; %#ok<AGROW> twelve joints
            pd = [pd; u(:)]; %#ok<AGROW>
            J.(P).F(:, use) = filtfilt(b, a, (j.F(:, use) + opts.learning_gain * u).').';
        end
        out.error(it) = sqrt(mean(err .^ 2));
        out.pd(it) = sqrt(mean(pd .^ 2));
        fprintf('track learn %d: angle RMS %.2f deg, PD RMS %.1f N*m\n', it, out.error(it), out.pd(it));
        if out.pd(it) <= min(out.pd(1:it))   % the feedforward this iteration ran with
            best = ran;
            out.best = it;
        end
        ran = J;
    end
    if isnan(out.best)   % no iteration completed: the start feedforward
        best = ran;
    end
    for P = prefixes
        out.feedforward.(P{1}) = best.(P{1}).F;
    end
end

function s = local_note(msg)
    s = sprintf(' (stopped: %s)', strtrim(extractBefore([msg newline], newline)));
end

function value = local_value(ws,name)
    assert(ws.hasVariable(name),'gs3dx:tracklearn','Workspace lacks %s',name);
    value = ws.getVariable(name);
end

function local_matrix(value,shape,name,nonnegative)
    assert(isnumeric(value) && isreal(value) && isequal(size(value),shape) && ...
        all(isfinite(value(:))), 'gs3dx:tracklearn', ...
        '%s must be real, finite and have shape %s',name,mat2str(shape));
    assert(~nonnegative || all(value(:)>=0), 'gs3dx:tracklearn', '%s must be nonnegative',name);
end

function check = gs3dx_contact_check(info, opts)
%GS3DX_CONTACT_CHECK  Simulate GS3DX_FullBodyContact and test its physics (#10986).
%
%   CHECK = GS3DX_CONTACT_CHECK(INFO) simulates GS3DX_FullBodyContact with
%   the drive (default "impact") for STOP_TIME (default 0.3 s), with
%   whole-mechanism mass/COM sensing added in memory (nothing is saved),
%   and returns:
%
%     .t, .com (3xN), .mass, .grf (3xN, all contacts, World axes, N),
%     .grf_L/.grf_R (3xN per foot, World axes)
%     .newton     struct: .residual (3xN, N*s) of
%                   M*(v_com(t) - v_com(0)) - integral(GRF + M*g) dt,
%                 .max, .bound (1% of M*|g|*T) and .pass.  The contacts
%                 and gravity are then the only external forces, so a
%                 failure means another external load acts (the pelvis
%                 drive is still on) or the contact log is wrong.
%     .support    min and max of GRF.up / (M*|g|) over the run
%     .feet       per side: .slip (max horizontal ankle travel, m),
%                 .lift (max vertical ankle rise, m) from the ankle joint
%                 GlobalPosition, relative to t = 0, and .p (3xN, World,
%                 m) the ankle position itself
%     .contacts   per-contact forces (3 x contacts x N, World axes, N) in
%                 the order of 'FootContactForces' (left contacts first)
%     .pelvis     max distance of the pelvis frame from its t = 0 position (m)
%     .pelvis_p   pelvis frame origin (3xN, World, m) on the grid of .t
%     .pelvis_R   pelvis frame orientation (3x3xN, World) on the grid of .t
%     .signals    struct of the logged signals named in option SIGNALS
%                 (width x N on the grid of .t)
%     .joints     with JOINTS > 0: the simulated joint positions every
%                 JOINTS seconds (GS3DX_SIMLOG_JOINTS), a pose that
%                 GS3DX_RENDER draws; the Simscape log keeps every 10th
%                 solver step
%     .feedback   with FEEDBACK true: the torque the run applied beyond
%                 its feedforward (GS3DX_FEEDBACK_TORQUE), its distance to
%                 pure forward dynamics (docs/FORWARD_DYNAMICS.md); needs a
%                 tracked variant (GS3DX_FitLegs or later).  The balance
%                 share uses the finite-difference COM rate, not the
%                 model's filtered one (BalanceCOMTau).
%     .status, .message    of the simulation
%
%   Options: drive ("impact"), stop_time (0.3 s), variables (struct of
%   model-workspace overrides applied after the drive), rest (false),
%   model (GS3DX_FullBodyContact; any model built from it, e.g. GS3DX_Golfer),
%   signals (string array of logged signal names to return, default none),
%   joints (0: pose sample interval in s for .joints, 0 = none),
%   feedback (false: add .feedback).
%   rest=true zeroes every *StartVelocity* variable, so the body starts
%   still in the drive's pose: the standing test.  The impact drive alone
%   starts mid-downswing with the whole-body momentum of a model whose
%   pelvis was driven, which no planted stance can hold (DATA_AUDIT.md).
%
%   v_com is the finite-difference derivative of the COM on a uniform
%   1 kHz grid (linear interpolation of the logged samples), so the
%   Newton check compares integrals, not raw accelerations.

    arguments
        info (1,1) struct %#ok<INUSA> kept for the common GS3DX tool signature
        opts.drive (1,1) string = "impact"
        opts.stop_time (1,1) double {mustBePositive} = 0.3
        opts.variables (1,1) struct = struct()
        opts.rest (1,1) logical = false
        opts.model (1,1) string = gs3dx_names().variants.contact
        opts.signals (1,:) string = strings(1, 0)
        opts.joints (1,1) double {mustBeNonnegative} = 0
        opts.feedback (1,1) logical = false
    end
    mdl = char(opts.model);
    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    vars = gs3dx_drive(info, opts.drive, mdl);
    ws = get_param(mdl, 'ModelWorkspace');
    if opts.rest
        known = {ws.whos.name};
        for n = known(contains(known, 'StartVelocity'))
            vars.(n{1}) = zeros(size(ws.getVariable(n{1})));
        end
    end
    for f = reshape(fieldnames(opts.variables), 1, [])
        vars.(f{1}) = opts.variables.(f{1});
    end
    ground_R = ws.getVariable('GroundRotation');
    g = str2num(get_param([mdl '/Hips and Torso Inputs/Mechanism Configuration'], 'GravityVector')); %#ok<ST2NM> vector literal
    g = g(:);
    simlog = opts.joints > 0 || opts.feedback;
    if simlog
        % before the sensors go in; the joints are the same with them
        jp = simscape.multibody.KinematicsSolver(mdl).jointPositionVariables;
    end
    if opts.feedback
        % GS3DX_STANCE_FRAMES closes the model, so take the workspace now
        p = struct();
        for n = {ws.whos.name}
            p.(n{1}) = ws.getVariable(n{1});
        end
        log_name = get_param(mdl, 'SimscapeLogName');
    end
    frames = gs3dx_stance_frames(mdl, vars, stop_time=opts.stop_time, mass=true, shoulders=false, ...
        simscape_log=10 * simlog);
    if simlog
        load_system(mdl);   % GS3DX_STANCE_FRAMES closed it; the Simscape log needs it loaded
    end
    s = frames.series;
    logs = frames.out.logsout;
    assert(norm(-g / norm(g) - frames.up) < 1e-12, 'gs3dx:contact_check', 'Gravity and up disagree');

    grid = 0:1e-3:opts.stop_time;
    at = @(t, x) interp1(t(:), x.', grid(:), 'linear').';
    [tf, f] = local_signal(logs, 'FootContactForces');
    n_contacts = size(f, 1) / 3;
    assert(mod(n_contacts, 2) == 0, 'gs3dx:contact_check', 'Expected an even number of contacts, found %g', n_contacts);
    f = at(tf, f);
    per = reshape(f, 3, n_contacts, []);
    world = @(x) ground_R * reshape(x, 3, []);
    half = n_contacts / 2;
    check.grf_L = world(sum(per(:, 1:half, :), 2));
    check.grf_R = world(sum(per(:, half + 1:end, :), 2));
    check.grf = check.grf_L + check.grf_R;
    check.contacts = reshape(world(per), 3, n_contacts, []);
    check.t = grid;
    check.com = at(s.t, s.com);
    check.mass = s.mass;

    M = check.mass;
    v = gradient(check.com, 1e-3);
    impulse = cumtrapz(grid, check.grf + M * g, 2);
    r = M * (v - v(:, 1)) - impulse;
    bound = 0.01 * M * norm(g) * opts.stop_time;
    check.newton = struct('residual', r, 'max', max(vecnorm(r)), 'bound', bound, ...
        'pass', max(vecnorm(r)) <= bound);
    up = -g / norm(g);
    share = (up.' * check.grf) / (M * norm(g));
    check.support = [min(share), max(share)];

    for P = 'LR'
        [ta, a] = local_bus_leaf(logs, [P 'AnkleLogs'], 'GlobalPosition');
        a = at(ta, a);
        d = a - a(:, 1);
        vert = up.' * d;
        horiz = d - up * vert;
        check.feet.(P) = struct('slip', max(vecnorm(horiz)), 'lift', max(vert), 'p', a);
    end
    check.signals = struct();
    for name = opts.signals
        [tn, xn] = local_signal(logs, char(name));
        check.signals.(char(name)) = at(tn, xn);
    end
    if opts.joints > 0
        check.joints = gs3dx_simlog_joints(frames.out.simlog, mdl, jp, 0:opts.joints:opts.stop_time);
    end
    if opts.feedback
        check.feedback = local_feedback(p, log_name, vars, jp, frames.out, check, v, at);
    end
    check.pelvis = max(vecnorm(s.pelvis_p - s.pelvis_p(:, 1)));
    check.pelvis_p = at(s.t, s.pelvis_p);
    check.pelvis_R = reshape(at(s.t, reshape(s.pelvis_R, 9, [])), 3, 3, []);
    check.status = "success";
    check.message = "";
end

function fb = local_feedback(p, log_name, vars, jp, out, check, com_rate, at)
% GS3DX_FEEDBACK_TORQUE of the run: the workspace P as run, the chart joints
% from the Simscape log and the leg servo's own angles from the leg buses.
    for f = reshape(fieldnames(vars), 1, [])
        p.(f{1}) = vars.(f{1});
    end
    assert(isfield(p, 'LegReferenceTime'), 'gs3dx:contact_check', ...
        'FEEDBACK needs a tracked variant (GS3DX_FitLegs or later)');
    t = check.t;
    state = struct('t', t, 'upper', struct());
    spec = gs3dx_upper_body_joints();
    spec = spec(arrayfun(@(j) isfield(p, [j.prefix 'TrackAngle']), spec));
    blocks = gs3dx_track_blocks(jp, spec);
    log = out.get(log_name);
    for k = 1:numel(spec)
        P = spec(k).prefix;
        j = struct('block', blocks{k}, 'axes', spec(k).axes, 'A', p.([P 'TrackAngle']));
        [q, qd] = gs3dx_track_state(log, j, t, p.UpperBodyTrackTime);
        state.upper.(P) = struct('q', q, 'qd', qd);
    end
    % leg servo state order (GS3DX_BUILD_CONTACT): [hip XYZ, knee, ankle XY] per side
    pos = {{'AngularPositionX', 'AngularPositionY', 'AngularPosition_Z'}, {'AngularPosition'}, ...
           {'AngularPositionX', 'AngularPositionY'}};
    vel = {{'AngularVelocityX', 'AngularVelocityY', 'AngularVelocityZ'}, {'AngularVelocity'}, ...
           {'AngularVelocityX', 'AngularVelocityY'}};
    joints = {'Hip', 'Knee', 'Ankle'};
    q = zeros(0, numel(t)); qd = q;
    for P = 'LR'
        for j = 1:3
            bus = [P joints{j} 'Logs'];
            for a = 1:numel(pos{j})
                [tb, x] = local_bus_leaf(out.logsout, bus, pos{j}{a});
                q(end + 1, :) = at(tb, x); %#ok<AGROW> twelve axes
                [tb, x] = local_bus_leaf(out.logsout, bus, vel{j}{a});
                qd(end + 1, :) = at(tb, x); %#ok<AGROW>
            end
        end
    end
    state.legs = struct('q', q, 'qd', qd);
    state.com = check.com;
    state.com_rate = com_rate;
    state.feet = [check.feet.L.p; check.feet.R.p];
    fb = gs3dx_feedback_torque(p, state);
end

function [t, x] = local_signal(logs, name)
    v = logs.get(name).Values;
    t = v.Time;
    x = local_samples(v.Data, numel(t));
end

function [t, x] = local_bus_leaf(logs, bus, leaf)
    v = logs.get(bus).Values.(leaf);
    t = v.Time;
    x = local_samples(v.Data, numel(t));
end

function x = local_samples(d, n)
% Logged data with N time samples as a width-by-N matrix.
    if ismatrix(d) && size(d, 1) == n
        x = d.';
    else
        x = reshape(d, [], n);
    end
end

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
%                 GlobalPosition, relative to t = 0
%     .pelvis     max distance of the pelvis frame from its t = 0 position (m)
%     .status, .message    of the simulation
%
%   Options: drive ("impact"), stop_time (0.3 s), variables (struct of
%   model-workspace overrides applied after the drive), rest (false).
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
    end
    names = gs3dx_names();
    mdl = char(names.variants.contact);
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
    frames = gs3dx_stance_frames(mdl, vars, stop_time=opts.stop_time, mass=true);
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
        check.feet.(P) = struct('slip', max(vecnorm(horiz)), 'lift', max(vert));
    end
    check.pelvis = max(vecnorm(s.pelvis_p - s.pelvis_p(:, 1)));
    check.status = "success";
    check.message = "";
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

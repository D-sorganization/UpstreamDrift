function ik = gs3dx_whole_body_ik(jc, opts)
%GS3DX_WHOLE_BODY_IK  Least-squares whole-body IK of GS3DX_Fit to the capture (#10979).
%
%   IK = GS3DX_WHOLE_BODY_IK(JC) fits the joint positions of GS3DX_Fit to
%   the joint-centre estimates JC (GS3DX_CAPTURE_JOINT_CENTRES), frame by
%   frame, in the least-squares sense.  The capture's address target frame
%   [facing, toward target, up] is used as the model World (both are Z-up;
%   the free pelvis joint absorbs the placement).
%
%   Forward kinematics comes from Simscape's KinematicsSolver on the model
%   itself, so the fit sees the model's real geometry.  The solver cannot
%   take more targets than degrees of freedom, so the least squares is done
%   here (lsqnonlin, Levenberg-Marquardt) over the 33 independent joint
%   coordinates.  Both hands are welded to the club, so the right shoulder,
%   elbow and wrist are not parameters: the solver closes that loop for
%   every evaluation.  Spherical joints are parameterized by rotation
%   vectors.
%
%   Gap-filled data (JC.gap) are not fitted by default: a target is
%   dropped from a frame's residual (unless gap_weight > 0), and always
%   from the offset calibration, where its markers were missing (the
%   club-head cluster drops out after impact).
%
%   Each target point is the model joint centre plus a constant offset in
%   the frame of the body that carries the marker (skin markers sit off the
%   joint centres).
%
%   Every frame set is tracked forward, then backward, each solve
%   warm-started from its neighbour, keeping the lower cost per frame (the
%   backward pass repairs frames the forward pass left in a local minimum).
%   The offsets are calibrated first, by alternating the tracking of the
%   calibration frames with the closed-form offset update (the mean, over
%   frames, of the data point minus the joint centre, in the body frame);
%   they are then held fixed while FRAMES are tracked.
%
%   Options:
%     model               (GS3DX_Fit)
%     frames              frames to track (default all)
%     calibration_frames  (default every 3rd frame of FRAMES)
%     calibration_rounds  (3)
%     offsets             struct of 3x1 offsets (m) per target: skips the
%                         calibration when given
%     verbose             (false) print every frame fit
%     posture_weight      (0, m/rad) pull the redundant trunk coordinates
%                         (spine tilt, torso twist, both scapulae) toward
%                         zero: the shoulder centres fix only part of these
%                         seven angles, and without the pull the fit may sit
%                         anywhere in the rest (torso twisted 100 deg at
%                         address, 90-260 deg steps between frames); the
%                         trunk then also starts at zero
%     smooth_weight       (0, m/rad) penalize each frame's change of every
%                         rotation coordinate from its warm start (the
%                         neighbour frame)
%     gap_weight          (0) residual weight of gap-filled targets; 0
%                         drops them.  Where the pelvis markers drop out
%                         (frames 447-451 of the trial: pelvis, hips and
%                         knees gap-filled) nothing else fixes the pelvis
%                         translation, and the fit wanders 15-36 cm; the
%                         interpolated targets hold it.  The offset
%                         calibration uses measured samples only.
%     rom_weight          (0, m/rad) penalize every joint angle outside the
%                         normal human range (GS3DX_JOINT_ROM rows with a
%                         neutral): the residual gains, per such joint,
%                         ROM_WEIGHT times how far (rad) the anatomical
%                         angle lies beyond the range.  Without it the fit
%                         may take anatomically impossible branches that
%                         place the joint centres alike (the knees
%                         hyperextended with the legs spun about their long
%                         axes; the spine bent 81 deg sideways) (#11158).
%                         Applied by continuation: every frame is first
%                         fitted without it (that chain gives the warm
%                         starts), then polished from that pose with it
%     rom                 (GS3DX_JOINT_ROM()) the range table ROM_WEIGHT
%                         holds the joints to; GS3DX_GOLF_ROM() narrows it
%                         to the golf-swing band (lead arm straight, #11156)
%     backward            (true) also track backward and keep the lower cost
%                         per frame; with the weights above, false keeps one
%                         continuous forward solution (the lower-cost frame
%                         of either pass may sit on another branch)
%
%   IK fields:
%     .model       the model fitted
%     .names       target names; .offsets (struct, m, body frame)
%     .frames      tracked frames; .t (s)
%     .joint_ids   KinematicsSolver joint position variables; .joint
%                  (ids x frames, solver units: m, deg, axis components)
%     .residual    targets x frames distance (m), NaN where gap-filled;
%                  .rms (1 x frames, m, over the measured targets)
%     .status      KinematicsSolver status per frame (1 = loop closed)
%     .regularization  struct of posture_weight, smooth_weight, backward,
%                  gap_weight, rom_weight

    arguments
        jc (1,1) struct
        opts.model (1,:) char = char(gs3dx_names().variants.fit)
        opts.frames (1,:) double {mustBeInteger, mustBePositive} = 1:size(jc.pelvis, 2)
        opts.calibration_frames (1,:) double {mustBeInteger, mustBePositive} = []
        opts.calibration_rounds (1,1) double {mustBeInteger, mustBeNonnegative} = 3
        opts.offsets struct = struct([])
        opts.verbose (1,1) logical = false
        opts.posture_weight (1,1) double {mustBeNonnegative} = 0
        opts.smooth_weight (1,1) double {mustBeNonnegative} = 0
        opts.backward (1,1) logical = true
        opts.rom_weight (1,1) double {mustBeNonnegative} = 0
        opts.rom table = gs3dx_joint_rom()
        opts.gap_weight (1,1) double {mustBeInRange(opts.gap_weight, 0, 1)} = 0
    end
    s = local_setup(opts.model);
    s.verbose = opts.verbose;
    s.backward = opts.backward;
    s.reg = local_regularization(s, opts.posture_weight, opts.smooth_weight);
    s.rom = local_rom(s, opts.rom_weight, opts.rom);
    nt = numel(s.names);
    data = @(f) cell2mat(cellfun(@(n) jc.(n)(:, f), s.names, 'UniformOutput', false).');   % 3 x nt
    valid = @(f) ~cellfun(@(n) jc.gap.(n)(f), s.names).';   % 1 x nt, false where gap-filled
    weight = @(f) valid(f) + opts.gap_weight * ~valid(f);   % residual weight per target
    assert(all(isfield(jc.gap, s.names)), 'gs3dx:ik', 'JC.gap lacks a target');
    lm = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'Display', 'off', ...
        'FiniteDifferenceStepSize', 1e-6, 'MaxIterations', 200);

    % With the posture pull, the seed has the trunk at zero: (a, b) and
    % (a + 180, 180 - b) of a universal joint place its distal point alike,
    % and the local fit stays on the branch it starts from.
    [p, g] = local_seed(s, jc.pelvis(:, opts.frames(1)), opts.posture_weight > 0);
    if isempty(opts.offsets)
        cal = opts.calibration_frames;
        if isempty(cal)
            cal = opts.frames(1:3:end);
        end
        off = zeros(3, nt);
        for round = 1:opts.calibration_rounds
            best = local_track(s, cal, p, g, off, data, weight, lm);
            acc = zeros(3, nt);
            cnt = zeros(1, nt);
            for i = 1:numel(cal)
                [P, R] = local_fk(s, best(i).p, best(i).g);
                d = data(cal(i));
                for k = find(valid(cal(i)))
                    acc(:, k) = acc(:, k) + R(:, :, s.body(k)).' * (d(:, k) - P(:, k));
                    cnt(k) = cnt(k) + 1;
                end
            end
            assert(all(cnt > 0), 'gs3dx:ik', 'A target has no measured calibration frame');
            off = acc ./ cnt;
            p = best(1).p;
            g = best(1).g;
        end
    else
        off = cell2mat(cellfun(@(n) opts.offsets.(n), s.names, 'UniformOutput', false).');
    end

    n = numel(opts.frames);
    best = local_track(s, opts.frames, p, g, off, data, weight, lm);

    ik.model = opts.model;
    ik.names = s.names;
    ik.offsets = cell2struct(num2cell(off, 1).', s.names, 1);
    ik.frames = opts.frames;
    ik.t = jc.t(opts.frames);
    ik.joint_ids = s.ids;
    ik.joint = zeros(numel(s.ids), n);
    ik.residual = zeros(nt, n);
    ik.status = zeros(1, n);
    for i = 1:n
        [P, R, st] = local_fk(s, best(i).p, best(i).g);
        ik.joint(~s.closed, i) = local_targets(s, best(i).p);
        ik.joint(s.closed, i) = best(i).g;
        pts = P + reshape(pagemtimes(R(:, :, s.body), reshape(off, 3, 1, nt)), 3, nt);
        ik.residual(:, i) = vecnorm(pts - data(opts.frames(i))).';
        ik.residual(~valid(opts.frames(i)), i) = NaN;   % gap-filled: not a measurement
        ik.status(i) = st;
    end
    ik.rms = sqrt(mean(ik.residual .^ 2, 1, 'omitnan'));
    ik.regularization = struct('posture_weight', opts.posture_weight, ...
        'smooth_weight', opts.smooth_weight, 'backward', opts.backward, 'gap_weight', opts.gap_weight, ...
        'rom_weight', opts.rom_weight);
end

function rom = local_rom(s, weight, t)
% The range-of-motion penalty: for each bounded row of the ROM table T
% (GS3DX_JOINT_ROM columns) of a joint of this model, where its angle is
% (parameter index, or index into the closed-loop outputs G) and its range
% in rad.
    t = t(~isnan(t.neutral_deg), :);
    [has, at] = ismember(t.key, s.jkeys);
    t = t(has, :);
    id = s.ids(at(has));
    rom.on = weight > 0 && height(t) > 0;
    rom.weight = weight;
    rom.closed = ismember(id, s.ids(s.closed));
    [~, rom.index] = ismember(id, s.ids(s.closed));   % into G where closed
    start = cumsum([0 s.layout.n]);
    for k = find(~rom.closed(:).')
        parts = split(id(k), '.');
        j = find(string({s.layout.key}) == parts(1) + "." + parts(2));
        assert(numel(j) == 1 && s.layout(j).n == 1, 'gs3dx:ik', 'ROM joint %s is not a single angle', id(k));
        rom.index(k) = start(j) + 1;
    end
    rom.sign = t.sign;
    rom.neutral = deg2rad(t.neutral_deg);
    rom.lo = deg2rad(t.min_deg);
    rom.hi = deg2rad(t.max_deg);
    rom.n = height(t) * rom.on;
end

function r = local_rom_residual(rom, p, g)
% ROM.weight times how far (rad) each anatomical angle lies outside its range.
    q = zeros(numel(rom.index), 1);
    q(~rom.closed) = p(rom.index(~rom.closed));
    q(rom.closed) = deg2rad(g(rom.index(rom.closed)));
    a = rom.sign .* (mod(q - rom.neutral + pi, 2 * pi) - pi);
    r = rom.weight * (max(0, a - rom.hi) + max(0, rom.lo - a));
end

function reg = local_regularization(s, posture, smooth)
% Weights per coordinate of the parameter vector: the posture pull on the
% redundant trunk coordinates, the smoothing on every rotation coordinate.
    keys = repelem(string({s.layout.key}), [s.layout.n]);
    trunk = startsWith(keys, ["j2.", "j3.", "j6.", "j17."]);   % spine, torso, scapulae
    assert(nnz(trunk) == 7, 'gs3dx:ik', 'Expected 7 trunk coordinates, found %d', nnz(trunk));
    reg.posture = posture * double(trunk(:));
    reg.smooth = smooth * double(~endsWith(keys(:), [".Px", ".Py", ".Pz"]));
    reg.on = posture > 0 || smooth > 0;
    reg.wrap = ~endsWith(keys(:), [".Px", ".Py", ".Pz"]) & ~contains(keys(:), ".S");   % single angles
end

function best = local_track(s, frames, p, g, off, data, weight, lm)
% Forward then backward over FRAMES, each fit warm-started from its
% neighbour; the lower cost per frame is kept.  With the range penalty the
% warm starts come from a chain fitted without it, and each frame is then
% polished from its own chain pose with the penalty on (continuation): a
% penalized pose that left the markers' basin never seeds the next frame,
% and the smoothing term holds the polish near the chain pose.
    n = numel(frames);
    best = struct('p', cell(1, n), 'g', cell(1, n), 'cost', num2cell(inf(1, n)));
    chain = best;
    free = s;
    free.rom.on = false;
    free.rom.n = 0;
    orders = {1:n, n:-1:1};
    if ~s.backward
        orders = orders(1);
    end
    for order = orders
        for i = order{1}
            t0 = tic;
            d = data(frames(i));
            w = weight(frames(i));
            [p, g, cost, iters] = local_fit(free, p, g, off, d, w, lm);
            if cost < chain(i).cost
                chain(i) = struct('p', p, 'g', g, 'cost', cost);
            end
            if s.rom.on
                [pr, gr, cost, it] = local_fit(s, chain(i).p, chain(i).g, off, d, w, lm);
                iters = iters + it;
            else
                [pr, gr] = deal(p, g);
            end
            if s.verbose
                fprintf('ik frame %d: cost %.3g, %d iterations, %.1f s\n', frames(i), cost, iters, toc(t0));
            end
            if cost < best(i).cost
                best(i) = struct('p', pr, 'g', gr, 'cost', cost);
            end
            p = chain(i).p;
            g = chain(i).g;
        end
    end
end

function s = local_setup(mdl)
% KinematicsSolver with the target points, the body rotations and the
% right-arm loop joints as outputs.
    load_system(mdl);
    wf = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'ReferenceBlock', 'sm_lib/Frames and Transforms/World Frame');
    world = [wf{1} '/W'];
    ks = simscape.multibody.KinematicsSolver(mdl);
    jp = ks.jointPositionVariables;
    ids = string(jp.ID);
    paths = string(unique(jp.BlockPath, 'stable'));
    joint = @(pat) char(paths(contains(paths, pat)));
    club = find_system(mdl, 'LookUnderMasks', 'all', 'Name', 'Clubhead');
    % target, point frame, frame of the body that carries the marker
    T = {
        'pelvis',     [joint('Hip Kinetically Driven/Hip Joint') '/F'], [joint('Hip Kinetically Driven/Hip Joint') '/F']
        'hip_L',      [joint('Left Hip Joint') '/F'],   [joint('Hip Kinetically Driven/Hip Joint') '/F']
        'hip_R',      [joint('Right Hip Joint') '/F'],  [joint('Hip Kinetically Driven/Hip Joint') '/F']
        'knee_L',     [joint('Left Knee') '/F'],        [joint('Left Knee') '/B']
        'knee_R',     [joint('Right Knee') '/F'],       [joint('Right Knee') '/B']
        'ankle_L',    [joint('Left Ankle') '/F'],       [joint('Left Ankle') '/B']
        'ankle_R',    [joint('Right Ankle') '/F'],      [joint('Right Ankle') '/B']
        'shoulder_L', [joint('Left Shoulder') '/F'],    [joint('Left Shoulder') '/B']
        'shoulder_R', [joint('Right Shoulder') '/F'],   [joint('Right Shoulder') '/B']
        'elbow_L',    [joint('Left Elbow') '/F'],       [joint('Left Elbow') '/B']
        'elbow_R',    [joint('Right Elbow') '/F'],      [joint('Right Elbow') '/B']
        'wrist_L',    [joint('Left Wrist') '/F'],       [joint('Left Wrist') '/B']
        'wrist_R',    [joint('Right Wrist') '/F'],      [joint('Right Wrist') '/B']
        'club_head',  [club{1} '/R'],                   [club{1} '/R']
        };
    s.names = T(:, 1);
    [bodies, ~, s.body] = unique(T(:, 3), 'stable');
    for k = 1:size(T, 1)
        addFrameVariables(ks, sprintf('p%d', k), 'Translation', world, T{k, 2});
    end
    for b = 1:numel(bodies)
        addFrameVariables(ks, sprintf('r%d', b), 'Rotation', world, bodies{b});
    end
    s.closed = startsWith(ids, ["j15.", "j18.", "j19."]);   % right elbow, shoulder, wrist
    s.tv = ids(~s.closed);
    addTargetVariables(ks, s.tv);
    addOutputVariables(ks, [string(ks.frameVariables.ID); ids(s.closed)]);
    addInitialGuessVariables(ks, ids(s.closed));
    s.ks = ks;
    s.ids = ids;
    s.jkeys = gs3dx_joint_keys(mdl, jp);
    s.nt = size(T, 1);
    s.nb = numel(bodies);
    s.isdeg = containers.Map(cellstr(ids), num2cell(jp.Unit == "deg"));
    parts = split(s.tv, '.');
    keys = unique(parts(:, 1) + "." + parts(:, 2), 'stable');
    s.layout = struct('key', num2cell(keys), 'n', num2cell(1 + 2 * endsWith(keys, '.S')));
    assert(sum([s.layout.n]) == 33, 'gs3dx:ik', 'Expected 33 independent coordinates, found %d', sum([s.layout.n]));
end

function [p, g] = local_seed(s, pelvis, trunk_zero)
% A pose that closes the grip loop (a target-free solve), moved onto PELVIS.
    ks0 = simscape.multibody.KinematicsSolver(s.ks.ModelName);
    addOutputVariables(ks0, s.ids);
    trunk = s.ids(startsWith(s.ids, ["j2.", "j3.", "j6.", "j17."]));
    zero = [];
    if trunk_zero
        addTargetVariables(ks0, trunk);
        zero = zeros(numel(trunk), 1);
    end
    [q, st] = solve(ks0, zero, []);
    assert(st == 1, 'gs3dx:ik', 'The target-free solve did not close the loop (status %d)', st);
    p = zeros(sum([s.layout.n]), 1);
    si = @(vid) q(s.ids == vid) * (1 + (pi / 180 - 1) * s.isdeg(char(vid)));
    i = 0;
    for k = 1:numel(s.layout)
        key = s.layout(k).key;
        if s.layout(k).n == 3
            p(i + 1:i + 3) = [q(s.ids == key + ".ax_x"); q(s.ids == key + ".ax_y"); q(s.ids == key + ".ax_z")] * si(key + ".q");
        else
            vid = s.ids(startsWith(s.ids, key + "."));
            p(i + 1) = si(vid(1));
        end
        i = i + s.layout(k).n;
    end
    g = q(s.closed);
    P = local_fk(s, p, g);
    p(1:3) = p(1:3) + pelvis - P(:, 1);   % j1 Px, Py, Pz lead the layout
end

function [p, g, cost, iters] = local_fit(s, p, g, off, d, w, lm)
    if s.reg.on   % the same pose, each single angle on (-pi, pi]: the branch the posture pull sees
        p(s.reg.wrap) = p(s.reg.wrap) - 2 * pi * round(p(s.reg.wrap) / (2 * pi));
    end
    p0 = p;   % the warm start: the neighbour frame's solution
    r = @(pp) local_residual(s, pp, g, off, d, w, p0);
    [p, cost, ~, ~, out] = lsqnonlin(r, p, [], [], lm);
    iters = out.iterations;
    [~, ~, ~, g] = local_fk(s, p, g);
end

function r = local_residual(s, p, g, off, d, w, p0)
    [P, R, st, gc] = local_fk(s, p, g);
    if st < 1
        r = 10 * ones(3 * s.nt + 2 * numel(p) * s.reg.on + s.rom.n, 1);   % loop not closed: reject the step
        return;
    end
    pts = P + reshape(pagemtimes(R(:, :, s.body), reshape(off, 3, 1, s.nt)), 3, s.nt);
    r = reshape((pts - d) .* w, [], 1);
    if s.reg.on
        r = [r; s.reg.posture .* p; s.reg.smooth .* (p - p0)];
    end
    if s.rom.on
        r = [r; local_rom_residual(s.rom, p, gc)];
    end
end

function [P, R, st, g] = local_fk(s, p, g)
    [o, st] = solve(s.ks, local_targets(s, p), g);
    P = reshape(o(1:3 * s.nt), 3, s.nt);
    a = reshape(o(3 * s.nt + 1:3 * (s.nt + s.nb)), 3, s.nb) * pi / 180;
    R = zeros(3, 3, s.nb);
    for b = 1:s.nb   % intrinsic X-Y-Z, the KinematicsSolver 'Rotation' convention
        R(:, :, b) = local_rx(a(1, b)) * local_ry(a(2, b)) * local_rz(a(3, b));
    end
    g = o(3 * (s.nt + s.nb) + 1:end);
end

function T = local_targets(s, p)
    T = zeros(numel(s.tv), 1);
    i = 0;
    for k = 1:numel(s.layout)
        key = s.layout(k).key;
        v = p(i + 1:i + s.layout(k).n);
        i = i + s.layout(k).n;
        if s.layout(k).n == 3
            a = norm(v);
            ax = [0; 0; 1];
            if a > 1e-12
                ax = v / a;
            end
            comp = ["ax_x", "ax_y", "ax_z", "q"];
            vals = [ax; a];
        else
            comp = extractAfter(s.tv(startsWith(s.tv, key + ".")), key + ".");
            vals = v;
        end
        for c = 1:numel(comp)
            vid = key + "." + comp(c);
            x = vals(c);
            if s.isdeg(char(vid))
                x = x * 180 / pi;
            end
            T(s.tv == vid) = x;
        end
    end
end

function R = local_rx(a)
    R = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
end

function R = local_ry(a)
    R = [cos(a) 0 sin(a); 0 1 0; -sin(a) 0 cos(a)];
end

function R = local_rz(a)
    R = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
end

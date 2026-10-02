function ik = gs3dx_whole_body_ik(jc, opts)
%GS3DX_WHOLE_BODY_IK  Least-squares whole-body IK of GS3DX_Human to the capture (#10979).
%
%   IK = GS3DX_WHOLE_BODY_IK(JC) fits the joint positions of GS3DX_Human to
%   the joint-centre estimates JC (GS3DX_CAPTURE_JOINT_CENTRES), frame by
%   frame, in the least-squares sense.  The capture's address target frame
%   [facing, toward target, up] is used as the model World (both are Z-up;
%   the free pelvis joint absorbs the placement).
%
%   Forward kinematics comes from Simscape's KinematicsSolver on the model
%   itself, so the fit sees the model's real geometry.  The solver cannot
%   take more targets than degrees of freedom, so the least squares is done
%   here (lsqnonlin, Levenberg-Marquardt) over the independent joint
%   coordinates (e.g. 37 for Human, 33 for Fit).  Both hands are welded to the club, so the
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
%     model               (GS3DX_Human)
%     frames              frames to track (default all)
%     calibration_frames  (default every 3rd frame of FRAMES)
%     calibration_rounds  (3)
%     offsets             struct of 3x1 offsets (m) per target: skips the
%                         calibration when given
%     initial_pose        (struct([])) optional keyed initial pose struct with
%                         canonical joint_keys, native joint values, explicit
%                         units, and status == 1. Warm-starts tracking from a
%                         valid GS3DX_Human pose, avoiding cold target-free solve.
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
%     target_weight       (ones per target) extra scale on each target's
%                         position residual, multiplied with the gap/valid
%                         weight inside the least-squares fit.  A struct with
%                         one field per target name, or a 1 x numel(names)
%                         vector in .names order.  Omitted names, wrong
%                         length, or negative values error with gs3dx:ik.
%                         Does not change reported .residual (Euclidean m).
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
%     foot_orientation_weight (0, m per normalized orientation error) identify
%                         the unconstrained ankle rotations from calibrated
%                         full 3D foot orientation targets (jc.foot_R_L/R).
%                         Evaluates normalized chordal SO(3) distance over 18
%                         residual components (9 per foot), constraining
%                         roll, pitch, and yaw (identifying inverted soles
%                         and backward flips alike).  Flat sole at address
%                         is a kinematic assumption, not ground contact or
%                         dynamics qualification; no knee-torsion magically
%                         solved.  Gap-filled or nonfinite measurements are
%                         masked with zero residual without inventing data.
%     head_orientation_weight (0, m per normalized orientation error) optional
%                         candidate term penalizing head solid orientation error
%                         relative to calibrated cluster-relative targets (jc.head_R).
%                         Evaluates normalized chordal SO(3) distance over 9
%                         residual components. The head has a 2-DOF neck in
%                         Human/Neck models; no full 3-DOF exact fit is promised
%                         and raw cluster is not anatomical.
%                         Defaults to 0. With no head targets, retains the
%                         original solver output layout. Positive weights require
%                         jc.head_R (3x3xN) and logical jc.gap.head_R (1xN).
%                         Validation covers calibration and tracking frames before
%                         model setup; missing measurements remain missing.
%                         Does not alter the 14 position target names,
%                         .residual, or .rms.
%
%   IK fields:
%     .model       the model fitted
%     .seed_source (explicit initial_pose only) kinematic seed provenance ('target_free_solve' or 'initial_pose');
%                  indicates solver warm-start origin, not physical acceptance
%     .names       target names; .offsets (struct, m, body frame)
%     .frames      tracked frames; .t (s)
%     .joint_ids   KinematicsSolver joint position variables; .joint
%                  (ids x frames, solver units: m, deg, axis components)
%     .residual    targets x frames distance (m), NaN where gap-filled;
%                  .rms (1 x frames, m, over the measured targets)
%     .points      3 x targets x frames predicted target positions (m);
%                  column order matches .names; .residual is the distance
%                  from each column to the measured joint centre (NaN where
%                  gap-filled)
%     .status      KinematicsSolver status per frame (1 = loop closed)
%     .regularization  struct of posture_weight, smooth_weight, backward,
%                  gap_weight, rom_weight, foot_orientation_weight,
%                  head_orientation_weight
%     .foot_orientation_error_deg  2 x frames 3D orientation error angle (deg,
%                  Left then Right), NaN where gap-filled or nonfinite
%     .head_orientation_error_deg  1 x frames 3D orientation error angle (deg),
%                  NaN where gap-filled or nonfinite

    arguments
        jc (1,1) struct
        opts.model (1,:) char = char(gs3dx_names().variants.human)
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
        opts.target_weight = []
        opts.foot_orientation_weight (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 0
        opts.head_orientation_weight (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 0
        opts.initial_pose = struct([])
    end
    assert(isscalar(opts.foot_orientation_weight) && isreal(opts.foot_orientation_weight) && ...
        isfinite(opts.foot_orientation_weight) && opts.foot_orientation_weight >= 0, ...
        'gs3dx:ik', 'foot_orientation_weight must be a finite real non-negative scalar');
    assert(isscalar(opts.head_orientation_weight) && isreal(opts.head_orientation_weight) && ...
        isfinite(opts.head_orientation_weight) && opts.head_orientation_weight >= 0, ...
        'gs3dx:ik', 'head_orientation_weight must be a finite real non-negative scalar');
    p_seed = gs3dx_ik_initial_pose(opts.initial_pose);
    if ~isempty(p_seed)
        assert(strcmp(opts.model, char(gs3dx_names().variants.human)), 'gs3dx:ik', ...
            'initial_pose keyed warm-start is restricted to model %s', char(gs3dx_names().variants.human));
    end
    head_data = gs3dx_head_input_data(jc, opts.frames, ...
        opts.calibration_frames, opts.head_orientation_weight);
    s = local_setup(opts.model, head_data.active);
    s.verbose = opts.verbose;
    s.backward = opts.backward;
    s.foot_orientation_weight = opts.foot_orientation_weight;
    s.head_orientation_weight = opts.head_orientation_weight;
    s.reg = local_regularization(s, opts.posture_weight, opts.smooth_weight);
    s.rom = local_rom(s, opts.rom_weight, opts.rom);
    has_foot_R = isfield(jc, 'foot_R_L') && isfield(jc, 'foot_R_R');
    if opts.foot_orientation_weight > 0
        assert(has_foot_R, 'gs3dx:ik', 'jc must contain foot_R_L and foot_R_R when foot_orientation_weight > 0');
    end
    feet = @(f) local_foot_data(jc, f, has_foot_R);
    head_fn = @(f) local_head_data(head_data, f);
    nt = numel(s.names);
    data = @(f) cell2mat(cellfun(@(n) jc.(n)(:, f), s.names, 'UniformOutput', false).');   % 3 x nt
    valid = @(f) ~cellfun(@(n) jc.gap.(n)(f), s.names).';   % 1 x nt, false where gap-filled
    tw = local_target_weight(s.names, opts.target_weight);
    weight = @(f) (valid(f) + opts.gap_weight * ~valid(f)) .* tw;   % residual weight per target
    assert(all(isfield(jc.gap, s.names)), 'gs3dx:ik', 'JC.gap lacks a target');
    lm = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'Display', 'off', ...
        'FiniteDifferenceStepSize', 1e-6, 'MaxIterations', 200);

    % Warm-start seed: if initial_pose is provided, use validated pose directly;
    % otherwise use standard target-free solve with pelvis translation shift.
    if isempty(p_seed)
        [p, g] = local_seed(s, jc.pelvis(:, opts.frames(1)), opts.posture_weight > 0);
        seed_source = 'target_free_solve';
    else
        [p, g] = local_apply_initial_pose(s, p_seed);
        seed_source = 'initial_pose';
    end
    if isempty(opts.offsets)
        cal = opts.calibration_frames;
        if isempty(cal)
            cal = opts.frames(1:3:end);
        end
        off = zeros(3, nt);
        for round = 1:opts.calibration_rounds
            best = local_track(s, cal, p, g, off, data, weight, feet, head_fn, lm);
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
    best = local_track(s, opts.frames, p, g, off, data, weight, feet, head_fn, lm);

    ik.model = opts.model;
    if ~isempty(p_seed), ik.seed_source = seed_source; end
    ik.independent_coordinate_count = s.roles.n_independent;
    ik.names = s.names;
    ik.offsets = cell2struct(num2cell(off, 1).', s.names, 1);
    ik.frames = opts.frames;
    ik.t = jc.t(opts.frames);
    ik.joint_ids = s.ids;
    ik.joint = zeros(numel(s.ids), n);
    ik.residual = zeros(nt, n);
    ik.points = zeros(3, nt, n);
    ik.status = zeros(1, n);
    ik.foot_orientation_error_deg = zeros(2, n);
    ik.head_orientation_error_deg = nan(1, n);
    ik.head_R = nan(3, 3, n);
    names_ref = ik.names;
    for i = 1:n
        [P, R, st, ~, R_feet, R_head] = local_fk(s, best(i).p, best(i).g);
        ik.joint(~s.closed, i) = local_targets(s, best(i).p);
        ik.joint(s.closed, i) = best(i).g;
        pts = P + reshape(pagemtimes(R(:, :, s.body), reshape(off, 3, 1, nt)), 3, nt);
        ik.points(:, :, i) = pts;
        ik.head_R(:, :, i) = R_head;
        d = data(opts.frames(i));
        ik.residual(:, i) = vecnorm(ik.points(:, :, i) - d).';
        ik.residual(~valid(opts.frames(i)), i) = NaN;   % gap-filled: not a measurement
        ik.status(i) = st;
        fd_i = feet(opts.frames(i));
        [~, finfo] = gs3dx_foot_orientation_residual(R_feet, fd_i.R_target, fd_i.gaps, 0);
        ik.foot_orientation_error_deg(:, i) = finfo.err_deg.';
        ik.foot_orientation_error_deg(~finfo.valid, i) = NaN;
        if head_data.active
            hd_i = head_fn(opts.frames(i));
            [~, hinfo] = gs3dx_head_orientation_residual(R_head, hd_i.R_target, hd_i.gap, 0);
            ik.head_orientation_error_deg(i) = hinfo.err_deg;
        end
    end
    assert(size(ik.points, 1) == 3 && size(ik.points, 2) == numel(ik.names) && ...
        size(ik.points, 3) == numel(ik.frames), 'gs3dx:ik', 'IK points size mismatch');
    assert(isequal(ik.names, names_ref), 'gs3dx:ik', 'IK names changed');
    ik.rms = sqrt(mean(ik.residual .^ 2, 1, 'omitnan'));
    ik.regularization = struct('posture_weight', opts.posture_weight, ...
        'smooth_weight', opts.smooth_weight, 'backward', opts.backward, 'gap_weight', opts.gap_weight, ...
        'rom_weight', opts.rom_weight, 'foot_orientation_weight', opts.foot_orientation_weight, ...
        'head_orientation_weight', opts.head_orientation_weight);
end

function tw = local_target_weight(names, tw_in)
% Per-target scale on the position residual (1 x numel(names)); default all ones.
    n = numel(names);
    if isempty(tw_in)
        tw = ones(1, n);
        return;
    end
    if isstruct(tw_in)
        tw = zeros(1, n);
        for k = 1:n
            nm = names{k};
            assert(isfield(tw_in, nm), 'gs3dx:ik', 'target_weight missing field %s', nm);
            tw(k) = tw_in.(nm);
        end
    elseif isnumeric(tw_in) && isvector(tw_in)
        assert(numel(tw_in) == n, 'gs3dx:ik', 'target_weight length must match targets (%d)', n);
        tw = tw_in(:).';
    else
        assert(false, 'gs3dx:ik', 'target_weight must be a struct or vector');
    end
    assert(all(isfinite(tw)) && all(tw >= 0), 'gs3dx:ik', 'target_weight must be non-negative');
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
    trunk = s.roles.is_trunk_coord;
    assert(nnz(trunk) == 7, 'gs3dx:ik', 'Expected 7 trunk coordinates, found %d', nnz(trunk));
    reg.posture = posture * double(trunk(:));
    reg.smooth = smooth * double(~endsWith(keys(:), [".Px", ".Py", ".Pz"]));
    reg.on = posture > 0 || smooth > 0;
    reg.wrap = ~endsWith(keys(:), [".Px", ".Py", ".Pz"]) & ~contains(keys(:), ".S");   % single angles
end

function best = local_track(s, frames, p, g, off, data, weight, feet, head, lm)
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
            fd = feet(frames(i));
            hd = head(frames(i));
            [p, g, cost, iters] = local_fit(free, p, g, off, d, w, fd, hd, lm);
            if cost < chain(i).cost
                chain(i) = struct('p', p, 'g', g, 'cost', cost);
            end
            if s.rom.on
                [pr, gr, cost, it] = local_fit(s, chain(i).p, chain(i).g, off, d, w, fd, hd, lm);
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

function s = local_setup(mdl, with_head)
% KinematicsSolver with the target points, the body rotations, the foot solid
% rotation frames and the right-arm loop joints as outputs.
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

    % Foot solid rotation frames (native foot frame +x is forward axis per Human / contact model)
    foot_l = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'RegExp', 'on', 'Name', '^(L Foot|Left Foot)$');
    foot_r = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'RegExp', 'on', 'Name', '^(R Foot|Right Foot)$');
    assert(~isempty(foot_l) && ~isempty(foot_r), 'gs3dx:ik', 'Could not locate L/R Foot solid blocks in %s', mdl);
    foot_l_blk = foot_l{1};
    for k = 1:numel(foot_l)
        ph = get_param(foot_l{k}, 'PortHandles');
        if isfield(ph, 'RConn') && ~isempty(ph.RConn)
            foot_l_blk = foot_l{k};
            break;
        end
    end
    foot_r_blk = foot_r{1};
    for k = 1:numel(foot_r)
        ph = get_param(foot_r{k}, 'PortHandles');
        if isfield(ph, 'RConn') && ~isempty(ph.RConn)
            foot_r_blk = foot_r{k};
            break;
        end
    end
    addFrameVariables(ks, 'r_foot_1', 'Rotation', world, [foot_l_blk '/R']);
    addFrameVariables(ks, 'r_foot_2', 'Rotation', world, [foot_r_blk '/R']);

    s.head_active = with_head;
    if with_head
        local_add_head_frame(ks, mdl, world);
    end

    % Model-independent identification of invariant joint roles
    roles = gs3dx_ik_joint_roles(jp);
    s.roles = roles;
    s.closed = roles.closed_mask;
    s.tv = roles.target_ids;
    addTargetVariables(ks, s.tv);
    addOutputVariables(ks, [string(ks.frameVariables.ID); roles.closed_ids]);
    addInitialGuessVariables(ks, roles.closed_ids);
    s.ks = ks;
    s.ids = ids;
    s.jkeys = gs3dx_joint_keys(mdl, jp);
    s.nt = size(T, 1);
    s.nb = numel(bodies);
    s.isdeg = containers.Map(cellstr(ids), num2cell(jp.Unit == "deg"));
    s.layout = roles.layout;
    assert(roles.n_independent > 0 && isfinite(roles.n_independent), 'gs3dx:ik', ...
        'Independent coordinates count must be positive and finite');
    assert(numel(roles.pelvis_trans_indices) == 3, 'gs3dx:ik', ...
        'Pelvis translation coordinate indices missing');
    assert(nnz(roles.is_trunk_coord) == 7, 'gs3dx:ik', ...
        'Expected exactly 7 trunk coordinates');
end

function [p, g] = local_seed(s, pelvis, trunk_zero)
% A pose that closes the grip loop (a target-free solve), moved onto PELVIS.
    ks0 = simscape.multibody.KinematicsSolver(s.ks.ModelName);
    addOutputVariables(ks0, s.ids);
    trunk = s.roles.trunk_ids;
    zero = [];
    if trunk_zero
        addTargetVariables(ks0, trunk);
        zero = zeros(numel(trunk), 1);
    end
    [q, st] = solve(ks0, zero, []);
    assert(st == 1, 'gs3dx:ik', 'The target-free solve did not close the loop (status %d)', st);
    [p, g] = local_pack_pose(s, q);
    P = local_fk(s, p, g);
    p(s.roles.pelvis_trans_indices) = p(s.roles.pelvis_trans_indices) + pelvis - P(:, 1);
end

function [p, g] = local_pack_pose(s, q)
% Pack KinematicsSolver variable vector q into parameter vector p and closed guess g.
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
end

function [p, g] = local_apply_initial_pose(s, p_seed)
% Map validated keyed initial pose into model coordinates and verify loop closure.
    if numel(p_seed.joint_keys) ~= numel(s.jkeys)
        error('gs3dx:ik', 'initial_pose key count (%d) does not match model joint count (%d)', ...
            numel(p_seed.joint_keys), numel(s.jkeys));
    end
    [found, loc] = ismember(s.jkeys, p_seed.joint_keys);
    if ~all(found) || any(~ismember(p_seed.joint_keys, s.jkeys))
        error('gs3dx:ik', 'initial_pose keys do not exactly match model joint keys');
    end
    expected_units = string(s.ks.jointPositionVariables.Unit);
    actual_units = p_seed.units(loc);
    if any(actual_units ~= expected_units)
        error('gs3dx:ik', 'initial_pose unit mismatch for model joint coordinates');
    end
    q = p_seed.joint(loc);
    [p, g] = local_pack_pose(s, q);
    [~, ~, st, g] = local_fk(s, p, g);
    if st ~= 1
        error('gs3dx:ik', 'initial_pose did not close the kinematic loop (status %d)', st);
    end
end

function [p, g, cost, iters] = local_fit(s, p, g, off, d, w, fd, hd, lm)
    if s.reg.on   % the same pose, each single angle on (-pi, pi]: the branch the posture pull sees
        p(s.reg.wrap) = p(s.reg.wrap) - 2 * pi * round(p(s.reg.wrap) / (2 * pi));
    end
    p0 = p;   % the warm start: the neighbour frame's solution
    r = @(pp) local_residual(s, pp, g, off, d, w, fd, hd, p0);
    [p, cost, ~, ~, out] = lsqnonlin(r, p, [], [], lm);
    iters = out.iterations;
    [~, ~, ~, g] = local_fk(s, p, g);
end

function r = local_residual(s, p, g, off, d, w, fd, hd, p0)
    [P, R, st, gc, R_feet, R_head] = local_fk(s, p, g);
    if st < 1
        r = 10 * ones(3 * s.nt + 2 * numel(p) * s.reg.on + s.rom.n + ...
            18 * (s.foot_orientation_weight > 0) + 9 * (s.head_orientation_weight > 0), 1);   % loop not closed: reject the step
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
    if s.foot_orientation_weight > 0
        r_feet = gs3dx_foot_orientation_residual(R_feet, fd.R_target, fd.gaps, s.foot_orientation_weight);
        r = [r; r_feet];
    end
    if s.head_orientation_weight > 0
        r_head = gs3dx_head_orientation_residual(R_head, hd.R_target, hd.gap, s.head_orientation_weight);
        r = [r; r_head];
    end
end

function [P, R, st, g, R_feet, R_head] = local_fk(s, p, g)
    [o, st] = solve(s.ks, local_targets(s, p), g);
    P = reshape(o(1:3 * s.nt), 3, s.nt);
    a = reshape(o(3 * s.nt + 1:3 * (s.nt + s.nb)), 3, s.nb) * pi / 180;
    R = zeros(3, 3, s.nb);
    for b = 1:s.nb   % intrinsic X-Y-Z, the KinematicsSolver 'Rotation' convention
        R(:, :, b) = local_rx(a(1, b)) * local_ry(a(2, b)) * local_rz(a(3, b));
    end
    idx_feet = 3 * (s.nt + s.nb) + (1:6);
    a_feet = reshape(o(idx_feet), 3, 2) * pi / 180;
    R_feet = zeros(3, 3, 2);
    for k = 1:2
        R_feet(:, :, k) = local_rx(a_feet(1, k)) * local_ry(a_feet(2, k)) * local_rz(a_feet(3, k));
    end
    R_head = nan(3, 3);
    if s.head_active
        idx_head = 3 * (s.nt + s.nb + 2) + (1:3);
        a_head = o(idx_head) * pi / 180;
        R_head = local_rx(a_head(1)) * local_ry(a_head(2)) * local_rz(a_head(3));
    end
    g = o(3 * (s.nt + s.nb + 2 + double(s.head_active)) + 1:end);
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

function fd = local_foot_data(jc, f, has_foot_R)
    if ~has_foot_R
        fd.R_target = repmat(eye(3), [1 1 2]);
        fd.gaps = [true, true];
        return;
    end
    fd.R_target = cat(3, jc.foot_R_L(:, :, f), jc.foot_R_R(:, :, f));
    gap_L = local_is_gap(jc, 'foot_R_L', f);
    gap_R = local_is_gap(jc, 'foot_R_R', f);
    fd.gaps = [gap_L, gap_R];
end

function g = local_is_gap(jc, name, f)
    g = false;
    if isfield(jc, 'gap') && isstruct(jc.gap) && isfield(jc.gap, name)
        gap_vec = jc.gap.(name);
        if f <= numel(gap_vec)
            g = logical(gap_vec(f));
        end
    end
end

function hd = local_head_data(data, f)
    hd.R_target = nan(3, 3);
    hd.gap = true;
    if data.active
        hd.R_target = data.R(:, :, f);
        hd.gap = data.gaps(f);
    end
end

function local_add_head_frame(ks, mdl, world)
% Optional output only; no head target means the original KS layout is retained.
    blocks = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'RegExp', 'on', 'Name', '^Head$');
    candidates = {};
    for k = 1:numel(blocks)
        ports = get_param(blocks{k}, 'PortHandles');
        if isfield(ports, 'RConn') && ~isempty(ports.RConn)
            candidates{end + 1} = blocks{k}; %#ok<AGROW>
        end
    end
    assert(numel(candidates) == 1, 'gs3dx:ik', ...
        'Expected exactly one Head solid reference port in %s', mdl);
    addFrameVariables(ks, 'r_head', 'Rotation', world, [candidates{1} '/R']);
end

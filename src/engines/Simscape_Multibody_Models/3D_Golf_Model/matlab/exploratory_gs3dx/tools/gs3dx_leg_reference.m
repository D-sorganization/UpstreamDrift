function ref = gs3dx_leg_reference(ik, jc, cap, opts)
%GS3DX_LEG_REFERENCE  Time-varying leg servo references from the capture (#10979).
%
%   REF = GS3DX_LEG_REFERENCE(IK, JC, CAP) turns a whole-trial
%   GS3DX_WHOLE_BODY_IK result into the leg angles the GS3DX leg servo
%   tracks, on the capture's time base (docs/ANTHROPOMETRY.md plan, step 2).
%   JC is GS3DX_CAPTURE_JOINT_CENTRES(CAP); CAP is GS3DX_CAPTURE_MARKERS.
%
%   1. Pelvis path.  The model's pelvis frame ('Lower Torso', the follower
%      of the pelvis joint) is evaluated at every tracked frame with the
%      model's own kinematics (KinematicsSolver on IK.model).
%   2. Foot path.  Each foot frame follows the measured foot: position =
%      the ankle joint centre JC.ankle_*, orientation = the address foot
%      frame (flat, turned by GS3DX_CAPTURE_STANCE foot_yaw, as
%      GS3DX_BUILD_CONTACT) carried by the rotation of the marker triad
%      (ankle centre, ToeIn, ToeOut) since address.  The feet are not
%      planted: the trail heel is already ~36 mm up at impact.  The
%      whole-body IK has no toe target, so its ankle angles are not used.
%   3. Floor.  The ankle joint centres are marker proxies whose heights
%      differ between the feet (the right sole sat 18.5 mm above the left
%      at address), so each foot path is shifted vertically by a constant
%      that puts both address soles (AnkleHeight below the ankle) on their
%      mean height (.foot_shift).
%   4. Filtering.  Pelvis and foot positions and quaternions are low-pass
%      filtered, 4th-order zero-phase Butterworth at CUTOFF_HZ (quaternions
%      renormalized).
%   5. Torsion.  The model ankle is a universal joint with no axial
%      rotation, so a foot pose fixes the knee's swivel about the
%      hip-ankle line.  Prescribing the measured foot yaw put the knees
%      4-5 cm (p95 8-9 cm) from the capture; leaving the yaw free (ankle
%      position, sole normal and knee fitted) matched the knees but let the
%      foot spin by up to 99 deg once the lead knee straightens and the
%      knee no longer fixes the swivel.  So a constant yaw offset per foot
%      (the torsion the model lacks) is fitted over the address frames:
%      weighted least squares on the ankle position (100), the sole normal
%      (0.3 m/rad) and the filtered knee joint centre JC.knee_* (1).
%   6. Leg angles.  The foot path turned by that offset about the vertical
%      is the target of GS3DX_LEG_IK for every filtered pelvis pose (exact,
%      six angles for six numbers); the rates are the time derivative of
%      the angles.

%   The capture frame is the model World, as in GS3DX_WHOLE_BODY_IK, so the
%   address target frame [facing, toward target, up] is also the leg frame.
%
%   Reach.  Near impact the lead knee is straight and the IK pelvis sits up
%   to ~1% of the leg length too far from the lead ankle (within the IK
%   leg residual).  Where the hip-to-ankle distance exceeds MAX_REACH of
%   the leg length, the ankle target moves toward the hip onto that
%   distance; REF.clamp records by how much.
%
%   Options: cutoff_hz (10), address_frames (1:150, the still address),
%   max_reach (0.999).
%
%   REF fields (angles in deg, the servo order [L hip X Y Z, knee, ankle X
%   Y, then R], as LegAngleReference):
%     .model, .frames, .t (s, from the first frame), .rate_hz (of the IK
%                          frames, which may skip capture frames evenly),
%                          .cutoff_hz
%     .q, .qd              12 x frames angles and rates
%     .pelvis_R, .pelvis_p filtered pelvis frame (3x3xN, 3xN, World)
%     .pelvis_joint        the same path in pelvis-joint coordinates:
%                          .translation (3xN m), .xyz (3xN deg, follower
%                          X-Y-Z) and their rates (the start variables),
%                          and .base (.R, .p) the joint's World base frame
%     .feet                per side (L/R): .R (3x3xN), .p (3xN) model foot
%                          frame (the measured one turned by .torsion)
%     .torsion             1 x 2 foot yaw offset, model minus measured (deg)
%     .foot_shift          1 x 2 vertical foot path shift (m)
%     .reach               2 x frames hip-to-ankle distance / leg length,
%                          before the clamp
%     .clamp               2 x frames ankle target moved toward the hip (m)
%     .knee                per side: 3xN knee joint centre of the reference
%     .knee_error          2 x frames distance from JC.knee_L/_R (m, NaN
%                          where gap-filled)
%     .torsion_fit         per side: the address fit (.knee_error m,
%                          .yaw_error deg per address frame)

    arguments
        ik (1,1) struct
        jc (1,1) struct
        cap (1,1) struct
        opts.cutoff_hz (1,1) double {mustBePositive} = 10
        opts.address_frames (1,:) double {mustBeInteger, mustBePositive} = 1:150
        opts.max_reach (1,1) double {mustBeInRange(opts.max_reach, 0.9, 1, 'exclude-upper')} = 0.999
    end
    frames = ik.frames;
    stride = max(1, frames(min(2, end)) - frames(1));
    assert(numel(frames) >= 16 && isequal(frames, frames(1):stride:frames(end)), 'gs3dx:legref', ...
        'IK frames must be at least 16, evenly spaced');
    assert(all(ik.status >= 1), 'gs3dx:legref', 'IK has frames whose grip loop did not close');
    rate = 1 / mean(diff(jc.t)) / stride;   % of the IK frames
    assert(opts.cutoff_hz < rate / 2, 'gs3dx:legref', 'Cutoff above Nyquist (%g Hz)', rate / 2);
    mdl = ik.model;
    load_system(mdl);
    ws = get_param(mdl, 'ModelWorkspace');

    [b, a] = butter(4, opts.cutoff_hz / (rate / 2));
    [R, p, base] = local_pelvis(ik);
    [ref.pelvis_R, ref.pelvis_p] = local_filter(b, a, R, p);
    ref.pelvis_joint = local_pelvis_joint(base, ref.pelvis_R, ref.pelvis_p, rate);

    stance = gs3dx_capture_stance();
    q0 = ws.getVariable('LegAngleReference');
    sides = 'LR';
    n = numel(frames);
    ref.q = zeros(12, n);
    ref.knee_error = nan(2, n);
    ref.reach = zeros(2, n);
    ref.clamp = zeros(2, n);
    ref.torsion = zeros(1, 2);
    at = find(ismember(frames, opts.address_frames));
    assert(~isempty(at), 'gs3dx:legref', 'The IK frames include no address frame');
    h = ws.getVariable('AnkleHeight');
    sole = zeros(1, 2);
    for k = 1:2
        P = sides(k);
        [R, p] = local_foot(jc, cap, P, frames, opts.address_frames, stance.(['foot_yaw_' P]));
        [feet.(P).R, feet.(P).p] = local_filter(b, a, R, p);
        sole(k) = median(feet.(P).p(3, at) - h * reshape(feet.(P).R(3, 3, at), 1, []));
    end
    ref.foot_shift = mean(sole) - sole;   % both address soles on one floor
    for k = 1:2
        P = sides(k);
        foot = feet.(P);
        foot.p(3, :) = foot.p(3, :) + ref.foot_shift(k);
        geom = struct('mount_R', ws.getVariable([P 'HipMountRotation']), ...
            'mount_p', ws.getVariable([P 'HipMountOffset']), ...
            'thigh', ws.getVariable('ThighLength'), 'shank', ws.getVariable('ShankLength'));
        hip = ref.pelvis_p + reshape(pagemtimes(ref.pelvis_R, geom.mount_p), 3, []);
        ref.reach(k, :) = vecnorm(hip - foot.p) / (geom.thigh + geom.shank);
        over = ref.reach(k, :) > opts.max_reach;   % a straight knee, the IK pelvis a little high
        d = foot.p(:, over) - hip(:, over);
        foot.p(:, over) = hip(:, over) + d ./ ref.reach(k, over) * opts.max_reach;
        ref.clamp(k, :) = 0;
        ref.clamp(k, over) = (ref.reach(k, over) - opts.max_reach) * (geom.thigh + geom.shank);
        [~, knee] = local_filter(b, a, repmat(eye(3), 1, 1, n), jc.(['knee_' P])(:, frames));
        rows = (k - 1) * 6 + (1:6);
        sub = struct('R', foot.R(:, :, at), 'p', foot.p(:, at));
        fit = local_leg_fit(geom, ref.pelvis_R(:, :, at), ref.pelvis_p(:, at), sub, knee(:, at), q0(rows));
        ref.torsion(k) = median(fit.yaw_error);
        ref.torsion_fit.(P) = struct('knee_error', vecnorm(fit.knee - knee(:, at)), 'yaw_error', fit.yaw_error);
        foot.R = pagemtimes(local_rz(ref.torsion(k)), foot.R);
        ref.q(rows, :) = local_leg_ik(geom, ref.pelvis_R, ref.pelvis_p, foot, fit.q(:, 1));
        ref.feet.(P) = foot;
        ref.knee.(P) = local_knee(geom, ref.pelvis_R, ref.pelvis_p, ref.q(rows, :));
        e = vecnorm(ref.knee.(P) - jc.(['knee_' P])(:, frames));
        e(jc.gap.(['knee_' P])(frames)) = NaN;
        ref.knee_error(k, :) = e;
    end
    ref.qd = gradient(ref.q, 1 / rate);
    ref.model = mdl;
    ref.frames = frames;
    ref.t = (frames - frames(1)) / (rate * stride);
    ref.rate_hz = rate;
    ref.cutoff_hz = opts.cutoff_hz;
end

function [R, p, base] = local_pelvis(ik)
% Pelvis frame ('Lower Torso') of IK.model in World at every IK frame, and
% the fixed base frame of the pelvis 6-DOF joint (World = base * joint).
    mdl = ik.model;
    wf = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'ReferenceBlock', 'sm_lib/Frames and Transforms/World Frame');
    ks = simscape.multibody.KinematicsSolver(mdl);
    ids = string(ks.jointPositionVariables.ID);
    assert(isequal(ids(:), string(ik.joint_ids(:))), 'gs3dx:legref', 'IK.joint_ids do not match %s', mdl);
    paths = string(unique(ks.jointPositionVariables.BlockPath, 'stable'));
    pelvis = [char(paths(contains(paths, 'Hip Kinetically Driven/Hip Joint'))) '/F'];   % 'Lower Torso'
    addFrameVariables(ks, 'pelvis', 'Translation', [wf{1} '/W'], pelvis);
    addFrameVariables(ks, 'pelvis', 'Rotation', [wf{1} '/W'], pelvis);
    closed = startsWith(ids, ["j15.", "j18.", "j19."]);   % the right-arm loop, as in the IK
    addTargetVariables(ks, ids(~closed));
    addOutputVariables(ks, string(ks.frameVariables.ID));
    addInitialGuessVariables(ks, ids(closed));
    n = numel(ik.frames);
    R = zeros(3, 3, n);
    p = zeros(3, n);
    for i = 1:n
        [o, st] = solve(ks, ik.joint(~closed, i), ik.joint(closed, i));
        assert(st >= 1, 'gs3dx:legref', 'Pelvis FK failed at IK frame %d (status %d)', ik.frames(i), st);
        p(:, i) = o(1:3);
        a = o(4:6);   % intrinsic X-Y-Z (deg), the KinematicsSolver 'Rotation' convention
        R(:, :, i) = local_rx(a(1)) * local_ry(a(2)) * local_rz(a(3));
    end
    j = @(id) ik.joint(ids == id, :);
    P = [j("j1.Px.p"); j("j1.Py.p"); j("j1.Pz.p")];
    S = [j("j1.S.ax_x"); j("j1.S.ax_y"); j("j1.S.ax_z"); j("j1.S.q") * pi / 180].';
    base.R = R(:, :, 1) / axang2rotm(S(1, :));
    base.p = p(:, 1) - base.R * P(:, 1);
    for i = 1:n
        err = norm(base.R * axang2rotm(S(i, :)) - R(:, :, i)) + norm(base.p + base.R * P(:, i) - p(:, i));
        assert(err < 1e-6, 'gs3dx:legref', 'The pelvis joint base frame moves (frame %d, %.3g)', ik.frames(i), err);
    end
end

function j = local_pelvis_joint(base, R, p, rate)
% Pelvis 6-DOF joint coordinates of a World pelvis path: translation (m)
% and the follower-axes X-Y-Z sequence (deg) of its Spherical primitive,
% as TranslationStartPosition* and HipStartPosition* take them, and rates.
    n = size(p, 2);
    j.base = base;
    j.translation = base.R.' * (p - base.p);
    j.xyz = zeros(3, n);
    for i = 1:n
        M = base.R.' * R(:, :, i);   % Rx(a) * Ry(b) * Rz(c)
        j.xyz(:, i) = [atan2d(-M(2, 3), M(3, 3)); asind(M(1, 3)); atan2d(-M(1, 2), M(1, 1))];
    end
    assert(all(abs(j.xyz(2, :)) < 80), 'gs3dx:legref', 'Pelvis X-Y-Z sequence near gimbal lock');
    j.xyz = rad2deg(unwrap(deg2rad(j.xyz), [], 2));
    j.translation_rate = gradient(j.translation, 1 / rate);
    j.xyz_rate = gradient(j.xyz, 1 / rate);
end

function [R, p] = local_filter(b, a, R, p)
% Zero-phase low-pass of a pose path: positions and unit quaternions (kept
% in one hemisphere so the filter sees a continuous signal).
    Q = rotm2quat(R).';   % 4 x N
    for k = 2:size(Q, 2)
        if dot(Q(:, k), Q(:, k - 1)) < 0
            Q(:, k) = -Q(:, k);
        end
    end
    Q = filtfilt(b, a, Q.').';
    R = quat2rotm((Q ./ vecnorm(Q)).');
    p = filtfilt(b, a, p.').';
end

function [R, p] = local_foot(jc, cap, P, frames, address, yaw)
% Foot frame path: the ankle joint centre, and the address foot frame
% rz(YAW) carried by the rotation of the (ankle, ToeIn, ToeOut) triad.
    S = cap.target_frame;
    w = (cap.marker("WaistLeft") + cap.marker("WaistRight") + ...
        cap.marker("WaistLBack") + cap.marker("WaistRBack")) / 4;
    origin = w(:, find(all(~isnan(w), 1), 1));   % the JC origin
    local = @(name) S.' * (fillmissing(cap.marker(string(P) + name), 'linear', 2) - origin);
    ankle = jc.(['ankle_' P]);
    toe_in = local("ToeIn");
    toe_out = local("ToeOut");
    F = zeros(3, 3, size(ankle, 2));
    for f = 1:size(ankle, 2)
        x = (toe_in(:, f) + toe_out(:, f)) / 2 - ankle(:, f);
        x = x / norm(x);
        z = cross(x, toe_in(:, f) - toe_out(:, f));
        z = z / norm(z);
        F(:, :, f) = [x, cross(z, x), z];
    end
    ok = address(~jc.gap.(['ankle_' P])(address));
    assert(~isempty(ok), 'gs3dx:legref', 'No measured %s ankle in the address frames', P);
    F0 = local_mean_rotation(F(:, :, ok));
    R0 = [cosd(yaw) -sind(yaw) 0; sind(yaw) cosd(yaw) 0; 0 0 1];
    R = pagemtimes(pagemtimes(F(:, :, frames), F0.'), R0);
    p = ankle(:, frames);
end

function R = local_mean_rotation(F)
    [U, ~, V] = svd(sum(F, 3));
    R = U * diag([1 1 det(U * V.')]) * V.';
end

function fit = local_leg_fit(geom, pelvis_R, pelvis_p, foot, knee, q0)
% Weighted least-squares leg angles with the foot yaw free, one pose at a
% time, seeded by the last.
    n = size(pelvis_p, 2);
    q = zeros(6, n);
    fit.knee = zeros(3, n);
    [fit.ankle_error, fit.sole_error, fit.yaw_error] = deal(zeros(1, n));
    for k = 1:n
        pR = pelvis_R(:, :, k);
        pp = pelvis_p(:, k);
        fR = foot.R(:, :, k);
        f = @(v) local_leg_residual(geom, pR, pp, v, foot.p(:, k), fR(:, 3), knee(:, k));
        q(:, k) = local_lm(f, q0);
        q0 = q(:, k);
        [R, p] = gs3dx_leg_fk(geom, pR, pp, q0);
        fit.knee(:, k) = local_knee(geom, pR, pp, q0);
        fit.ankle_error(k) = norm(p - foot.p(:, k));
        fit.sole_error(k) = acosd(min(1, R(:, 3).' * fR(:, 3)));
        fit.yaw_error(k) = atan2d(R(2, 1), R(1, 1)) - atan2d(fR(2, 1), fR(1, 1));
    end
    fit.yaw_error = mod(fit.yaw_error + 180, 360) - 180;
    fit.q = q;
end

function q = local_leg_ik(geom, pelvis_R, pelvis_p, foot, q0)
% GS3DX_LEG_IK for a moving foot: one pose at a time, seeded by the last.
    n = size(pelvis_p, 2);
    q = zeros(6, n);
    for k = 1:n
        q(:, k) = gs3dx_leg_ik(geom, pelvis_R(:, :, k), pelvis_p(:, k), foot.R(:, :, k), foot.p(:, k), q0);
        q0 = q(:, k);
    end
end

function e = local_leg_residual(geom, pR, pp, v, ankle, normal, knee)
    [R, p] = gs3dx_leg_fk(geom, pR, pp, v);
    e = [100 * (p - ankle); 0.3 * cross(R(:, 3), normal); local_knee(geom, pR, pp, v) - knee];
end

function x = local_lm(f, x)
% Levenberg-Marquardt with a central-difference Jacobian (deg).
    e = f(x);
    lambda = 1e-3;
    for it = 1:200
        J = zeros(numel(e), numel(x));
        for c = 1:numel(x)
            d = zeros(size(x));
            d(c) = 1e-6;
            J(:, c) = (f(x + d) - f(x - d)) / 2e-6;
        end
        step = -(J.' * J + lambda * eye(numel(x))) \ (J.' * e);
        et = f(x + step);
        if norm(et) < norm(e)
            x = x + step;
            done = norm(e) - norm(et) < 1e-12 * max(1, norm(e));
            e = et;
            lambda = max(lambda / 10, 1e-12);
            if done
                return;
            end
        else
            lambda = lambda * 10;
            if lambda > 1e8
                return;
            end
        end
    end
end

function k = local_knee(geom, pelvis_R, pelvis_p, q)
% Knee joint centre: the GS3DX_LEG_FK chain up to the knee mount.
    n = size(q, 2);
    k = zeros(3, n);
    for i = 1:n
        Rh = pelvis_R(:, :, i) * geom.mount_R * local_rx(q(1, i)) * local_ry(q(2, i)) * local_rz(q(3, i));
        k(:, i) = pelvis_p(:, i) + pelvis_R(:, :, i) * geom.mount_p + Rh * [0; 0; -geom.thigh];
    end
end

function R = local_rx(a)
    R = [1 0 0; 0 cosd(a) -sind(a); 0 sind(a) cosd(a)];
end

function R = local_ry(a)
    R = [cosd(a) 0 sind(a); 0 1 0; -sind(a) 0 cosd(a)];
end

function R = local_rz(deg)
    R = [cosd(deg) -sind(deg) 0; sind(deg) cosd(deg) 0; 0 0 1];
end

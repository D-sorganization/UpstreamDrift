function jc = gs3dx_capture_joint_centres(cap, opts)
%GS3DX_CAPTURE_JOINT_CENTRES  Joint-centre estimates from the capture markers (#10979).
%
%   JC = GS3DX_CAPTURE_JOINT_CENTRES(CAP) turns the skin and shoe markers
%   of CAP (GS3DX_CAPTURE_MARKERS) into per-frame joint-centre estimates in
%   the address target frame [facing, toward target, up] with the origin
%   at the address waist centre (as GS3DX_CAPTURE_STANCE).  Gaps are
%   filled linearly.  Every point is 3 x frames, in m:
%
%     pelvis      waist centre (mean of the four waist markers)
%     pelvis_R    3 x 3 x frames pelvis axes [forward, left, up]: left from
%                 the right to the left waist markers, forward from the
%                 back to the front ones made normal to left, up = f x l
%     hip_L/R     pelvis + pelvis_R * [0; +-hip_half_width; -hip_drop]
%                 (the GS3DX leg mount convention, GS3DX_LEG_TABLE)
%     knee_L/R    KneeOut moved knee_inset toward the body midline along
%                 the pelvis left axis
%     ankle_L/R   AnkleOut moved ankle_inset toward the midline likewise
%     toe_L/R     mean of ToeIn and ToeOut
%     c7          BackTop
%     shoulder_L/R  acromion marker lowered shoulder_drop along the thorax
%                 up axis; RShoulderTop is missing in 80% of frames, so it
%                 is rebuilt from RShoulderBack with the mean
%                 RShoulderBack -> RShoulderTop offset in the thorax frame
%                 (BackTop, BackLeft, BackRight) over the frames that have both
%     elbow_L/R   ElbowOut (lateral epicondyle; no inset: the lead arm is
%                 nearly straight, so the flexion axis is ill-defined)
%     wrist_L/R   WristTop
%     club_grip, club_head   club marker cluster centroids
%
%   JC.gap.(point) is 1 x frames, true where a marker the point is built
%   from was missing (the point is then gap-filled, not measured).
%
%   JC.lengths holds the median and the standard deviation over frames of
%   the distance between the two ends of each segment (a rigid segment
%   keeps its length, so the SD measures how well the proxies track).
%   JC.t (s) and JC.impact_frame come from CAP.
%
%   Every offset is a proxy from typical anatomy, not measured on this
%   golfer; options: hip_half_width (0.09 m), hip_drop (0.10 m),
%   knee_inset (0.05 m), ankle_inset (0.035 m), shoulder_drop (0.04 m).

    arguments
        cap (1,1) struct
        opts.hip_half_width (1,1) double {mustBeNonnegative} = 0.09
        opts.hip_drop (1,1) double {mustBeNonnegative} = 0.10
        opts.knee_inset (1,1) double {mustBeNonnegative} = 0.05
        opts.ankle_inset (1,1) double {mustBeNonnegative} = 0.035
        opts.shoulder_drop (1,1) double {mustBeNonnegative} = 0.04
    end
    tf = gs3dx_capture_address_transform(cap);
    n = cap.n_frames;
    fill = tf.fill;
    local = tf.local;
    m = tf.m;

    front = (m("WaistLeft") + m("WaistRight")) / 2;
    back = (m("WaistLBack") + m("WaistRBack")) / 2;
    left = local_unit((m("WaistLeft") + m("WaistLBack")) / 2 - (m("WaistRight") + m("WaistRBack")) / 2);
    fwd = front - back;
    fwd = local_unit(fwd - sum(fwd .* left, 1) .* left);
    up = cross(fwd, left, 1);
    jc.pelvis = (front + back) / 2;
    jc.pelvis_R = permute(cat(3, fwd, left, up), [1 3 2]);   % 3 x 3 x n

    thorax = local_thorax(m("BackTop"), m("BackLeft"), m("BackRight"));
    r_top = local_rebuild(cap.marker("RShoulderTop"), m("RShoulderBack"), thorax, local);
    for s = ["L", "R"]
        sgn = 1 - 2 * (s == "R");   % +1 left, -1 right
        mount = pagemtimes(jc.pelvis_R, [0; sgn * opts.hip_half_width; -opts.hip_drop]);
        jc.("hip_" + s) = jc.pelvis + reshape(mount, 3, n);
        jc.("knee_" + s) = m(s + "KneeOut") - sgn * opts.knee_inset * left;
        jc.("ankle_" + s) = m(s + "AnkleOut") - sgn * opts.ankle_inset * left;
        jc.("toe_" + s) = (m(s + "ToeIn") + m(s + "ToeOut")) / 2;
        jc.("elbow_" + s) = m(s + "ElbowOut");
        jc.("wrist_" + s) = m(s + "WristTop");
    end
    thorax_up = reshape(thorax(:, 3, :), 3, n);
    jc.shoulder_L = m("LShoulderTop") - opts.shoulder_drop * thorax_up;
    jc.shoulder_R = r_top - opts.shoulder_drop * thorax_up;
    jc.c7 = m("BackTop");
    jc.club_grip = local(fill(cap.club_grip));
    jc.club_head = local(fill(cap.club_head));

    gap = @(names) any(cell2mat(arrayfun(@(nm) any(isnan(cap.marker(nm)), 1), names(:), 'UniformOutput', false)), 1);
    waist_gap = gap(["WaistLeft", "WaistRight", "WaistLBack", "WaistRBack"]);
    jc.gap = struct('pelvis', waist_gap, 'c7', gap("BackTop"), ...
        'shoulder_L', gap(["LShoulderTop", "BackTop", "BackLeft", "BackRight"]), ...
        'shoulder_R', gap(["RShoulderBack", "BackTop", "BackLeft", "BackRight"]), ...
        'club_grip', any(isnan(cap.club_grip), 1), 'club_head', any(isnan(cap.club_head), 1));
    for s = ["L", "R"]
        jc.gap.("hip_" + s) = waist_gap;
        jc.gap.("knee_" + s) = gap([s + "KneeOut", "WaistLeft", "WaistRight", "WaistLBack", "WaistRBack"]);
        jc.gap.("ankle_" + s) = gap([s + "AnkleOut", "WaistLeft", "WaistRight", "WaistLBack", "WaistRBack"]);
        jc.gap.("toe_" + s) = gap([s + "ToeIn", s + "ToeOut"]);
        jc.gap.("elbow_" + s) = gap(s + "ElbowOut");
        jc.gap.("wrist_" + s) = gap(s + "WristTop");
    end
    jc.t = (0:n - 1) / cap.rate_hz;
    jc.impact_frame = cap.impact_frame;

    segs = struct( ...
        'thigh_L', {{'hip_L', 'knee_L'}}, 'thigh_R', {{'hip_R', 'knee_R'}}, ...
        'shank_L', {{'knee_L', 'ankle_L'}}, 'shank_R', {{'knee_R', 'ankle_R'}}, ...
        'upper_arm_L', {{'shoulder_L', 'elbow_L'}}, 'upper_arm_R', {{'shoulder_R', 'elbow_R'}}, ...
        'forearm_L', {{'elbow_L', 'wrist_L'}}, 'forearm_R', {{'elbow_R', 'wrist_R'}}, ...
        'shoulders', {{'shoulder_L', 'shoulder_R'}}, 'club', {{'club_grip', 'club_head'}});
    jc.lengths = struct();
    for f = fieldnames(segs).'
        ends = segs.(f{1});
        d = vecnorm(jc.(ends{1}) - jc.(ends{2}));
        jc.lengths.(f{1}) = [median(d), std(d)];
    end
    mid = (jc.shoulder_L + jc.shoulder_R) / 2;
    d = vecnorm(mid - jc.pelvis);
    jc.lengths.pelvis_to_shoulders = [median(d), std(d)];
end

function u = local_unit(v)
    u = v ./ vecnorm(v);
end

function T = local_thorax(c7, back_left, back_right)
% Thorax axes [forward, left, up] per frame: left from BackRight to
% BackLeft, up from the BackLeft/BackRight midpoint to C7 made normal to left.
    left = local_unit(back_left - back_right);
    up = c7 - (back_left + back_right) / 2;
    up = local_unit(up - sum(up .* left, 1) .* left);
    fwd = cross(left, up, 1);
    T = permute(cat(3, fwd, left, up), [1 3 2]);
end

function top = local_rebuild(top_raw, back, thorax, local)
% RShoulderTop from RShoulderBack plus its mean offset in the thorax frame.
    have = all(~isnan(top_raw), 1);
    assert(nnz(have) >= 20, 'gs3dx:jc', 'RShoulderTop is present in only %d frames', nnz(have));
    top_raw = local(top_raw);
    off = pagemtimes(permute(thorax(:, :, have), [2 1 3]), reshape(top_raw(:, have) - back(:, have), 3, 1, []));
    top = back + reshape(pagemtimes(thorax, mean(off, 3)), 3, []);
    top(:, have) = top_raw(:, have);
end

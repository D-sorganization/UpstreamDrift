function pred = gs3dx_club_prediction(ik, jc, cap)
%GS3DX_CLUB_PREDICTION  Club head, grip and face normal from whole-body IK (#11160).
%
%   PRED = GS3DX_CLUB_PREDICTION(IK, JC, CAP) maps a GS3DX_WHOLE_BODY_IK
%   result onto the club track GS3DX_CLUB_MATCH expects: head and grip
%   positions and a club-face normal, all in the capture address target
%   frame (the IK world, m).
%
%   CAP is GS3DX_CAPTURE_MARKERS (six club markers for the Kabsch cluster
%   frame and the reference face normal).  JC supplies measured club_head
%   and club_grip for calibrating the grip offset and n_body; they are not
%   added to the IK residual.
%
%   PRED fields (each 3 x numel(IK.frames)):
%     .head          club-head position (IK club_head target point)
%     .grip          hand-sphere midpoint plus a constant club-head offset
%     .face_normal   unit face normal from the club-head body rotation
%     .ref_face_normal  capture reference normal (cluster-frame axis), for
%                    acceptance tests; NaN where the six club markers drop out

    arguments
        ik (1,1) struct
        jc (1,1) struct
        cap (1,1) struct
    end
    need = {'points', 'names', 'frames', 'status', 'joint', 'joint_ids', 'model'};
    assert(all(isfield(ik, need)), 'gs3dx:club', 'IK must be a GS3DX_WHOLE_BODY_IK result');
    assert(all(isfield(jc, {'club_head', 'club_grip'})), 'gs3dx:club', ...
        'JC must carry club_head and club_grip');
    ih = find(strcmp(ik.names, 'club_head'), 1);
    assert(~isempty(ih), 'gs3dx:club', 'IK.names must include club_head');

    n = numel(ik.frames);
    pred.head = squeeze(ik.points(:, ih, :));
    assert(isequal(size(pred.head), [3 n]), 'gs3dx:club', 'club_head points size mismatch');

    [hand, ikf] = gs3dx_hand_centres(ik);
    mid = squeeze(mean(hand, 2));
    R = local_club_rotations(ik);

    [X, ok_cap] = gs3dx_club_markers_address(cap);
    ref = find(ok_cap, 1);
    M0 = X(:, :, ref);
    c0 = mean(M0, 2);
    pose = @(f) gs3dx_kabsch(M0, c0, X(:, :, f));

    addr_col = find(arrayfun(@(i) ok_cap(ik.frames(i)), 1:n), 1);
    assert(~isempty(addr_col), 'gs3dx:club', 'No IK frame has all six club markers');
    f_addr = ik.frames(addr_col);
    e_cluster = local_cluster_face_axis(pose, jc, f_addr);

    ref_face = nan(3, n);
    for i = 1:n
        f = ik.frames(i);
        if ok_cap(f)
            [Rk, ~] = pose(f);
            ref_face(:, i) = Rk * e_cluster;
        end
    end
    pred.ref_face_normal = ref_face;

    [Rk_addr, ~] = pose(f_addr);
    assert(norm(ref_face(:, addr_col) - Rk_addr * e_cluster) < 1e-12, 'gs3dx:club', ...
        'Postcondition: ref_face must equal Rk*e_cluster on the axis frame');
    shaft = jc.club_head(:, f_addr) - jc.club_grip(:, f_addr);
    shaft = local_unit(shaft);
    n_plane = ref_face(:, addr_col);
    n_plane = n_plane - shaft * dot(n_plane, shaft);
    n_plane = local_unit(n_plane);
    assert(dot(n_plane, [0; 1; 0]) > 0, 'gs3dx:club', ...
        'Postcondition: ref_face normal must align with +Y after shaft removal');

    off_acc = zeros(3, 0);
    n_acc = zeros(3, 0);
    for j = 1:numel(ikf)
        i = ikf(j);
        f = ik.frames(i);
        if all(isfinite(jc.club_grip(:, f)))
            d = jc.club_grip(:, f) - mid(:, j);
            off_acc(:, end + 1) = R(:, :, i).' * d; %#ok<AGROW>
        end
        if all(isfinite(ref_face(:, i)))
            n_acc(:, end + 1) = R(:, :, i).' * ref_face(:, i); %#ok<AGROW>
        end
    end
    assert(size(off_acc, 2) > 0, 'gs3dx:club', 'No frames to calibrate the grip offset');
    assert(size(n_acc, 2) > 0, 'gs3dx:club', 'No frames to calibrate the face normal');
    off_body = mean(off_acc, 2);
    n_body = local_unit(mean(n_acc, 2));

    pred.grip = nan(3, n);
    pred.face_normal = nan(3, n);
    for j = 1:numel(ikf)
        i = ikf(j);
        pred.grip(:, i) = mid(:, j) + R(:, :, i) * off_body;
        pred.face_normal(:, i) = R(:, :, i) * n_body;
    end

    assert(all(abs(vecnorm(pred.face_normal(:, ikf)) - 1) < 1e-9, 'all'), 'gs3dx:club', ...
        'Postcondition: face_normal is not unit length');
    assert(all(isfinite(pred.grip(:, ikf)), 'all'), 'gs3dx:club', ...
        'Postcondition: grip is not finite on frames with status >= 1');

    for i = 1:n
        f = ik.frames(i);
        if ok_cap(f)
            jc_mid = (jc.club_head(:, f) + jc.club_grip(:, f)) / 2;
            assert(norm(mean(X(:, :, f), 2) - jc_mid) < 1e-9, 'gs3dx:club', ...
                'Postcondition: Kabsch markers not in address frame at frame %d', f);
        end
    end
end

function e = local_cluster_face_axis(pose, jc, addr_frame)
% Unit axis in the Kabsch cluster frame used as the capture face normal.
    f = addr_frame;
    shaft = jc.club_head(:, f) - jc.club_grip(:, f);
    shaft = local_unit(shaft);
    target = [0; 1; 0];
    target = target - shaft * dot(target, shaft);
    target = local_unit(target);
    [R, ~] = pose(f);
    best = -inf;
    pick = zeros(3, 1);
    for ax = 1:3
        for sgn = [1, -1]
            e = zeros(3, 1);
            e(ax) = sgn;
            n = R * e;
            n = n - shaft * dot(n, shaft);
            n = local_unit(n);
            s = dot(n, target);
            if s > best
                best = s;
                pick = e;
            end
        end
    end
    e = pick;
end

function R = local_club_rotations(ik)
    mdl = ik.model;
    load_system(mdl);
    wf = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'ReferenceBlock', 'sm_lib/Frames and Transforms/World Frame');
    club = find_system(mdl, 'LookUnderMasks', 'all', 'Name', 'Clubhead');
    ks = simscape.multibody.KinematicsSolver(mdl);
    ids = string(ks.jointPositionVariables.ID);
    assert(isequal(ids(:), string(ik.joint_ids(:))), 'gs3dx:club', 'IK.joint_ids do not match %s', mdl);
    addFrameVariables(ks, 'club', 'Rotation', [wf{1} '/W'], [club{1} '/R']);
    closed = startsWith(ids, ["j15.", "j18.", "j19."]);
    addTargetVariables(ks, ids(~closed));
    addOutputVariables(ks, string(ks.frameVariables.ID));
    addInitialGuessVariables(ks, ids(closed));
    n = numel(ik.frames);
    R = nan(3, 3, n);
    for i = find(ik.status >= 1)
        o = solve(ks, ik.joint(~closed, i), ik.joint(closed, i));
        a = o(1:3) * pi / 180;
        R(:, :, i) = local_rx(a(1)) * local_ry(a(2)) * local_rz(a(3));
    end
end

function u = local_unit(v)
    u = v / norm(v);
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

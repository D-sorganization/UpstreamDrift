function grip = gs3dx_fit_grip(cap, ik)
%GS3DX_FIT_GRIP  Hand-on-grip geometry of GS3DX_Fit from the capture (#10979).
%
%   GRIP = GS3DX_FIT_GRIP(CAP, IK) estimates where the golfer's wrist joint
%   centres sit on the club, and maps that onto the GS3DX grip variables.
%   CAP is GS3DX_CAPTURE_MARKERS; IK is a GS3DX_WHOLE_BODY_IK result (the
%   club pose it tracked gives the shaft axis).
%
%   1. Club frame.  The two three-marker club clusters are rigid (pairwise
%      distances within 4 mm), so each frame's club pose is the least-
%      squares rigid fit (Kabsch) of the six markers to a reference frame.
%   2. Wrist centres.  Both hands are welded to the club in the model, so
%      each wrist joint centre is a fixed point in the club frame, and the
%      WristTop marker (on the forearm, next to the joint) moves on a sphere
%      about it.  A least-squares sphere fit of each WristTop trajectory in
%      the club frame gives the functional wrist centre.
%   3. Shaft axis.  The model's two hand-sphere centres lie on the grip
%      axis.  They are carried into the club-marker frame for every tracked
%      frame and averaged.  Only the axis is used, not the model's roll of
%      the club about it, which the hands pin down only weakly.
%   4. Mapping.  The wrist centres' positions along the axis give the lead
%      hand position and the hand spacing.  Across the axis, only their
%      separation is identifiable: moving the shaft sideways relative to
%      both wrists is a translation of the club head in the club frame,
%      which the club-head marker offset of the IK absorbs (there is no
%      marker on the shaft axis).  The separation is split equally, one
%      standoff on each side of the grip, as the palms face each other
%      across it; GS3DX_BUILD_FIT flips the lead standoff to match.
%
%   GRIP fields:
%     .vars      model-workspace values (in) for GS3DX_BUILD_FIT:
%                  FitButtToLeadHand      butt end to the lead hand
%                  FitHandSpacing         lead hand to trail hand
%                  FitGripToShaft         trail hand to the shaft (keeps the
%                                         original butt-to-shaft 10.5 in)
%                  FitLeftWristStandoff   half the wrist centres' separation
%                  FitRightWristStandoff  across the axis (each side)
%     .centre    3x2 wrist centres (lead, trail) in the club-marker frame (m)
%     .sphere    per wrist: radius and fit rms (m)
%     .axial     per wrist: along the axis from the model's lead hand (m)
%     .radial    per wrist: distance from the mean model axis (m); not
%                identifiable on its own, reported for diagnosis
%     .across    the wrist centres' separation across the axis (m)
%     .axis_spread  median angle (deg) of the per-frame axis from the mean
%
%   The model's lead hand position comes from IK.model's FitButtToLeadHand
%   when that variable exists, else the original 2.5 in.

    arguments
        cap (1,1) struct
        ik (1,1) struct
    end
    assert(all(isfield(ik, {'frames', 'joint', 'joint_ids', 'model'})), 'gs3dx:grip', ...
        'IK must be a GS3DX_WHOLE_BODY_IK result');
    in = 0.0254;
    [X, ok] = local_club_markers(cap);
    ref = find(ok, 1);
    M0 = X(:, :, ref);
    c0 = mean(M0, 2);
    pose = @(f) local_kabsch(M0, c0, X(:, :, f));

    % 2. Functional wrist centres in the club-marker frame.
    wrists = {cap.marker("LWristTop"), cap.marker("RWristTop")};
    grip.centre = zeros(3, 2);
    grip.sphere = struct('radius', {0, 0}, 'rms', {0, 0});
    for s = 1:2
        P = nan(3, cap.n_frames);
        for f = find(ok & all(~isnan(wrists{s}), 1))
            [R, c] = pose(f);
            P(:, f) = R.' * (wrists{s}(:, f) - c) + c0;
        end
        P = P(:, all(~isnan(P), 1));
        [grip.centre(:, s), grip.sphere(s).radius, grip.sphere(s).rms] = local_sphere(P);
    end

    % 3. Shaft axis (the model's hand-sphere centres) in the club-marker frame.
    [hand, ikf] = local_model_hands(ik);
    S = cap.target_frame;
    origin = local_origin(cap);
    keep = ok(ik.frames(ikf));
    ikf = ikf(keep);
    hand = hand(:, :, keep);
    U = zeros(3, numel(ikf));
    A = zeros(3, numel(ikf));
    for i = 1:numel(ikf)
        [R, c] = pose(ik.frames(ikf(i)));
        lab = S * hand(:, :, i) + origin;   % address target frame -> capture
        U(:, i) = R.' * (lab(:, 2) - lab(:, 1));
        U(:, i) = U(:, i) / norm(U(:, i));
        A(:, i) = R.' * (lab(:, 1) - c) + c0;
    end
    u = mean(U, 2);
    u = u / norm(u);
    a = mean(A, 2);
    grip.axis_spread = median(acosd(min(1, u.' * U)));

    % 4. Axial position and distance from the axis.
    v = grip.centre - a;
    grip.axial = u.' * v;
    r = v - u * grip.axial;
    grip.radial = vecnorm(r);
    grip.across = norm(r(:, 2) - r(:, 1));

    butt = local_model_var(ik.model, 'FitButtToLeadHand', 2.5);
    g.FitButtToLeadHand = butt + grip.axial(1) / in;
    g.FitHandSpacing = (grip.axial(2) - grip.axial(1)) / in;
    g.FitGripToShaft = 10.5 - g.FitButtToLeadHand - g.FitHandSpacing;
    g.FitLeftWristStandoff = grip.across / 2 / in;
    g.FitRightWristStandoff = grip.across / 2 / in;
    grip.vars = g;

    assert(all(structfun(@(x) x > 0, g)), 'gs3dx:grip', ...
        'Postcondition: every grip length is positive (%s)', jsonencode(g));
end

function [X, ok] = local_club_markers(cap)
% The six club markers, 3 x 6 x frames; OK marks frames with all six.
    idx = find(startsWith(cap.labels, ["Marker_2:2:", "Marker_3:3:"]));
    assert(numel(idx) == 6, 'gs3dx:grip', 'Expected six club markers, found %d', numel(idx));
    X = zeros(3, 6, cap.n_frames);
    for k = 1:6
        X(:, k, :) = reshape(cap.marker(cap.labels(idx(k))), 3, 1, []);
    end
    ok = reshape(all(~isnan(X), [1 2]), 1, []);
    assert(nnz(ok) >= 3, 'gs3dx:grip', 'Too few frames with all six club markers');
end

function [R, c] = local_kabsch(M0, c0, M)
% Rotation R and centroid c with M - c = R (M0 - c0) in the least squares.
    c = mean(M, 2);
    [U, ~, V] = svd((M0 - c0) * (M - c).');
    R = V * diag([1 1 det(V * U.')]) * U.';
end

function [c, r, e] = local_sphere(P)
% Algebraic least-squares sphere through the columns of P.
    x = [2 * P.', ones(size(P, 2), 1)] \ sum(P .^ 2, 1).';
    c = x(1:3);
    r = sqrt(x(4) + sum(c .^ 2));
    e = rms(vecnorm(P - c) - r);
end

function origin = local_origin(cap)
% Address waist centre: the origin of GS3DX_CAPTURE_JOINT_CENTRES.
    w = (cap.marker("WaistLeft") + cap.marker("WaistRight") + ...
        cap.marker("WaistLBack") + cap.marker("WaistRBack")) / 4;
    origin = w(:, find(all(~isnan(w), 1), 1));
end

function [hand, ikf] = local_model_hands(ik)
% Lead and trail hand-sphere centres of IK.model (address target frame),
% 3 x 2 x frames, for the IK frames whose grip loop closed.
    mdl = ik.model;
    load_system(mdl);
    wf = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'ReferenceBlock', 'sm_lib/Frames and Transforms/World Frame');
    ks = simscape.multibody.KinematicsSolver(mdl);
    ids = string(ks.jointPositionVariables.ID);
    assert(isequal(ids(:), string(ik.joint_ids(:))), 'gs3dx:grip', 'IK.joint_ids do not match %s', mdl);
    addFrameVariables(ks, 'lead', 'Translation', [wf{1} '/W'], [mdl '/Grip/LHand/R']);
    addFrameVariables(ks, 'trail', 'Translation', [wf{1} '/W'], [mdl '/Grip/RHand/R']);
    closed = startsWith(ids, ["j15.", "j18.", "j19."]);   % the right-arm loop, as in the IK
    addTargetVariables(ks, ids(~closed));
    addOutputVariables(ks, string(ks.frameVariables.ID));
    addInitialGuessVariables(ks, ids(closed));
    ikf = find(ik.status >= 1);
    hand = zeros(3, 2, numel(ikf));
    for i = 1:numel(ikf)
        o = solve(ks, ik.joint(~closed, ikf(i)), ik.joint(closed, ikf(i)));
        hand(:, :, i) = reshape(o(1:6), 3, 2);
    end
end

function v = local_model_var(mdl, name, default)
    ws = get_param(mdl, 'ModelWorkspace');
    v = default;
    if hasVariable(ws, name)
        v = getVariable(ws, name);
    end
end

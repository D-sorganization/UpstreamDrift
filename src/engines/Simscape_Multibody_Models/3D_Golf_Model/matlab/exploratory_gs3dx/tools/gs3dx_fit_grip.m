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
    tf = gs3dx_capture_address_transform(cap);
    [X, ok] = gs3dx_club_markers_address(cap);
    assert(nnz(ok) >= 3, 'gs3dx:grip', 'Too few frames with all six club markers');
    ref = find(ok, 1);
    M0 = X(:, :, ref);
    c0 = mean(M0, 2);
    pose = @(f) gs3dx_kabsch(M0, c0, X(:, :, f));

    % 2. Functional wrist centres in the club-marker frame.
    wrists = {cap.marker("LWristTop"), cap.marker("RWristTop")};
    grip.centre = zeros(3, 2);
    grip.sphere = struct('radius', {0, 0}, 'rms', {0, 0});
    for s = 1:2
        P = nan(3, cap.n_frames);
        for f = find(ok & all(isfinite(wrists{s}), 1))
            [R, c] = pose(f);
            P(:, f) = R.' * (tf.local(wrists{s}(:, f)) - c) + c0;
        end
        P = P(:, all(~isnan(P), 1));
        [grip.centre(:, s), grip.sphere(s).radius, grip.sphere(s).rms] = local_sphere(P);
    end

    % 3. Shaft axis (the model's hand-sphere centres) in the club-marker frame.
    [hand, ikf] = gs3dx_hand_centres(ik);
    keep = ok(ik.frames(ikf));
    ikf = ikf(keep);
    hand = hand(:, :, keep);
    U = zeros(3, numel(ikf));
    A = zeros(3, numel(ikf));
    for i = 1:numel(ikf)
        [R, c] = pose(ik.frames(ikf(i)));
        U(:, i) = R.' * (hand(:, 2, i) - hand(:, 1, i));
        U(:, i) = U(:, i) / norm(U(:, i));
        A(:, i) = R.' * (hand(:, 1, i) - c) + c0;
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

function [c, r, e] = local_sphere(P)
% Algebraic least-squares sphere through the columns of P.
    x = [2 * P.', ones(size(P, 2), 1)] \ sum(P .^ 2, 1).';
    c = x(1:3);
    r = sqrt(x(4) + sum(c .^ 2));
    e = rms(vecnorm(P - c) - r);
end

function v = local_model_var(mdl, name, default)
    ws = get_param(mdl, 'ModelWorkspace');
    v = default;
    if hasVariable(ws, name)
        v = getVariable(ws, name);
    end
end

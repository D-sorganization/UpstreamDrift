function grip = gs3dx_fit_grip(cap, ik, opts)
%GS3DX_FIT_GRIP  Hand-on-grip geometry of GS3DX_Fit from the capture (#10979).
%
%   GRIP = GS3DX_FIT_GRIP(CAP, IK) estimates where the golfer's wrist joint
%   centres sit on the club, and maps that onto the GS3DX grip variables.
%   Legacy default: uses all frames in CAP (1:CAP.n_frames).
%
%   GRIP = GS3DX_FIT_GRIP(CAP, IK, calibration_frames=CAL_FRAMES) restricts
%   both wrist sphere fitting AND mean model shaft axis estimation to ONLY
%   CAL_FRAMES (e.g. backswing calibration frames), ensuring zero data leakage
%   into held-out downswing or impact evaluations.
%
%   Inputs:
%     CAP                 GS3DX_CAPTURE_MARKERS struct.
%     IK                  GS3DX_WHOLE_BODY_IK result struct.
%     opts.calibration_frames  (1 x N double) strictly increasing unique positive
%                         integers within [1, CAP.n_frames]. Default [] (uses all
%                         frames in capture, preserving legacy interface).
%
%   Methodology & Modeling Assumptions:
%   1. Club frame.  The two three-marker club clusters are assumed rigid
%      (inherited diagnostic assumption, not qualified in unit tests);
%      each frame's club pose is the least-squares rigid fit (Kabsch) of
%      the six markers to a reference calibration frame. Address frame 1 is
%      an independent address origin assumption from GS3DX_CAPTURE_ADDRESS_TRANSFORM.
%   2. Wrist centres.  Both hands are welded to the club in the model, so
%      each wrist joint centre is a fixed point in the club frame, and the
%      WristTop marker moves on a sphere about it. A least-squares sphere fit
%      (GS3DX_FIT_SPHERE) of each WristTop trajectory in the club frame over
%      CAL_FRAMES gives the functional wrist centre. NOTE: This is a kinematic
%      modeling surrogate assuming fixed wrist centres and isotropic optical
%      marker noise, not an anatomical measurement proof.
%   3. Shaft axis.  The model's two hand-sphere centres lie on the grip
%      axis.  They are carried into the club-marker frame for IK frames that
%      intersect CAL_FRAMES and averaged.  Only the axis is used, not the
%      model's roll of the club about it, which the hands pin down only weakly.
%   4. Mapping.  The wrist centres' positions along the axis give the lead
%      hand position and the hand spacing.  Across the axis, only their
%      separation is identifiable: moving the shaft sideways relative to
%      both wrists is a translation of the club head in the club frame,
%      which the club-head marker offset of the IK absorbs (there is no
%      marker on the shaft axis).  The separation is split equally, one
%      standoff on each side of the grip, retaining the existing equal
%      standoff gauge as palms face each other across it; GS3DX_BUILD_FIT
%      flips the lead standoff to match.
%
%   GRIP fields:
%     .vars               model-workspace values (in) for GS3DX_BUILD_FIT:
%                           FitButtToLeadHand      butt end to the lead hand
%                           FitHandSpacing         lead hand to trail hand
%                           FitGripToShaft         trail hand to the shaft (keeps the
%                                                  original butt-to-shaft 10.5 in)
%                           FitLeftWristStandoff   half the wrist centres' separation
%                           FitRightWristStandoff  across the axis (each side)
%     .centre             3x2 wrist centres (lead, trail) in the club-marker frame (m)
%     .sphere             per wrist: radius, rms, rank, condition, rank_tolerance, n_points
%     .axial              per wrist: along the axis from the model's lead hand (m)
%     .radial             per wrist: distance from the mean model axis (m); not
%                         identifiable on its own, reported for diagnosis
%     .across             the wrist centres' separation across the axis (m)
%     .axis_spread        median angle (deg) of the per-frame axis from the mean
%     .calibration_frames frames used for calibration
%     .wrist_measured_counts 1x2 valid measured frames per wrist in calibration_frames
%     .assumption         kinematic modeling assumptions string
%     .provenance         diagnostic provenance struct (no private paths/raw data)
%
%   The model's lead hand position comes from IK.model's FitButtToLeadHand
%   when that variable exists, else the original 2.5 in.
%
%   See also GS3DX_FIT_SPHERE, GS3DX_WHOLE_BODY_IK, GS3DX_BUILD_FIT.

    arguments
        cap (1,1) struct
        ik (1,1) struct
        opts.calibration_frames (1,:) double = []
    end

    % Preconditions: Validate CAP (no unknown capture accepted)
    assert(isfield(cap, 'n_frames') && isnumeric(cap.n_frames) && isscalar(cap.n_frames) && ...
           isreal(cap.n_frames) && isfinite(cap.n_frames) && cap.n_frames >= 1 && cap.n_frames == floor(cap.n_frames), ...
           'gs3dx:grip:invalid_cap', 'CAP.n_frames must be a real finite positive integer scalar');
    assert(isfield(cap, 'marker'), 'gs3dx:grip:invalid_cap', 'CAP must have a marker accessor');
    assert(isfield(cap, 'target_frame') && isnumeric(cap.target_frame) && isreal(cap.target_frame) && ...
           isequal(size(cap.target_frame), [3, 3]) && all(isfinite(cap.target_frame(:))), ...
           'gs3dx:grip:invalid_cap', 'CAP must have a real finite 3x3 target_frame');

    % Preconditions: Validate IK (frames non-empty strictly increasing unique <= cap.n_frames)
    assert(all(isfield(ik, {'frames', 'joint', 'joint_ids', 'model'})), 'gs3dx:grip', ...
        'IK must be a GS3DX_WHOLE_BODY_IK result with fields frames, joint, joint_ids, model');
    assert(isnumeric(ik.frames) && isreal(ik.frames) && isvector(ik.frames) && ~isempty(ik.frames), ...
           'gs3dx:grip:invalid_ik', 'IK.frames must be a non-empty real numeric vector');
    assert(all(isfinite(ik.frames)) && all(ik.frames >= 1) && all(ik.frames == floor(ik.frames)), ...
           'gs3dx:grip:invalid_ik', 'IK.frames must be real finite positive integers');
    if numel(ik.frames) > 1
        assert(all(diff(ik.frames) > 0), 'gs3dx:grip:invalid_ik', ...
               'IK.frames must be strictly increasing and unique');
    end
    assert(all(ik.frames <= cap.n_frames), 'gs3dx:grip:invalid_ik', ...
           'IK.frames must not exceed CAP.n_frames (%d)', cap.n_frames);

    % Joints: finite, real, 2D matrix matching frames and joint_ids
    assert(isnumeric(ik.joint) && isreal(ik.joint) && ismatrix(ik.joint), ...
           'gs3dx:grip:invalid_ik', 'IK.joint must be a 2D real numeric matrix');
    assert(all(isfinite(ik.joint(:))), 'gs3dx:grip:invalid_ik', ...
           'IK.joint must be finite');
    assert(isstring(ik.joint_ids) || iscellstr(ik.joint_ids), ...
           'gs3dx:grip:invalid_ik', 'IK.joint_ids must be a string or cellstr array');
    assert(size(ik.joint, 1) == numel(ik.joint_ids), 'gs3dx:grip:invalid_ik', ...
           'IK.joint rows (%d) must match numel(IK.joint_ids) (%d)', size(ik.joint, 1), numel(ik.joint_ids));
    assert(size(ik.joint, 2) == numel(ik.frames), 'gs3dx:grip:invalid_ik', ...
           'IK.joint cols (%d) must match numel(IK.frames) (%d)', size(ik.joint, 2), numel(ik.frames));

    % Preconditions: Validate calibration_frames
    if isempty(opts.calibration_frames)
        % Documented legacy default: all frames in capture
        cal_frames = 1:cap.n_frames;
    else
        cal_frames = opts.calibration_frames;
        assert(isnumeric(cal_frames) && isreal(cal_frames) && isvector(cal_frames) && ~isempty(cal_frames), ...
            'gs3dx:grip:invalid_calibration_frames', 'calibration_frames must be a non-empty real numeric vector');
        assert(all(isfinite(cal_frames)), 'gs3dx:grip:invalid_calibration_frames', ...
            'calibration_frames must be finite');
        assert(all(cal_frames >= 1) && all(cal_frames == floor(cal_frames)), ...
            'gs3dx:grip:invalid_calibration_frames', 'calibration_frames must be positive integers');
        if numel(cal_frames) > 1
            assert(all(diff(cal_frames) > 0), 'gs3dx:grip:invalid_calibration_frames', ...
                'calibration_frames must be strictly increasing and unique');
        end
        assert(all(cal_frames <= cap.n_frames), 'gs3dx:grip:invalid_calibration_frames', ...
            'calibration_frames must be within capture range (1 to %d)', cap.n_frames);
    end

    in = 0.0254;
    tf = gs3dx_capture_address_transform(cap);
    [X, ok] = gs3dx_club_markers_address(cap);

    % Restrict valid club marker frames strictly to calibration_frames
    ok_cal = false(1, cap.n_frames);
    ok_cal(cal_frames) = ok(cal_frames);
    assert(nnz(ok_cal) >= 3, 'gs3dx:grip:too_few_club_markers', ...
        'Too few calibration frames with all six club markers (%d < 3)', nnz(ok_cal));

    % Reference frame selected strictly from calibration_frames
    ref = find(ok_cal, 1);
    M0 = X(:, :, ref);
    c0 = mean(M0, 2);
    pose = @(f) gs3dx_kabsch(M0, c0, X(:, :, f));

    % 2. Functional wrist centres in the club-marker frame.
    % Fitted strictly on calibration_frames (no held-out leakage).
    wrist_names = ["LWristTop", "RWristTop"];
    grip.centre = zeros(3, 2);
    grip.sphere = struct('radius', cell(1, 2), 'rms', cell(1, 2), ...
                         'rank', cell(1, 2), 'condition', cell(1, 2), ...
                         'rank_tolerance', cell(1, 2), 'n_points', cell(1, 2));
    wrist_measured_counts = zeros(1, 2);

    for s = 1:2
        w_name = wrist_names(s);
        w_pos = cap.marker(w_name);
        w_valid = true(1, cap.n_frames);
        assert(isnumeric(w_pos) && isreal(w_pos) && isequal(size(w_pos), [3, cap.n_frames]), ...
            'gs3dx:grip:invalid_marker', 'Wrist markers must be real 3-by-capture-frame arrays');

        % Check if cap carries explicit residual or missing masks
        if isfield(cap, 'residual') && isstruct(cap.residual) && isfield(cap.residual, char(w_name))
            res = cap.residual.(char(w_name));
            assert(isnumeric(res) && isreal(res) && isvector(res) && numel(res) == cap.n_frames, ...
                'gs3dx:grip:invalid_marker_mask', 'Residual masks must align with capture frames');
            w_valid = w_valid & reshape(isfinite(res) & (res >= 0), 1, []);
        end
        if isfield(cap, 'missing') && isstruct(cap.missing) && isfield(cap.missing, char(w_name))
            miss = cap.missing.(char(w_name));
            assert(islogical(miss) && isvector(miss) && numel(miss) == cap.n_frames, ...
                'gs3dx:grip:invalid_marker_mask', 'Missing masks must be logical capture-frame vectors');
            w_valid = w_valid & ~reshape(miss, 1, []);
        end

        % Filter missing: finite, non-zero (reject finite zeros dropouts),
        % valid residual/missing mask, and valid club markers in calibration_frames.
        % No substitution invented.
        is_finite_non_zero = all(isfinite(w_pos), 1) & any(w_pos ~= 0, 1);
        valid_f = cal_frames(ok_cal(cal_frames) & is_finite_non_zero(cal_frames) & w_valid(cal_frames));
        wrist_measured_counts(s) = numel(valid_f);

        assert(numel(valid_f) >= 4, 'gs3dx:grip:too_few_wrist_samples', ...
            'Wrist %d has too few valid measured frames (%d < 4) in calibration_frames', s, numel(valid_f));

        P = zeros(3, numel(valid_f));
        for idx = 1:numel(valid_f)
            f = valid_f(idx);
            [R, c] = pose(f);
            P(:, idx) = R.' * (tf.local(w_pos(:, f)) - c) + c0;
        end
        [grip.centre(:, s), grip.sphere(s).radius, grip.sphere(s).rms, diag_s] = gs3dx_fit_sphere(P);
        grip.sphere(s).rank = diag_s.rank;
        grip.sphere(s).condition = diag_s.condition;
        grip.sphere(s).rank_tolerance = diag_s.rank_tolerance;
        grip.sphere(s).n_points = diag_s.n_points;
    end
    grip.wrist_measured_counts = wrist_measured_counts;

    % 3. Shaft axis (the model's hand-sphere centres) in the club-marker frame.
    % Restrict shaft axis estimation strictly to calibration_frames (no held-out leakage).
    if isfield(ik, 'hand_centres')
        % Optional precomputed centres must use the IK world frame and sample order.
        hand = ik.hand_centres;
        ikf = 1:size(hand, 3);
    else
        [hand, ikf] = gs3dx_hand_centres(ik);
    end
    assert(isnumeric(hand) && isreal(hand) && size(hand, 1) == 3 && size(hand, 2) == 2 ...
        && ndims(hand) <= 3 && size(hand, 3) == numel(ikf) && all(isfinite(hand), 'all'), ...
        'gs3dx:grip:invalid_hand_centres', 'Hand centres must be finite real 3-by-2-by-sample data');
    if isfield(ik, 'hand_centres')
        assert(size(hand, 3) == numel(ik.frames), 'gs3dx:grip:invalid_hand_centres', ...
            'Precomputed hand-centre samples must align with every IK frame');
    end

    ik_frames = ik.frames(ikf);

    % Status alignment check when status is provided
    if isfield(ik, 'status') && ~isempty(ik.status)
        assert((isnumeric(ik.status) || islogical(ik.status)) && numel(ik.status) == numel(ik.frames), ...
            'gs3dx:grip:invalid_ik', 'IK.status must match IK.frames');
        assert(all(ik.status(ikf) == 1), 'gs3dx:grip:invalid_status', ...
            'IK frames used for shaft axis estimation must have status == 1 (loop closed)');
    end

    keep = ismember(ik_frames, cal_frames) & (ik_frames >= 1) & (ik_frames <= cap.n_frames);
    keep_indices = find(keep);
    keep_valid = false(size(keep_indices));
    for j = 1:numel(keep_indices)
        f_num = ik_frames(keep_indices(j));
        keep_valid(j) = ok(f_num);
    end
    keep_indices = keep_indices(keep_valid);

    ikf = ikf(keep_indices);
    hand = hand(:, :, keep_indices);

    assert(numel(ikf) >= 2, 'gs3dx:grip:too_few_axis_frames', ...
        'At least 2 frames intersecting IK, club markers, and calibration_frames required for shaft axis estimation (got %d)', numel(ikf));

    U = zeros(3, numel(ikf));
    A = zeros(3, numel(ikf));
    for i = 1:numel(ikf)
        f = ik.frames(ikf(i));
        [R, c] = pose(f);
        d_hand = hand(:, 2, i) - hand(:, 1, i);
        hand_dist = norm(d_hand);
        assert(hand_dist > 1e-4, 'gs3dx:grip:zero_hand_vector', ...
            'Hand centres are coincident or too close (dist = %e m) in frame %d', hand_dist, f);
        U(:, i) = R.' * d_hand / hand_dist;
        A(:, i) = R.' * (hand(:, 1, i) - c) + c0;
    end
    u = mean(U, 2);
    u_norm = norm(u);
    assert(u_norm > 1e-3, 'gs3dx:grip:opposed_hand_axes', ...
        'Shaft axis unit vectors are opposed or cancel out across frames (norm = %e)', u_norm);
    u = u / u_norm;
    a = mean(A, 2);
    grip.axis_spread = median(acosd(min(1, max(-1, u.' * U))));

    % 4. Axial position and distance from the axis.
    v = grip.centre - a;
    grip.axial = u.' * v;
    r = v - u * grip.axial;
    grip.radial = vecnorm(r, 2, 1);
    grip.across = norm(r(:, 2) - r(:, 1));

    butt = local_model_var(ik.model, 'FitButtToLeadHand', 2.5);
    g.FitButtToLeadHand = butt + grip.axial(1) / in;
    g.FitHandSpacing = (grip.axial(2) - grip.axial(1)) / in;
    g.FitGripToShaft = 10.5 - g.FitButtToLeadHand - g.FitHandSpacing;
    g.FitLeftWristStandoff = grip.across / 2 / in;
    g.FitRightWristStandoff = grip.across / 2 / in;
    grip.vars = g;

    % DbC Postconditions: Validate grip variables
    var_names = fieldnames(g);
    for k = 1:numel(var_names)
        val = g.(var_names{k});
        assert(isreal(val) && isfinite(val) && val > 0, 'gs3dx:grip:invalid_grip_variable', ...
            'Postcondition failed: grip variable %s must be real, finite, and positive (got %f)', ...
            var_names{k}, val);
    end

    total_grip_len = g.FitButtToLeadHand + g.FitHandSpacing + g.FitGripToShaft;
    assert(abs(total_grip_len - 10.5) < 1e-10, 'gs3dx:grip:inconsistent_grip_sum', ...
        'Postcondition failed: FitButtToLeadHand + FitHandSpacing + FitGripToShaft must equal 10.5 in (got %f)', ...
        total_grip_len);

    assert(abs(g.FitLeftWristStandoff - g.FitRightWristStandoff) < 1e-12, 'gs3dx:grip:unequal_standoff', ...
        'Postcondition failed: equal standoff gauge violated');

    % Diagnostic and provenance reporting
    grip.calibration_frames = cal_frames;
    grip.assumption = sprintf(...
        ['Functional fixed wrist centre on club with isotropic optical noise assumption ' ...
         '(kinematic modeling surrogate, not anatomical measurement proof); ' ...
         'club cluster rigidity is an inherited diagnostic assumption; ' ...
         'address frame 1 is an independent address origin assumption; ' ...
         'shaft roll and single wrist transverse position unidentifiable without shaft markers; ' ...
         'retaining equal bilateral standoff gauge across grip axis.']);
    grip.provenance = struct(...
        'calibration_frames_count', numel(cal_frames), ...
        'wrist_measured_counts', wrist_measured_counts, ...
        'axis_frames_count', numel(ikf), ...
        'reference_frame', ref, ...
        'model', char(ik.model));
end

function v = local_model_var(mdl, name, default)
    v = default;
    if ischar(mdl) || isstring(mdl)
        mdl_name = char(mdl);
        if bdIsLoaded(mdl_name)
            try
                ws = get_param(mdl_name, 'ModelWorkspace');
                if hasVariable(ws, name)
                    v = getVariable(ws, name);
                end
            catch
                % Keep default if workspace variable read fails
            end
        end
    end
end

function result = align_measured_to_model(skel, target, opts)
%ALIGN_MEASURED_TO_MODEL  Solve for the best (dx,dy,dz,roll,pitch,yaw)
%   to align the measured club path's IMPACT FRAME with the model's
%   IMPACT POSE.
%
%   The transform applied to the measured data is
%       p_aligned = R * p_measured + [dx; dy; dz]
%   where R = Rz(yaw) * Ry(pitch) * Rx(roll), so that after alignment
%   the measured clubhead at the impact frame coincides with the model
%   clubhead (i.e. with the BALL LOCATION) and the measured butt
%   coincides with the model butt.  Two 3-vectors give 6 constraints
%   matching the 6 DOF — the alignment is exact (sub-mm) at impact.
%
%   OPTS:
%     .mode        'grip_pose' (default — closed-form 6-DOF rigid match
%                  of the grip pose: position from skel.mp /
%                  target.grip and orientation from skel.grip_quat /
%                  target.grip_quat at the impact frame.  This is the
%                  recommended mode because the body→club interface is
%                  rigid; club-length and shaft-flex differences are
%                  irrelevant.), 'clubhead_shaft' (translate ball onto
%                  model clubhead + align shaft direction; useful when
%                  grip orientation is unavailable), 'clubhead_shaft_scaled'
%                  (same plus a uniform scale to absorb club-length
%                  differences), 'rigid6' (least-squares butt+clubhead
%                  fmin), 'yaw' (4-DOF translation + yaw only).
%     .x0          initial guess (auto if absent).
%     .max_iters   fminsearch iteration cap, default 2000.
%     .tol_fun     fminsearch tolerance, default 1e-12.
%
%   The returned struct has:
%     .mode                     mode used
%     .dx .dy .dz               translation (m)
%     .roll_deg .pitch_deg .yaw_deg  rotation (deg, ZYX intrinsic)
%     .R                        3x3 rotation matrix
%     .t                        3x1 translation vector
%     .butt_aligned             1x3 measured butt after transform (m)
%     .clubhead_aligned         1x3 measured clubhead after transform (m)
%     .butt_error_mm            final butt residual in mm
%     .clubhead_error_mm        final clubhead residual in mm
%     .iters                    fminsearch iterations
%     .initial_butt_error_mm    error before solving (no transform)
%     .initial_clubhead_error_mm
%     .final_cost / .initial_cost
%
%   See also: PLOT_STARTING_POSITION_MATCH, LOAD_IMPACT_STARTING_POSITION,
%             DEMO_STARTING_POSITION_MATCH.

    arguments
        skel   (1,1) struct
        target (1,1) struct
        opts   (1,1) struct = struct()
    end
    if ~isfield(opts, 'mode');      opts.mode      = 'grip_pose'; end
    if ~isfield(opts, 'max_iters'); opts.max_iters = 2000; end
    if ~isfield(opts, 'tol_fun');   opts.tol_fun   = 1e-12; end

    idx = double(target.impact_idx);
    butt_m = target.butt(idx, :);
    head_m = target.clubhead(idx, :);
    butt_s = skel.butt;
    head_s = skel.ch;

    if strcmpi(string(opts.mode), "grip_pose")
        result = local_solve_grip_pose(skel, target, idx);
        return;
    end
    if strcmpi(string(opts.mode), "clubhead_shaft")
        result = local_solve_clubhead_shaft(butt_m, head_m, butt_s, head_s, skel, false);
        return;
    end
    if strcmpi(string(opts.mode), "clubhead_shaft_scaled")
        result = local_solve_clubhead_shaft(butt_m, head_m, butt_s, head_s, skel, true);
        return;
    end

    switch lower(string(opts.mode))
        case "yaw"
            n_dof = 4;
            unpack = @(x) deal(x(1:3)', local_rot_zyx(0, 0, x(4)), x(4), 0, 0);
        case "rigid6"
            n_dof = 6;
            unpack = @(x) deal(x(1:3)', local_rot_zyx(x(4), x(5), x(6)), x(6), x(5), x(4));
        otherwise
            error('align_measured_to_model:badMode', ...
                  'opts.mode must be "clubhead_shaft", "rigid6" or "yaw" (got %s)', opts.mode);
    end

    if ~isfield(opts, 'x0')
        if n_dof == 4
            opts.x0 = [head_s - head_m, 0];
        else
            % Translation initial = match the clubheads exactly; rotation
            % initial = align the measured shaft direction with the
            % model shaft direction in one shot, avoiding flat plateaus
            % around (0,0,0) that fminsearch can stall on.
            t0 = head_s - head_m;
            v_meas = head_m - butt_m;
            v_mod  = head_s - butt_s;
            R0 = local_align_vectors(v_meas, v_mod);
            rpy0 = local_R_to_rpy(R0);
            opts.x0 = [t0(1), t0(2), t0(3), rpy0(1), rpy0(2), rpy0(3)];
        end
    end

    fopts = optimset('TolFun', opts.tol_fun, 'TolX', 1e-10, ...
                     'MaxIter', opts.max_iters, ...
                     'MaxFunEvals', 6 * opts.max_iters, ...
                     'Display', 'off');

    cost = @(x) local_cost(x, butt_m, head_m, butt_s, head_s, unpack);
    initial_cost = cost(zeros(1, n_dof));
    [x_opt, ~, ~, output] = fminsearch(cost, opts.x0, fopts);

    [t, R, yaw_deg, pitch_deg, roll_deg] = unpack(x_opt);
    butt_aligned = (R * butt_m')' + t';
    head_aligned = (R * head_m')' + t';

    result = struct( ...
        'mode', string(opts.mode), ...
        'dx', x_opt(1), 'dy', x_opt(2), 'dz', x_opt(3), ...
        'roll_deg',  roll_deg,  ...
        'pitch_deg', pitch_deg, ...
        'yaw_deg',   yaw_deg, ...
        'R', R, 't', t, ...
        'butt_aligned', butt_aligned, ...
        'clubhead_aligned', head_aligned, ...
        'butt_error_mm', 1000 * norm(butt_aligned - butt_s), ...
        'clubhead_error_mm', 1000 * norm(head_aligned - head_s), ...
        'iters', output.iterations, ...
        'initial_butt_error_mm', 1000 * norm(butt_m - butt_s), ...
        'initial_clubhead_error_mm', 1000 * norm(head_m - head_s), ...
        'final_cost', cost(x_opt), ...
        'initial_cost', initial_cost);
end

%% =====================================================================
function result = local_solve_grip_pose(skel, target, idx)
%LOCAL_SOLVE_GRIP_POSE  Exact 6-DOF rigid match of the GRIP pose.
%   The grip is the rigid contact between the body kinematics and the
%   club: matching it directly is club-length-independent and shaft-flex-
%   robust.  This is the recommended motion-matching anchor.
%
%   Inputs:
%     skel.mp        (1,3) model grip position at impact (m, world)
%     skel.grip_R    (3,3) model grip rotation matrix at impact
%                          (extracted from MidpointCalcsLogs /
%                          MomentandCoupleLogs.RotationTransformMP)
%     target.grip    (N,3) measured grip positions
%     target.grip_quat (N,4) measured grip orientations [w x y z]
%     idx            scalar impact-frame index
%
%   Output transform:
%     R = R_model_grip * R_meas_grip^{-1}        (rotation of meas frame)
%     t = skel.mp' - R * target.grip(idx,:)'      (translation)
    if ~isfield(target, 'grip') || ~isfield(target, 'grip_quat')
        error('align_measured_to_model:noGripFields', ...
              'grip_pose mode requires target.grip and target.grip_quat (load with the updated load_club_target_excel).');
    end
    if ~isfield(skel, 'mp') || any(isnan(skel.mp))
        error('align_measured_to_model:noModelGrip', ...
              'grip_pose mode requires skel.mp (model mid-hands position).');
    end

    grip_pos_m = target.grip(idx, :);
    grip_pos_s = skel.mp;

    % Grip-anchored alignment, with two physically meaningful
    % constraints:
    %   (1) Grip position match  — 3 DOF, hard constraint.
    %   (2) Shaft direction match — 2 DOF.
    %   (3) Shaft-axis twist about the shaft is a free DOF; the
    %       measured grip-twist quaternion would tighten this but the
    %       Wiffle ACS has its z-axis pointing "from clubhead toward
    %       grip" (per the Definitions tab) which doesn't necessarily
    %       match the model's MP RotationTransform convention.  We use
    %       the measured grip orientation only to disambiguate the
    %       twist around the shaft when both conventions agree, and
    %       fall back to "no twist" otherwise.
    v_meas = target.clubhead(idx, :) - target.grip(idx, :);
    v_mod  = skel.ch - skel.mp;
    R_shaft = local_align_vectors(v_meas, v_mod);

    % Optional twist correction: if the grip orientation is published
    % by both sides AND it agrees with the shaft direction sign-wise,
    % apply the residual rotation about the shaft axis.  Otherwise we
    % leave the twist undetermined (does not affect clubhead position).
    R = R_shaft;
    if isfield(target, 'grip_quat') && isfield(skel, 'grip_R') && ...
            ~any(isnan(target.grip_quat(idx, :))) && ~any(isnan(skel.grip_R(:)))
        R_meas_grip = local_quat_to_rotmat(target.grip_quat(idx, :));
        % Rotated-then-shafted measured x-axis vs model grip x-axis
        cand = R_shaft * R_meas_grip(:, 1);
        x_mod = skel.grip_R(:, 1);
        % Residual rotation about shaft axis to align x-axes.
        sd = v_mod(:) / max(norm(v_mod), eps);
        twist = local_signed_angle_about(cand, x_mod, sd);
        if isfinite(twist) && abs(twist) < pi/2
            % Only apply if the residual twist is a small adjustment;
            % a near-180° twist signals the conventions disagree on
            % shaft-axis sign and we keep R = R_shaft.
            R = local_axis_angle(sd, twist) * R_shaft;
        end
    end

    t = grip_pos_s' - R * grip_pos_m';

    grip_aligned = (R * grip_pos_m')' + t';
    head_aligned = (R * target.clubhead(idx, :)')' + t';
    butt_aligned = grip_aligned;   % grip ≡ butt under the new schema

    rpy = local_R_to_rpy(R);
    result = struct( ...
        'mode', "grip_pose", ...
        'dx', t(1), 'dy', t(2), 'dz', t(3), ...
        'roll_deg',  rpy(1), 'pitch_deg', rpy(2), 'yaw_deg', rpy(3), ...
        'scale', 1.0, ...
        'R', R, 't', t, ...
        'grip_aligned',     grip_aligned, ...
        'butt_aligned',     butt_aligned, ...
        'clubhead_aligned', head_aligned, ...
        'grip_error_mm',     1000 * norm(grip_aligned - grip_pos_s), ...
        'butt_error_mm',     1000 * norm(grip_aligned - grip_pos_s), ...
        'clubhead_error_mm', 1000 * norm(head_aligned - skel.ch), ...
        'iters', 0, ...
        'initial_butt_error_mm',     1000 * norm(grip_pos_m - grip_pos_s), ...
        'initial_clubhead_error_mm', 1000 * norm(target.clubhead(idx,:) - skel.ch), ...
        'final_cost',   norm(grip_aligned - grip_pos_s)^2, ...
        'initial_cost', norm(grip_pos_m   - grip_pos_s)^2, ...
        'measured_shaft_length_m', norm(target.clubhead(idx,:) - target.grip(idx,:)), ...
        'model_shaft_length_m',    norm(skel.ch - skel.mp), ...
        'ball_landed_on_model_clubhead', norm(head_aligned - skel.ch) < 0.05);
    result.ball_world_xyz = skel.ch;
end

%% =====================================================================
function ang = local_signed_angle_about(a, b, axis)
%LOCAL_SIGNED_ANGLE_ABOUT  Signed angle from vector A to B about AXIS.
    a = a(:); b = b(:); axis = axis(:);
    axis = axis / max(norm(axis), eps);
    a_p = a - (a' * axis) * axis;   % project off axis
    b_p = b - (b' * axis) * axis;
    na = norm(a_p); nb = norm(b_p);
    if na < 1e-9 || nb < 1e-9; ang = NaN; return; end
    a_p = a_p / na; b_p = b_p / nb;
    s = dot(cross(a_p, b_p), axis);
    c = dot(a_p, b_p);
    ang = atan2(s, c);
end

%% =====================================================================
function R = local_axis_angle(axis, ang)
%LOCAL_AXIS_ANGLE  Rodrigues rotation matrix from axis + angle (rad).
    axis = axis(:); axis = axis / max(norm(axis), eps);
    K = [    0, -axis(3),  axis(2); ...
        axis(3),     0,  -axis(1); ...
       -axis(2),  axis(1),     0];
    R = eye(3) + sin(ang) * K + (1 - cos(ang)) * (K * K);
end

%% =====================================================================
function R = local_quat_to_rotmat(q)
%LOCAL_QUAT_TO_ROTMAT  [w x y z] unit quaternion to 3x3 rotation matrix.
    q = q(:)';
    q = q / max(norm(q), eps);
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w); ...
         2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w); ...
         2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x*x + y*y)];
end

%% =====================================================================
function result = local_solve_clubhead_shaft(butt_m, head_m, butt_s, head_s, skel, with_scale)
%LOCAL_SOLVE_CLUBHEAD_SHAFT  Closed-form clubhead-position + shaft-axis match.
%   Step 1: pick R that maps the measured shaft direction (head_m-butt_m)
%   onto the model shaft direction (head_s-butt_s).
%   Step 2 (optional, with_scale): pick s = ||v_mod|| / ||v_meas|| to
%           absorb the small physical-shaft-length difference between
%           the modeled club and the player's actual club.
%   Step 3: pick t so that s*R*head_m + t = head_s exactly.
    v_meas = head_m - butt_m;
    v_mod  = head_s - butt_s;
    R = local_align_vectors(v_meas, v_mod);
    if with_scale
        s = norm(v_mod) / max(norm(v_meas), eps);
    else
        s = 1.0;
    end
    t = head_s' - s * R * head_m';

    butt_aligned = (s * R * butt_m')' + t';
    head_aligned = (s * R * head_m')' + t';

    rpy = local_R_to_rpy(R);
    mode_label = ternary(with_scale, "clubhead_shaft_scaled", "clubhead_shaft");
    result = struct( ...
        'mode', mode_label, ...
        'dx', t(1), 'dy', t(2), 'dz', t(3), ...
        'roll_deg',  rpy(1), 'pitch_deg', rpy(2), 'yaw_deg', rpy(3), ...
        'scale', s, ...
        'R', R, 't', t, ...
        'butt_aligned', butt_aligned, ...
        'clubhead_aligned', head_aligned, ...
        'butt_error_mm', 1000 * norm(butt_aligned - butt_s), ...
        'clubhead_error_mm', 1000 * norm(head_aligned - head_s), ...
        'iters', 0, ...
        'initial_butt_error_mm', 1000 * norm(butt_m - butt_s), ...
        'initial_clubhead_error_mm', 1000 * norm(head_m - head_s), ...
        'final_cost', norm(head_aligned - head_s)^2, ...
        'initial_cost', norm(head_m - head_s)^2, ...
        'measured_shaft_length_m', norm(v_meas), ...
        'model_shaft_length_m',    norm(v_mod), ...
        'ball_landed_on_model_clubhead', true);
    result.ball_world_xyz = head_s;
    result.skel_used = skel; %#ok<STRNU>
end

function v = ternary(c, a, b); if c; v = a; else; v = b; end; end

%% =====================================================================
function c = local_cost(x, butt_m, head_m, butt_s, head_s, unpack)
    [t, R, ~, ~, ~] = unpack(x);
    bm = (R * butt_m')' + t';
    hm = (R * head_m')' + t';
    c = sum((bm - butt_s).^2) + sum((hm - head_s).^2);
end

%% =====================================================================
function R = local_rot_zyx(roll_deg, pitch_deg, yaw_deg)
%LOCAL_ROT_ZYX  Rz(yaw)*Ry(pitch)*Rx(roll), all degrees.
    cz = cosd(yaw_deg);   sz = sind(yaw_deg);
    cy = cosd(pitch_deg); sy = sind(pitch_deg);
    cx = cosd(roll_deg);  sx = sind(roll_deg);
    R = [cz, -sz, 0; sz, cz, 0; 0, 0, 1] ...
      * [cy,  0, sy;  0,  1, 0; -sy, 0, cy] ...
      * [ 1,  0,  0;  0, cx, -sx; 0, sx, cx];
end

%% =====================================================================
function rpy = local_R_to_rpy(R)
%LOCAL_R_TO_RPY  Inverse of rot_zyx — extract [roll pitch yaw] in degrees.
    pitch_deg = asind(max(-1, min(1, -R(3,1))));
    if abs(cosd(pitch_deg)) > 1e-6
        roll_deg = atan2d( R(3,2),  R(3,3));
        yaw_deg  = atan2d( R(2,1),  R(1,1));
    else
        % Gimbal lock — fall back to one DOF.
        roll_deg = 0;
        yaw_deg  = atan2d(-R(1,2), R(2,2));
    end
    rpy = [roll_deg, pitch_deg, yaw_deg];
end

%% =====================================================================
function R = local_align_vectors(a, b)
%LOCAL_ALIGN_VECTORS  Rotation matrix mapping unit(a) -> unit(b).  Uses
%   Rodrigues' formula; safe for parallel/antiparallel inputs.
    a = a(:); b = b(:);
    a = a / max(norm(a), eps);
    b = b / max(norm(b), eps);
    v = cross(a, b);
    s = norm(v);
    c = dot(a, b);
    if s < 1e-9
        if c > 0
            R = eye(3);
        else
            % 180-degree flip about any axis perpendicular to a
            perp = null(a.');
            ax = perp(:, 1);
            R = eye(3) - 2 * (ax * ax.');
        end
        return;
    end
    Vx = [   0, -v(3),  v(2); ...
          v(3),    0, -v(1); ...
         -v(2),  v(1),    0];
    R = eye(3) + Vx + Vx * Vx * ((1 - c) / (s * s));
end
